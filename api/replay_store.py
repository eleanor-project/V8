"""
ReplayStore — persistent store for deliberation inputs and outputs.

Provides:
 - append-only JSONL log for replay
 - in-memory index for fast lookup
 - retrieval by trace_id
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, cast
from uuid import uuid4

_fcntl: Optional[ModuleType]
try:
    import fcntl as _fcntl  # type: ignore
except Exception:  # pragma: no cover - platform specific
    _fcntl = None

fcntl: Optional[ModuleType] = _fcntl

logger = logging.getLogger(__name__)


REVIEW_PACKET_DIR = "logs/review_packets"
REVIEW_RECORD_DIR = "logs/reviews"

os.makedirs(REVIEW_PACKET_DIR, exist_ok=True)
os.makedirs(REVIEW_RECORD_DIR, exist_ok=True)


def _atomic_write_json(path: str, record: Dict[str, Any]) -> None:
    """Write JSON to path atomically to avoid partial writes."""
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(record, f, indent=2, default=str)
    os.replace(tmp_path, path)


def _as_dict(record: Any) -> Dict[str, Any]:
    if hasattr(record, "model_dump"):
        return cast(Dict[str, Any], record.model_dump())
    if hasattr(record, "dict"):
        return cast(Dict[str, Any], record.dict())
    return dict(record)


def _env_int(name: str) -> Optional[int]:
    raw = os.getenv(name)
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


class ReplayStore:
    def __init__(
        self,
        path: str = "replay_log.jsonl",
        max_cache: int = 1000,
        max_log_bytes: Optional[int] = None,
        max_index_entries: Optional[int] = None,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_cache = max_cache
        env_max_bytes = _env_int("REPLAY_LOG_MAX_BYTES")
        if max_log_bytes is None:
            max_log_bytes = env_max_bytes
        self.max_log_bytes = max_log_bytes
        env_index_entries = _env_int("REPLAY_INDEX_MAX")
        if max_index_entries is None:
            max_index_entries = env_index_entries
        self.max_index_entries = max_index_entries or max_cache
        if self.max_index_entries < max_cache:
            self.max_index_entries = max_cache
        self._lock = threading.Lock()
        self._cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._offsets: "OrderedDict[str, int]" = OrderedDict()
        self._file_size = 0
        self._trimmed_records = 0
        self._load_existing()

    def _load_existing(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as f:
                while True:
                    offset = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    trace_id = item.get("trace_id")
                    if trace_id:
                        self._touch_cache_locked(trace_id, item)
                        self._touch_index_locked(trace_id, offset)
            try:
                self._file_size = self.path.stat().st_size
            except OSError:
                self._file_size = 0
        except Exception:
            # On load failure, start with empty cache but keep file intact.
            self._cache = OrderedDict()
            self._offsets = OrderedDict()
            self._file_size = 0

    def _touch_cache_locked(self, trace_id: str, record: Dict[str, Any]) -> None:
        self._cache[trace_id] = record
        self._cache.move_to_end(trace_id)
        self._prune_cache_locked()

    def _touch_index_locked(self, trace_id: str, offset: int) -> None:
        self._offsets[trace_id] = offset
        self._offsets.move_to_end(trace_id)
        self._prune_index_locked()

    def _prune_cache_locked(self) -> None:
        while len(self._cache) > self.max_cache:
            self._cache.popitem(last=False)

    def _prune_index_locked(self) -> None:
        while len(self._offsets) > self.max_index_entries:
            self._offsets.popitem(last=False)

    @property
    def trimmed_records(self) -> int:
        return self._trimmed_records

    def _maybe_trim_log_locked(self) -> None:
        if not self.max_log_bytes:
            return
        if self._file_size <= self.max_log_bytes:
            return
        self._rewrite_log_locked()

    def _rewrite_log_locked(self) -> None:
        records = list(self._cache.values())
        if not records:
            try:
                self.path.write_text("", encoding="utf-8")
            except Exception:
                pass
            self._offsets = OrderedDict()
            self._file_size = 0
            return

        max_bytes = self.max_log_bytes
        kept: List[tuple[Dict[str, Any], str]] = []
        total = 0
        oversized_skipped = 0
        for record in reversed(records):
            line = json.dumps(record, ensure_ascii=False)
            size = len((line + "\n").encode("utf-8"))
            if max_bytes is not None and size > max_bytes:
                oversized_skipped += 1
                continue
            if max_bytes is not None and total + size > max_bytes and kept:
                break
            kept.append((record, line))
            total += size
        kept.reverse()

        tmp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        new_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        new_offsets: "OrderedDict[str, int]" = OrderedDict()
        with tmp_path.open("w", encoding="utf-8") as f:
            for record, line in kept:
                offset = f.tell()
                f.write(line + "\n")
                trace_id = record.get("trace_id")
                if trace_id:
                    new_cache[trace_id] = record
                    new_offsets[trace_id] = offset

        os.replace(tmp_path, self.path)
        self._cache = new_cache
        self._offsets = new_offsets
        try:
            self._file_size = self.path.stat().st_size
        except OSError:
            self._file_size = 0
        trimmed = len(records) - len(kept)
        if max_bytes is not None and trimmed > 0:
            self._trimmed_records += trimmed
            logger.info(
                "replay_store_trimmed_log",
                extra={
                    "trimmed_records": trimmed,
                    "oversized_skipped": oversized_skipped,
                    "max_log_bytes": max_bytes,
                },
            )

    def save(self, record: Dict[str, Any]) -> None:
        trace_id = record.get("trace_id")
        if not trace_id:
            return
        text = json.dumps(record, ensure_ascii=False)
        payload = text + "\n"
        payload_size = len(payload.encode("utf-8"))
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                if fcntl is not None:
                    fcntl.flock(f, fcntl.LOCK_EX)
                offset = f.tell()
                f.write(payload)
                if fcntl is not None:
                    fcntl.flock(f, fcntl.LOCK_UN)
            self._file_size += payload_size
            self._touch_cache_locked(trace_id, record)
            self._touch_index_locked(trace_id, offset)
            self._maybe_trim_log_locked()

    async def save_async(self, record: Dict[str, Any]) -> None:
        await asyncio.to_thread(self.save, record)

    def get(self, trace_id: str) -> Optional[Dict[str, Any]]:
        cached = self._cache.get(trace_id)
        if cached is not None:
            return cached

        if not self.path.exists():
            return None

        with self._lock:
            cached = self._cache.get(trace_id)
            if cached is not None:
                return cached

            offset = self._offsets.get(trace_id)
            if offset is not None:
                item = self._read_at_offset_locked(trace_id, offset)
                if item is not None:
                    return item

            try:
                with self.path.open("r", encoding="utf-8") as f:
                    if fcntl is not None:
                        fcntl.flock(f, fcntl.LOCK_SH)
                    while True:
                        offset = f.tell()
                        line = f.readline()
                        if not line:
                            break
                        try:
                            item = cast(Dict[str, Any], json.loads(line))
                        except json.JSONDecodeError:
                            continue
                        if item.get("trace_id") == trace_id:
                            self._touch_cache_locked(trace_id, item)
                            self._touch_index_locked(trace_id, offset)
                            if fcntl is not None:
                                fcntl.flock(f, fcntl.LOCK_UN)
                            return item
                    if fcntl is not None:
                        fcntl.flock(f, fcntl.LOCK_UN)
            except FileNotFoundError:
                return None
        return None

    async def get_async(self, trace_id: str) -> Optional[Dict[str, Any]]:
        return await asyncio.to_thread(self.get, trace_id)

    def _read_at_offset_locked(self, trace_id: str, offset: int) -> Optional[Dict[str, Any]]:
        try:
            with self.path.open("r", encoding="utf-8") as f:
                if fcntl is not None:
                    fcntl.flock(f, fcntl.LOCK_SH)
                f.seek(offset)
                line = f.readline()
                if fcntl is not None:
                    fcntl.flock(f, fcntl.LOCK_UN)
            item = cast(Dict[str, Any], json.loads(line))
            if item.get("trace_id") != trace_id:
                return None
            self._touch_cache_locked(trace_id, item)
            self._touch_index_locked(trace_id, offset)
            return item
        except Exception:
            return None


def store_review_packet(packet: Any) -> Optional[str]:
    """
    Stores immutable review packets for human adjudication.
    These are NEVER modified once written.
    """
    record = _as_dict(packet)
    record["stored_at"] = datetime.now(timezone.utc).isoformat()
    record["packet_id"] = str(uuid4())

    case_id = record.get("case_id")
    if not case_id:
        return None

    path = os.path.join(REVIEW_PACKET_DIR, f"{case_id}_{record['packet_id']}.json")
    _atomic_write_json(path, record)

    return path


def store_human_review(review_record: Any) -> Optional[str]:
    """
    Stores completed human reviews.
    These are append-only governance artifacts.
    """
    record = _as_dict(review_record)
    record["stored_at"] = datetime.now(timezone.utc).isoformat()

    review_id = record.get("review_id")
    if not review_id:
        return None

    path = os.path.join(REVIEW_RECORD_DIR, f"{review_id}.json")
    _atomic_write_json(path, record)

    return path


def load_review_packet(case_id: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(REVIEW_PACKET_DIR):
        return None

    latest: Optional[Dict[str, Any]] = None
    latest_ts: Optional[float] = None

    for fname in os.listdir(REVIEW_PACKET_DIR):
        if not fname.endswith(".json"):
            continue
        if not (fname == f"{case_id}.json" or fname.startswith(f"{case_id}_")):
            continue

        path = os.path.join(REVIEW_PACKET_DIR, fname)
        try:
            with open(path, "r") as f:
                record = json.load(f)
        except Exception as exc:
            print(f"[REPLAY STORE] Failed to load review packet {path}: {exc}")
            continue

        ts_val = record.get("stored_at")
        try:
            ts = datetime.fromisoformat(ts_val).timestamp() if ts_val else os.path.getmtime(path)
        except Exception:
            ts = os.path.getmtime(path)

        if latest_ts is None or ts > latest_ts:
            latest_ts = ts
            latest = record

    return latest


def list_review_packets(case_id: str) -> List[Dict[str, Any]]:
    """
    Return all review packets for a given case_id (most recent first).
    """
    if not os.path.exists(REVIEW_PACKET_DIR):
        return []

    packets: List[Dict[str, Any]] = []
    for fname in os.listdir(REVIEW_PACKET_DIR):
        if not fname.endswith(".json"):
            continue
        if not (fname == f"{case_id}.json" or fname.startswith(f"{case_id}_")):
            continue
        path = os.path.join(REVIEW_PACKET_DIR, fname)
        try:
            with open(path, "r") as f:
                record = json.load(f)
                packets.append(record)
        except Exception as exc:
            print(f"[REPLAY STORE] Failed to load review packet {path}: {exc}")
            continue

    packets.sort(key=lambda r: r.get("stored_at") or "", reverse=True)
    return packets


def load_human_reviews(case_id: str) -> List[Dict[str, Any]]:
    reviews: List[Dict[str, Any]] = []

    if not os.path.exists(REVIEW_RECORD_DIR):
        return reviews

    for fname in os.listdir(REVIEW_RECORD_DIR):
        path = os.path.join(REVIEW_RECORD_DIR, fname)
        with open(path, "r") as f:
            review = json.load(f)
            if review.get("case_id") == case_id:
                reviews.append(review)

    return reviews


__all__ = [
    "ReplayStore",
    "store_review_packet",
    "store_human_review",
    "load_review_packet",
    "list_review_packets",
    "load_human_reviews",
    "REVIEW_PACKET_DIR",
    "REVIEW_RECORD_DIR",
]
