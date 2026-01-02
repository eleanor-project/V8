"""
ReplayStore — persistent store for deliberation inputs and outputs.

Provides:
 - append-only JSONL log for replay
 - in-memory index for fast lookup
 - retrieval by trace_id
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, cast
from types import ModuleType
from uuid import uuid4

_fcntl: Optional[ModuleType]
try:
    import fcntl as _fcntl  # type: ignore
except Exception:  # pragma: no cover - platform specific
    _fcntl = None

fcntl: Optional[ModuleType] = _fcntl


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


class ReplayStore:
    def __init__(self, path: str = "replay_log.jsonl", max_cache: int = 1000):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_cache = max_cache
        self._lock = threading.Lock()
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        trace_id = item.get("trace_id")
                        if trace_id:
                            self._cache[trace_id] = item
                    except json.JSONDecodeError:
                        continue
        except Exception:
            # On load failure, start with empty cache but keep file intact.
            self._cache = {}

    def save(self, record: Dict[str, Any]) -> None:
        trace_id = record.get("trace_id")
        if not trace_id:
            return
        text = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                if fcntl is not None:
                    fcntl.flock(f, fcntl.LOCK_EX)
                f.write(text + "\n")
                if fcntl is not None:
                    fcntl.flock(f, fcntl.LOCK_UN)
            self._cache[trace_id] = record
            # prune cache if oversized
            if len(self._cache) > self.max_cache:
                # drop arbitrary first item
                first_key = next(iter(self._cache.keys()))
                self._cache.pop(first_key, None)

    def get(self, trace_id: str) -> Optional[Dict[str, Any]]:
        if trace_id in self._cache:
            return self._cache[trace_id]

        # fallback to file scan
        if not self.path.exists():
            return None
        with self._lock:
            try:
                with self.path.open("r", encoding="utf-8") as f:
                    if fcntl is not None:
                        fcntl.flock(f, fcntl.LOCK_SH)
                    for line in f:
                        try:
                            item = cast(Dict[str, Any], json.loads(line))
                        except json.JSONDecodeError:
                            continue
                        if item.get("trace_id") == trace_id:
                            self._cache[trace_id] = item
                            if fcntl is not None:
                                fcntl.flock(f, fcntl.LOCK_UN)
                            return item
                    if fcntl is not None:
                        fcntl.flock(f, fcntl.LOCK_UN)
            except FileNotFoundError:
                return None
        return None


def store_review_packet(packet: Any) -> Optional[str]:
    """
    Stores immutable review packets for human adjudication.
    These are NEVER modified once written.
    """
    record = packet.dict() if hasattr(packet, "dict") else dict(packet)
    record["stored_at"] = datetime.utcnow().isoformat()
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
    record = review_record.dict() if hasattr(review_record, "dict") else dict(review_record)
    record["stored_at"] = datetime.utcnow().isoformat()

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
