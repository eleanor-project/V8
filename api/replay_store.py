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
from typing import Any, Dict, Optional
from uuid import uuid4


REVIEW_PACKET_DIR = "logs/review_packets"
REVIEW_RECORD_DIR = "logs/reviews"

os.makedirs(REVIEW_PACKET_DIR, exist_ok=True)
os.makedirs(REVIEW_RECORD_DIR, exist_ok=True)


class ReplayStore:
    def __init__(self, path: str = "replay_log.jsonl", max_cache: int = 1000):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_cache = max_cache
        self._lock = threading.Lock()
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._load_existing()

    def _load_existing(self):
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
                f.write(text + "\n")
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
                    for line in f:
                        try:
                            item = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if item.get("trace_id") == trace_id:
                            self._cache[trace_id] = item
                            return item
            except FileNotFoundError:
                return None
        return None


def store_review_packet(packet):
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

    path = os.path.join(REVIEW_PACKET_DIR, f"{case_id}.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2, default=str)

    return path


def store_human_review(review_record):
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
    with open(path, "w") as f:
        json.dump(record, f, indent=2, default=str)

    return path


def load_review_packet(case_id: str):
    path = os.path.join(REVIEW_PACKET_DIR, f"{case_id}.json")
    if not os.path.exists(path):
        return None

    with open(path, "r") as f:
        return json.load(f)


def load_human_reviews(case_id: str):
    reviews = []

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
    "load_human_reviews",
]
