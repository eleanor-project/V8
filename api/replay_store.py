"""
ReplayStore — persistent store for deliberation inputs and outputs.

Provides:
 - append-only JSONL log for replay
 - in-memory index for fast lookup
 - retrieval by trace_id
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional


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


__all__ = ["ReplayStore"]
