"""
ELEANOR V8 — Router Selection Cache

Caches router selections for identical or highly similar inputs.
"""

import hashlib
import json
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, Optional

from cachetools import TTLCache  # type: ignore[import-untyped]


@dataclass(frozen=True)
class RouterCacheEntry:
    normalized_text: str
    context_hash: str
    selection: Dict[str, Any]


class RouterSelectionCache:
    """L1 cache with optional similarity-based lookup for router selections."""

    def __init__(
        self,
        *,
        maxsize: int = 500,
        ttl: int = 120,
        similarity_threshold: float = 0.95,
    ) -> None:
        self.cache = TTLCache(maxsize=maxsize, ttl=ttl)
        self.similarity_threshold = similarity_threshold
        self.hits = 0
        self.misses = 0
        self.similarity_hits = 0
        self.sets = 0

    def _normalize_text(self, text: str) -> str:
        return " ".join(text.lower().split())

    def _context_hash(self, context: Optional[Dict[str, Any]]) -> str:
        if not context:
            return "none"
        try:
            payload = json.dumps(context, sort_keys=True, default=str)
        except Exception:
            payload = str(context)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _cache_key(self, normalized_text: str, context_hash: str) -> str:
        digest = hashlib.sha256(f"{normalized_text}|{context_hash}".encode()).hexdigest()[:16]
        return f"router:{digest}"

    def get(self, text: str, context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        normalized = self._normalize_text(text)
        ctx_hash = self._context_hash(context)
        key = self._cache_key(normalized, ctx_hash)
        entry = self.cache.get(key)
        if entry is not None:
            self.hits += 1
            return entry.selection
        self.misses += 1
        return None

    def get_similar(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        normalized = self._normalize_text(text)
        ctx_hash = self._context_hash(context)

        for entry in self.cache.values():
            if entry.context_hash != ctx_hash:
                continue
            ratio = SequenceMatcher(None, normalized, entry.normalized_text).ratio()
            if ratio >= self.similarity_threshold:
                self.similarity_hits += 1
                return entry.selection
        return None

    def set(
        self,
        text: str,
        context: Optional[Dict[str, Any]],
        selection: Dict[str, Any],
    ) -> None:
        normalized = self._normalize_text(text)
        ctx_hash = self._context_hash(context)
        key = self._cache_key(normalized, ctx_hash)
        self.cache[key] = RouterCacheEntry(
            normalized_text=normalized,
            context_hash=ctx_hash,
            selection=selection,
        )
        self.sets += 1

    def stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        hit_rate = self.hits / total if total else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "similarity_hits": self.similarity_hits,
            "sets": self.sets,
            "hit_rate": hit_rate,
            "entries": len(self.cache),
            "ttl_seconds": self.cache.ttl,
            "maxsize": self.cache.maxsize,
        }


__all__ = ["RouterSelectionCache"]
