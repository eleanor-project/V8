"""
ELEANOR V8 — Precedent Retrieval
------------------------------------

Retrieves precedent entries from a vector database or local store, and
computes semantic + normative alignment scores.

In production, this module connects to:
    - Weaviate
    - pgvector
    - ChromaDB
    - Elastic vector search
    - or a simple JSON store for local mode

Output:
{
    "precedent_cases": [...],
    "alignment_score": float,
    "top_case": {...},
}
"""

from typing import Dict, Any, List, Optional, Callable
import inspect
import math

from engine.schemas.pipeline_types import (
    CriticResult,
    PrecedentRetrievalResult,
    PrecedentCaseResult,
)

class PrecedentRetrievalV8:

    def __init__(
        self,
        store_client,
        embedding_fn: Optional[Callable[[str], List[float]]] = None,
        embedding_cache: Optional[Any] = None,
    ):
        """
        store_client: abstraction for vector DB or local JSON.
        Must implement:
            search(query: str, top_k: int) -> List[dict]
        """
        self.store = store_client
        self.embedding_fn = embedding_fn
        self.embedding_cache = embedding_cache

        if self.embedding_fn is None:
            embedder = getattr(store_client, "embedder", None)
            embed_fn = getattr(embedder, "embed", None)
            if callable(embed_fn):
                self.embedding_fn = embed_fn

    # ---------------------------------------------------------------
    #  Semantic + normative alignment scoring
    # ---------------------------------------------------------------
    def _score_alignment(self, case: PrecedentCaseResult, critic_outputs: List[CriticResult]) -> float:
        """
        Compute a rough alignment score based on:
          - overlap of values invoked
          - similarity of violation patterns
          - similarity of critic scores

        Returns float between 0 and 1.
        """
        case_values = set(case.get("values", []))
        current_values = set()
        for output in critic_outputs:
            value = output.get("value")
            if value:
                current_values.add(value)

        # Value overlap ratio
        if not current_values:
            value_alignment = 0.0
        else:
            value_alignment = len(case_values & current_values) / float(len(current_values))

        # Score similarity (very rough)
        case_score = case.get("aggregate_score", 0.5)
        curr_scores = [o.get("score", 0.0) for o in critic_outputs]
        curr_avg = sum(curr_scores) / len(curr_scores) if curr_scores else 0.5

        score_dist = abs(case_score - curr_avg)
        score_alignment = max(0.0, 1 - score_dist)

        # Combined harmonic-like weighting
        return float((value_alignment + score_alignment) / 2)

    # ---------------------------------------------------------------
    #  Main retrieval function
    # ---------------------------------------------------------------
    def retrieve(
        self,
        query_text: str,
        critic_outputs: List[CriticResult],
        top_k: int = 5,
    ) -> PrecedentRetrievalResult:
        query_embedding = self._get_cached_embedding(query_text)
        if not query_embedding and self.embedding_fn:
            try:
                query_embedding = self.embedding_fn(query_text)
            except Exception:
                query_embedding = []
            if query_embedding:
                self._cache_embedding(query_text, query_embedding)

        results = self._search_store(query_text, top_k, query_embedding) or []

        if not results:
            return {
                "precedent_cases": [],
                "alignment_score": 1.0,  # neutral when no precedent exists
                "top_case": None,
                "query_embedding": query_embedding or [],
            }

        scored = []
        for case in results:
            score = self._score_alignment(case, critic_outputs)
            scored.append((case, score))

        # Pick top case
        scored.sort(key=lambda x: x[1], reverse=True)
        top_case, top_score = scored[0]

        if not query_embedding:
            candidate_embedding = results[0].get("embedding")
            if isinstance(candidate_embedding, list):
                query_embedding = candidate_embedding

        return {
            "precedent_cases": [c for c, s in scored],
            "alignment_score": float(top_score),
            "top_case": top_case,
            "query_embedding": query_embedding or [],
        }

    def _search_store(
        self,
        query_text: str,
        top_k: int,
        query_embedding: Optional[List[float]],
    ) -> List[Dict[str, Any]]:
        search_fn = getattr(self.store, "search", None)
        if not callable(search_fn):
            return []

        if query_embedding:
            try:
                params = inspect.signature(search_fn).parameters
            except (TypeError, ValueError):
                params = {}
            if "embedding" in params:
                return search_fn(query_text, top_k=top_k, embedding=query_embedding) or []
            try:
                return search_fn(query_text, top_k=top_k, embedding=query_embedding) or []
            except TypeError:
                return search_fn(query_text, top_k=top_k) or []

        return search_fn(query_text, top_k=top_k) or []

    def _get_cached_embedding(self, text: str) -> List[float]:
        if not self.embedding_cache:
            return []
        getter = getattr(self.embedding_cache, "get_cached_embedding", None)
        if not callable(getter):
            return []
        try:
            cached = getter(text)
        except Exception:
            return []
        if cached is None:
            return []
        try:
            if hasattr(cached, "detach"):
                cached = cached.detach().cpu()
            if hasattr(cached, "tolist"):
                return cached.tolist()
        except Exception:
            return []
        return []

    def _cache_embedding(self, text: str, embedding: List[float]) -> None:
        if not self.embedding_cache:
            return
        cache_fn = getattr(self.embedding_cache, "cache_embedding", None)
        if not callable(cache_fn):
            return
        try:
            import torch
        except Exception:
            return
        try:
            tensor = torch.tensor(embedding, dtype=torch.float32)
            cache_fn(text, tensor)
        except Exception:
            return

    async def close(self) -> None:
        """Close underlying store connection if supported."""
        close_fn = getattr(self.store, "close", None)
        if callable(close_fn):
            result = close_fn()
            if inspect.isawaitable(result):
                await result
