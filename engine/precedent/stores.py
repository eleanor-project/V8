"""
Compatibility shims for precedent stores.

Prefer importing from engine.precedent.store.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from engine.precedent.store import PgVectorStore, WeaviatePrecedentStore

logger = logging.getLogger(__name__)

TABLE_NAME_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


class PGVectorPrecedentStore(PgVectorStore):
    def __init__(self, conn_string: str, table_name: str = "precedent", embed_fn=None):
        if not TABLE_NAME_PATTERN.match(table_name):
            raise ValueError(
                f"Invalid table name '{table_name}'. "
                "Table names must start with a letter or underscore and contain only "
                "alphanumeric characters and underscores."
            )
        super().__init__(connection_string=conn_string, table_name=table_name)
        self._embed_fn = embed_fn

    def search(
        self,
        query_text: str,
        top_k: int = 5,
        embedding: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        if embedding is None and self._embed_fn:
            embedding = self._embed_fn(query_text)
        if not embedding:
            logger.warning(
                "pgvector_search_missing_embedding",
                extra={"table": self.table_name},
            )
            return []

        results = super().search(query_text, top_k=top_k, embedding=embedding)
        normalized: List[Dict[str, Any]] = []
        for case in results:
            case = dict(case)
            case.setdefault("text", case.get("query_text", ""))
            case["embedding"] = embedding or case.get("embedding", [])
            normalized.append(case)
        return normalized


__all__ = ["PGVectorPrecedentStore", "WeaviatePrecedentStore"]
