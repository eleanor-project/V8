"""
ELEANOR V8 — Precedent Store Integrations
-------------------------------------------

This module provides real vector database clients for storing and
retrieving precedent cases used by the ELEANOR V8 governance engine.

Supported backends:

1. Weaviate (local or cloud)
2. pgvector (Postgres + pgvector extension)

Each backend must implement:

    search(query_text: str, top_k: int = 5) -> List[dict]

Returned objects MUST include:
{
    "text": <precedent text>,
    "embedding": <vector>,
    "metadata": {...}
}
"""

from __future__ import annotations

import json
import logging
import os
import re
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

try:
    import psycopg2  # type: ignore[import-untyped]
    from psycopg2 import sql  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - optional dependency
    psycopg2 = SimpleNamespace(connect=None, Error=Exception)  # type: ignore[assignment]
    sql = None  # type: ignore[assignment]

try:
    from psycopg2.pool import ThreadedConnectionPool  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - optional dependency
    ThreadedConnectionPool = None  # type: ignore[assignment]

try:
    import weaviate  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    weaviate = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# Valid table name pattern (alphanumeric and underscore only)
TABLE_NAME_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


# ----------------------------------------------------------
# Utility: Embedding provider (LLM-agnostic)
# ----------------------------------------------------------


class Embedder:
    """LLM-based embedding generator for precedent search."""

    def __init__(self, embed_fn: Optional[Callable[[str], List[float]]] = None):
        self.embed_fn = embed_fn

    def embed(self, text: str) -> List[float]:
        if not self.embed_fn:
            return []
        return self.embed_fn(text)


# ----------------------------------------------------------
# 1. Weaviate Backend
# ----------------------------------------------------------


class WeaviatePrecedentStore:
    def __init__(self, client: Any, class_name: str = "Precedent", embed_fn=None):
        if weaviate is None:
            logger.warning("weaviate_client_unavailable")
        self.client = client
        self.class_name = class_name
        self.embedder = Embedder(embed_fn)

    def search(
        self,
        query_text: str,
        top_k: int = 5,
        embedding: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        if embedding is None:
            embedding = self.embedder.embed(query_text)
        if not embedding:
            logger.warning(
                "weaviate_search_missing_embedding",
                extra={"class_name": self.class_name},
            )
            return []

        result = (
            self.client.query.get(self.class_name, ["text", "metadata"])  # type: ignore[attr-defined]  # type: ignore[attr-defined]
            .with_near_vector({"vector": embedding})  # type: ignore[attr-defined]
            .with_limit(top_k)  # type: ignore[attr-defined]
            .do()
        )

        if "data" not in result or "Get" not in result["data"]:
            return []

        hits = result["data"]["Get"].get(self.class_name, [])
        output = []

        for hit in hits:
            output.append(
                {
                    "text": hit.get("text", ""),
                    "embedding": embedding,
                    "metadata": hit.get("metadata", {}),
                }
            )

        return output


# ----------------------------------------------------------
# 2. pgvector Backend
# ----------------------------------------------------------


class PGVectorPrecedentStore:
    def __init__(self, conn_string: str, table_name: str = "precedent", embed_fn=None):
        # Validate table name to prevent SQL injection
        if not TABLE_NAME_PATTERN.match(table_name):
            raise ValueError(
                f"Invalid table name '{table_name}'. "
                "Table names must start with a letter or underscore and contain only "
                "alphanumeric characters and underscores."
            )

        if ThreadedConnectionPool is None and getattr(psycopg2, "connect", None) is None:
            raise ImportError("psycopg2 required for pgvector support. Install psycopg2-binary.")

        self.pool = None
        self.conn = None
        if ThreadedConnectionPool is not None:
            min_size = _env_int("PG_POOL_MIN", 1)
            max_size = _env_int("PG_POOL_MAX", 5)
            if max_size < min_size:
                max_size = min_size
            self.pool = ThreadedConnectionPool(min_size, max_size, conn_string)
        else:
            self.conn = psycopg2.connect(conn_string)
        self.table = table_name
        self.embedder = Embedder(embed_fn)

    def search(
        self,
        query_text: str,
        top_k: int = 5,
        embedding: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        if embedding is None:
            embedding = self.embedder.embed(query_text)
        if not embedding:
            logger.warning(
                "pgvector_search_missing_embedding",
                extra={"table": self.table},
            )
            return []

        # Use psycopg2.sql module for safe identifier handling
        if sql is None:
            query = f"""
            SELECT text, metadata
            FROM {self.table}
            ORDER BY embedding <-> %s
            LIMIT %s;
        """
        else:
            query = sql.SQL(
                """
                SELECT text, metadata
                FROM {}
                ORDER BY embedding <-> %s
                LIMIT %s;
            """
            ).format(sql.Identifier(self.table))

        if self.pool is not None:
            conn = self.pool.getconn()
            try:
                with conn.cursor() as cur:
                    cur.execute(query, (embedding, top_k))
                    rows = cur.fetchall()
            finally:
                self.pool.putconn(conn)
        elif self.conn is not None:
            with self.conn.cursor() as cur:
                cur.execute(query, (embedding, top_k))
                rows = cur.fetchall()
        else:
            raise RuntimeError("No pgvector connection available")

        output = []
        for text, metadata in rows:
            output.append(
                {
                    "text": text,
                    "embedding": embedding,
                    "metadata": json.loads(metadata) if isinstance(metadata, str) else metadata,
                }
            )

        return output

    def close(self):
        """Close the database connection."""
        if self.pool is not None:
            self.pool.closeall()
        if self.conn:
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(1, value)
