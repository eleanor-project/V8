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

import json
import re
import psycopg2  # type: ignore[import-untyped]
from psycopg2 import sql  # type: ignore[import-untyped]
import weaviate  # type: ignore[import-not-found]
from typing import Any, Callable, Dict, List


# Valid table name pattern (alphanumeric and underscore only)
TABLE_NAME_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')


# ----------------------------------------------------------
# Utility: Embedding provider (LLM-agnostic)
# ----------------------------------------------------------

class Embedder:
    """LLM-based embedding generator for precedent search."""

    def __init__(self, embed_fn: Callable[[str], List[float]]):
        self.embed_fn = embed_fn

    def embed(self, text: str) -> List[float]:
        return self.embed_fn(text)


# ----------------------------------------------------------
# 1. Weaviate Backend
# ----------------------------------------------------------

class WeaviatePrecedentStore:
    def __init__(self, client: weaviate.Client, class_name="Precedent", embed_fn=None):
        self.client = client
        self.class_name = class_name
        self.embedder = Embedder(embed_fn)

    def search(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        embedding = self.embedder.embed(query_text)

        result = (
            self.client.query  # type: ignore[attr-defined]
            .get(self.class_name, ["text", "metadata"])  # type: ignore[attr-defined]
            .with_near_vector({"vector": embedding})  # type: ignore[attr-defined]
            .with_limit(top_k)  # type: ignore[attr-defined]
            .do()
        )

        if "data" not in result or "Get" not in result["data"]:
            return []

        hits = result["data"]["Get"].get(self.class_name, [])
        output = []

        for hit in hits:
            output.append({
                "text": hit.get("text", ""),
                "embedding": embedding,
                "metadata": hit.get("metadata", {})
            })

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

        self.conn = psycopg2.connect(conn_string)
        self.table = table_name
        self.embedder = Embedder(embed_fn)

    def search(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        embedding = self.embedder.embed(query_text)

        # Use psycopg2.sql module for safe identifier handling
        query = sql.SQL("""
            SELECT text, metadata
            FROM {}
            ORDER BY embedding <-> %s
            LIMIT %s;
        """).format(sql.Identifier(self.table))

        with self.conn.cursor() as cur:
            cur.execute(query, (embedding, top_k))
            rows = cur.fetchall()

        output = []
        for text, metadata in rows:
            output.append({
                "text": text,
                "embedding": embedding,
                "metadata": json.loads(metadata) if isinstance(metadata, str) else metadata
            })

        return output

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
