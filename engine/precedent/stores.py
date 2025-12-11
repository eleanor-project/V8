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
import psycopg2
import weaviate
from typing import List, Dict, Any


# ----------------------------------------------------------
# Utility: Embedding provider (LLM-agnostic)
# ----------------------------------------------------------

class Embedder:
    """LLM-based embedding generator for precedent search."""

    def __init__(self, embed_fn):
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
            self.client.query
            .get(self.class_name, ["text", "metadata"])
            .with_near_vector({"vector": embedding})
            .with_limit(top_k)
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
    def __init__(self, conn_string: str, table_name="precedent", embed_fn=None):
        self.conn = psycopg2.connect(conn_string)
        self.table = table_name
        self.embedder = Embedder(embed_fn)

    def search(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        embedding = self.embedder.embed(query_text)

        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT text, metadata
                FROM {self.table}
                ORDER BY embedding <-> %s
                LIMIT %s;
                """,
                (embedding, top_k)
            )
            rows = cur.fetchall()

        output = []
        for text, metadata in rows:
            output.append({
                "text": text,
                "embedding": embedding,
                "metadata": json.loads(metadata) if isinstance(metadata, str) else metadata
            })

        return output
