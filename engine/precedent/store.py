"""
ELEANOR V8 — Precedent Store Implementations
----------------------------------------------

Provides vector database backends for precedent storage and retrieval.
Supports multiple backends for flexibility in deployment.

Supported Backends:
1. PgVectorStore - PostgreSQL with pgvector extension
2. ChromaStore - ChromaDB for local/development use
3. WeaviateStore - Weaviate cloud or self-hosted
4. InMemoryStore - For testing and development
5. JSONFileStore - Simple file-based storage
"""

import json
import os
import hashlib
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, cast
from dataclasses import dataclass, field, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class PrecedentCase:
    """
    Represents a constitutional precedent case for jurisprudence.
    """
    case_id: str
    query_text: str
    decision: str  # allow, deny, escalate, constrained_allow
    values: List[str] = field(default_factory=list)
    aggregate_score: float = 0.0
    critic_outputs: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding embedding for storage efficiency."""
        d = asdict(self)
        # Don't store embedding in the case dict (stored separately in vector DB)
        d.pop("embedding", None)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PrecedentCase":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class BasePrecedentStore(ABC):
    """Abstract base class for precedent stores."""

    @abstractmethod
    def add(self, case: PrecedentCase, embedding: List[float]) -> str:
        """Add a precedent case with its embedding. Returns case_id."""
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 5, embedding: Optional[List[float]] = None) -> List[Dict[str, Any]]:
        """Search for similar precedent cases. Returns list of case dicts."""
        pass

    @abstractmethod
    def get(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific case by ID."""
        pass

    @abstractmethod
    def delete(self, case_id: str) -> bool:
        """Delete a case by ID."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Return total number of cases."""
        pass


class InMemoryStore(BasePrecedentStore):
    """
    Simple in-memory store for testing and development.
    Uses brute-force cosine similarity for search.
    """

    def __init__(self):
        self._cases: Dict[str, Dict[str, Any]] = {}
        self._embeddings: Dict[str, List[float]] = {}

    def add(self, case: PrecedentCase, embedding: List[float]) -> str:
        case_id = case.case_id or self._generate_id(case.query_text)
        case.case_id = case_id
        self._cases[case_id] = case.to_dict()
        self._embeddings[case_id] = embedding
        return case_id

    def search(self, query: str, top_k: int = 5, embedding: Optional[List[float]] = None) -> List[Dict[str, Any]]:
        if not self._cases:
            return []

        if embedding is None:
            # Without embedding, return most recent cases
            sorted_cases = sorted(
                self._cases.values(),
                key=lambda c: c.get("timestamp", 0),
                reverse=True
            )
            return sorted_cases[:top_k]

        # Compute similarities
        scores = []
        for case_id, case_embedding in self._embeddings.items():
            sim = self._cosine_similarity(embedding, case_embedding)
            scores.append((case_id, sim))

        # Sort by similarity
        scores.sort(key=lambda x: x[1], reverse=True)

        # Return top_k cases with similarity scores
        results = []
        for case_id, sim in scores[:top_k]:
            case = self._cases[case_id].copy()
            case["similarity_score"] = sim
            results.append(case)

        return results

    def get(self, case_id: str) -> Optional[Dict[str, Any]]:
        return self._cases.get(case_id)

    def delete(self, case_id: str) -> bool:
        if case_id in self._cases:
            del self._cases[case_id]
            self._embeddings.pop(case_id, None)
            return True
        return False

    def count(self) -> int:
        return len(self._cases)

    @staticmethod
    def _generate_id(text: str) -> str:
        """Generate a unique ID from text."""
        timestamp = str(time.time())
        return hashlib.sha256((text + timestamp).encode()).hexdigest()[:16]

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0
        dot = float(sum(x * y for x, y in zip(a, b)))
        norm_a = float(sum(x * x for x in a) ** 0.5)
        norm_b = float(sum(x * x for x in b) ** 0.5)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


class JSONFileStore(BasePrecedentStore):
    """
    File-based JSON store for simple persistence.
    Suitable for development and small-scale deployments.
    """

    def __init__(self, file_path: str = "precedents.json"):
        self.file_path = file_path
        self._cases: Dict[str, Dict[str, Any]] = {}
        self._embeddings: Dict[str, List[float]] = {}
        self._load()

    def _load(self):
        """Load cases from file."""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r") as f:
                    data = json.load(f)
                    self._cases = data.get("cases", {})
                    self._embeddings = data.get("embeddings", {})
            except Exception as e:
                logger.warning(f"Failed to load precedent store: {e}")
                self._cases = {}
                self._embeddings = {}

    def _save(self):
        """Save cases to file."""
        try:
            with open(self.file_path, "w") as f:
                json.dump({
                    "cases": self._cases,
                    "embeddings": self._embeddings
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save precedent store: {e}")

    def add(self, case: PrecedentCase, embedding: List[float]) -> str:
        case_id = case.case_id or self._generate_id(case.query_text)
        case.case_id = case_id
        self._cases[case_id] = case.to_dict()
        self._embeddings[case_id] = embedding
        self._save()
        return case_id

    def search(self, query: str, top_k: int = 5, embedding: Optional[List[float]] = None) -> List[Dict[str, Any]]:
        if not self._cases:
            return []

        if embedding is None:
            sorted_cases = sorted(
                self._cases.values(),
                key=lambda c: c.get("timestamp", 0),
                reverse=True
            )
            return sorted_cases[:top_k]

        scores = []
        for case_id, case_embedding in self._embeddings.items():
            sim = InMemoryStore._cosine_similarity(embedding, case_embedding)
            scores.append((case_id, sim))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for case_id, sim in scores[:top_k]:
            case = self._cases[case_id].copy()
            case["similarity_score"] = sim
            results.append(case)

        return results

    def get(self, case_id: str) -> Optional[Dict[str, Any]]:
        return self._cases.get(case_id)

    def delete(self, case_id: str) -> bool:
        if case_id in self._cases:
            del self._cases[case_id]
            self._embeddings.pop(case_id, None)
            self._save()
            return True
        return False

    def count(self) -> int:
        return len(self._cases)

    @staticmethod
    def _generate_id(text: str) -> str:
        timestamp = str(time.time())
        return hashlib.sha256((text + timestamp).encode()).hexdigest()[:16]


class PgVectorStore(BasePrecedentStore):
    """
    PostgreSQL with pgvector extension for production-grade vector search.

    Requires:
        - PostgreSQL with pgvector extension installed
        - psycopg2 or asyncpg for database connection

    Table schema:
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE TABLE precedent_cases (
            case_id VARCHAR(64) PRIMARY KEY,
            query_text TEXT NOT NULL,
            decision VARCHAR(32) NOT NULL,
            values TEXT[],
            aggregate_score FLOAT,
            critic_outputs JSONB,
            rationale TEXT,
            timestamp TIMESTAMP DEFAULT NOW(),
            metadata JSONB,
            embedding vector(1536)  -- Adjust dimension as needed
        );
        CREATE INDEX ON precedent_cases USING ivfflat (embedding vector_cosine_ops);
    """

    def __init__(
        self,
        connection_string: Optional[str] = None,
        table_name: str = "precedent_cases",
        embedding_dim: int = 1536,
        pool_size: int = 5
    ):
        """
        Initialize PgVector store.

        Args:
            connection_string: PostgreSQL connection string
            table_name: Name of the precedent table
            embedding_dim: Dimension of embedding vectors
            pool_size: Connection pool size
        """
        self.connection_string = connection_string or os.getenv(
            "POSTGRES_URL",
            "postgresql://localhost:5432/eleanor"
        )
        self.table_name = table_name
        self.embedding_dim = embedding_dim
        self.pool_size = pool_size
        self._pool = None
        self._initialized = False

    def _get_connection(self):
        """Get a database connection."""
        try:
            import psycopg2  # type: ignore[import-untyped]
            from psycopg2.extras import RealDictCursor  # type: ignore[import-untyped]
            return psycopg2.connect(self.connection_string, cursor_factory=RealDictCursor)
        except ImportError:
            raise ImportError("psycopg2 not installed. Run: pip install psycopg2-binary")

    def _ensure_table(self, conn):
        """Ensure the precedent table exists."""
        if self._initialized:
            return

        with conn.cursor() as cur:
            # Create extension if not exists
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

            # Create table
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    case_id VARCHAR(64) PRIMARY KEY,
                    query_text TEXT NOT NULL,
                    decision VARCHAR(32) NOT NULL,
                    values TEXT[],
                    aggregate_score FLOAT,
                    critic_outputs JSONB,
                    rationale TEXT,
                    timestamp TIMESTAMP DEFAULT NOW(),
                    metadata JSONB,
                    embedding vector({self.embedding_dim})
                )
            """)

            # Create index for fast similarity search
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx
                ON {self.table_name} USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """)

            conn.commit()
            self._initialized = True

    def add(self, case: PrecedentCase, embedding: List[float]) -> str:
        """Add a precedent case with embedding."""
        case_id = case.case_id or self._generate_id(case.query_text)
        case.case_id = case_id

        conn = self._get_connection()
        try:
            self._ensure_table(conn)

            with conn.cursor() as cur:
                cur.execute(f"""
                    INSERT INTO {self.table_name}
                    (case_id, query_text, decision, values, aggregate_score,
                     critic_outputs, rationale, timestamp, metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, to_timestamp(%s), %s, %s)
                    ON CONFLICT (case_id) DO UPDATE SET
                        query_text = EXCLUDED.query_text,
                        decision = EXCLUDED.decision,
                        values = EXCLUDED.values,
                        aggregate_score = EXCLUDED.aggregate_score,
                        critic_outputs = EXCLUDED.critic_outputs,
                        rationale = EXCLUDED.rationale,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding
                """, (
                    case_id,
                    case.query_text,
                    case.decision,
                    case.values,
                    case.aggregate_score,
                    json.dumps(case.critic_outputs),
                    case.rationale,
                    case.timestamp,
                    json.dumps(case.metadata),
                    embedding
                ))
                conn.commit()
        finally:
            conn.close()

        return case_id

    def search(self, query: str, top_k: int = 5, embedding: Optional[List[float]] = None) -> List[Dict[str, Any]]:
        """Search for similar precedent cases using vector similarity."""
        conn = self._get_connection()
        try:
            self._ensure_table(conn)

            with conn.cursor() as cur:
                if embedding:
                    # Vector similarity search
                    cur.execute(f"""
                        SELECT case_id, query_text, decision, values, aggregate_score,
                               critic_outputs, rationale, timestamp, metadata,
                               1 - (embedding <=> %s::vector) as similarity_score
                        FROM {self.table_name}
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                    """, (embedding, embedding, top_k))
                else:
                    # Fall back to most recent
                    cur.execute(f"""
                        SELECT case_id, query_text, decision, values, aggregate_score,
                               critic_outputs, rationale, timestamp, metadata,
                               1.0 as similarity_score
                        FROM {self.table_name}
                        ORDER BY timestamp DESC
                        LIMIT %s
                    """, (top_k,))

                rows = cur.fetchall()
                results = []
                for row in rows:
                    case = dict(row)
                    case["critic_outputs"] = json.loads(case["critic_outputs"]) if case["critic_outputs"] else {}
                    case["metadata"] = json.loads(case["metadata"]) if case["metadata"] else {}
                    results.append(case)

                return results
        finally:
            conn.close()

    def get(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific case by ID."""
        conn = self._get_connection()
        try:
            self._ensure_table(conn)

            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT case_id, query_text, decision, values, aggregate_score,
                           critic_outputs, rationale, timestamp, metadata
                    FROM {self.table_name}
                    WHERE case_id = %s
                """, (case_id,))

                row = cur.fetchone()
                if row:
                    case = dict(row)
                    case["critic_outputs"] = json.loads(case["critic_outputs"]) if case["critic_outputs"] else {}
                    case["metadata"] = json.loads(case["metadata"]) if case["metadata"] else {}
                    return case
                return None
        finally:
            conn.close()

    def delete(self, case_id: str) -> bool:
        """Delete a case by ID."""
        conn = self._get_connection()
        try:
            self._ensure_table(conn)

            with conn.cursor() as cur:
                cur.execute(f"DELETE FROM {self.table_name} WHERE case_id = %s", (case_id,))
                deleted = bool(cur.rowcount and cur.rowcount > 0)
                conn.commit()
                return deleted
        finally:
            conn.close()

    def count(self) -> int:
        """Return total number of cases."""
        conn = self._get_connection()
        try:
            self._ensure_table(conn)

            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) as cnt FROM {self.table_name}")
                row = cur.fetchone()
                return int(row["cnt"]) if row else 0
        finally:
            conn.close()

    @staticmethod
    def _generate_id(text: str) -> str:
        timestamp = str(time.time())
        return hashlib.sha256((text + timestamp).encode()).hexdigest()[:16]


class ChromaStore(BasePrecedentStore):
    """
    ChromaDB store for local/development vector search.

    Requires:
        - chromadb package installed
    """

    def __init__(
        self,
        collection_name: str = "precedent_cases",
        persist_directory: Optional[str] = None,
    ):
        """
        Initialize ChromaDB store.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory for persistent storage (None for in-memory)
        """
        try:
            import chromadb  # type: ignore[import-not-found]
        except ImportError:
            raise ImportError("chromadb not installed. Run: pip install chromadb")

        if persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.Client()

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add(self, case: PrecedentCase, embedding: List[float]) -> str:
        case_id = case.case_id or self._generate_id(case.query_text)
        case.case_id = case_id

        self._collection.add(
            ids=[case_id],
            embeddings=[embedding],
            documents=[case.query_text],
            metadatas=[{
                "decision": case.decision,
                "values": json.dumps(case.values),
                "aggregate_score": case.aggregate_score,
                "critic_outputs": json.dumps(case.critic_outputs),
                "rationale": case.rationale,
                "timestamp": case.timestamp,
                "metadata": json.dumps(case.metadata),
            }]
        )
        return case_id

    def search(self, query: str, top_k: int = 5, embedding: Optional[List[float]] = None) -> List[Dict[str, Any]]:
        if embedding:
            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
        else:
            results = self._collection.query(
                query_texts=[query],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )

        cases = []
        if results["ids"] and results["ids"][0]:
            for i, case_id in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0

                case = {
                    "case_id": case_id,
                    "query_text": results["documents"][0][i] if results["documents"] else "",
                    "decision": meta.get("decision", ""),
                    "values": json.loads(meta.get("values", "[]")),
                    "aggregate_score": meta.get("aggregate_score", 0.0),
                    "critic_outputs": json.loads(meta.get("critic_outputs", "{}")),
                    "rationale": meta.get("rationale", ""),
                    "timestamp": meta.get("timestamp", 0),
                    "metadata": json.loads(meta.get("metadata", "{}")),
                    "similarity_score": 1 - distance,  # Convert distance to similarity
                }
                cases.append(case)

        return cases

    def get(self, case_id: str) -> Optional[Dict[str, Any]]:
        results = self._collection.get(ids=[case_id], include=["documents", "metadatas"])

        if results["ids"]:
            meta = results["metadatas"][0] if results["metadatas"] else {}
            return {
                "case_id": case_id,
                "query_text": results["documents"][0] if results["documents"] else "",
                "decision": meta.get("decision", ""),
                "values": json.loads(meta.get("values", "[]")),
                "aggregate_score": meta.get("aggregate_score", 0.0),
                "critic_outputs": json.loads(meta.get("critic_outputs", "{}")),
                "rationale": meta.get("rationale", ""),
                "timestamp": meta.get("timestamp", 0),
                "metadata": json.loads(meta.get("metadata", "{}")),
            }
        return None

    def delete(self, case_id: str) -> bool:
        try:
            self._collection.delete(ids=[case_id])
            return True
        except Exception:
            return False

    def count(self) -> int:
        return int(self._collection.count())

    @staticmethod
    def _generate_id(text: str) -> str:
        timestamp = str(time.time())
        return hashlib.sha256((text + timestamp).encode()).hexdigest()[:16]


def create_store(backend: str = "memory", **kwargs) -> BasePrecedentStore:
    """
    Factory function to create a precedent store.

    Args:
        backend: Store backend type ("memory", "json", "pgvector", "chroma")
        **kwargs: Backend-specific configuration

    Returns:
        Configured precedent store instance
    """
    backends: Dict[str, Any] = {
        "memory": InMemoryStore,
        "json": JSONFileStore,
        "pgvector": PgVectorStore,
        "chroma": ChromaStore,
    }

    if backend not in backends:
        raise ValueError(f"Unknown backend: {backend}. Available: {list(backends.keys())}")

    store_cls = backends[backend]
    return cast(BasePrecedentStore, store_cls(**kwargs))
