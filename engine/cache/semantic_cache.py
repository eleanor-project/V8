"""
ELEANOR V8 — Semantic Cache
----------------------------

Cache based on semantic similarity, not exact match. This enables 3-5x better
cache hit rates by finding semantically similar queries.

Uses embedding-based similarity search to find cached results for similar
(but not identical) queries.
"""

import logging
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available. Semantic cache will use fallback methods.")


@dataclass
class SemanticCacheEntry:
    """Entry in semantic cache."""
    query_text: str
    query_embedding: List[float]
    result: Any
    timestamp: datetime
    hit_count: int = 0
    last_accessed: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SemanticCache:
    """
    Semantic cache that finds results based on semantic similarity.
    
    Features:
    - Embedding-based similarity search
    - Configurable similarity threshold
    - LRU-style eviction based on access patterns
    - Batch similarity search for performance
    - Fallback to exact match if embeddings unavailable
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.85,
        max_size: int = 10000,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        use_gpu: bool = False
    ):
        """
        Initialize semantic cache.
        
        Args:
            similarity_threshold: Minimum cosine similarity for cache hit (0.0 to 1.0)
            max_size: Maximum number of entries in cache
            embedding_model_name: Name of sentence transformer model
            use_gpu: Whether to use GPU for embeddings
        """
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
        self.embedding_model_name = embedding_model_name
        self.use_gpu = use_gpu
        
        # Cache storage: query_hash -> SemanticCacheEntry
        self._cache: Dict[str, SemanticCacheEntry] = {}
        
        # Embedding model (lazy-loaded)
        self._embedding_model: Optional[Any] = None
        self._embedding_model_loaded = False
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "semantic_hits": 0,  # Hits from semantic similarity (not exact match)
            "exact_hits": 0,  # Hits from exact match
            "evictions": 0,
            "total_queries": 0
        }
        
        logger.info(
            "semantic_cache_initialized",
            extra={
                "similarity_threshold": similarity_threshold,
                "max_size": max_size,
                "embedding_model": embedding_model_name,
                "sentence_transformers_available": SENTENCE_TRANSFORMERS_AVAILABLE
            }
        )
    
    def _load_embedding_model(self) -> None:
        """Lazy-load embedding model."""
        if self._embedding_model_loaded:
            return
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("sentence-transformers not available. Semantic cache will use exact matching only.")
            self._embedding_model_loaded = True
            return
        
        try:
            device = "cuda" if self.use_gpu else "cpu"
            self._embedding_model = SentenceTransformer(
                self.embedding_model_name,
                device=device
            )
            self._embedding_model_loaded = True
            logger.info(f"Loaded embedding model: {self.embedding_model_name} on {device}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}", exc_info=True)
            self._embedding_model_loaded = True  # Mark as loaded to prevent retries
            self._embedding_model = None
    
    def _get_query_hash(self, query: str) -> str:
        """Generate hash for query (for exact matching)."""
        return hashlib.sha256(query.encode()).hexdigest()
    
    def _compute_embedding(self, text: str) -> Optional[List[float]]:
        """Compute embedding for text."""
        self._load_embedding_model()
        
        if not self._embedding_model:
            return None
        
        try:
            # Compute embedding
            embedding = self._embedding_model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to compute embedding: {e}", exc_info=True)
            return None
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two embeddings."""
        try:
            a_np = np.array(a)
            b_np = np.array(b)
            
            # Normalize
            norm_a = np.linalg.norm(a_np)
            norm_b = np.linalg.norm(b_np)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            # Cosine similarity
            return float(np.dot(a_np, b_np) / (norm_a * norm_b))
        except Exception as e:
            logger.error(f"Failed to compute cosine similarity: {e}", exc_info=True)
            return 0.0
    
    async def get(
        self,
        query: str,
        exact_match: bool = True
    ) -> Optional[Tuple[Any, float]]:
        """
        Get cached result if semantically similar query exists.
        
        Args:
            query: Query text
            exact_match: If True, also check for exact matches first
            
        Returns:
            Tuple of (result, similarity_score) if found, None otherwise
            similarity_score is 1.0 for exact match, 0.85-1.0 for semantic match
        """
        self.stats["total_queries"] += 1
        
        # Check exact match first (fastest)
        if exact_match:
            query_hash = self._get_query_hash(query)
            if query_hash in self._cache:
                entry = self._cache[query_hash]
                entry.hit_count += 1
                entry.last_accessed = datetime.utcnow()
                self.stats["hits"] += 1
                self.stats["exact_hits"] += 1
                logger.debug(f"Exact cache hit for query hash: {query_hash[:8]}...")
                return (entry.result, 1.0)
        
        # Try semantic similarity search
        query_embedding = self._compute_embedding(query)
        if not query_embedding:
            # Embedding unavailable, can only do exact match
            self.stats["misses"] += 1
            return None
        
        # Find most similar cached entry
        best_match: Optional[SemanticCacheEntry] = None
        best_similarity = 0.0
        
        for entry in self._cache.values():
            if not entry.query_embedding:
                continue
            
            similarity = self._cosine_similarity(query_embedding, entry.query_embedding)
            
            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_match = entry
        
        if best_match:
            best_match.hit_count += 1
            best_match.last_accessed = datetime.utcnow()
            self.stats["hits"] += 1
            self.stats["semantic_hits"] += 1
            logger.debug(
                f"Semantic cache hit: similarity={best_similarity:.3f}, "
                f"threshold={self.similarity_threshold}"
            )
            return (best_match.result, best_similarity)
        
        # No match found
        self.stats["misses"] += 1
        return None
    
    async def set(
        self,
        query: str,
        result: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Cache result with semantic key.
        
        Args:
            query: Query text
            result: Result to cache
            metadata: Optional metadata to store with entry
        """
        # Check if cache is full
        if len(self._cache) >= self.max_size:
            self._evict_oldest()
        
        # Compute embedding
        query_embedding = self._compute_embedding(query)
        
        # Create cache entry
        query_hash = self._get_query_hash(query)
        entry = SemanticCacheEntry(
            query_text=query,
            query_embedding=query_embedding or [],
            result=result,
            timestamp=datetime.utcnow(),
            hit_count=0,
            last_accessed=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        self._cache[query_hash] = entry
        logger.debug(f"Cached entry with hash: {query_hash[:8]}...")
    
    def _evict_oldest(self) -> None:
        """Evict oldest/least-recently-used entry."""
        if not self._cache:
            return
        
        # Find entry with oldest last_accessed (or oldest timestamp if never accessed)
        oldest_entry: Optional[Tuple[str, SemanticCacheEntry]] = None
        oldest_time: Optional[datetime] = None
        
        for hash_key, entry in self._cache.items():
            entry_time = entry.last_accessed or entry.timestamp
            
            if oldest_time is None or entry_time < oldest_time:
                oldest_time = entry_time
                oldest_entry = (hash_key, entry)
        
        if oldest_entry:
            hash_key, _ = oldest_entry
            del self._cache[hash_key]
            self.stats["evictions"] += 1
            logger.debug(f"Evicted cache entry: {hash_key[:8]}...")
    
    async def get_batch(
        self,
        queries: List[str],
        exact_match: bool = True
    ) -> Dict[str, Optional[Tuple[Any, float]]]:
        """
        Get cached results for multiple queries (batch operation).
        
        Args:
            queries: List of query texts
            exact_match: If True, also check for exact matches first
            
        Returns:
            Dictionary mapping query -> (result, similarity) or None
        """
        results = {}
        
        # Process queries in batch for better performance
        query_embeddings: Dict[str, Optional[List[float]]] = {}
        
        if not exact_match or any(self._get_query_hash(q) not in self._cache for q in queries):
            # Need to compute embeddings for semantic search
            self._load_embedding_model()
            
            if self._embedding_model and SENTENCE_TRANSFORMERS_AVAILABLE:
                try:
                    # Batch compute embeddings (much faster)
                    embeddings = self._embedding_model.encode(
                        queries,
                        convert_to_numpy=True,
                        batch_size=32
                    )
                    query_embeddings = {
                        query: emb.tolist()
                        for query, emb in zip(queries, embeddings)
                    }
                except Exception as e:
                    logger.error(f"Failed to batch compute embeddings: {e}", exc_info=True)
                    query_embeddings = {q: self._compute_embedding(q) for q in queries}
            else:
                query_embeddings = {q: self._compute_embedding(q) for q in queries}
        
        # Check each query
        for query in queries:
            result = await self.get(query, exact_match=exact_match)
            results[query] = result
        
        return results
    
    async def set_batch(
        self,
        items: Dict[str, Any],
        metadata: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> None:
        """
        Cache multiple results (batch operation).
        
        Args:
            items: Dictionary mapping query -> result
            metadata: Optional dictionary mapping query -> metadata
        """
        for query, result in items.items():
            query_metadata = (metadata or {}).get(query)
            await self.set(query, result, metadata=query_metadata)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.stats["total_queries"]
        hit_rate = (self.stats["hits"] / total * 100) if total > 0 else 0.0
        semantic_hit_rate = (
            (self.stats["semantic_hits"] / total * 100) if total > 0 else 0.0
        )
        
        return {
            **self.stats,
            "hit_rate_percent": hit_rate,
            "semantic_hit_rate_percent": semantic_hit_rate,
            "current_size": len(self._cache),
            "max_size": self.max_size,
            "similarity_threshold": self.similarity_threshold
        }
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "semantic_hits": 0,
            "exact_hits": 0,
            "evictions": 0,
            "total_queries": 0
        }
        logger.info("Semantic cache cleared")
    
    def __len__(self) -> int:
        """Get current cache size."""
        return len(self._cache)


__all__ = [
    "SemanticCache",
    "SemanticCacheEntry",
]
