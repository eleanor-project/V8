"""
ELEANOR V8 — GPU-Accelerated Embeddings

High-performance embedding computation and similarity search on GPU.
"""

import logging
import hashlib
from typing import List, Optional, Dict, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class GPUEmbeddingCache:
    """
    GPU-accelerated embedding cache with fast similarity search.
    
    Keeps embeddings on GPU for fast vector operations.
    """
    
    def __init__(self, gpu_manager: 'GPUManager', cache_size: int = 10000):
        """
        Initialize GPU embedding cache.
        
        Args:
            gpu_manager: GPU manager instance
            cache_size: Maximum number of embeddings to cache
        """
        self.gpu_manager = gpu_manager
        self.cache_size = cache_size
        self.device = gpu_manager.get_device()
        
        # Cache: text_hash -> (embedding_tensor, text)
        self.cache: Dict[str, Tuple[Any, str]] = {}
        self.access_order: List[str] = []
        
        logger.info(
            "GPU embedding cache initialized",
            cache_size=cache_size,
            device=str(self.device)
        )
    
    def _hash_text(self, text: str) -> str:
        """Generate hash for text."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]
    
    def get(self, text: str) -> Optional[Any]:
        """
        Get cached embedding.
        
        Args:
            text: Input text
        
        Returns:
            Embedding tensor or None if not cached
        """
        text_hash = self._hash_text(text)
        
        if text_hash in self.cache:
            # Update access order (LRU)
            self.access_order.remove(text_hash)
            self.access_order.append(text_hash)
            
            embedding, _ = self.cache[text_hash]
            logger.debug(f"Embedding cache hit: {text[:50]}...")
            return embedding
        
        return None
    
    def put(self, text: str, embedding: Any) -> None:
        """
        Cache embedding on GPU.
        
        Args:
            text: Input text
            embedding: Embedding tensor or array
        """
        text_hash = self._hash_text(text)
        
        # Convert to GPU tensor if needed
        if self.gpu_manager.torch_available:
            if not isinstance(embedding, self.gpu_manager.torch.Tensor):
                embedding = self.gpu_manager.torch.tensor(
                    embedding, 
                    device=self.device,
                    dtype=self.gpu_manager.torch.float16 if self.gpu_manager.config.mixed_precision else self.gpu_manager.torch.float32
                )
            elif embedding.device != self.device:
                embedding = embedding.to(self.device)
        
        # Evict oldest if cache full
        if len(self.cache) >= self.cache_size:
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        
        self.cache[text_hash] = (embedding, text)
        self.access_order.append(text_hash)
        
        logger.debug(f"Embedding cached: {text[:50]}...")
    
    def compute_similarity(self, query_embedding: Any, candidate_embeddings: Any) -> Any:
        """
        Compute cosine similarity on GPU (vectorized).
        
        Args:
            query_embedding: Query embedding tensor
            candidate_embeddings: Candidate embeddings tensor (batch)
        
        Returns:
            Similarity scores tensor
        """
        if not self.gpu_manager.torch_available:
            # NumPy fallback
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            candidate_norms = candidate_embeddings / np.linalg.norm(
                candidate_embeddings, axis=1, keepdims=True
            )
            return np.dot(candidate_norms, query_norm)
        
        # GPU-accelerated computation
        torch = self.gpu_manager.torch
        
        # Ensure on GPU
        if not isinstance(query_embedding, torch.Tensor):
            query_embedding = torch.tensor(query_embedding, device=self.device)
        if not isinstance(candidate_embeddings, torch.Tensor):
            candidate_embeddings = torch.tensor(candidate_embeddings, device=self.device)
        
        # Normalize
        query_norm = query_embedding / query_embedding.norm()
        candidate_norms = candidate_embeddings / candidate_embeddings.norm(dim=1, keepdim=True)
        
        # Compute similarity (single GPU operation)
        similarities = torch.mm(candidate_norms, query_norm.unsqueeze(1)).squeeze()
        
        return similarities
    
    def batch_similarity(
        self,
        query_embeddings: List[Any],
        candidate_embeddings: Any
    ) -> Any:
        """
        Compute similarity for multiple queries at once.
        
        Args:
            query_embeddings: List of query embeddings
            candidate_embeddings: Candidate embeddings tensor
        
        Returns:
            Similarity matrix (queries × candidates)
        """
        if not self.gpu_manager.torch_available:
            return np.array([
                self.compute_similarity(q, candidate_embeddings)
                for q in query_embeddings
            ])
        
        torch = self.gpu_manager.torch
        
        # Stack queries
        query_batch = torch.stack([
            q if isinstance(q, torch.Tensor) else torch.tensor(q, device=self.device)
            for q in query_embeddings
        ])
        
        # Normalize
        query_norms = query_batch / query_batch.norm(dim=1, keepdim=True)
        candidate_norms = candidate_embeddings / candidate_embeddings.norm(dim=1, keepdim=True)
        
        # Batch matrix multiplication
        similarities = torch.mm(query_norms, candidate_norms.t())
        
        return similarities
    
    def top_k_similar(
        self,
        query_embedding: Any,
        candidate_embeddings: Any,
        k: int = 10
    ) -> Tuple[Any, Any]:
        """
        Find top-k most similar embeddings.
        
        Args:
            query_embedding: Query embedding
            candidate_embeddings: Candidate embeddings
            k: Number of results
        
        Returns:
            (indices, scores) of top-k matches
        """
        similarities = self.compute_similarity(query_embedding, candidate_embeddings)
        
        if self.gpu_manager.torch_available:
            # GPU topk operation
            top_scores, top_indices = self.gpu_manager.torch.topk(similarities, k)
            return top_indices.cpu().numpy(), top_scores.cpu().numpy()
        else:
            # NumPy fallback
            top_indices = np.argsort(similarities)[-k:][::-1]
            top_scores = similarities[top_indices]
            return top_indices, top_scores
    
    def clear(self) -> None:
        """Clear cache and free GPU memory."""
        self.cache.clear()
        self.access_order.clear()
        
        if self.gpu_manager.cuda_available:
            self.gpu_manager.empty_cache()
        
        logger.info("Embedding cache cleared")
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        return {
            'cache_size': len(self.cache),
            'max_size': self.cache_size,
            'utilization': len(self.cache) / self.cache_size if self.cache_size > 0 else 0.0,
            'device': str(self.device),
        }


__all__ = ["GPUEmbeddingCache"]
