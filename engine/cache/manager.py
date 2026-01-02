"""
ELEANOR V8 — Cache Manager Implementation

Multi-level caching with L1 (in-memory LRU) and L2 (Redis) support.
"""

import asyncio
import hashlib
import json
import logging
from typing import Optional, Any, Dict, Callable
from dataclasses import dataclass
from cachetools import TTLCache
import time

logger = logging.getLogger(__name__)


@dataclass
class CacheKey:
    """Structured cache key with prefix and content hash."""
    
    prefix: str
    content_hash: str
    
    @classmethod
    def from_data(cls, prefix: str, *args, **kwargs) -> 'CacheKey':
        """Generate cache key from arbitrary data."""
        # Combine args and kwargs into single dict
        data = {
            'args': args,
            'kwargs': kwargs,
        }
        
        # Create stable JSON representation
        serialized = json.dumps(data, sort_keys=True, default=str)
        
        # Hash for compact key
        content_hash = hashlib.sha256(serialized.encode()).hexdigest()[:16]
        
        return cls(prefix=prefix, content_hash=content_hash)
    
    def __str__(self) -> str:
        return f"{self.prefix}:{self.content_hash}"


class CacheStats:
    """Track cache statistics."""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.errors = 0
        self.sets = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'hits': self.hits,
            'misses': self.misses,
            'errors': self.errors,
            'sets': self.sets,
            'hit_rate': self.hit_rate,
        }


class CacheManager:
    """
    Multi-level cache manager with L1 (memory) and L2 (Redis) support.
    
    L1: Fast in-memory TTL cache with LRU eviction
    L2: Persistent Redis cache for cross-instance sharing
    """
    
    def __init__(
        self,
        redis_client: Optional[Any] = None,
        l1_sizes: Optional[Dict[str, int]] = None,
        l1_ttls: Optional[Dict[str, int]] = None,
        l2_ttls: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize cache manager.
        
        Args:
            redis_client: Optional Redis client for L2 caching
            l1_sizes: L1 cache sizes by prefix (default: 1000 per prefix)
            l1_ttls: L1 TTLs in seconds by prefix (default: 300)
            l2_ttls: L2 TTLs in seconds by prefix (default: 3600)
        """
        self.redis = redis_client
        
        # Default configurations
        default_l1_sizes = {
            'precedent': 1000,
            'embedding': 500,
            'router': 200,
            'critic': 1000,
            'detector': 500,
        }
        default_l1_ttls = {
            'precedent': 3600,
            'embedding': 7200,
            'router': 1800,
            'critic': 1800,
            'detector': 600,
        }
        default_l2_ttls = {
            'precedent': 7200,
            'embedding': 14400,
            'router': 3600,
            'critic': 3600,
            'detector': 1200,
        }
        
        self.l1_sizes = l1_sizes or default_l1_sizes
        self.l1_ttls = l1_ttls or default_l1_ttls
        self.l2_ttls = l2_ttls or default_l2_ttls
        
        # L1 caches by prefix
        self.l1_caches: Dict[str, TTLCache] = {}
        for prefix in self.l1_sizes:
            self.l1_caches[prefix] = TTLCache(
                maxsize=self.l1_sizes[prefix],
                ttl=self.l1_ttls[prefix]
            )
        
        # Statistics
        self.stats: Dict[str, CacheStats] = {}
        for prefix in self.l1_sizes:
            self.stats[prefix] = CacheStats()
    
    def _get_l1(self, prefix: str) -> TTLCache:
        """Get or create L1 cache for prefix."""
        if prefix not in self.l1_caches:
            self.l1_caches[prefix] = TTLCache(
                maxsize=self.l1_sizes.get(prefix, 1000),
                ttl=self.l1_ttls.get(prefix, 300)
            )
        return self.l1_caches[prefix]
    
    def _get_stats(self, prefix: str) -> CacheStats:
        """Get or create statistics for prefix."""
        if prefix not in self.stats:
            self.stats[prefix] = CacheStats()
        return self.stats[prefix]
    
    async def get(self, key: CacheKey) -> Optional[Any]:
        """
        Get value from cache (L1 -> L2).
        
        Args:
            key: Cache key
        
        Returns:
            Cached value or None if not found
        """
        stats = self._get_stats(key.prefix)
        key_str = str(key)
        
        # Try L1 first
        l1_cache = self._get_l1(key.prefix)
        if key_str in l1_cache:
            stats.hits += 1
            logger.debug(f"Cache L1 hit: {key_str}")
            return l1_cache[key_str]
        
        # Try L2 (Redis)
        if self.redis:
            try:
                cached_bytes = await self._redis_get(key_str)
                if cached_bytes:
                    # Deserialize and populate L1
                    value = json.loads(cached_bytes)
                    l1_cache[key_str] = value
                    stats.hits += 1
                    logger.debug(f"Cache L2 hit: {key_str}")
                    return value
            except Exception as e:
                logger.error(f"Redis get failed for {key_str}: {e}")
                stats.errors += 1
        
        # Cache miss
        stats.misses += 1
        logger.debug(f"Cache miss: {key_str}")
        return None
    
    async def set(self, key: CacheKey, value: Any) -> None:
        """
        Set value in cache (L1 + L2).
        
        Args:
            key: Cache key
            value: Value to cache
        """
        stats = self._get_stats(key.prefix)
        key_str = str(key)
        
        # Set in L1
        l1_cache = self._get_l1(key.prefix)
        l1_cache[key_str] = value
        
        # Set in L2 (Redis)
        if self.redis:
            try:
                ttl = self.l2_ttls.get(key.prefix, 3600)
                serialized = json.dumps(value, default=str)
                await self._redis_setex(key_str, ttl, serialized)
            except Exception as e:
                logger.error(f"Redis set failed for {key_str}: {e}")
                stats.errors += 1
        
        stats.sets += 1
        logger.debug(f"Cache set: {key_str}")
    
    async def get_or_compute(
        self,
        key: CacheKey,
        compute_fn: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Get from cache or compute and cache result.
        
        Args:
            key: Cache key
            compute_fn: Function to compute value if not cached
            *args: Arguments to compute_fn
            **kwargs: Keyword arguments to compute_fn
        
        Returns:
            Cached or computed value
        """
        # Try cache first
        cached = await self.get(key)
        if cached is not None:
            return cached
        
        # Compute value
        start_time = time.time()
        if asyncio.iscoroutinefunction(compute_fn):
            value = await compute_fn(*args, **kwargs)
        else:
            value = compute_fn(*args, **kwargs)
        
        compute_time = (time.time() - start_time) * 1000
        logger.debug(f"Computed {key} in {compute_time:.2f}ms")
        
        # Cache result
        await self.set(key, value)
        
        return value
    
    async def _redis_get(self, key: str) -> Optional[bytes]:
        """Get from Redis (async wrapper)."""
        if hasattr(self.redis, 'get'):
            # aioredis style
            return await self.redis.get(key)
        else:
            # redis-py style (sync)
            return await asyncio.get_event_loop().run_in_executor(
                None, self.redis.get, key
            )
    
    async def _redis_setex(self, key: str, ttl: int, value: str) -> None:
        """Set with expiration in Redis (async wrapper)."""
        if hasattr(self.redis, 'setex'):
            # aioredis style
            await self.redis.setex(key, ttl, value)
        else:
            # redis-py style (sync)
            await asyncio.get_event_loop().run_in_executor(
                None, self.redis.setex, key, ttl, value
            )
    
    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get cache statistics for all prefixes."""
        return {
            prefix: stats.to_dict()
            for prefix, stats in self.stats.items()
        }
    
    def clear_l1(self, prefix: Optional[str] = None) -> None:
        """Clear L1 cache for prefix or all prefixes."""
        if prefix:
            if prefix in self.l1_caches:
                self.l1_caches[prefix].clear()
        else:
            for cache in self.l1_caches.values():
                cache.clear()
        logger.info(f"Cleared L1 cache: {prefix or 'all'}")


__all__ = ["CacheManager", "CacheKey", "CacheStats"]
