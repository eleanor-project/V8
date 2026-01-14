"""
ELEANOR V8 — Cache Warming Strategy
------------------------------------

Proactive cache warming for frequently accessed data.
"""

import asyncio
import inspect
import logging
from typing import List, Optional, Dict, Any

from engine.cache import CacheKey

logger = logging.getLogger(__name__)


class CacheWarmer:
    """
    Warm cache with frequently accessed data.
    
    Proactively loads common queries and embeddings into cache
    to improve response times.
    """
    
    def __init__(
        self,
        precedent_retriever: Optional[Any] = None,
        embedding_service: Optional[Any] = None,
        cache_manager: Optional[Any] = None,
        router: Optional[Any] = None,
        router_cache: Optional[Any] = None,
        engine: Optional[Any] = None,
    ):
        self.precedent_retriever = precedent_retriever
        self.embedding_service = embedding_service
        self.cache_manager = cache_manager
        self.router = router
        self.router_cache = router_cache
        self.engine = engine
        self._warming_in_progress = False
    
    async def warm_precedent_cache(self, common_queries: List[str]) -> None:
        """
        Pre-warm precedent cache with common queries.
        
        Args:
            common_queries: List of common query strings
        """
        if not self.precedent_retriever:
            logger.warning("precedent_retriever_not_available_for_warming")
            return
        
        if self._warming_in_progress:
            logger.debug("cache_warming_already_in_progress")
            return
        
        self._warming_in_progress = True
        
        try:
            logger.info(
                "warming_precedent_cache",
                extra={"query_count": len(common_queries)},
            )
            
            # Warm cache in parallel (with limit)
            semaphore = asyncio.Semaphore(5)  # Limit concurrent operations
            
            async def warm_query(query: str):
                async with semaphore:
                    try:
                        await self.precedent_retriever.retrieve(query, [])
                        logger.debug(f"Warmed precedent cache for query: {query[:50]}")
                    except Exception as exc:
                        logger.warning(
                            "precedent_cache_warming_failed",
                            extra={"query": query[:50], "error": str(exc)},
                        )
            
            await asyncio.gather(*[warm_query(query) for query in common_queries], return_exceptions=True)
            
            logger.info("precedent_cache_warming_complete")
        
        finally:
            self._warming_in_progress = False
    
    async def warm_embedding_cache(self, texts: List[str]) -> None:
        """
        Pre-warm embedding cache.
        
        Args:
            texts: List of texts to generate embeddings for
        """
        if not self.embedding_service:
            logger.warning("embedding_service_not_available_for_warming")
            return
        
        if self._warming_in_progress:
            logger.debug("cache_warming_already_in_progress")
            return
        
        self._warming_in_progress = True
        
        try:
            logger.info(
                "warming_embedding_cache",
                extra={"text_count": len(texts)},
            )
            
            # Warm cache in parallel (with limit)
            semaphore = asyncio.Semaphore(10)  # Limit concurrent operations
            
            async def warm_embedding(text: str):
                async with semaphore:
                    try:
                        if hasattr(self.embedding_service, "get_embedding"):
                            await self.embedding_service.get_embedding(text)
                        elif hasattr(self.embedding_service, "embed"):
                            await self.embedding_service.embed([text])
                        logger.debug(f"Warmed embedding cache for text: {text[:50]}")
                    except Exception as exc:
                        logger.warning(
                            "embedding_cache_warming_failed",
                            extra={"text": text[:50], "error": str(exc)},
                        )
            
            await asyncio.gather(*[warm_embedding(text) for text in texts], return_exceptions=True)
            
            logger.info("embedding_cache_warming_complete")
        
        finally:
            self._warming_in_progress = False
    
    async def warm_router_cache(self, common_inputs: List[str]) -> None:
        """
        Pre-warm router selection cache.
        
        Args:
            common_inputs: List of common input texts
        """
        router = self.router or getattr(self.engine, "router", None)
        router_cache = self.router_cache or getattr(self.engine, "router_cache", None)
        cache_manager = self.cache_manager or getattr(self.engine, "cache_manager", None)

        if not router:
            logger.warning("router_not_available_for_warming")
            return
        if not router_cache and not cache_manager:
            logger.warning("router_cache_not_available_for_warming")
            return

        logger.info("warming_router_cache", extra={"input_count": len(common_inputs)})
        semaphore = asyncio.Semaphore(5)

        async def warm_input(text: str) -> None:
            async with semaphore:
                context: Dict[str, Any] = {}
                try:
                    result = router.route(text=text, context=context)
                    if inspect.isawaitable(result):
                        result = await result
                    selection = {
                        "model_info": {
                            "model_name": result.get("model_name"),
                            "model_version": result.get("model_version"),
                            "router_selection_reason": result.get("reason"),
                            "health_score": result.get("health_score"),
                            "cost_estimate": result.get("cost"),
                        },
                        "response_text": result.get("response_text") or "",
                    }
                    cache_key = CacheKey.from_data("router", text=text, context=context)
                    if cache_manager:
                        await cache_manager.set(cache_key, selection)
                    if router_cache:
                        router_cache.set(text, context, selection)
                    logger.debug(
                        "router_cache_warmed",
                        extra={"text_excerpt": text[:50], "model": selection["model_info"]["model_name"]},
                    )
                except Exception as exc:
                    logger.warning(
                        "router_cache_warming_failed",
                        extra={"text": text[:50], "error": str(exc)},
                    )

        await asyncio.gather(*[warm_input(text) for text in common_inputs], return_exceptions=True)
        logger.info("router_cache_warming_complete")
    
    async def warm_all(
        self,
        common_queries: Optional[List[str]] = None,
        common_texts: Optional[List[str]] = None,
    ) -> None:
        """
        Warm all caches.
        
        Args:
            common_queries: Common precedent queries
            common_texts: Common texts for embeddings
        """
        tasks = []
        
        if common_queries:
            tasks.append(self.warm_precedent_cache(common_queries))
        
        if common_texts:
            tasks.append(self.warm_embedding_cache(common_texts))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.info("cache_warming_complete")


__all__ = ["CacheWarmer"]
