import asyncio
import inspect
import logging
from typing import Any, Optional

logger = logging.getLogger("engine.engine")


async def setup_resources(engine: Any) -> None:
    """Initialize engine resources that require async setup."""
    
    # Setup signal handlers for graceful shutdown
    try:
        from engine.resource_manager import ShutdownHandler
        shutdown_handler = ShutdownHandler(lambda: engine.shutdown(timeout=engine.config.shutdown_timeout_seconds))
        shutdown_handler.setup_handlers()
        engine._shutdown_handler = shutdown_handler
        logger.info("signal_handlers_registered")
    except Exception as exc:
        logger.warning("signal_handler_setup_failed", extra={"error": str(exc)})

    async def _maybe_call(obj: Any, method_name: str) -> None:
        method = getattr(obj, method_name, None)
        if not callable(method):
            return
        result = method()
        if inspect.isawaitable(result):
            await result

    if engine.recorder:
        await _maybe_call(engine.recorder, "initialize")
    if engine.cache_manager:
        await _maybe_call(engine.cache_manager, "connect")
    if engine.precedent_retriever:
        await _maybe_call(engine.precedent_retriever, "connect")

    resource_manager = getattr(engine, "resource_manager", None)
    if resource_manager is not None:
        try:
            await resource_manager.initialize(engine=engine)
        except Exception as exc:
            logger.warning("resource_manager_start_failed", extra={"error": str(exc)})


async def shutdown_engine(engine: Any, *, timeout: Optional[float] = None) -> None:
    """Gracefully shutdown engine and cleanup resources."""
    logger.info("engine_shutdown_initiated", extra={"instance_id": engine.instance_id})
    engine._shutdown_event.set()
    if timeout is None:
        timeout = engine.config.shutdown_timeout_seconds

    async def _close_resource(name: str, obj: Any, method_name: str) -> None:
        method = getattr(obj, method_name, None)
        if not callable(method):
            return
        try:
            result = method()
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            logger.warning(
                "resource_close_failed",
                extra={"resource": name, "error": str(exc)},
            )

    cleanup_coros = []
    if engine.recorder:
        cleanup_coros.append(_close_resource("recorder", engine.recorder, "close"))
    if engine.precedent_retriever:
        cleanup_coros.append(
            _close_resource("precedent_retriever", engine.precedent_retriever, "close")
        )
    if engine.cache_manager:
        cleanup_coros.append(_close_resource("cache_manager", engine.cache_manager, "close"))

    resource_manager = getattr(engine, "resource_manager", None)
    if resource_manager is not None:
        cleanup_coros.append(resource_manager.shutdown(timeout=timeout))

    if cleanup_coros:
        if timeout is None or timeout <= 0:
            await asyncio.gather(*cleanup_coros)
        else:
            try:
                await asyncio.wait_for(asyncio.gather(*cleanup_coros), timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning("engine_shutdown_timeout", extra={"timeout": timeout})

    for task in list(engine._cleanup_tasks):
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    if engine.gpu_embedding_cache is not None:
        try:
            engine.gpu_embedding_cache.clear_cache()
        except Exception as exc:
            logger.debug("gpu_cache_clear_failed", extra={"error": str(exc)})
    if engine.gpu_manager is not None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            logger.debug("gpu_cleanup_failed", extra={"error": str(exc)})

    try:
        from engine.utils.http_client import aclose_async_client

        await aclose_async_client()
    except Exception as exc:
        logger.debug("http_client_close_failed", extra={"error": str(exc)})

    logger.info("engine_shutdown_complete", extra={"instance_id": engine.instance_id})
