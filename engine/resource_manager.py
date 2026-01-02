"""
Resource Manager for ELEANOR V8 Engine

Provides:
- Async context management
- Graceful shutdown coordination
- Resource lifecycle tracking
- Signal handling for production deployments
"""

import asyncio
import logging
import signal
from typing import Set, Callable, Any, Dict

logger = logging.getLogger(__name__)


class ResourceManager:
    """Manages lifecycle of engine resources"""
    
    def __init__(self):
        self._shutdown_event = asyncio.Event()
        self._cleanup_tasks: Set[asyncio.Task] = set()
        self._resources_initialized = False
        self._shutdown_in_progress = False
        
    def is_shutdown(self) -> bool:
        """Check if shutdown has been signaled"""
        return self._shutdown_event.is_set()
    
    def track_task(self, task: asyncio.Task) -> None:
        """Track a background task for cleanup"""
        self._cleanup_tasks.add(task)
        task.add_done_callback(self._cleanup_tasks.discard)
    
    async def wait_for_shutdown(self):
        """Wait for shutdown signal"""
        await self._shutdown_event.wait()
    
    def signal_shutdown(self):
        """Signal all resources to begin shutdown"""
        self._shutdown_event.set()
    
    async def cleanup_tasks(self, timeout: float = 5.0):
        """Cancel and await all tracked tasks"""
        if not self._cleanup_tasks:
            return
        
        logger.info(f"Cancelling {len(self._cleanup_tasks)} background tasks")
        
        for task in self._cleanup_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks with timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._cleanup_tasks, return_exceptions=True),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"Task cleanup exceeded {timeout}s timeout")
        except Exception as exc:
            logger.error(f"Task cleanup failed: {exc}", exc_info=True)
        
        self._cleanup_tasks.clear()


class ShutdownHandler:
    """Handle graceful shutdown on signals"""
    
    def __init__(self, shutdown_callback: Callable[[], Any]):
        """
        Args:
            shutdown_callback: Async or sync function to call on shutdown signal
        """
        self.shutdown_callback = shutdown_callback
        self._shutdown_event = asyncio.Event()
        self._original_handlers: Dict[signal.Signals, Any] = {}
        
    def setup_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        loop = asyncio.get_event_loop()
        
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                # Store original handler
                self._original_handlers[sig] = signal.getsignal(sig)
                
                # Set new handler
                loop.add_signal_handler(
                    sig,
                    lambda s=sig: asyncio.create_task(self._handle_shutdown(s))
                )
                logger.debug(f"Registered signal handler for {sig.name}")
            except (ValueError, RuntimeError) as exc:
                # On Windows or non-main thread, signal handlers may not work
                logger.warning(f"Could not register signal handler for {sig.name}: {exc}")
    
    async def _handle_shutdown(self, sig: signal.Signals):
        """Handle shutdown signal"""
        if self._shutdown_event.is_set():
            logger.warning(f"Shutdown signal {sig.name} received, but shutdown already in progress")
            return
        
        logger.info(f"Shutdown signal received: {sig.name}")
        self._shutdown_event.set()
        
        try:
            result = self.shutdown_callback()
            if asyncio.iscoroutine(result):
                await result
        except Exception as exc:
            logger.error(f"Shutdown callback failed: {exc}", exc_info=True)
    
    def restore_handlers(self):
        """Restore original signal handlers"""
        for sig, handler in self._original_handlers.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, RuntimeError):
                pass


class TimeoutProtection:
    """Protect operations with timeouts"""
    
    @staticmethod
    async def with_timeout(
        coro,
        timeout: float,
        operation: str,
        raise_on_timeout: bool = True
    ):
        """
        Execute coroutine with timeout protection.
        
        Args:
            coro: Coroutine to execute
            timeout: Timeout in seconds
            operation: Operation name for logging
            raise_on_timeout: Whether to raise TimeoutError or return None
        
        Returns:
            Result of coroutine or None if timeout and raise_on_timeout=False
        """
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            logger.error(
                "operation_timeout",
                extra={
                    "operation": operation,
                    "timeout_seconds": timeout,
                }
            )
            if raise_on_timeout:
                raise
            return None
