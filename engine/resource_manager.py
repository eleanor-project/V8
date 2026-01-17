"""
Resource Manager for ELEANOR V8 Engine

Provides:
- Async context management
- Graceful shutdown coordination
- Resource lifecycle tracking
- Signal handling for production deployments
- Connection pooling for external resources
- Memory pressure monitoring with cleanup callbacks
- Resource limit enforcement
- Health reporting for system resources
"""

import asyncio
import logging
import signal
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Set, Callable, Any, Dict, List, Optional

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None

try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False
    asyncpg = None

try:
    import redis.asyncio as redis_async
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis_async = None

logger = logging.getLogger(__name__)


class ResourceManager:
    """Manages lifecycle of engine resources"""

    def __init__(self, config: Optional[Any] = None):
        self._shutdown_event: asyncio.Event = asyncio.Event()
        self._cleanup_tasks: Set[asyncio.Task] = set()
        self._resources_initialized = False
        self._shutdown_in_progress = False
        self._cleanup_callbacks: List[Callable[[bool], Any]] = []
        self.config = config
        self.connection_pools: Optional[ConnectionPoolManager] = None
        self.memory_monitor: Optional[MemoryPressureMonitor] = None
        self.resource_limits: Optional[ResourceLimits] = None
        self.health_checker: Optional[ResourceHealthChecker] = None
        self.evidence_config = getattr(config, "evidence", None)
        self._engine_ref: Optional[Any] = None

        if config is not None:
            self.connection_pools = ConnectionPoolManager(getattr(config, "connections", None))
            self.memory_monitor = MemoryPressureMonitor(getattr(config, "memory", None))
            self.resource_limits = ResourceLimits(getattr(config, "limits", None))
            self.health_checker = ResourceHealthChecker(self, getattr(config, "health", None))

    def is_shutdown(self) -> bool:
        """Check if shutdown has been signaled"""
        return self._shutdown_event.is_set()

    def track_task(self, task: asyncio.Task) -> None:
        """Track a background task for cleanup"""
        self._cleanup_tasks.add(task)
        task.add_done_callback(self._cleanup_tasks.discard)

    def register_cleanup_callback(self, callback: Callable[[bool], Any]) -> None:
        """Register cleanup callback for memory pressure events."""
        self._cleanup_callbacks.append(callback)

    def create_evidence_buffer(self) -> "BoundedEvidenceBuffer":
        """Create a bounded evidence buffer based on config."""
        max_size = 1000
        if self.evidence_config is not None:
            max_size = getattr(self.evidence_config, "max_buffer_size", max_size)
        return BoundedEvidenceBuffer(max_size=max_size)

    def get_forensic_evidence(self, buffer: "BoundedEvidenceBuffer") -> List[Dict[str, Any]]:
        """Get bounded forensic output."""
        size = 200
        if self.evidence_config is not None:
            size = getattr(self.evidence_config, "forensic_output_size", size)
        return buffer.get_recent(size)

    async def initialize(self, engine: Optional[Any] = None) -> None:
        """Initialize resource manager components."""
        if self._resources_initialized:
            return
        self._resources_initialized = True
        self._engine_ref = engine

        if self.connection_pools:
            await self.connection_pools.initialize()

        if self.memory_monitor:
            if engine is not None:
                self.register_cleanup_callback(self._make_engine_cleanup(engine))
            self.memory_monitor.register_cleanup_callback(self._run_cleanup_callbacks)
            monitor_task = asyncio.create_task(self.memory_monitor.monitor())
            self.track_task(monitor_task)

    async def shutdown(self, timeout: float = 5.0) -> None:
        """Shutdown resource manager components."""
        if self._shutdown_in_progress:
            return
        self._shutdown_in_progress = True

        if self.memory_monitor:
            self.memory_monitor.stop()

        await self.cleanup_tasks(timeout=timeout)

        if self.connection_pools:
            await self.connection_pools.close()

    async def _run_cleanup_callbacks(self, emergency: bool = False) -> None:
        for callback in list(self._cleanup_callbacks):
            try:
                result = callback(emergency)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:
                logger.warning(
                    "resource_cleanup_failed",
                    extra={"error": str(exc), "emergency": emergency},
                )

    def _make_engine_cleanup(self, engine: Any) -> Callable[[bool], Any]:
        async def _cleanup(emergency: bool = False) -> None:
            cache_manager = getattr(engine, "cache_manager", None)
            if cache_manager is not None:
                try:
                    # Clear in-memory caches if available.
                    l1_caches = getattr(cache_manager, "l1_caches", None)
                    if isinstance(l1_caches, dict):
                        for cache in l1_caches.values():
                            cache.clear()
                    clear_fn = getattr(cache_manager, "clear", None)
                    if callable(clear_fn):
                        result = clear_fn()
                        if asyncio.iscoroutine(result):
                            await result
                except Exception as exc:
                    logger.debug("cache_cleanup_failed", extra={"error": str(exc)})

            gpu_cache = getattr(engine, "gpu_embedding_cache", None)
            if gpu_cache is not None:
                try:
                    gpu_cache.clear_cache()
                except Exception as exc:
                    logger.debug("gpu_cache_cleanup_failed", extra={"error": str(exc)})

            if emergency:
                logger.warning("emergency_resource_cleanup_triggered")

        return _cleanup
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
                asyncio.gather(*self._cleanup_tasks, return_exceptions=True), timeout=timeout
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
                    sig, lambda s=sig: asyncio.create_task(self._handle_shutdown(s))
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
    async def with_timeout(coro, timeout: float, operation: str, raise_on_timeout: bool = True):
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
                },
            )
            if raise_on_timeout:
                raise
            return None


class BoundedEvidenceBuffer:
    """Evidence buffer with bounded memory usage."""

    def __init__(self, max_size: int = 1000):
        self.buffer = deque(maxlen=max_size)
        self.overflow_count = 0

    def append(self, record: Any) -> None:
        if len(self.buffer) == self.buffer.maxlen:
            self.overflow_count += 1
        self.buffer.append(record)

    def __len__(self) -> int:
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)

    def get_recent(self, count: int = 200) -> List[Any]:
        return list(self.buffer)[-count:]

    def to_list(self) -> List[Any]:
        return list(self.buffer)


@dataclass
class ResourceUsageSnapshot:
    """Snapshot of current resource usage."""

    memory_percent: Optional[float] = None
    memory_rss_bytes: Optional[int] = None
    cpu_percent: Optional[float] = None
    open_files: Optional[int] = None
    thread_count: Optional[int] = None
    coroutine_count: Optional[int] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class ResourceLimitResult:
    """Result of enforcing resource limits."""

    compliant: bool
    violations: List[str]
    snapshot: ResourceUsageSnapshot


class MemoryPressureMonitor:
    """Monitor memory usage and trigger cleanup callbacks."""

    def __init__(self, config: Optional[Any] = None):
        warning = 0.75
        critical = 0.90
        interval = 30.0
        if config is not None:
            warning = getattr(config, "warning_threshold", warning)
            critical = getattr(config, "critical_threshold", critical)
            interval = getattr(config, "check_interval", interval)
        self.warning_threshold = warning
        self.critical_threshold = critical
        self.check_interval = interval
        self._monitoring = False
        self._cleanup_callbacks: List[Callable[[bool], Any]] = []
        self.last_state: Optional[str] = None

    def register_cleanup_callback(self, callback: Callable[[bool], Any]) -> None:
        self._cleanup_callbacks.append(callback)

    def get_memory_usage(self) -> Optional[float]:
        if not PSUTIL_AVAILABLE:
            return None
        process = psutil.Process()
        mem_info = process.memory_info()
        system_mem = psutil.virtual_memory()
        return mem_info.rss / system_mem.total

    async def monitor(self) -> None:
        if not PSUTIL_AVAILABLE:
            return
        self._monitoring = True
        while self._monitoring:
            await self.check_and_cleanup()
            await asyncio.sleep(self.check_interval)

    async def check_and_cleanup(self) -> None:
        usage = self.get_memory_usage()
        if usage is None:
            return
        if usage >= self.critical_threshold:
            if self.last_state != "critical":
                logger.critical("critical_memory_pressure", extra={"usage": usage})
            await self._trigger_cleanup(emergency=True)
            self.last_state = "critical"
        elif usage >= self.warning_threshold:
            if self.last_state != "warning":
                logger.warning("memory_pressure_warning", extra={"usage": usage})
            await self._trigger_cleanup(emergency=False)
            self.last_state = "warning"
        else:
            self.last_state = "ok"

    async def _trigger_cleanup(self, emergency: bool) -> None:
        for callback in list(self._cleanup_callbacks):
            try:
                result = callback(emergency)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:
                logger.warning(
                    "memory_cleanup_failed",
                    extra={"error": str(exc), "emergency": emergency},
                )

    def stop(self) -> None:
        self._monitoring = False


class ConnectionPoolManager:
    """Manage connection pools for external resources."""

    def __init__(self, config: Optional[Any] = None):
        self.config = config
        self.http_pool: Optional[Any] = None
        self.db_pool: Optional[Any] = None
        self.redis_pool: Optional[Any] = None

    async def initialize(self) -> None:
        if self.config is None:
            return

        if HTTPX_AVAILABLE:
            try:
                limits = httpx.Limits(
                    max_connections=getattr(self.config, "http_pool_size", 100),
                    max_keepalive_connections=getattr(self.config, "http_per_host_limit", 10),
                )
                self.http_pool = httpx.AsyncClient(
                    limits=limits,
                    timeout=None,
                    transport=httpx.AsyncHTTPTransport(
                        retries=0,
                        keepalive_expiry=getattr(self.config, "http_keepalive_timeout", 30.0),
                    ),
                )
            except Exception as exc:
                logger.warning("http_pool_init_failed", extra={"error": str(exc)})
        else:
            logger.debug("http_pool_unavailable")

        database_url = getattr(self.config, "database_url", None)
        if ASYNCPG_AVAILABLE and database_url:
            try:
                self.db_pool = await asyncpg.create_pool(
                    dsn=database_url,
                    min_size=getattr(self.config, "db_pool_min", 2),
                    max_size=getattr(self.config, "db_pool_max", 20),
                    command_timeout=getattr(self.config, "db_timeout", 10.0),
                )
            except Exception as exc:
                logger.warning("db_pool_init_failed", extra={"error": str(exc)})
        elif database_url:
            logger.warning("db_pool_unavailable", extra={"reason": "asyncpg_not_installed"})

        redis_url = getattr(self.config, "redis_url", None)
        if REDIS_AVAILABLE and redis_url:
            try:
                self.redis_pool = redis_async.ConnectionPool.from_url(
                    redis_url,
                    max_connections=getattr(self.config, "redis_pool_size", 50),
                )
            except Exception as exc:
                logger.warning("redis_pool_init_failed", extra={"error": str(exc)})
        elif redis_url:
            logger.warning("redis_pool_unavailable", extra={"reason": "redis_not_installed"})

    async def close(self) -> None:
        if self.http_pool is not None:
            try:
                await self.http_pool.aclose()
            except Exception as exc:
                logger.debug("http_pool_close_failed", extra={"error": str(exc)})
            self.http_pool = None

        if self.db_pool is not None:
            try:
                await self.db_pool.close()
            except Exception as exc:
                logger.debug("db_pool_close_failed", extra={"error": str(exc)})
            self.db_pool = None

        if self.redis_pool is not None:
            try:
                result = self.redis_pool.disconnect(inuse_connections=True)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:
                logger.debug("redis_pool_close_failed", extra={"error": str(exc)})
            self.redis_pool = None

    async def health_check(self) -> Dict[str, bool]:
        status = {"http": True, "database": True, "redis": True}
        if self.http_pool is not None:
            status["http"] = not getattr(self.http_pool, "is_closed", False)
        if self.db_pool is not None:
            status["database"] = not getattr(self.db_pool, "_closed", False)
        if self.redis_pool is not None:
            status["redis"] = True
        return status


class ResourceLimits:
    """Enforce resource usage limits."""

    def __init__(self, config: Optional[Any] = None):
        self.config = config

    def snapshot(self) -> ResourceUsageSnapshot:
        snapshot = ResourceUsageSnapshot()
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                snapshot.memory_percent = psutil.virtual_memory().percent / 100.0
                snapshot.memory_rss_bytes = process.memory_info().rss
                snapshot.cpu_percent = process.cpu_percent(interval=0.0) / 100.0
                try:
                    snapshot.open_files = process.num_fds()
                except Exception:
                    try:
                        snapshot.open_files = len(process.open_files())
                    except Exception:
                        snapshot.open_files = None
                snapshot.thread_count = process.num_threads()
            except Exception as exc:
                logger.debug("resource_snapshot_failed", extra={"error": str(exc)})
        try:
            snapshot.coroutine_count = len(asyncio.all_tasks())
        except RuntimeError:
            snapshot.coroutine_count = None
        return snapshot

    def enforce(self) -> ResourceLimitResult:
        violations: List[str] = []
        snapshot = self.snapshot()
        if self.config is None:
            return ResourceLimitResult(compliant=True, violations=[], snapshot=snapshot)

        max_memory_gb = getattr(self.config, "max_memory_gb", None)
        if max_memory_gb and snapshot.memory_rss_bytes is not None:
            memory_gb = snapshot.memory_rss_bytes / (1024 ** 3)
            if memory_gb > max_memory_gb:
                violations.append(f"memory {memory_gb:.2f}GB > {max_memory_gb:.2f}GB")

        max_open_files = getattr(self.config, "max_open_files", None)
        if max_open_files and snapshot.open_files is not None:
            if snapshot.open_files > max_open_files:
                violations.append(f"open_files {snapshot.open_files} > {max_open_files}")

        max_threads = getattr(self.config, "max_threads", None)
        if max_threads and snapshot.thread_count is not None:
            if snapshot.thread_count > max_threads:
                violations.append(f"threads {snapshot.thread_count} > {max_threads}")

        max_coroutines = getattr(self.config, "max_coroutines", None)
        if max_coroutines and snapshot.coroutine_count is not None:
            if snapshot.coroutine_count > max_coroutines:
                violations.append(f"coroutines {snapshot.coroutine_count} > {max_coroutines}")

        return ResourceLimitResult(compliant=not violations, violations=violations, snapshot=snapshot)


@dataclass
class ResourceHealthReport:
    """Aggregated health report for resources."""

    healthy: bool
    checks: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)


class ResourceHealthChecker:
    """Comprehensive resource health checks."""

    def __init__(self, manager: ResourceManager, config: Optional[Any] = None):
        self.manager = manager
        self.config = config

    async def check_all(self) -> ResourceHealthReport:
        checks = {
            "memory": await self.check_memory(),
            "connections": await self.check_connections(),
            "disk": await self.check_disk_space(),
            "limits": await self.check_limits(),
        }
        healthy = all(value in (True, "ok", "not_configured") for value in checks.values())
        return ResourceHealthReport(healthy=healthy, checks=checks)

    async def check_memory(self) -> Any:
        if not PSUTIL_AVAILABLE:
            return "not_configured"
        usage = psutil.virtual_memory().percent / 100.0
        warning = 0.90
        if self.manager.memory_monitor is not None:
            warning = self.manager.memory_monitor.warning_threshold
        return usage < warning

    async def check_connections(self) -> Any:
        if not self.manager.connection_pools:
            return "not_configured"
        status = await self.manager.connection_pools.health_check()
        return all(status.values())

    async def check_disk_space(self) -> Any:
        try:
            usage = psutil.disk_usage("/").percent / 100.0 if PSUTIL_AVAILABLE else None
        except Exception:
            usage = None
        if usage is None:
            return "not_configured"
        warning = getattr(self.config, "disk_warning_threshold", 0.90)
        critical = getattr(self.config, "disk_critical_threshold", 0.95)
        if usage >= critical:
            return False
        return usage < warning

    async def check_limits(self) -> Any:
        if not self.manager.resource_limits:
            return "not_configured"
        result = self.manager.resource_limits.enforce()
        return result.compliant
