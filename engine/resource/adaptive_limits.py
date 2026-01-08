"""
ELEANOR V8 — Adaptive Resource Limits
--------------------------------------

Adapt resource limits based on system load and conditions.
"""

import asyncio
import logging
import time
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None


@dataclass
class SystemMetrics:
    """System resource metrics."""
    cpu_percent: float
    memory_percent: float
    load_average: Optional[float] = None
    timestamp: float = 0.0
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


class AdaptiveResourceLimiter:
    """
    Adapt resource limits based on system load.
    
    Automatically adjusts:
    - Max concurrency
    - Cache sizes
    - Timeout values
    - Batch sizes
    """
    
    def __init__(
        self,
        base_concurrency: int = 6,
        max_concurrency: int = 50,
        min_concurrency: int = 1,
        cpu_threshold_high: float = 0.8,
        cpu_threshold_low: float = 0.3,
        memory_threshold_high: float = 0.85,
        memory_threshold_low: float = 0.4,
    ):
        """
        Initialize adaptive resource limiter.
        
        Args:
            base_concurrency: Base concurrency limit
            max_concurrency: Maximum concurrency limit
            min_concurrency: Minimum concurrency limit
            cpu_threshold_high: CPU threshold for reducing concurrency
            cpu_threshold_low: CPU threshold for increasing concurrency
            memory_threshold_high: Memory threshold for reducing concurrency
            memory_threshold_low: Memory threshold for increasing concurrency
        """
        self.base_concurrency = base_concurrency
        self.max_concurrency = max_concurrency
        self.min_concurrency = min_concurrency
        self.cpu_threshold_high = cpu_threshold_high
        self.cpu_threshold_low = cpu_threshold_low
        self.memory_threshold_high = memory_threshold_high
        self.memory_threshold_low = memory_threshold_low
        
        self._current_concurrency = base_concurrency
        self._last_adjustment = time.time()
        self._adjustment_cooldown = 30.0  # Seconds between adjustments
    
    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """Get current system metrics."""
        if not PSUTIL_AVAILABLE:
            return None
        
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent
            
            # Get load average if available
            try:
                load_avg = psutil.getloadavg()[0] if hasattr(psutil, "getloadavg") else None
            except AttributeError:
                load_avg = None
            
            return SystemMetrics(
                cpu_percent=cpu_percent / 100.0,  # Normalize to 0-1
                memory_percent=memory_percent / 100.0,  # Normalize to 0-1
                load_average=load_avg,
            )
        except Exception as exc:
            logger.warning(
                "failed_to_get_system_metrics",
                extra={"error": str(exc)},
            )
            return None
    
    def adjust_concurrency(self, current_load: Optional[float] = None) -> int:
        """
        Adjust max concurrency based on system load.
        
        Args:
            current_load: Optional current load (0-1), if None will be measured
        
        Returns:
            Adjusted concurrency limit
        """
        # Cooldown check
        now = time.time()
        if now - self._last_adjustment < self._adjustment_cooldown:
            return self._current_concurrency
        
        metrics = self.get_current_metrics()
        if not metrics:
            return self._current_concurrency
        
        # Determine load (use provided or measured)
        if current_load is None:
            # Use CPU as primary indicator, memory as secondary
            current_load = max(metrics.cpu_percent, metrics.memory_percent)
        
        # Adjust concurrency
        old_concurrency = self._current_concurrency
        
        if current_load > self.cpu_threshold_high or metrics.memory_percent > self.memory_threshold_high:
            # High load: reduce concurrency
            self._current_concurrency = max(
                self.min_concurrency,
                int(self._current_concurrency * 0.7),  # Reduce by 30%
            )
        elif current_load < self.cpu_threshold_low and metrics.memory_percent < self.memory_threshold_low:
            # Low load: increase concurrency
            self._current_concurrency = min(
                self.max_concurrency,
                int(self._current_concurrency * 1.5),  # Increase by 50%
            )
        # Otherwise keep current
        
        if self._current_concurrency != old_concurrency:
            logger.info(
                "concurrency_adjusted",
                extra={
                    "old": old_concurrency,
                    "new": self._current_concurrency,
                    "cpu_percent": metrics.cpu_percent * 100,
                    "memory_percent": metrics.memory_percent * 100,
                    "load": current_load,
                },
            )
            self._last_adjustment = now
        
        return self._current_concurrency
    
    def get_current_concurrency(self) -> int:
        """Get current concurrency limit."""
        return self._current_concurrency
    
    def reset(self) -> None:
        """Reset to base concurrency."""
        self._current_concurrency = self.base_concurrency
        self._last_adjustment = time.time()


class MemoryMonitor:
    """Monitor memory usage and trigger cleanup."""
    
    def __init__(
        self,
        threshold: float = 0.85,
        cleanup_threshold: float = 0.90,
        check_interval: float = 30.0,
    ):
        """
        Initialize memory monitor.
        
        Args:
            threshold: Memory threshold to start monitoring (0-1)
            cleanup_threshold: Memory threshold to trigger cleanup (0-1)
            check_interval: Interval between checks in seconds
        """
        self.threshold = threshold
        self.cleanup_threshold = cleanup_threshold
        self.check_interval = check_interval
        self._monitoring = False
        self._cleanup_callbacks: list = []
    
    def register_cleanup_callback(self, callback) -> None:
        """Register callback for memory cleanup."""
        self._cleanup_callbacks.append(callback)
    
    async def monitor(self) -> None:
        """Monitor memory and trigger cleanup if needed."""
        if not PSUTIL_AVAILABLE:
            return
        
        self._monitoring = True
        
        while self._monitoring:
            try:
                memory_percent = psutil.virtual_memory().percent / 100.0
                
                if memory_percent > self.cleanup_threshold:
                    logger.warning(
                        "memory_pressure_detected",
                        extra={"memory_percent": memory_percent * 100},
                    )
                    await self.trigger_cleanup()
                elif memory_percent > self.threshold:
                    logger.debug(
                        "memory_approaching_threshold",
                        extra={"memory_percent": memory_percent * 100},
                    )
                
                await asyncio.sleep(self.check_interval)
            
            except Exception as exc:
                logger.error(
                    "memory_monitoring_error",
                    extra={"error": str(exc)},
                    exc_info=True,
                )
                await asyncio.sleep(self.check_interval)
    
    async def trigger_cleanup(self) -> None:
        """Trigger cleanup callbacks."""
        logger.info("triggering_memory_cleanup")
        
        for callback in self._cleanup_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as exc:
                logger.error(
                    "cleanup_callback_failed",
                    extra={"error": str(exc)},
                    exc_info=True,
                )
    
    def stop(self) -> None:
        """Stop monitoring."""
        self._monitoring = False


__all__ = [
    "SystemMetrics",
    "AdaptiveResourceLimiter",
    "MemoryMonitor",
]
