"""
ELEANOR V8 — Adaptive Batch Sizer
-----------------------------------

Dynamically adjust batch sizes based on performance metrics.
"""

import time
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for batch sizing."""
    latency: float
    success_rate: float
    timestamp: float = field(default_factory=time.time)


class AdaptiveBatchSizer:
    """
    Dynamically adjust batch sizes based on performance.
    
    Automatically tunes batch size to optimize for:
    - Latency (target: < 2 seconds)
    - Success rate (target: > 95%)
    - Throughput
    """
    
    def __init__(
        self,
        initial_batch_size: int = 5,
        min_batch_size: int = 2,
        max_batch_size: int = 10,
        target_latency: float = 2.0,
        min_success_rate: float = 0.90,
        adjustment_cooldown: float = 30.0,
    ):
        """
        Initialize adaptive batch sizer.
        
        Args:
            initial_batch_size: Starting batch size
            min_batch_size: Minimum batch size
            max_batch_size: Maximum batch size
            target_latency: Target latency in seconds
            min_success_rate: Minimum acceptable success rate
            adjustment_cooldown: Seconds between adjustments
        """
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.target_latency = target_latency
        self.min_success_rate = min_success_rate
        self.adjustment_cooldown = adjustment_cooldown
        
        self.performance_history: List[PerformanceMetrics] = []
        self.last_adjustment = time.time()
        self.adjustment_count = 0
    
    def record_performance(
        self,
        latency: float,
        success_rate: float,
    ) -> None:
        """Record performance metrics."""
        metrics = PerformanceMetrics(
            latency=latency,
            success_rate=success_rate,
        )
        self.performance_history.append(metrics)
        
        # Keep only recent history (last 100 entries)
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
    
    def adjust_batch_size(
        self,
        current_latency: Optional[float] = None,
        current_success_rate: Optional[float] = None,
    ) -> int:
        """
        Adjust batch size based on recent performance.
        
        Args:
            current_latency: Current latency (if None, uses recent average)
            current_success_rate: Current success rate (if None, uses recent average)
        
        Returns:
            Adjusted batch size
        """
        # Cooldown check
        now = time.time()
        if now - self.last_adjustment < self.adjustment_cooldown:
            return self.current_batch_size
        
        # Calculate metrics from history if not provided
        if current_latency is None or current_success_rate is None:
            if not self.performance_history:
                return self.current_batch_size
            
            # Use average of last 10 entries
            recent = self.performance_history[-10:]
            current_latency = current_latency or sum(m.latency for m in recent) / len(recent)
            current_success_rate = current_success_rate or sum(m.success_rate for m in recent) / len(recent)
        
        old_batch_size = self.current_batch_size
        
        # Decision logic
        if current_latency < self.target_latency and current_success_rate > self.min_success_rate:
            # Performance is good: try increasing batch size
            if current_success_rate > 0.98 and current_latency < self.target_latency * 0.7:
                # Excellent performance: increase more aggressively
                self.current_batch_size = min(
                    self.max_batch_size,
                    self.current_batch_size + 2,
                )
            else:
                # Good performance: increase conservatively
                self.current_batch_size = min(
                    self.max_batch_size,
                    self.current_batch_size + 1,
                )
        elif current_latency > self.target_latency * 1.5 or current_success_rate < self.min_success_rate:
            # Performance is poor: decrease batch size
            if current_latency > self.target_latency * 2.0 or current_success_rate < 0.80:
                # Very poor performance: decrease aggressively
                self.current_batch_size = max(
                    self.min_batch_size,
                    self.current_batch_size - 2,
                )
            else:
                # Poor performance: decrease conservatively
                self.current_batch_size = max(
                    self.min_batch_size,
                    self.current_batch_size - 1,
                )
        # Otherwise: keep current batch size
        
        if self.current_batch_size != old_batch_size:
            self.last_adjustment = now
            self.adjustment_count += 1
            logger.info(
                "batch_size_adjusted",
                extra={
                    "old_size": old_batch_size,
                    "new_size": self.current_batch_size,
                    "latency": current_latency,
                    "success_rate": current_success_rate,
                    "adjustment_count": self.adjustment_count,
                },
            )
        
        return self.current_batch_size
    
    def get_current_batch_size(self) -> int:
        """Get current batch size."""
        return self.current_batch_size
    
    def reset(self) -> None:
        """Reset to initial batch size."""
        self.current_batch_size = self.min_batch_size + (self.max_batch_size - self.min_batch_size) // 2
        self.performance_history.clear()
        self.last_adjustment = time.time()
        self.adjustment_count = 0


__all__ = ["AdaptiveBatchSizer", "PerformanceMetrics"]
