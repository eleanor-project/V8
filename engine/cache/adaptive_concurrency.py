"""
ELEANOR V8 — Adaptive Concurrency Management

Dynamically adjust concurrency limits based on observed latency.
"""

import asyncio
import logging
from collections import deque

logger = logging.getLogger(__name__)


class AdaptiveConcurrencyManager:
    """
    Adaptive concurrency control based on latency feedback.

    Increases concurrency when latencies are low and stable.
    Decreases concurrency when latencies degrade.
    """

    def __init__(
        self,
        initial_limit: int = 6,
        min_limit: int = 2,
        max_limit: int = 20,
        target_latency_ms: float = 500.0,
        window_size: int = 100,
    ):
        """
        Initialize adaptive concurrency manager.

        Args:
            initial_limit: Starting concurrency limit
            min_limit: Minimum concurrency limit
            max_limit: Maximum concurrency limit
            target_latency_ms: Target P95 latency in milliseconds
            window_size: Number of observations for adjustment decisions
        """
        self.current_limit = initial_limit
        self.min_limit = min_limit
        self.max_limit = max_limit
        self.target_latency_ms = target_latency_ms
        self.window_size = window_size

        # Latency observations
        self.latencies: deque = deque(maxlen=window_size)

        # Semaphore for concurrency control
        self.semaphore = asyncio.Semaphore(initial_limit)

        # Adjustment counter
        self.observations_since_adjust = 0
        self.adjust_every = 20  # Adjust after N observations

    def record_latency(self, latency_ms: float) -> None:
        """
        Record operation latency.

        Args:
            latency_ms: Observed latency in milliseconds
        """
        self.latencies.append(latency_ms)
        self.observations_since_adjust += 1

        # Adjust concurrency periodically
        if self.observations_since_adjust >= self.adjust_every:
            self._adjust_concurrency()
            self.observations_since_adjust = 0

    def _adjust_concurrency(self) -> None:
        """
        Adjust concurrency limit based on latency observations.
        """
        if len(self.latencies) < 10:
            return  # Not enough data

        # Calculate statistics
        sorted_latencies = sorted(self.latencies)
        p50 = sorted_latencies[len(sorted_latencies) // 2]
        p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)]
        old_limit = self.current_limit

        # Decision logic
        if p95 > self.target_latency_ms * 1.5:
            # Latency too high, reduce concurrency
            self.current_limit = max(self.min_limit, self.current_limit - 1)
            logger.info(
                f"Reducing concurrency: {old_limit} -> {self.current_limit} "
                f"(P95={p95:.1f}ms, target={self.target_latency_ms}ms)"
            )
        elif p95 < self.target_latency_ms * 0.7 and p50 < self.target_latency_ms * 0.5:
            # Latency good, can increase concurrency
            self.current_limit = min(self.max_limit, self.current_limit + 1)
            logger.info(
                f"Increasing concurrency: {old_limit} -> {self.current_limit} "
                f"(P95={p95:.1f}ms, P50={p50:.1f}ms)"
            )

        # Recreate semaphore if limit changed
        if old_limit != self.current_limit:
            self.semaphore = asyncio.Semaphore(self.current_limit)

    async def __aenter__(self):
        """Context manager entry."""
        await self.semaphore.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.semaphore.release()

    def get_stats(self) -> dict:
        """Get current statistics."""
        if not self.latencies:
            return {
                "current_limit": self.current_limit,
                "observations": 0,
            }

        sorted_latencies = sorted(self.latencies)
        return {
            "current_limit": self.current_limit,
            "observations": len(self.latencies),
            "avg_latency_ms": sum(self.latencies) / len(self.latencies),
            "p50_latency_ms": sorted_latencies[len(sorted_latencies) // 2],
            "p95_latency_ms": sorted_latencies[int(len(sorted_latencies) * 0.95)],
            "target_latency_ms": self.target_latency_ms,
        }


__all__ = ["AdaptiveConcurrencyManager"]
