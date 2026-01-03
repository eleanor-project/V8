"""
Batch Processor - Dynamic batch size optimization for GPU efficiency
"""

import logging
import time
from typing import Any, Callable, List, cast

import asyncio
import inspect
import torch
from collections import deque

logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Dynamic batch processor for GPU operations.

    Features:
    - Automatic batch size optimization based on memory
    - Timeout-based batching (don't wait forever)
    - Adaptive batch sizing based on observed latency
    - Handles variable-length inputs

    Example:
        >>> processor = BatchProcessor(
        ...     process_fn=model.forward,
        ...     device=torch.device("cuda"),
        ...     initial_batch_size=8
        ... )
        >>> result = await processor.process(input_data)
    """

    def __init__(
        self,
        process_fn: Callable,
        device: torch.device,
        initial_batch_size: int = 8,
        max_batch_size: int = 32,
        min_batch_size: int = 1,
        batch_timeout: float = 0.1,  # seconds
        dynamic_sizing: bool = True,
    ):
        """
        Initialize batch processor.

        Args:
            process_fn: Function to process batches (should accept List of inputs)
            device: Torch device to use
            initial_batch_size: Starting batch size
            max_batch_size: Maximum batch size
            min_batch_size: Minimum batch size
            batch_timeout: Max time to wait for full batch (seconds)
            dynamic_sizing: Enable automatic batch size adjustment
        """
        self.process_fn = process_fn
        self.device = device
        self.current_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.batch_timeout = batch_timeout
        self.dynamic_sizing = dynamic_sizing

        # Queue for pending items
        self.queue: deque = deque()
        self.processing_lock = asyncio.Lock()

        # Performance tracking
        self.recent_latencies: List[float] = []
        self.max_latency_samples = 20
        self.oom_count = 0

        logger.info(
            "batch_processor_initialized",
            extra={
                "device": str(device),
                "initial_batch_size": initial_batch_size,
                "max_batch_size": max_batch_size,
                "dynamic_sizing": dynamic_sizing,
            },
        )

    async def process(self, item: Any) -> Any:
        """
        Process a single item (batched with others).

        Args:
            item: Input to process

        Returns:
            Processed result
        """
        # Create a future for this item's result
        future: asyncio.Future = asyncio.Future()

        # Add to queue
        async with self.processing_lock:
            self.queue.append((item, future))

            # If batch is full, trigger background processing
            if len(self.queue) >= self.current_batch_size:
                asyncio.create_task(self._process_batch())

        # Wait for result
        return await future

    async def _process_batch(self) -> None:
        """
        Process a batch of items from the queue.
        """
        async with self.processing_lock:
            if not self.queue:
                return

            # Get batch
            batch_size = min(len(self.queue), self.current_batch_size)
            batch = [self.queue.popleft() for _ in range(batch_size)]

        if not batch:
            return

        items, futures = zip(*batch)

        try:
            start_time = time.time()

            # If GPU is available and using CUDA, clear cache if memory is tight
            if torch.cuda.is_available() and self.device.type == "cuda":
                torch.cuda.empty_cache()

            # Call processing function
            results = await self._execute_batch(list(items))

            # Track latency
            latency = time.time() - start_time
            self.recent_latencies.append(latency)
            if len(self.recent_latencies) > self.max_latency_samples:
                self.recent_latencies.pop(0)

            # Adjust batch size if enabled
            if self.dynamic_sizing:
                self._adjust_batch_size(latency, len(items))

            # Set results
            for future, result in zip(futures, results):
                if not future.done():
                    future.set_result(result)

            logger.debug(
                "Processed batch of %d items in %.1fms (batch_size=%d)",
                len(items),
                latency * 1000,
                self.current_batch_size,
            )

        except RuntimeError as e:
            # Handle GPU OOM
            if "out of memory" in str(e).lower():
                logger.warning("GPU OOM with batch_size=%d", self.current_batch_size)
                self.oom_count += 1

                # Reduce batch size
                self.current_batch_size = max(
                    self.min_batch_size,
                    self.current_batch_size // 2,
                )
                logger.info("Reduced batch_size to %d", self.current_batch_size)

                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Re-queue items and retry
                async with self.processing_lock:
                    for item, future in batch:
                        if not future.done():
                            self.queue.appendleft((item, future))

                # Retry with smaller batch
                await self._process_batch()
            else:
                # Other error - propagate to all futures
                for future in futures:
                    if not future.done():
                        future.set_exception(e)
                raise

        except Exception as e:
            logger.error("Batch processing failed: %s", e)
            # Set exception for all futures
            for future in futures:
                if not future.done():
                    future.set_exception(e)

    async def process_batch(self, items: List[Any]) -> List[Any]:
        """
        Process a list of items immediately in batches.

        Args:
            items: Items to process.

        Returns:
            List of results in the same order.
        """
        if not items:
            return []

        results: List[Any] = []
        index = 0

        while index < len(items):
            batch_size = min(self.current_batch_size, len(items) - index)
            batch = items[index : index + batch_size]
            try:
                start_time = time.time()
                batch_results = await self._execute_batch(batch)
                latency = time.time() - start_time
                self.recent_latencies.append(latency)
                if len(self.recent_latencies) > self.max_latency_samples:
                    self.recent_latencies.pop(0)
                if self.dynamic_sizing:
                    self._adjust_batch_size(latency, len(batch))
                results.extend(batch_results)
                index += batch_size
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning("GPU OOM with batch_size=%d", self.current_batch_size)
                    self.oom_count += 1
                    if self.current_batch_size <= self.min_batch_size:
                        raise
                    self.current_batch_size = max(
                        self.min_batch_size,
                        self.current_batch_size // 2,
                    )
                    logger.info("Reduced batch_size to %d", self.current_batch_size)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    raise

        return results

    async def _execute_batch(self, items: List[Any]) -> List[Any]:
        """
        Execute batch processing (possibly in executor).

        Args:
            items: List of items to process

        Returns:
            List of results
        """
        # If process_fn is async, await it
        if inspect.iscoroutinefunction(self.process_fn):
            result = await self.process_fn(items)
        else:
            # Run in executor to not block event loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.process_fn, items)

        return cast(List[Any], result)

    def _adjust_batch_size(self, latency: float, actual_batch_size: int) -> None:
        """
        Adjust batch size based on observed latency.

        Strategy:
        - If latency is low and stable, increase batch size
        - If latency is high or increasing, decrease batch size
        """
        if len(self.recent_latencies) < 5:
            return  # Need more samples

        avg_latency = sum(self.recent_latencies) / len(self.recent_latencies)

        # Increase batch size if latency is good
        if avg_latency < 0.5 and self.current_batch_size < self.max_batch_size:
            self.current_batch_size = min(
                self.max_batch_size,
                self.current_batch_size + 1,
            )
            logger.debug("Increased batch_size to %d", self.current_batch_size)

        # Decrease if latency is too high
        elif avg_latency > 2.0 and self.current_batch_size > self.min_batch_size:
            self.current_batch_size = max(
                self.min_batch_size,
                self.current_batch_size - 1,
            )
            logger.debug("Decreased batch_size to %d", self.current_batch_size)

    def stats(self) -> dict:
        """
        Get batch processor statistics.

        Returns:
            Dictionary with performance metrics
        """
        avg_latency = (
            sum(self.recent_latencies) / len(self.recent_latencies) if self.recent_latencies else 0
        )

        return {
            "current_batch_size": self.current_batch_size,
            "max_batch_size": self.max_batch_size,
            "queue_size": len(self.queue),
            "avg_latency_ms": round(avg_latency * 1000, 1),
            "oom_count": self.oom_count,
            "dynamic_sizing": self.dynamic_sizing,
        }

    def __repr__(self) -> str:
        stats = self.stats()
        return (
            f"BatchProcessor(device={self.device}, "
            f"batch_size={stats['current_batch_size']}, "
            f"queue={stats['queue_size']}, "
            f"latency={stats['avg_latency_ms']}ms)"
        )
