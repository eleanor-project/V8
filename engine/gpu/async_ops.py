"""
Async GPU Operations for ELEANOR V8

Coordinate async GPU operations with CPU tasks.
"""

import asyncio
import logging
from functools import partial
from typing import Any, Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)

torch: Optional[Any]
try:
    import torch as _torch
except ImportError:  # pragma: no cover - torch optional
    torch = None
else:
    torch = _torch


class AsyncGPUExecutor:
    """
    Coordinate async GPU operations with CPU tasks.

    Features:
    - Multiple CUDA streams for parallel operations
    - Async/await integration
    - Non-blocking GPU execution
    - Stream synchronization management

    Example:
        >>> executor = AsyncGPUExecutor(device, num_streams=4)
        >>> result = await executor.execute_async(gpu_operation, tensor)
        >>>
        >>> # Batch execution
        >>> operations = [(op1, args1, kwargs1), (op2, args2, kwargs2)]
        >>> results = await executor.batch_execute(operations)
    """

    def __init__(self, device, num_streams: int = 4):
        """
        Initialize async GPU executor.

        Args:
            device: torch.device to use
            num_streams: Number of CUDA streams (ignored for CPU/MPS)
        """
        self.device = device
        self._torch_available = torch is not None
        self.streams: List[Optional[Any]] = [None] * num_streams
        self.current_stream_idx = 0

        if not self._torch_available or torch is None:
            logger.warning("PyTorch not installed. Async GPU operations unavailable.")
            return

        assert torch is not None

        # Create CUDA streams if using CUDA
        if device is not None and device.type == "cuda" and torch.cuda.is_available():
            self.streams = [torch.cuda.Stream(device) for _ in range(num_streams)]
            logger.info(
                "async_gpu_executor_initialized",
                extra={
                    "device": str(device),
                    "num_streams": num_streams,
                },
            )
        else:
            # No streams for CPU/MPS
            self.streams = [None] * num_streams
            logger.info(
                "async_gpu_executor_initialized_cpu",
                extra={
                    "device": str(device) if device else "none",
                    "note": "No streams (CPU/MPS mode)",
                },
            )

    def get_stream(self) -> Any:
        """
        Get next available CUDA stream (round-robin).

        Returns:
            CUDA stream or None for CPU/MPS
        """
        if not self._torch_available or self.device is None:
            return None

        if self.device.type != "cuda":
            return None

        stream = self.streams[self.current_stream_idx]
        self.current_stream_idx = (self.current_stream_idx + 1) % len(self.streams)
        return stream

    async def execute_async(self, operation: Callable, *args, **kwargs) -> Any:
        """
        Execute GPU operation asynchronously.

        GPU operations are non-blocking by default, but we need to
        synchronize when returning results to CPU.

        Args:
            operation: Callable to execute on GPU
            *args: Positional arguments for operation
            **kwargs: Keyword arguments for operation

        Returns:
            Result of operation
        """
        if not self._torch_available or torch is None:
            # Fallback to sync execution without torch
            return operation(*args, **kwargs)

        assert torch is not None

        stream = self.get_stream()
        call = partial(operation, *args, **kwargs)

        if stream is not None:
            # CUDA: Use stream
            with torch.cuda.stream(stream):
                result = call()

                # Synchronize stream in executor to avoid blocking event loop
                await asyncio.get_event_loop().run_in_executor(None, stream.synchronize)
        else:
            # CPU or MPS: no stream management needed
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, call)

        return result

    async def batch_execute(self, operations: List[Tuple[Callable, tuple, dict]]) -> List[Any]:
        """
        Execute multiple GPU operations in parallel using different streams.

        Args:
            operations: List of (callable, args, kwargs) tuples

        Returns:
            List of results in same order

        Example:
            >>> ops = [
            ...     (compute_embedding, (text1,), {}),
            ...     (compute_embedding, (text2,), {}),
            ...     (compute_embedding, (text3,), {}),
            ... ]
            >>> results = await executor.batch_execute(ops)
        """
        tasks = []

        for op, args, kwargs in operations:
            task = self.execute_async(op, *args, **kwargs)
            tasks.append(task)

        return await asyncio.gather(*tasks)

    def synchronize_all(self):
        """
        Synchronize all CUDA streams.

        Blocks until all streams complete their operations.
        """
        if not self._torch_available or self.device is None:
            return

        if self.device.type == "cuda":
            for stream in self.streams:
                if stream is not None:
                    stream.synchronize()

    def __repr__(self) -> str:
        """String representation."""
        if not self._torch_available:
            return "AsyncGPUExecutor(torch_unavailable)"

        device_str = str(self.device) if self.device else "none"
        num_streams = len(self.streams)

        return f"AsyncGPUExecutor(device={device_str}, num_streams={num_streams})"
