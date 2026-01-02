"""
Async GPU Operations - CUDA stream management and async execution
"""

import asyncio
import torch
from typing import Callable, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class AsyncGPUExecutor:
    """
    Coordinate async GPU operations with CPU tasks.
    
    Features:
    - Multiple CUDA streams for parallel operations
    - Async/await integration with asyncio
    - Non-blocking GPU execution
    - Automatic stream synchronization
    
    Example:
        >>> executor = AsyncGPUExecutor(device=torch.device("cuda"), num_streams=4)
        >>> result = await executor.execute_async(my_gpu_function, tensor_input)
    """
    
    def __init__(self, device: torch.device, num_streams: int = 4):
        """
        Initialize async GPU executor.
        
        Args:
            device: Torch device to use
            num_streams: Number of CUDA streams for parallel operations
        """
        self.device = device
        
        if device.type == "cuda":
            # Create multiple CUDA streams for parallel operations
            self.streams = [torch.cuda.Stream(device) for _ in range(num_streams)]
            logger.info(f"Created {num_streams} CUDA streams for device {device}")
        else:
            # No streams for CPU/MPS
            self.streams = [None] * num_streams
            logger.info(f"Running on {device.type} - no stream management")
        
        self.current_stream_idx = 0
        self.num_streams = num_streams
        
        logger.info(
            "async_gpu_executor_initialized",
            device=str(device),
            num_streams=num_streams if device.type == "cuda" else 0
        )
    
    def get_stream(self) -> Optional[torch.cuda.Stream]:
        """
        Get next available CUDA stream (round-robin).
        
        Returns:
            CUDA stream or None for CPU/MPS
        """
        if self.device.type != "cuda":
            return None
        
        stream = self.streams[self.current_stream_idx]
        self.current_stream_idx = (self.current_stream_idx + 1) % self.num_streams
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
            Result of the GPU operation
        """
        stream = self.get_stream()
        
        if stream is not None:
            # CUDA: Use stream for parallel execution
            with torch.cuda.stream(stream):
                result = operation(*args, **kwargs)
                
                # Synchronize stream in executor to avoid blocking event loop
                # Run synchronization in thread pool to not block asyncio
                await asyncio.get_event_loop().run_in_executor(
                    None, stream.synchronize
                )
        else:
            # CPU or MPS: no stream management needed
            result = operation(*args, **kwargs)
        
        return result
    
    async def batch_execute(self, operations: List[Tuple[Callable, tuple, dict]]) -> List[Any]:
        """
        Execute multiple GPU operations in parallel using different streams.
        
        Args:
            operations: List of (callable, args, kwargs) tuples
        
        Returns:
            List of results in same order as operations
            
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
        
        # Execute all operations concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check for exceptions
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Operation {idx} failed: {result}")
        
        return results
    
    def synchronize_all(self):
        """
        Synchronize all CUDA streams.
        
        Blocks until all pending GPU operations complete.
        """
        if self.device.type == "cuda":
            for stream in self.streams:
                if stream is not None:
                    stream.synchronize()
            logger.debug("Synchronized all CUDA streams")
    
    async def synchronize_all_async(self):
        """
        Asynchronously synchronize all CUDA streams.
        """
        if self.device.type == "cuda":
            await asyncio.get_event_loop().run_in_executor(
                None, self.synchronize_all
            )
    
    def __repr__(self) -> str:
        return (
            f"AsyncGPUExecutor(device={self.device}, "
            f"num_streams={self.num_streams})"
        )
