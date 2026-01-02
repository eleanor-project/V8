"""
ELEANOR V8 — Async GPU Operations

Coordinate GPU operations with Python asyncio for non-blocking execution.
"""

import asyncio
import logging
import time
from typing import Any, Callable, List, Optional, TYPE_CHECKING
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .manager import GPUManager

class AsyncGPUExecutor:
    """
    Execute GPU operations asynchronously with proper coordination.
    
    Uses multiple CUDA streams for parallel GPU operations while
    maintaining async/await compatibility.
    """
    
    def __init__(self, gpu_manager: 'GPUManager', num_streams: int = 4):
        """
        Initialize async GPU executor.
        
        Args:
            gpu_manager: GPU manager instance
            num_streams: Number of CUDA streams for parallelism
        """
        self.gpu_manager = gpu_manager
        self.num_streams = num_streams
        self.streams = []
        self.current_stream_idx = 0
        
        # Create CUDA streams if available
        if gpu_manager.cuda_available:
            self.streams = [
                gpu_manager.torch.cuda.Stream()
                for _ in range(num_streams)
            ]
            logger.info(f"Created {num_streams} CUDA streams for async operations")
    
    def get_next_stream(self) -> Optional[Any]:
        """
        Get next available CUDA stream (round-robin).
        
        Returns:
            CUDA stream or None if not using CUDA
        """
        if not self.streams:
            return None
        
        stream = self.streams[self.current_stream_idx]
        self.current_stream_idx = (self.current_stream_idx + 1) % self.num_streams
        return stream
    
    @asynccontextmanager
    async def stream_context(self, stream_id: Optional[int] = None):
        """
        Context manager for CUDA stream execution.
        
        Args:
            stream_id: Specific stream ID or None for auto-selection
        """
        if not self.gpu_manager.cuda_available:
            yield None
            return
        
        if stream_id is not None and stream_id < len(self.streams):
            stream = self.streams[stream_id]
        else:
            stream = self.get_next_stream()
        
        try:
            with self.gpu_manager.torch.cuda.stream(stream):
                yield stream
        finally:
            # Stream cleanup happens automatically
            pass
    
    async def run_async(self, operation: Callable, *args, **kwargs) -> Any:
        """
        Run GPU operation asynchronously.
        
        Args:
            operation: Function to execute on GPU
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            Operation result
        """
        if not self.gpu_manager.is_available():
            # CPU fallback
            return await self._run_cpu(operation, *args, **kwargs)
        
        stream = self.get_next_stream()
        
        async with self.stream_context():
            start_time = time.time()
            
            # Execute operation
            result = operation(*args, **kwargs)
            
            # Wait for GPU completion without blocking event loop
            await self._sync_stream(stream)
            
            duration_ms = (time.time() - start_time) * 1000
            
            logger.debug(
                "GPU operation completed",
                extra={
                    "operation": operation.__name__,
                    "duration_ms": duration_ms,
                    "stream_id": self.current_stream_idx,
                },
            )
            
            return result
    
    async def _sync_stream(self, stream: Optional[Any]) -> None:
        """
        Synchronize CUDA stream without blocking event loop.
        
        Args:
            stream: CUDA stream to synchronize
        """
        if stream is None:
            return
        
        # Run synchronization in thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, stream.synchronize)
    
    async def _run_cpu(self, operation: Callable, *args, **kwargs) -> Any:
        """
        Run operation on CPU in thread pool.
        
        Args:
            operation: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            Operation result
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            lambda: operation(*args, **kwargs)
        )
    
    async def run_batch_parallel(
        self,
        operations: List[Callable],
        *args_list,
        **kwargs
    ) -> List[Any]:
        """
        Run multiple GPU operations in parallel using different streams.
        
        Args:
            operations: List of functions to execute
            *args_list: List of argument tuples
            **kwargs: Common keyword arguments
        
        Returns:
            List of results
        """
        if not operations:
            return []
        
        # Create tasks for parallel execution
        tasks = []
        for i, operation in enumerate(operations):
            args = args_list[i] if i < len(args_list) else ()
            task = asyncio.create_task(
                self.run_async(operation, *args, **kwargs)
            )
            tasks.append(task)
        
        # Wait for all operations
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Log any errors
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    "GPU operation %s failed",
                    i,
                    extra={
                        "operation": operations[i].__name__,
                        "error": str(result),
                    },
                )
        
        return results
    
    def get_stream_info(self) -> dict:
        """Get information about CUDA streams."""
        return {
            'num_streams': self.num_streams,
            'current_stream': self.current_stream_idx,
            'cuda_available': self.gpu_manager.cuda_available,
        }


__all__ = ["AsyncGPUExecutor"]
