"""
ELEANOR V8 - Async GPU Operations

Coordinate async GPU operations with CPU tasks using CUDA streams.
"""

import asyncio
import logging
from typing import Callable, Any, List, Tuple, Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

logger = logging.getLogger(__name__)


class AsyncGPUExecutor:
    """
    Coordinate async GPU operations with CPU tasks.
    
    Features:
    - Multiple CUDA streams for parallel operations
    - Async/await integration
    - Non-blocking GPU execution
    - Stream synchronization management
    
    Example:
        >>> executor = AsyncGPUExecutor(device=torch.device("cuda"), num_streams=4)
        >>> result = await executor.execute_async(model.forward, input_tensor)
        
        >>> # Batch execution on different streams
        >>> operations = [(model1.forward, (x1,), {}), (model2.forward, (x2,), {})]
        >>> results = await executor.batch_execute(operations)
    """
    
    def __init__(self, device: Any, num_streams: int = 4):
        """
        Initialize async GPU executor.
        
        Args:
            device: torch.device to use
            num_streams: Number of CUDA streams for parallel operations
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available. AsyncGPUExecutor will run operations on CPU.")
            self.device = None
            self.streams = [None] * num_streams
            self.current_stream_idx = 0
            return
        
        self.device = device
        
        if device.type == "cuda":
            # Create CUDA streams for parallel operations
            self.streams = [torch.cuda.Stream(device) for _ in range(num_streams)]
        else:
            # No streams for CPU or MPS
            self.streams = [None] * num_streams
        
        self.current_stream_idx = 0
        
        logger.info(
            "async_gpu_executor_initialized",
            extra={
                "device": str(device),
                "num_streams": num_streams if device.type == "cuda" else 0,
                "device_type": device.type
            }
        )
    
    def get_stream(self) -> Optional[Any]:
        """
        Get next available CUDA stream (round-robin).
        
        Returns:
            torch.cuda.Stream or None for CPU/MPS
        """
        if not TORCH_AVAILABLE or self.device is None or self.device.type != "cuda":
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
            operation: Callable to execute (e.g., model forward pass)
            *args: Positional arguments for operation
            **kwargs: Keyword arguments for operation
            
        Returns:
            Result of the operation
        """
        if not TORCH_AVAILABLE or self.device is None:
            # Run on CPU without streams
            return operation(*args, **kwargs)
        
        stream = self.get_stream()
        
        if stream is not None:
            # Execute on CUDA stream
            with torch.cuda.stream(stream):
                result = operation(*args, **kwargs)
                
                # Synchronize stream in executor to avoid blocking event loop
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
            List of results in same order
            
        Example:
            >>> ops = [
            ...     (model1.forward, (input1,), {}),
            ...     (model2.forward, (input2,), {}),
            ...     (model3.forward, (input3,), {}),
            ... ]
            >>> results = await executor.batch_execute(ops)
        """
        tasks = []
        
        for op, args, kwargs in operations:
            task = self.execute_async(op, *args, **kwargs)
            tasks.append(task)
        
        return await asyncio.gather(*tasks)
    
    def synchronize_all(self):
        """Synchronize all CUDA streams."""
        if not TORCH_AVAILABLE or self.device is None or self.device.type != "cuda":
            return
        
        for stream in self.streams:
            if stream is not None:
                stream.synchronize()
    
    def __repr__(self) -> str:
        return (
            f"AsyncGPUExecutor(device={self.device}, "
            f"num_streams={len(self.streams)}, "
            f"torch_available={TORCH_AVAILABLE})"
        )
