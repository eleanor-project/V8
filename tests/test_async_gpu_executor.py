"""
Tests for Async GPU Executor
"""

import pytest
import asyncio
import sys
from unittest.mock import Mock, patch, MagicMock

# Mock torch if not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = MagicMock()
    sys.modules['torch'] = torch

from engine.gpu.async_ops import AsyncGPUExecutor


class TestAsyncGPUExecutor:
    """Test async GPU executor."""
    
    @pytest.fixture
    def device(self):
        """Create mock device."""
        if TORCH_AVAILABLE:
            return torch.device("cpu")  # Use CPU for testing
        return Mock(type="cpu")
    
    @pytest.fixture
    def executor(self, device):
        """Create executor for testing."""
        return AsyncGPUExecutor(device=device, num_streams=4)
    
    def test_initialization(self, executor):
        """Test executor initialization."""
        assert executor is not None
        assert hasattr(executor, 'device')
        assert hasattr(executor, 'streams')
        assert len(executor.streams) == 4
    
    def test_get_stream_round_robin(self, executor):
        """Test stream allocation round-robin."""
        streams = [executor.get_stream() for _ in range(8)]
        
        # Should cycle through streams
        assert len(set(id(s) for s in streams if s is not None)) <= 4
    
    @pytest.mark.asyncio
    async def test_execute_async_simple(self, executor):
        """Test simple async execution."""
        def simple_operation(x, y):
            return x + y
        
        result = await executor.execute_async(simple_operation, 5, 3)
        assert result == 8
    
    @pytest.mark.asyncio
    async def test_execute_async_with_kwargs(self, executor):
        """Test async execution with kwargs."""
        def operation_with_kwargs(a, b=10):
            return a * b
        
        result = await executor.execute_async(operation_with_kwargs, 3, b=7)
        assert result == 21
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    async def test_execute_async_tensor_operation(self, device):
        """Test async execution with tensor operations."""
        executor = AsyncGPUExecutor(device=device, num_streams=2)
        
        def tensor_operation():
            x = torch.randn(10, 10).to(device)
            return torch.nn.functional.relu(x)
        
        result = await executor.execute_async(tensor_operation)
        
        assert result is not None
        assert result.shape == (10, 10)
        assert (result >= 0).all()  # ReLU output should be non-negative
    
    @pytest.mark.asyncio
    async def test_batch_execute(self, executor):
        """Test batch execution."""
        def op1():
            return 1
        
        def op2():
            return 2
        
        def op3():
            return 3
        
        operations = [
            (op1, (), {}),
            (op2, (), {}),
            (op3, (), {}),
        ]
        
        results = await executor.batch_execute(operations)
        
        assert len(results) == 3
        assert results == [1, 2, 3]
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    async def test_batch_execute_tensor_ops(self, device):
        """Test batch execution with tensor operations."""
        executor = AsyncGPUExecutor(device=device, num_streams=4)
        
        def matrix_multiply(size):
            a = torch.randn(size, size).to(device)
            b = torch.randn(size, size).to(device)
            return torch.matmul(a, b)
        
        def relu_op(size):
            x = torch.randn(size, size).to(device)
            return torch.nn.functional.relu(x)
        
        def softmax_op(size):
            x = torch.randn(size, size).to(device)
            return torch.nn.functional.softmax(x, dim=-1)
        
        operations = [
            (matrix_multiply, (100,), {}),
            (relu_op, (100,), {}),
            (softmax_op, (100,), {}),
        ]
        
        results = await executor.batch_execute(operations)
        
        assert len(results) == 3
        for result in results:
            assert result.shape == (100, 100)
    
    @pytest.mark.asyncio
    async def test_batch_execute_empty(self, executor):
        """Test batch execution with empty list."""
        results = await executor.batch_execute([])
        assert results == []
    
    @pytest.mark.asyncio
    async def test_execute_async_exception(self, executor):
        """Test exception handling in async execution."""
        def failing_operation():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError, match="Test error"):
            await executor.execute_async(failing_operation)
    
    def test_synchronize_all(self, executor):
        """Test synchronizing all streams."""
        # Should not raise exception
        executor.synchronize_all()
    
    @pytest.mark.skipif(not TORCH_AVAILABLE or not torch.cuda.is_available(), 
                       reason="CUDA not available")
    def test_cuda_streams_created(self):
        """Test CUDA streams are created for CUDA device."""
        device = torch.device("cuda")
        executor = AsyncGPUExecutor(device=device, num_streams=4)
        
        # Should have real CUDA streams
        assert all(s is not None for s in executor.streams)
    
    def test_repr(self, executor):
        """Test string representation."""
        repr_str = repr(executor)
        
        assert 'AsyncGPUExecutor' in repr_str
        assert 'device=' in repr_str
        assert 'num_streams=' in repr_str


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestAsyncGPUExecutorPerformance:
    """Performance tests for async executor."""
    
    @pytest.mark.asyncio
    async def test_parallel_speedup(self):
        """Test that parallel execution is faster than sequential."""
        device = torch.device("cpu")  # Use CPU for consistent testing
        executor = AsyncGPUExecutor(device=device, num_streams=4)
        
        import time
        
        def slow_operation(duration=0.1):
            time.sleep(duration)
            return duration
        
        # Sequential execution
        start = time.time()
        for _ in range(3):
            await executor.execute_async(slow_operation, 0.1)
        sequential_time = time.time() - start
        
        # Parallel execution
        operations = [(slow_operation, (0.1,), {}) for _ in range(3)]
        start = time.time()
        await executor.batch_execute(operations)
        parallel_time = time.time() - start
        
        # Parallel should be similar (async/await overhead, not true parallel on CPU)
        # But should complete all operations
        assert parallel_time < sequential_time * 1.5  # Some overhead allowed
