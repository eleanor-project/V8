"""
Tests for GPU acceleration components.
"""

import pytest
from unittest.mock import Mock, patch


class TestGPUManager:
    """Tests for GPUManager"""
    
    def test_gpu_manager_cpu_fallback(self):
        """Test GPU manager with CPU fallback (no torch)"""
        with patch.dict('sys.modules', {'torch': None}):
            from engine.gpu import GPUManager
            
            manager = GPUManager()
            assert not manager.is_gpu_available()
            assert manager.device is None
            assert manager.devices_available == 0
    
    @patch('engine.gpu.manager.torch')
    def test_gpu_manager_cuda_detection(self, mock_torch):
        """Test CUDA GPU detection"""
        # Mock CUDA available
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 2
        mock_torch.device.return_value = Mock(type="cuda")
        mock_torch.version.cuda = "12.1"
        
        from engine.gpu import GPUManager
        
        manager = GPUManager()
        assert manager.is_gpu_available()
        assert manager.devices_available == 2
    
    @patch('engine.gpu.manager.torch')
    def test_gpu_manager_mps_detection(self, mock_torch):
        """Test Apple MPS detection"""
        # Mock MPS available, CUDA not
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True
        mock_torch.device.return_value = Mock(type="mps")
        
        from engine.gpu import GPUManager
        
        manager = GPUManager()
        assert manager.is_gpu_available()
    
    @patch('engine.gpu.manager.torch')
    def test_memory_stats(self, mock_torch):
        """Test GPU memory statistics"""
        # Mock CUDA device
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.device.return_value = Mock(type="cuda")
        
        # Mock memory stats
        mock_torch.cuda.memory_allocated.return_value = 1024 * 1024 * 100  # 100 MB
        mock_torch.cuda.memory_reserved.return_value = 1024 * 1024 * 150  # 150 MB
        mock_torch.cuda.max_memory_allocated.return_value = 1024 * 1024 * 200  # 200 MB
        
        mock_props = Mock()
        mock_props.total_memory = 1024 * 1024 * 1024 * 8  # 8 GB
        mock_torch.cuda.get_device_properties.return_value = mock_props
        
        from engine.gpu import GPUManager
        
        manager = GPUManager()
        stats = manager.memory_stats(0)
        
        assert stats["available"]
        assert stats["device_id"] == 0
        assert stats["allocated_mb"] == 100
        assert stats["reserved_mb"] == 150
        assert stats["max_allocated_mb"] == 200
        assert stats["total_mb"] == 8192
        assert 0 <= stats["utilization_pct"] <= 100
    
    @patch('engine.gpu.manager.torch')
    def test_health_check(self, mock_torch):
        """Test GPU health check"""
        # Mock CUDA device
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 2
        mock_torch.device.return_value = Mock(type="cuda")
        
        # Mock memory stats for health check
        mock_torch.cuda.memory_allocated.return_value = 1024 * 1024 * 100
        mock_props = Mock()
        mock_props.total_memory = 1024 * 1024 * 1024 * 8
        mock_torch.cuda.get_device_properties.return_value = mock_props
        mock_torch.cuda.memory_reserved.return_value = 0
        mock_torch.cuda.max_memory_allocated.return_value = 0
        
        from engine.gpu import GPUManager
        
        manager = GPUManager()
        health = manager.health_check()
        
        assert "healthy" in health
        assert health["mode"] == "cuda"
        assert "devices" in health
        assert len(health["devices"]) == 2
    
    @patch('engine.gpu.manager.torch')
    def test_get_device(self, mock_torch):
        """Test device selection"""
        # Mock CUDA with multiple devices
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 4
        mock_device = Mock(type="cuda")
        mock_torch.device.return_value = mock_device
        
        from engine.gpu import GPUManager
        
        manager = GPUManager()
        
        # Get default device
        device = manager.get_device()
        assert device is not None
        
        # Get specific device
        device = manager.get_device(preferred_gpu=2)
        assert device is not None


class TestAsyncGPUExecutor:
    """Tests for AsyncGPUExecutor"""
    
    @pytest.mark.asyncio
    async def test_execute_async_cpu(self):
        """Test async execution on CPU"""
        from engine.gpu import AsyncGPUExecutor
        
        # Mock CPU device
        mock_device = Mock(type="cpu")
        
        executor = AsyncGPUExecutor(mock_device, num_streams=4)
        
        # Test operation
        def test_op(x):
            return x * 2
        
        result = await executor.execute_async(test_op, 5)
        assert result == 10
    
    @pytest.mark.asyncio
    async def test_batch_execute(self):
        """Test batch execution"""
        from engine.gpu import AsyncGPUExecutor
        
        mock_device = Mock(type="cpu")
        executor = AsyncGPUExecutor(mock_device)
        
        # Test operations
        def add(x, y):
            return x + y
        
        def multiply(x, y):
            return x * y
        
        operations = [
            (add, (1, 2), {}),
            (multiply, (3, 4), {}),
            (add, (5, 6), {}),
        ]
        
        results = await executor.batch_execute(operations)
        
        assert len(results) == 3
        assert results[0] == 3
        assert results[1] == 12
        assert results[2] == 11
    
    @patch('engine.gpu.async_ops.torch')
    def test_cuda_stream_creation(self, mock_torch):
        """Test CUDA stream creation"""
        from engine.gpu import AsyncGPUExecutor
        
        # Mock CUDA device
        mock_device = Mock(type="cuda")
        mock_stream = Mock()
        mock_torch.cuda.Stream.return_value = mock_stream
        
        executor = AsyncGPUExecutor(mock_device, num_streams=4)
        
        # Should have created 4 streams
        assert len(executor.streams) == 4
        assert mock_torch.cuda.Stream.call_count == 4
    
    def test_get_stream_round_robin(self):
        """Test round-robin stream allocation"""
        from engine.gpu import AsyncGPUExecutor
        
        mock_device = Mock(type="cuda")
        
        with patch('engine.gpu.async_ops.torch') as mock_torch:
            mock_streams = [Mock(), Mock(), Mock()]
            mock_torch.cuda.Stream.side_effect = mock_streams
            
            executor = AsyncGPUExecutor(mock_device, num_streams=3)
            
            # Get streams in round-robin order
            stream1 = executor.get_stream()
            stream2 = executor.get_stream()
            stream3 = executor.get_stream()
            stream4 = executor.get_stream()  # Should wrap around
            
            assert stream1 == mock_streams[0]
            assert stream2 == mock_streams[1]
            assert stream3 == mock_streams[2]
            assert stream4 == mock_streams[0]  # Wrapped around


class TestGPUIntegration:
    """Integration tests for GPU components"""
    
    @patch('engine.gpu.manager.torch')
    def test_manager_and_executor_integration(self, mock_torch):
        """Test GPUManager and AsyncGPUExecutor work together"""
        # Mock CUDA
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_device = Mock(type="cuda")
        mock_torch.device.return_value = mock_device
        
        from engine.gpu import GPUManager, AsyncGPUExecutor
        
        # Initialize manager
        manager = GPUManager()
        assert manager.is_gpu_available()
        
        # Get device for executor
        device = manager.get_device()
        
        # Initialize executor with device from manager
        executor = AsyncGPUExecutor(device, num_streams=4)
        
        assert executor.device == device
        assert len(executor.streams) == 4
