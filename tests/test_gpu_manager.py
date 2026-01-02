"""
Tests for GPU Manager
"""

import pytest
import sys
from unittest.mock import patch, MagicMock

# Mock torch if not available
try:
    import torch
    TORCH_AVAILABLE = not isinstance(torch, MagicMock)
except ImportError:
    TORCH_AVAILABLE = False
    torch = MagicMock()
    sys.modules['torch'] = torch

from engine.gpu.manager import GPUManager, GPUConfig


class TestGPUConfig:
    """Test GPU configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = GPUConfig()
        
        assert config.enabled is True
        assert config.device_preference == ["cuda", "mps", "cpu"]
        assert config.mixed_precision is True
        assert config.num_streams == 4
        assert config.default_batch_size == 8
        assert config.max_batch_size == 32
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = GPUConfig(
            enabled=False,
            device_preference=["cpu"],
            mixed_precision=False,
            num_streams=2,
            default_batch_size=4,
        )
        
        assert config.enabled is False
        assert config.device_preference == ["cpu"]
        assert config.mixed_precision is False
        assert config.num_streams == 2
        assert config.default_batch_size == 4


class TestGPUManager:
    """Test GPU manager."""
    
    @pytest.fixture
    def gpu_manager(self):
        """Create GPU manager for testing."""
        return GPUManager()
    
    def test_initialization(self, gpu_manager):
        """Test GPU manager initialization."""
        assert gpu_manager is not None
        assert hasattr(gpu_manager, 'device')
        assert hasattr(gpu_manager, 'devices_available')
        assert hasattr(gpu_manager, 'config')
    
    def test_device_detection_cpu_fallback(self):
        """Test CPU fallback when no GPU available."""
        with patch('torch.cuda.is_available', return_value=False), \
             patch('torch.backends.mps.is_available', return_value=False, create=True):
            gpu_manager = GPUManager()
            
            if TORCH_AVAILABLE:
                assert gpu_manager.device.type == "cpu"
            else:
                assert gpu_manager.device is None
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_device_detection_cuda(self):
        """Test CUDA device detection."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.device_count', return_value=1):
                gpu_manager = GPUManager()
                
                assert gpu_manager.device.type in ["cuda", "cpu"]
                assert gpu_manager.devices_available >= 0
    
    def test_get_device_default(self, gpu_manager):
        """Test getting default device."""
        device = gpu_manager.get_device()
        
        if TORCH_AVAILABLE:
            assert device is not None
        else:
            assert device is None
    
    @pytest.mark.skipif(not TORCH_AVAILABLE or not torch.cuda.is_available(), 
                       reason="CUDA not available")
    def test_get_device_specific_gpu(self):
        """Test getting specific GPU device."""
        gpu_manager = GPUManager()
        device = gpu_manager.get_device(preferred_gpu=0)
        
        assert device.type == "cuda"
        assert device.index == 0 or device.index is None
    
    @pytest.mark.skipif(not TORCH_AVAILABLE or not torch.cuda.is_available(), 
                       reason="CUDA not available")
    def test_memory_stats_cuda(self):
        """Test GPU memory statistics for CUDA."""
        gpu_manager = GPUManager()
        
        if gpu_manager.is_available():
            stats = gpu_manager.memory_stats(device_id=0)
            
            assert stats['available'] is True
            assert 'device_id' in stats
            assert 'allocated_mb' in stats
            assert 'reserved_mb' in stats
            assert 'total_mb' in stats
            assert 'utilization_pct' in stats
            assert stats['allocated_mb'] >= 0
            assert stats['total_mb'] > 0
    
    def test_memory_stats_unavailable(self):
        """Test memory stats when GPU unavailable."""
        with patch('torch.cuda.is_available', return_value=False):
            gpu_manager = GPUManager()
            stats = gpu_manager.memory_stats(device_id=0)
            
            assert stats['available'] is False
    
    def test_health_check(self, gpu_manager):
        """Test GPU health check."""
        health = gpu_manager.health_check()
        
        assert 'healthy' in health
        assert 'mode' in health
        assert 'devices' in health
        assert isinstance(health['healthy'], bool)
        assert isinstance(health['devices'], list)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE or not torch.cuda.is_available(), 
                       reason="CUDA not available")
    def test_health_check_cuda(self):
        """Test health check with CUDA devices."""
        gpu_manager = GPUManager()
        
        if gpu_manager.is_available():
            health = gpu_manager.health_check()
            
            assert health['mode'] == 'cuda'
            assert len(health['devices']) > 0
            
            for device in health['devices']:
                assert 'device_id' in device
                assert 'healthy' in device
                assert 'memory_stats' in device
    
    def test_is_available(self, gpu_manager):
        """Test GPU availability check."""
        available = gpu_manager.is_available()
        assert isinstance(available, bool)
    
    def test_repr(self, gpu_manager):
        """Test string representation."""
        repr_str = repr(gpu_manager)
        
        assert 'GPUManager' in repr_str
        assert 'device=' in repr_str
        assert 'devices_available=' in repr_str


class TestGPUManagerWithConfig:
    """Test GPU manager with custom configuration."""
    
    def test_custom_config(self):
        """Test initialization with custom config."""
        config = GPUConfig(
            enabled=True,
            device_preference=["cuda", "cpu"],
            mixed_precision=True,
            num_streams=8,
        )
        
        gpu_manager = GPUManager(config=config)
        
        assert gpu_manager.config == config
        assert gpu_manager.config.num_streams == 8
    
    def test_disabled_gpu(self):
        """Test with GPU disabled in config."""
        config = GPUConfig(enabled=False)
        gpu_manager = GPUManager(config=config)
        
        # Should still initialize but may prefer CPU
        assert gpu_manager is not None
    
    def test_preferred_devices(self):
        """Test with preferred device list."""
        config = GPUConfig(preferred_devices=[0, 1])
        gpu_manager = GPUManager(config=config)
        
        assert gpu_manager.preferred_devices == [0, 1]


@pytest.mark.skipif(not TORCH_AVAILABLE or not torch.cuda.is_available(), 
                   reason="CUDA not available")
class TestGPUManagerMemoryOperations:
    """Test GPU memory operations with real GPU."""
    
    def test_memory_allocation_tracking(self):
        """Test memory tracking during allocations."""
        gpu_manager = GPUManager()
        device = gpu_manager.get_device()
        
        if not gpu_manager.is_available():
            pytest.skip("GPU not available")
        
        # Get initial memory
        initial_stats = gpu_manager.memory_stats(device_id=0)
        initial_allocated = initial_stats['allocated_mb']
        
        # Allocate tensor
        tensor = torch.randn(1000, 1000).to(device)
        
        # Check memory increased
        after_stats = gpu_manager.memory_stats(device_id=0)
        after_allocated = after_stats['allocated_mb']
        
        assert after_allocated > initial_allocated
        
        # Cleanup
        del tensor
        torch.cuda.empty_cache()
    
    def test_memory_cleanup(self):
        """Test memory cleanup."""
        gpu_manager = GPUManager()
        device = gpu_manager.get_device()
        
        if not gpu_manager.is_available():
            pytest.skip("GPU not available")
        
        # Allocate and cleanup
        tensors = [torch.randn(1000, 1000).to(device) for _ in range(5)]
        
        allocated_stats = gpu_manager.memory_stats(device_id=0)
        allocated_mb = allocated_stats['allocated_mb']
        
        # Clear
        tensors.clear()
        torch.cuda.empty_cache()
        
        cleared_stats = gpu_manager.memory_stats(device_id=0)
        cleared_mb = cleared_stats['allocated_mb']
        
        # Memory should be reduced (may not be exact due to caching)
        assert cleared_mb <= allocated_mb
