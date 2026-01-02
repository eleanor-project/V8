"""
ELEANOR V8 — GPU Manager

Core GPU resource management and device allocation.
"""

import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Supported device types."""
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon
    CPU = "cpu"


@dataclass
class GPUMemoryStats:
    """GPU memory statistics."""
    device_id: int
    allocated_mb: float
    reserved_mb: float
    max_allocated_mb: float
    total_mb: float
    utilization: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'device_id': self.device_id,
            'allocated_mb': self.allocated_mb,
            'reserved_mb': self.reserved_mb,
            'max_allocated_mb': self.max_allocated_mb,
            'total_mb': self.total_mb,
            'utilization': self.utilization,
        }


@dataclass
class GPUConfig:
    """GPU configuration."""
    enabled: bool = False
    device_ids: Optional[List[int]] = None
    mixed_precision: bool = True
    batch_size: int = 8
    max_batch_size: int = 32
    memory_limit_gb: Optional[float] = None
    fallback_to_cpu: bool = True
    log_memory_stats: bool = True
    
    # Optimization settings
    use_flash_attention: bool = True
    quantization_bits: Optional[int] = None  # None, 4, 8
    enable_kv_cache: bool = True


class GPUManager:
    """
    Manage GPU resources and device allocation.
    
    Handles device detection, memory monitoring, and resource allocation.
    """
    
    def __init__(self, config: Optional[GPUConfig] = None):
        """
        Initialize GPU manager.
        
        Args:
            config: GPU configuration
        """
        self.config = config or GPUConfig()
        self.torch_available = False
        self.cuda_available = False
        self.mps_available = False
        self.device_count = 0
        self.primary_device = ""
        self.torch: Optional[Any] = None
        
        # Try to import PyTorch
        try:
            import torch
            self.torch = torch
            self.torch_available = True
            self.cuda_available = torch.cuda.is_available()
            self.mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            
            if self.cuda_available:
                self.device_count = torch.cuda.device_count()
            
            logger.info(
                "GPU manager initialized",
                extra={
                    "torch_available": True,
                    "cuda_available": self.cuda_available,
                    "mps_available": self.mps_available,
                    "device_count": self.device_count,
                },
            )
            
        except ImportError:
            logger.warning(
                "PyTorch not available. GPU acceleration disabled. "
                "Install with: pip install torch torchvision"
            )
            self.torch = None
        
        # Detect primary device
        self.primary_device = self._detect_primary_device()
        
        # Log device info
        if self.config.enabled:
            self._log_device_info()
    
    def _detect_primary_device(self) -> str:
        """
        Auto-detect best available device.
        
        Returns:
            Device string ('cuda', 'cuda:0', 'mps', or 'cpu')
        """
        if not self.config.enabled or not self.torch_available:
            return 'cpu'
        
        if self.cuda_available:
            # Use first GPU by default
            device_id = self.config.device_ids[0] if self.config.device_ids else 0
            if device_id < self.device_count:
                return f'cuda:{device_id}'
            return 'cuda:0'
        
        if self.mps_available:
            return 'mps'
        
        if self.config.fallback_to_cpu:
            logger.warning("No GPU available, falling back to CPU")
            return 'cpu'
        
        raise RuntimeError("GPU requested but not available")
    
    def get_device(self, preferred_id: Optional[int] = None) -> Any:
        """
        Get torch device for computation.
        
        Args:
            preferred_id: Preferred GPU ID
        
        Returns:
            torch.device instance
        """
        if not self.torch_available or self.torch is None:
            raise RuntimeError("PyTorch not available")
        
        if preferred_id is not None and self.cuda_available:
            if preferred_id < self.device_count:
                return self.torch.device(f'cuda:{preferred_id}')
        
        return self.torch.device(self.primary_device)
    
    def get_memory_stats(self, device_id: int = 0) -> Optional[GPUMemoryStats]:
        """
        Get GPU memory statistics.
        
        Args:
            device_id: GPU device ID
        
        Returns:
            Memory statistics or None if not available
        """
        if not self.cuda_available or device_id >= self.device_count or self.torch is None:
            return None
        
        try:
            allocated = self.torch.cuda.memory_allocated(device_id)
            reserved = self.torch.cuda.memory_reserved(device_id)
            max_allocated = self.torch.cuda.max_memory_allocated(device_id)
            total = self.torch.cuda.get_device_properties(device_id).total_memory
            
            # Convert to MB
            allocated_mb = allocated / (1024 ** 2)
            reserved_mb = reserved / (1024 ** 2)
            max_allocated_mb = max_allocated / (1024 ** 2)
            total_mb = total / (1024 ** 2)
            
            utilization = (allocated / total) * 100 if total > 0 else 0.0
            
            return GPUMemoryStats(
                device_id=device_id,
                allocated_mb=allocated_mb,
                reserved_mb=reserved_mb,
                max_allocated_mb=max_allocated_mb,
                total_mb=total_mb,
                utilization=utilization,
            )
            
        except Exception as e:
            logger.error(f"Failed to get memory stats for device {device_id}: {e}")
            return None
    
    def reset_peak_memory_stats(self, device_id: int = 0) -> None:
        """Reset peak memory statistics."""
        if self.cuda_available and self.torch is not None and device_id < self.device_count:
            self.torch.cuda.reset_peak_memory_stats(device_id)
    
    def empty_cache(self) -> None:
        """Empty CUDA cache to free memory."""
        if self.cuda_available and self.torch is not None:
            self.torch.cuda.empty_cache()
            logger.debug("CUDA cache emptied")
    
    def _log_device_info(self) -> None:
        """Log detailed device information."""
        if self.cuda_available and self.torch is not None:
            for i in range(self.device_count):
                props = self.torch.cuda.get_device_properties(i)
                logger.info(
                    "GPU %s detected",
                    i,
                    extra={
                        "device_id": i,
                        "name": props.name,
                        "total_memory_gb": props.total_memory / (1024 ** 3),
                        "compute_capability": f"{props.major}.{props.minor}",
                        "multi_processor_count": props.multi_processor_count,
                    },
                )
        elif self.mps_available:
            logger.info("Apple Silicon GPU (MPS) detected")
        else:
            logger.info("Using CPU for computation")
    
    def is_available(self) -> bool:
        """Check if GPU acceleration is available."""
        return self.config.enabled and (self.cuda_available or self.mps_available)
    
    def synchronize(self, device_id: Optional[int] = None) -> None:
        """Synchronize CUDA device."""
        if self.cuda_available and self.torch is not None:
            if device_id is not None:
                self.torch.cuda.synchronize(device_id)
            else:
                self.torch.cuda.synchronize()


__all__ = ["GPUManager", "GPUConfig", "GPUMemoryStats", "DeviceType"]
