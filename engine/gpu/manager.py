"""
GPU Manager - Device detection, allocation, and monitoring
"""

import torch
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class GPUManager:
    """
    Manage GPU resources and device allocation.
    
    Features:
    - Auto-detect CUDA, MPS (Apple Silicon), or CPU
    - Multi-GPU device management
    - Memory statistics and monitoring
    - Device health checks
    
    Example:
        >>> gpu_manager = GPUManager()
        >>> device = gpu_manager.get_device()
        >>> print(f"Using device: {device}")
        >>> stats = gpu_manager.memory_stats()
        >>> print(f"GPU Memory: {stats['allocated_mb']:.1f}MB / {stats['total_mb']:.1f}MB")
    """
    
    def __init__(self, preferred_devices: Optional[List[int]] = None):
        """
        Initialize GPU manager.
        
        Args:
            preferred_devices: List of GPU device IDs to use (None = use all available)
        """
        self.device = self._detect_device()
        self.devices_available = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.preferred_devices = preferred_devices or list(range(self.devices_available))
        
        logger.info(
            "gpu_manager_initialized",
            device=str(self.device),
            cuda_available=torch.cuda.is_available(),
            device_count=self.devices_available,
            cuda_version=torch.version.cuda if torch.cuda.is_available() else None,
            pytorch_version=torch.__version__
        )
    
    def _detect_device(self) -> torch.device:
        """Auto-detect best available device."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Detected CUDA device: {torch.cuda.get_device_name(0)}")
            return device
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Detected Apple MPS (Metal Performance Shaders) device")
            return device
        else:
            logger.warning("No GPU detected, falling back to CPU")
            return torch.device("cpu")
    
    def get_device(self, preferred_gpu: Optional[int] = None) -> torch.device:
        """
        Get specific GPU device or default.
        
        Args:
            preferred_gpu: Specific GPU ID to use (None = use default)
            
        Returns:
            torch.device: Device to use for computation
        """
        if preferred_gpu is not None and self.devices_available > preferred_gpu:
            return torch.device(f"cuda:{preferred_gpu}")
        return self.device
    
    def memory_stats(self, device_id: int = 0) -> Dict[str, Any]:
        """
        Get GPU memory statistics.
        
        Args:
            device_id: GPU device ID (default: 0)
            
        Returns:
            Dictionary with memory statistics:
            - allocated_mb: Currently allocated memory in MB
            - reserved_mb: Reserved memory in MB
            - max_allocated_mb: Peak allocated memory in MB
            - total_mb: Total GPU memory in MB
            - utilization_pct: Memory utilization percentage
        """
        if not torch.cuda.is_available() or device_id >= self.devices_available:
            return {"available": False, "reason": "CUDA not available or invalid device_id"}
        
        try:
            allocated = torch.cuda.memory_allocated(device_id)
            reserved = torch.cuda.memory_reserved(device_id)
            max_allocated = torch.cuda.max_memory_allocated(device_id)
            total = torch.cuda.get_device_properties(device_id).total_memory
            
            return {
                "available": True,
                "device_id": device_id,
                "device_name": torch.cuda.get_device_name(device_id),
                "allocated_mb": allocated / 1024**2,
                "reserved_mb": reserved / 1024**2,
                "max_allocated_mb": max_allocated / 1024**2,
                "total_mb": total / 1024**2,
                "utilization_pct": (allocated / total * 100) if total > 0 else 0
            }
        except Exception as e:
            logger.error(f"Failed to get memory stats for device {device_id}: {e}")
            return {"available": False, "error": str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check GPU health status.
        
        Returns:
            Dictionary with health information:
            - healthy: Overall health status (bool)
            - mode: Operation mode (cuda/mps/cpu)
            - devices: List of device health info (CUDA only)
        """
        if not torch.cuda.is_available():
            return {
                "healthy": True,
                "mode": "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu",
                "message": "Running in CPU/MPS mode"
            }
        
        health = {"healthy": True, "mode": "cuda", "devices": []}
        
        for device_id in range(self.devices_available):
            try:
                stats = self.memory_stats(device_id)
                
                if not stats.get("available", False):
                    device_health = {
                        "device_id": device_id,
                        "healthy": False,
                        "reason": stats.get("reason", "Unknown error")
                    }
                else:
                    # Flag if GPU memory is >95% utilized
                    utilization = stats["utilization_pct"]
                    device_healthy = utilization < 95
                    
                    device_health = {
                        "device_id": device_id,
                        "device_name": stats["device_name"],
                        "healthy": device_healthy,
                        "memory_stats": {
                            "allocated_mb": round(stats["allocated_mb"], 1),
                            "total_mb": round(stats["total_mb"], 1),
                            "utilization_pct": round(utilization, 1)
                        },
                        "warning": "High memory usage" if utilization > 85 else None
                    }
                
                health["devices"].append(device_health)
                
                if not device_health["healthy"]:
                    health["healthy"] = False
                    
            except Exception as e:
                logger.error(f"GPU {device_id} health check failed: {e}")
                health["healthy"] = False
                health["devices"].append({
                    "device_id": device_id,
                    "healthy": False,
                    "error": str(e)
                })
        
        return health
    
    def reset_peak_memory_stats(self, device_id: Optional[int] = None):
        """
        Reset peak memory statistics.
        
        Args:
            device_id: GPU device ID (None = reset all devices)
        """
        if not torch.cuda.is_available():
            return
        
        if device_id is not None:
            torch.cuda.reset_peak_memory_stats(device_id)
            logger.debug(f"Reset peak memory stats for device {device_id}")
        else:
            for dev_id in range(self.devices_available):
                torch.cuda.reset_peak_memory_stats(dev_id)
            logger.debug("Reset peak memory stats for all devices")
    
    def clear_cache(self, device_id: Optional[int] = None):
        """
        Clear GPU memory cache.
        
        Args:
            device_id: GPU device ID (None = clear all devices)
        """
        if not torch.cuda.is_available():
            return
        
        torch.cuda.empty_cache()
        logger.info("gpu_cache_cleared", device_id=device_id or "all")
    
    def __repr__(self) -> str:
        return (
            f"GPUManager(device={self.device}, "
            f"devices_available={self.devices_available})"
        )
