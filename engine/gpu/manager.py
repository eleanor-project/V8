"""
GPU Manager for ELEANOR V8

Manages GPU resources, device allocation, and monitoring.
"""

import logging
from typing import Dict, List, Optional, Any

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
        Using device: cuda:0
        
        >>> if gpu_manager.is_gpu_available():
        ...     stats = gpu_manager.memory_stats()
        ...     print(f"GPU Memory: {stats['allocated_mb']:.1f} MB")
    """
    
    def __init__(self, preferred_devices: Optional[List[int]] = None):
        """
        Initialize GPU Manager.
        
        Args:
            preferred_devices: List of GPU device IDs to prefer (None = use all)
        """
        # Import torch here to avoid import errors if not installed
        try:
            import torch
            self.torch = torch
            self._torch_available = True
        except ImportError:
            logger.warning(
                "PyTorch not installed. GPU acceleration unavailable. "
                "Install with: pip install torch"
            )
            self.torch = None
            self._torch_available = False
            self.device = None
            self.devices_available = 0
            self.preferred_devices = []
            return
        
        # Detect device
        self.device = self._detect_device()
        self.devices_available = (
            torch.cuda.device_count() if torch.cuda.is_available() else 0
        )
        self.preferred_devices = preferred_devices or list(range(self.devices_available))
        
        # Log initialization
        logger.info(
            "gpu_manager_initialized",
            extra={
                "device": str(self.device),
                "device_type": self.device.type,
                "cuda_available": torch.cuda.is_available(),
                "device_count": self.devices_available,
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
                "preferred_devices": self.preferred_devices,
            }
        )
    
    def _detect_device(self):
        """
        Auto-detect best available device.
        
        Returns:
            torch.device: Best available device (cuda, mps, or cpu)
        """
        if not self._torch_available:
            return None
        
        # Try CUDA first (NVIDIA GPUs)
        if self.torch.cuda.is_available():
            device = self.torch.device("cuda")
            logger.info("detected_cuda_gpu", extra={"device": str(device)})
            return device
        
        # Try MPS (Apple Silicon)
        if hasattr(self.torch.backends, 'mps') and self.torch.backends.mps.is_available():
            device = self.torch.device("mps")
            logger.info("detected_apple_mps", extra={"device": str(device)})
            return device
        
        # Fallback to CPU
        device = self.torch.device("cpu")
        logger.info(
            "using_cpu_fallback",
            extra={
                "device": str(device),
                "reason": "No GPU detected"
            }
        )
        return device
    
    def is_gpu_available(self) -> bool:
        """
        Check if GPU acceleration is available.
        
        Returns:
            bool: True if CUDA or MPS available
        """
        if not self._torch_available or self.device is None:
            return False
        return self.device.type in ("cuda", "mps")
    
    def get_device(self, preferred_gpu: Optional[int] = None):
        """
        Get device for computation.
        
        Args:
            preferred_gpu: Specific GPU ID to use (None = use default)
        
        Returns:
            torch.device: Device to use for computation
        """
        if not self._torch_available:
            return None
        
        if preferred_gpu is not None and self.devices_available > preferred_gpu:
            return self.torch.device(f"cuda:{preferred_gpu}")
        
        return self.device
    
    def memory_stats(self, device_id: int = 0) -> Dict[str, Any]:
        """
        Get GPU memory statistics.
        
        Args:
            device_id: GPU device ID (default: 0)
        
        Returns:
            Dict with memory statistics:
            - device_id: GPU device ID
            - allocated_mb: Currently allocated memory (MB)
            - reserved_mb: Reserved memory (MB)
            - max_allocated_mb: Peak allocated memory (MB)
            - total_mb: Total GPU memory (MB)
            - utilization_pct: Memory utilization percentage
            - available: Whether statistics are available
        """
        if not self._torch_available:
            return {"available": False, "reason": "PyTorch not installed"}
        
        if not self.torch.cuda.is_available():
            return {"available": False, "reason": "CUDA not available"}
        
        if device_id >= self.devices_available:
            return {
                "available": False,
                "reason": f"Device {device_id} not found (only {self.devices_available} devices)"
            }
        
        try:
            allocated = self.torch.cuda.memory_allocated(device_id)
            reserved = self.torch.cuda.memory_reserved(device_id)
            max_allocated = self.torch.cuda.max_memory_allocated(device_id)
            total = self.torch.cuda.get_device_properties(device_id).total_memory
            
            return {
                "available": True,
                "device_id": device_id,
                "allocated_mb": allocated / 1024**2,
                "reserved_mb": reserved / 1024**2,
                "max_allocated_mb": max_allocated / 1024**2,
                "total_mb": total / 1024**2,
                "utilization_pct": (allocated / total * 100) if total > 0 else 0.0,
            }
        except Exception as e:
            logger.error(
                "memory_stats_error",
                extra={"device_id": device_id, "error": str(e)},
                exc_info=True
            )
            return {"available": False, "reason": str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check GPU health status.
        
        Returns:
            Dict with health status:
            - healthy: Overall health status
            - mode: cpu, cuda, or mps
            - devices: List of device health stats (CUDA only)
        """
        if not self._torch_available:
            return {
                "healthy": True,
                "mode": "cpu",
                "reason": "PyTorch not installed"
            }
        
        if not self.torch.cuda.is_available():
            return {
                "healthy": True,
                "mode": self.device.type if self.device else "cpu",
                "message": "No CUDA devices for detailed health check"
            }
        
        health = {
            "healthy": True,
            "mode": "cuda",
            "devices": []
        }
        
        for device_id in range(self.devices_available):
            try:
                stats = self.memory_stats(device_id)
                
                if not stats.get("available"):
                    device_health = {
                        "device_id": device_id,
                        "healthy": False,
                        "reason": stats.get("reason", "Unknown")
                    }
                else:
                    # Flag if >95% memory used
                    utilization = stats["utilization_pct"]
                    device_health = {
                        "device_id": device_id,
                        "healthy": utilization < 95,
                        "utilization_pct": utilization,
                        "allocated_mb": stats["allocated_mb"],
                        "total_mb": stats["total_mb"],
                    }
                    
                    if utilization >= 95:
                        device_health["warning"] = "High memory utilization"
                        health["healthy"] = False
                
                health["devices"].append(device_health)
                
            except Exception as e:
                logger.error(
                    "device_health_check_failed",
                    extra={"device_id": device_id, "error": str(e)},
                    exc_info=True
                )
                health["healthy"] = False
                health["devices"].append({
                    "device_id": device_id,
                    "healthy": False,
                    "error": str(e)
                })
        
        return health
    
    def reset_peak_stats(self, device_id: Optional[int] = None):
        """
        Reset peak memory statistics.
        
        Args:
            device_id: GPU device ID (None = all devices)
        """
        if not self._torch_available or not self.torch.cuda.is_available():
            return
        
        if device_id is not None:
            self.torch.cuda.reset_peak_memory_stats(device_id)
        else:
            for i in range(self.devices_available):
                self.torch.cuda.reset_peak_memory_stats(i)
    
    def __repr__(self) -> str:
        """String representation."""
        if not self._torch_available:
            return "GPUManager(torch_unavailable)"
        
        return (
            f"GPUManager("
            f"device={self.device}, "
            f"devices={self.devices_available}, "
            f"type={self.device.type if self.device else 'none'})"
        )
