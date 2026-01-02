"""
ELEANOR V8 - GPU Manager

Manages GPU resources, device allocation, and health monitoring.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

logger = logging.getLogger(__name__)


@dataclass
class GPUConfig:
    """GPU configuration settings."""

    enabled: bool = True
    device_preference: List[str] = field(default_factory=lambda: ["cuda", "mps", "cpu"])
    preferred_devices: Optional[List[int]] = None
    mixed_precision: bool = True
    num_streams: int = 4

    # Memory settings
    max_memory_per_gpu: Optional[str] = "24GB"
    log_memory_stats: bool = True
    memory_check_interval: int = 60

    # Batch settings
    default_batch_size: int = 8
    max_batch_size: int = 32
    dynamic_batching: bool = True


@dataclass
class GPUMemoryStats:
    """Simple GPU memory statistics container."""
    available: bool
    device_id: Optional[int] = None
    allocated_mb: float = 0.0
    reserved_mb: float = 0.0
    max_allocated_mb: float = 0.0
    total_mb: float = 0.0
    utilization_pct: float = 0.0


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

        >>> stats = gpu_manager.memory_stats(device_id=0)
        >>> print(f"GPU memory: {stats['allocated_mb']:.2f}MB")
    """

    def __init__(
        self,
        config: Optional[GPUConfig] = None,
        preferred_devices: Optional[List[int]] = None,
    ):
        """
        Initialize GPU manager.

        Args:
            config: GPU configuration settings
            preferred_devices: List of GPU device IDs to prefer (deprecated, use config)
        """
        if not TORCH_AVAILABLE:
            logger.warning(
                "PyTorch not available. GPU acceleration disabled. "
                "Install with: pip install torch"
            )
            self.device: Optional[Any] = None
            self.devices_available = 0
            self.preferred_devices: List[int] = []
            self.config = config or GPUConfig()
            return

        self.config = config or GPUConfig()

        # Detect device
        self.device = self._detect_device()
        self.devices_available = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.preferred_devices = (
            self.config.preferred_devices
            or preferred_devices
            or list(range(self.devices_available))
        )

        logger.info(
            "gpu_manager_initialized",
            extra={
                "device": str(self.device),
                "cuda_available": torch.cuda.is_available(),
                "device_count": self.devices_available,
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
                "pytorch_version": torch.__version__,
            },
        )

    def _detect_device(self) -> Any:
        """Auto-detect best available device."""
        if not TORCH_AVAILABLE:
            return None

        for device_type in self.config.device_preference:
            if device_type == "cuda" and torch.cuda.is_available():
                return torch.device("cuda")
            elif (
                device_type == "mps"
                and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                return torch.device("mps")
            elif device_type == "cpu":
                return torch.device("cpu")

        # Fallback to CPU
        return torch.device("cpu")

    def get_device(self, preferred_gpu: Optional[int] = None) -> Any:
        """
        Get specific GPU device or default.

        Args:
            preferred_gpu: Specific GPU device ID to use

        Returns:
            torch.device: Device to use for computation, or None if unavailable
        """
        if not TORCH_AVAILABLE or self.device is None:
            return None

        if preferred_gpu is not None and self.devices_available > preferred_gpu:
            return torch.device(f"cuda:{preferred_gpu}")
        return self.device

    def memory_stats(self, device_id: int = 0) -> Dict[str, Any]:
        """
        Get GPU memory statistics.

        Args:
            device_id: GPU device ID to query

        Returns:
            Dictionary with memory statistics:
            - available: bool
            - device_id: GPU device number
            - allocated_mb: Currently allocated memory in MB
            - reserved_mb: Reserved memory in MB
            - max_allocated_mb: Peak memory allocation in MB
            - total_mb: Total GPU memory in MB
            - utilization_pct: Memory utilization percentage
        """
        if (
            not TORCH_AVAILABLE
            or not torch.cuda.is_available()
            or device_id >= self.devices_available
        ):
            return {"available": False}

        allocated = torch.cuda.memory_allocated(device_id) / 1024**2
        reserved = torch.cuda.memory_reserved(device_id) / 1024**2
        max_allocated = torch.cuda.max_memory_allocated(device_id) / 1024**2
        total = torch.cuda.get_device_properties(device_id).total_memory / 1024**2
        utilization = (allocated / total * 100) if total > 0 else 0.0

        return {
            "available": True,
            "device_id": device_id,
            "allocated_mb": allocated,
            "reserved_mb": reserved,
            "max_allocated_mb": max_allocated,
            "total_mb": total,
            "utilization_pct": utilization,
        }

    def health_check(self) -> Dict[str, Any]:
        """
        Check GPU health status.

        Returns:
            Dictionary with health status:
            - healthy: Overall health boolean
            - mode: "cuda", "mps", or "cpu"
            - devices: List of device health info (for CUDA)
        """
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return {
                "healthy": True,
                "mode": str(self.device) if self.device else "cpu",
                "devices": [],
            }

        health: Dict[str, Any] = {"healthy": True, "mode": "cuda", "devices": []}

        for device_id in range(self.devices_available):
            try:
                stats = self.memory_stats(device_id)
                if not stats.get("available", False):
                    dev_health = {
                        "device_id": device_id,
                        "healthy": False,
                        "memory_stats": stats,
                    }
                else:
                    dev_health = {
                        "device_id": device_id,
                        "healthy": stats["utilization_pct"] < 95,
                        "memory_stats": stats,
                    }

                health["devices"].append(dev_health)

                if not dev_health["healthy"]:
                    health["healthy"] = False

            except Exception as e:
                logger.error(f"GPU {device_id} health check failed: {e}")
                health["healthy"] = False
                health["devices"].append(
                    {
                        "device_id": device_id,
                        "healthy": False,
                        "error": str(e),
                    }
                )

        return health

    def is_available(self) -> bool:
        """Check if GPU is available."""
        return TORCH_AVAILABLE and torch.cuda.is_available()

    def __repr__(self) -> str:
        return (
            f"GPUManager(device={self.device}, "
            f"devices_available={self.devices_available}, "
            f"torch_available={TORCH_AVAILABLE})"
        )


__all__ = ["GPUManager", "GPUConfig", "GPUMemoryStats"]
