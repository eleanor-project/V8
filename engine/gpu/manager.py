"""
GPU Manager for ELEANOR V8

Manages GPU resources, device allocation, and monitoring.
"""

import logging
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

torch: Optional[Any]
try:
    import torch as _torch
except ImportError:  # pragma: no cover - torch optional
    torch = None
else:
    torch = _torch


@dataclass
class GPUConfig:
    """Configuration for GPU usage and tuning."""

    enabled: bool = True
    device_preference: List[str] = field(default_factory=lambda: ["cuda", "mps", "cpu"])
    preferred_devices: Optional[List[int]] = None
    mixed_precision: bool = True
    num_streams: int = 4
    max_memory_per_gpu: Optional[float] = None
    log_memory_stats: bool = False
    memory_check_interval: float = 60.0
    default_batch_size: int = 8
    max_batch_size: int = 32
    dynamic_batching: bool = True


@dataclass
class GPUMemoryStats:
    """Structured GPU memory stats."""

    available: bool
    device_id: int
    allocated_mb: float = 0.0
    reserved_mb: float = 0.0
    max_allocated_mb: float = 0.0
    total_mb: float = 0.0
    utilization_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "device_id": self.device_id,
            "allocated_mb": self.allocated_mb,
            "reserved_mb": self.reserved_mb,
            "max_allocated_mb": self.max_allocated_mb,
            "total_mb": self.total_mb,
            "utilization_pct": self.utilization_pct,
        }


class GPUManager:
    """
    Manage GPU resources and device allocation.

    Features:
    - Auto-detect CUDA, MPS (Apple Silicon), or CPU
    - Multi-GPU device management
    - Memory statistics and monitoring
    - Device health checks
    """

    def __init__(
        self,
        config: Optional[GPUConfig] = None,
        preferred_devices: Optional[List[int]] = None,
    ):
        self.config = config or GPUConfig()

        if preferred_devices is not None:
            self.preferred_devices = list(preferred_devices)
        elif self.config.preferred_devices is not None:
            self.preferred_devices = list(self.config.preferred_devices)
        else:
            self.preferred_devices = []

        torch_available = torch is not None and sys.modules.get("torch") is not None

        if not torch_available or torch is None:
            logger.warning(
                "PyTorch not installed. GPU acceleration unavailable. "
                "Install with: pip install torch"
            )
            self.device = None
            self.devices_available = 0
            return

        assert torch is not None

        self.devices_available = torch.cuda.device_count() if torch.cuda.is_available() else 0

        if not self.config.enabled:
            self.device = torch.device("cpu")
        else:
            self.device = self._detect_device()

        if not self.preferred_devices:
            self.preferred_devices = list(range(self.devices_available))

        logger.info(
            "gpu_manager_initialized",
            extra={
                "device": str(self.device) if self.device is not None else None,
                "device_type": getattr(self.device, "type", None),
                "cuda_available": torch.cuda.is_available(),
                "device_count": self.devices_available,
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
                "preferred_devices": self.preferred_devices,
                "enabled": self.config.enabled,
            },
        )

    def _detect_device(self):
        """Auto-detect best available device based on preference list."""
        if torch is None:
            return None

        preferences = self.config.device_preference or ["cuda", "mps", "cpu"]

        for pref in preferences:
            pref_lower = pref.lower()
            if pref_lower == "cuda" and torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info("detected_cuda_gpu", extra={"device": str(device)})
                return device

            if pref_lower == "mps" and hasattr(torch.backends, "mps"):
                if torch.backends.mps.is_available():
                    device = torch.device("mps")
                    logger.info("detected_apple_mps", extra={"device": str(device)})
                    return device

            if pref_lower == "cpu":
                device = torch.device("cpu")
                logger.info(
                    "using_cpu_fallback",
                    extra={"device": str(device), "reason": "Preference order"},
                )
                return device

        return torch.device("cpu")

    def is_available(self) -> bool:
        """Check if GPU acceleration is available."""
        if torch is None or not self.config.enabled or self.device is None:
            return False
        return self.device.type in ("cuda", "mps")

    def is_gpu_available(self) -> bool:
        """Backward-compatible GPU availability check."""
        return self.is_available()

    def get_device(self, preferred_gpu: Optional[int] = None):
        """Get best device, optionally selecting a specific CUDA index."""
        if torch is None or self.device is None:
            return None

        if self.device.type == "cuda" and preferred_gpu is not None:
            if torch.cuda.is_available() and 0 <= preferred_gpu < torch.cuda.device_count():
                return torch.device(f"cuda:{preferred_gpu}")

        return self.device

    def memory_stats(self, device_id: int = 0) -> Dict[str, Any]:
        """Return memory stats for a CUDA device (if available)."""
        if (
            torch is None
            or not self.config.enabled
            or self.device is None
            or self.device.type != "cuda"
            or not torch.cuda.is_available()
        ):
            return GPUMemoryStats(available=False, device_id=device_id).to_dict()

        try:
            allocated = torch.cuda.memory_allocated(device_id)
            reserved = torch.cuda.memory_reserved(device_id)
            max_allocated = torch.cuda.max_memory_allocated(device_id)
            total = torch.cuda.get_device_properties(device_id).total_memory
        except Exception as exc:
            logger.warning(
                "gpu_memory_stats_failed",
                extra={"error": str(exc), "device_id": device_id},
            )
            return GPUMemoryStats(available=False, device_id=device_id).to_dict()

        allocated_mb = allocated / (1024**2)
        reserved_mb = reserved / (1024**2)
        max_allocated_mb = max_allocated / (1024**2)
        total_mb = total / (1024**2)
        utilization_pct = (allocated / total * 100.0) if total else 0.0

        return GPUMemoryStats(
            available=True,
            device_id=device_id,
            allocated_mb=allocated_mb,
            reserved_mb=reserved_mb,
            max_allocated_mb=max_allocated_mb,
            total_mb=total_mb,
            utilization_pct=utilization_pct,
        ).to_dict()

    def health_check(self) -> Dict[str, Any]:
        """Return a health snapshot for all available devices."""
        if not self.is_available():
            mode = "cpu" if torch is not None else "unavailable"
            return {"healthy": False, "mode": mode, "devices": []}

        if self.device is None:
            mode = "cpu" if torch is not None else "unavailable"
            return {"healthy": False, "mode": mode, "devices": []}

        if self.device.type == "mps":
            return {"healthy": True, "mode": "mps", "devices": []}

        devices = []
        healthy = True
        for device_id in range(self.devices_available):
            stats = self.memory_stats(device_id)
            device_healthy = bool(stats.get("available"))
            healthy = healthy and device_healthy
            devices.append(
                {
                    "device_id": device_id,
                    "healthy": device_healthy,
                    "memory_stats": stats,
                }
            )

        return {"healthy": healthy, "mode": "cuda", "devices": devices}

    def __repr__(self) -> str:
        device_str = str(self.device) if self.device is not None else "none"
        return (
            f"GPUManager(device={device_str}, devices_available={self.devices_available}, "
            f"enabled={self.config.enabled})"
        )


__all__ = ["GPUConfig", "GPUMemoryStats", "GPUManager"]
