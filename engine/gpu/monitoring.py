"""
ELEANOR V8 - GPU Monitoring

Lightweight GPU metrics collection and health reporting.
"""

import logging
from typing import Any, Dict, List, Optional

from .manager import GPUManager

logger = logging.getLogger(__name__)


def collect_gpu_metrics(gpu_manager: Optional[GPUManager]) -> Dict[str, Any]:
    """Collect GPU health and memory metrics."""
    if gpu_manager is None:
        return {
            "enabled": False,
            "available": False,
            "device": None,
            "devices_available": 0,
            "health": {"healthy": False, "mode": "cpu", "devices": []},
            "memory_stats": [],
        }

    available = gpu_manager.is_available()
    device = str(gpu_manager.device) if getattr(gpu_manager, "device", None) else None
    devices_available = getattr(gpu_manager, "devices_available", 0)

    health = gpu_manager.health_check()
    memory_stats: List[Dict[str, Any]] = []
    if available and getattr(gpu_manager.device, "type", None) == "cuda":
        for device_id in range(devices_available):
            memory_stats.append(gpu_manager.memory_stats(device_id))

    return {
        "enabled": True,
        "available": available,
        "device": device,
        "devices_available": devices_available,
        "health": health,
        "memory_stats": memory_stats,
    }


__all__ = ["collect_gpu_metrics"]
