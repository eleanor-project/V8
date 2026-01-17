"""
Tests for GPU module initialization and monitoring helpers.
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import Mock

from engine.gpu.monitoring import collect_gpu_metrics


def test_collect_gpu_metrics_none():
    """Test metrics when no GPU manager is provided."""
    metrics = collect_gpu_metrics(None)

    assert metrics["enabled"] is False
    assert metrics["available"] is False
    assert metrics["health"]["mode"] == "cpu"
    assert metrics["memory_stats"] == []


def test_collect_gpu_metrics_cuda_manager():
    """Test metrics collection when GPU manager is available."""
    manager = Mock()
    manager.is_available.return_value = True
    manager.device = Mock(type="cuda")
    manager.devices_available = 2
    manager.health_check.return_value = {"healthy": True, "mode": "cuda", "devices": []}
    manager.memory_stats.side_effect = [{"device_id": 0}, {"device_id": 1}]

    metrics = collect_gpu_metrics(manager)

    assert metrics["enabled"] is True
    assert metrics["available"] is True
    assert metrics["device"] is not None
    assert len(metrics["memory_stats"]) == 2
    manager.memory_stats.assert_any_call(0)
    manager.memory_stats.assert_any_call(1)


def test_gpu_init_optional_imports_missing(monkeypatch):
    """Test optional GPU modules gracefully handle missing torch."""
    import engine.gpu as gpu

    import importlib._bootstrap as bootstrap

    blocked = {
        "engine.gpu.embeddings",
        "engine.gpu.batch_processor",
        "engine.gpu.parallelization",
    }

    real_handle_fromlist = bootstrap._handle_fromlist

    def blocked_handle_fromlist(module, fromlist, *args, **kwargs):
        if module.__name__ == "engine.gpu" and any(
            name in {"embeddings", "batch_processor", "parallelization"} for name in fromlist
        ):
            raise ImportError("optional gpu module missing")
        return real_handle_fromlist(module, fromlist, *args, **kwargs)

    monkeypatch.setattr(bootstrap, "_handle_fromlist", blocked_handle_fromlist)
    for module_name in blocked:
        sys.modules.pop(module_name, None)
    reloaded = importlib.reload(gpu)

    assert reloaded.GPUEmbeddingCache is None
    assert reloaded.BatchProcessor is None
    assert reloaded.MultiGPUManager is None
    assert reloaded.GPUBatchProcessor is None

    for name in ("GPUEmbeddingCache", "BatchProcessor", "MultiGPUManager", "GPUBatchProcessor"):
        assert name in reloaded.__all__

    # Restore import and reload to avoid side effects for other tests.
    monkeypatch.setattr(bootstrap, "_handle_fromlist", real_handle_fromlist)
    importlib.reload(reloaded)
