"""
GPU Acceleration for ELEANOR V8

Provides GPU acceleration infrastructure:
- GPUManager: Device detection and management
- AsyncGPUExecutor: Async GPU operations
- GPUEmbeddingCache: GPU-accelerated embeddings
- BatchProcessor: Dynamic batching for GPU workloads
"""

from .manager import GPUManager, GPUConfig, GPUMemoryStats
from .async_ops import AsyncGPUExecutor
from .monitoring import collect_gpu_metrics

try:
    from .embeddings import GPUEmbeddingCache
except Exception:
    GPUEmbeddingCache = None  # type: ignore[assignment]

try:
    from .batch_processor import BatchProcessor
except Exception:
    BatchProcessor = None  # type: ignore[assignment]

try:
    from .parallelization import MultiGPURouter as MultiGPUManager
except Exception:
    MultiGPUManager = None  # type: ignore[assignment]

GPUBatchProcessor = BatchProcessor

__all__ = [
    "GPUManager",
    "GPUConfig",
    "GPUMemoryStats",
    "AsyncGPUExecutor",
    "collect_gpu_metrics",
    "GPUEmbeddingCache",
    "BatchProcessor",
    "GPUBatchProcessor",
    "MultiGPUManager",
]
