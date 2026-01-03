"""
GPU Acceleration for ELEANOR V8

Provides GPU acceleration infrastructure:
- GPUManager: Device detection and management
- AsyncGPUExecutor: Async GPU operations
- GPUEmbeddingCache: GPU-accelerated embeddings
- BatchProcessor: Dynamic batching for GPU workloads
"""

from typing import Any, Optional

from .manager import GPUManager, GPUConfig, GPUMemoryStats
from .async_ops import AsyncGPUExecutor
from .monitoring import collect_gpu_metrics

GPUEmbeddingCache: Optional[Any] = None
BatchProcessor: Optional[Any] = None
MultiGPUManager: Optional[Any] = None

_embeddings: Optional[Any]
try:
    from . import embeddings as _embeddings
except Exception:
    _embeddings = None
if _embeddings is not None:
    GPUEmbeddingCache = _embeddings.GPUEmbeddingCache

_batch_processor: Optional[Any]
try:
    from . import batch_processor as _batch_processor
except Exception:
    _batch_processor = None
if _batch_processor is not None:
    BatchProcessor = _batch_processor.BatchProcessor

_parallelization: Optional[Any]
try:
    from . import parallelization as _parallelization
except Exception:
    _parallelization = None
if _parallelization is not None:
    MultiGPUManager = _parallelization.MultiGPURouter

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
