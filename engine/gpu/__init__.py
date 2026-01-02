"""
ELEANOR V8 — GPU Acceleration Framework

Provides GPU acceleration for LLM inference, embeddings, and critic operations.
"""

from .manager import GPUManager, GPUConfig, GPUMemoryStats
from .async_ops import AsyncGPUExecutor
from .embeddings import GPUEmbeddingCache
from .batch_processor import GPUBatchProcessor
from .parallelization import MultiGPURouter as MultiGPUManager

__all__ = [
    "GPUManager",
    "GPUConfig",
    "GPUMemoryStats",
    "AsyncGPUExecutor",
    "GPUEmbeddingCache",
    "GPUBatchProcessor",
    "MultiGPUManager",
]
