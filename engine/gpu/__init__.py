"""
GPU Acceleration for ELEANOR V8

Provides GPU acceleration infrastructure:
- GPUManager: Device detection and management
- AsyncGPUExecutor: Async GPU operations
- GPUEmbeddingCache: GPU-accelerated embeddings (coming soon)
- GPU-based batch processing (coming soon)
"""

from engine.gpu.manager import GPUManager
from engine.gpu.async_ops import AsyncGPUExecutor

__all__ = [
    "GPUManager",
    "AsyncGPUExecutor",
]
