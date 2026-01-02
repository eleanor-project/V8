"""
ELEANOR V8 - GPU Acceleration Module

Provides GPU acceleration for LLM inference, embeddings, and critic evaluation.

Components:
- GPUManager: Device detection, allocation, and health monitoring
- AsyncGPUExecutor: Async GPU operations with CUDA streams
- GPUEmbeddingCache: GPU-accelerated embeddings and similarity search
- Memory optimization utilities

Usage:
    from engine.gpu import GPUManager, AsyncGPUExecutor

    gpu_manager = GPUManager()
    if gpu_manager.device.type == "cuda":
        executor = AsyncGPUExecutor(gpu_manager.device)
        result = await executor.execute_async(model_forward, inputs)
"""

from .manager import GPUManager, GPUConfig, GPUMemoryStats
from .async_ops import AsyncGPUExecutor
from .embeddings import GPUEmbeddingCache
from .batch_processor import GPUBatchProcessor
from .parallelization import MultiGPURouter as MultiGPUManager

__all__ = [
    "GPUManager",
    "GPUConfig",
    "AsyncGPUExecutor",
]

__version__ = "1.0.0"
