"""
GPU Acceleration for ELEANOR V8

Provides GPU acceleration infrastructure:
- GPUManager: Device detection and management
- AsyncGPUExecutor: Async GPU operations (coming soon)
- GPUEmbeddingCache: GPU-accelerated embeddings (coming soon)
- GPU-based batch processing (coming soon)
"""

from engine.gpu.manager import GPUManager

__all__ = [
    "GPUManager",
]
