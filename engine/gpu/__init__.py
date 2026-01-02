"""
ELEANOR V8 - GPU Acceleration Module

Provides GPU acceleration for:
- LLM inference
- Embedding computations
- Critic evaluations
- Precedent retrieval similarity search

Supports:
- NVIDIA GPUs (CUDA)
- Apple Silicon (MPS)
- CPU fallback
"""

from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Check GPU availability
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available() or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())
    CUDA_AVAILABLE = torch.cuda.is_available()
    MPS_AVAILABLE = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
except ImportError:
    GPU_AVAILABLE = False
    CUDA_AVAILABLE = False
    MPS_AVAILABLE = False
    logger.warning("PyTorch not installed - GPU acceleration unavailable")

# Conditionally import GPU components
if GPU_AVAILABLE:
    try:
        from engine.gpu.manager import GPUManager
        from engine.gpu.async_ops import AsyncGPUExecutor
        from engine.gpu.embeddings import GPUEmbeddingCache
        from engine.gpu.batch_processor import BatchProcessor
        
        __all__ = [
            "GPUManager",
            "AsyncGPUExecutor",
            "GPUEmbeddingCache",
            "BatchProcessor",
            "GPU_AVAILABLE",
            "CUDA_AVAILABLE",
            "MPS_AVAILABLE",
        ]
        
        logger.info(
            "gpu_module_loaded",
            cuda_available=CUDA_AVAILABLE,
            mps_available=MPS_AVAILABLE
        )
    except ImportError as e:
        logger.error(f"Failed to import GPU components: {e}")
        GPU_AVAILABLE = False
        __all__ = ["GPU_AVAILABLE", "CUDA_AVAILABLE", "MPS_AVAILABLE"]
else:
    __all__ = ["GPU_AVAILABLE", "CUDA_AVAILABLE", "MPS_AVAILABLE"]
    logger.info("gpu_acceleration_disabled", reason="No GPU detected")
