"""
GPU Memory Optimization for ELEANOR V8

Provides techniques for reducing GPU memory footprint:
- Mixed precision (FP16/BF16)
- Model quantization (INT8/INT4)
- Gradient checkpointing (training)
- KV cache optimization
"""

import logging
import torch
import torch.nn as nn
from typing import Optional, Any, Dict, cast
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class MemoryOptimizer:
    """
    Optimize GPU memory usage for inference.
    """

    def __init__(
        self,
        enable_mixed_precision: bool = True,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize memory optimizer.

        Args:
            enable_mixed_precision: Use automatic mixed precision
            dtype: Specific dtype to use (float16, bfloat16, etc.)
        """
        self.enable_mixed_precision = enable_mixed_precision

        # Auto-select best dtype for hardware
        if dtype is None:
            if torch.cuda.is_available():
                # Use bfloat16 on Ampere+ GPUs, float16 otherwise
                major, minor = torch.cuda.get_device_capability()
                if major >= 8:  # Ampere (A100, RTX 30xx) or newer
                    self.dtype = torch.bfloat16
                else:
                    self.dtype = torch.float16
            else:
                self.dtype = torch.float32
        else:
            self.dtype = dtype

        logger.info(
            f"MemoryOptimizer initialized: mixed_precision={enable_mixed_precision}, "
            f"dtype={self.dtype}"
        )

    @contextmanager
    def inference_mode(self):
        """
        Context manager for optimized inference.

        Example:
            with optimizer.inference_mode():
                output = model(input)
        """
        if self.enable_mixed_precision and torch.cuda.is_available():
            with torch.cuda.amp.autocast(dtype=self.dtype):
                with torch.inference_mode():
                    yield
        else:
            with torch.inference_mode():
                yield

    def optimize_model(
        self, model: nn.Module, quantization_bits: Optional[int] = None
    ) -> nn.Module:
        """
        Apply optimizations to a model.

        Args:
            model: PyTorch model to optimize
            quantization_bits: Quantize to 8-bit or 4-bit (None to disable)

        Returns:
            Optimized model
        """
        # Convert to appropriate dtype
        if self.enable_mixed_precision:
            model = model.to(dtype=self.dtype)
            logger.info(f"Model converted to {self.dtype}")

        # Apply quantization if requested
        if quantization_bits is not None:
            model = self._quantize_model(model, quantization_bits)

        # Set to eval mode
        model.eval()

        return model

    def _quantize_model(self, model: nn.Module, bits: int) -> nn.Module:
        """
        Quantize model to reduce memory footprint.

        Args:
            model: Model to quantize
            bits: 8 or 4

        Returns:
            Quantized model
        """
        if bits == 8:
            return self._quantize_8bit(model)
        elif bits == 4:
            return self._quantize_4bit(model)
        else:
            raise ValueError(f"Unsupported quantization: {bits} bits")

    def _quantize_8bit(self, model: nn.Module) -> nn.Module:
        """
        Apply 8-bit quantization using torch.quantization.
        """
        try:
            # Dynamic quantization (easier, good for inference)
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear},  # Quantize linear layers
                dtype=torch.qint8,
            )
            logger.info("Applied 8-bit dynamic quantization")
            return cast(nn.Module, quantized_model)
        except Exception as e:
            logger.warning(f"8-bit quantization failed: {e}, using original model")
            return model

    def _quantize_4bit(self, model: nn.Module) -> nn.Module:
        """
        Apply 4-bit quantization using bitsandbytes (if available).
        """
        try:
            import bitsandbytes as bnb  # type: ignore[import-not-found]

            # Replace linear layers with 4-bit versions
            for name, module in model.named_children():
                if isinstance(module, nn.Linear):
                    # Replace with 4-bit linear
                    setattr(
                        model,
                        name,
                        bnb.nn.Linear4bit(
                            module.in_features,
                            module.out_features,
                            bias=module.bias is not None,
                        ),
                    )

            logger.info("Applied 4-bit quantization")
            return model

        except ImportError:
            logger.warning(
                "bitsandbytes not available for 4-bit quantization, " "falling back to 8-bit"
            )
            return self._quantize_8bit(model)
        except Exception as e:
            logger.warning(f"4-bit quantization failed: {e}, using original model")
            return model


class KVCacheManager:
    """
    Manage key-value caches for transformer models.

    KV caching reduces computation by storing intermediate results
    for autoregressive generation.
    """

    def __init__(self, max_cache_size_mb: int = 4096, device: Optional[torch.device] = None):
        """
        Initialize KV cache manager.

        Args:
            max_cache_size_mb: Maximum cache size in MB
            device: Device to store cache on
        """
        self.max_cache_size_mb = max_cache_size_mb
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.cache: Dict[str, Any] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]

        self.cache_misses += 1
        return None

    def put(self, key: str, value: Any):
        """Store value in cache."""
        # Check size before adding
        current_size = self._get_cache_size_mb()

        if current_size >= self.max_cache_size_mb:
            # Evict oldest entry (simple FIFO)
            self._evict_oldest()

        self.cache[key] = value

    def clear(self):
        """Clear all cached values."""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0

    def _get_cache_size_mb(self) -> int:
        """Estimate cache size in MB."""
        total_bytes = 0

        for value in self.cache.values():
            if isinstance(value, torch.Tensor):
                total_bytes += value.element_size() * value.nelement()

        return total_bytes // (1024**2)

    def _evict_oldest(self):
        """Evict the oldest cache entry."""
        if self.cache:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            logger.debug(f"Evicted cache entry: {oldest_key}")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0

        return {
            "cache_size_mb": self._get_cache_size_mb(),
            "max_cache_size_mb": self.max_cache_size_mb,
            "num_entries": len(self.cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
        }


__all__ = [
    "MemoryOptimizer",
    "KVCacheManager",
]
