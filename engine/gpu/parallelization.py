"""
GPU Model Parallelization for ELEANOR V8

Provides DataParallel and model parallelization across multiple GPUs
for large models that benefit from multi-GPU inference.
"""

import logging
import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class ModelParallelWrapper:
    """
    Wrap models for multi-GPU parallelization.
    
    Supports:
    - DataParallel: Replicate model across GPUs, split batches
    - Manual device placement for pipeline parallelism
    """
    
    def __init__(
        self,
        model: Any,
        device_ids: Optional[List[int]] = None,
        output_device: Optional[int] = None,
        strategy: str = "data_parallel"
    ):
        """
        Initialize model parallelization.
        
        Args:
            model: PyTorch model to parallelize
            device_ids: List of GPU device IDs to use
            output_device: GPU to collect outputs (default: device_ids[0])
            strategy: "data_parallel" or "pipeline"
        """
        self.original_model = model
        self.strategy = strategy
        
        # Auto-detect GPUs if not specified
        if device_ids is None:
            device_count = torch.cuda.device_count()
            device_ids = list(range(device_count)) if device_count > 0 else []
        
        self.device_ids = device_ids
        self.output_device = output_device or (device_ids[0] if device_ids else None)
        
        # Apply parallelization strategy
        if len(self.device_ids) > 1:
            self._apply_parallelization()
        elif len(self.device_ids) == 1:
            # Single GPU
            self.model = model.to(f"cuda:{self.device_ids[0]}")
            logger.info(f"Model loaded on single GPU: cuda:{self.device_ids[0]}")
        else:
            # CPU fallback
            self.model = model
            logger.warning("No GPUs available, using CPU")
    
    def _apply_parallelization(self):
        """Apply the chosen parallelization strategy."""
        if self.strategy == "data_parallel":
            self._apply_data_parallel()
        elif self.strategy == "pipeline":
            self._apply_pipeline_parallel()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _apply_data_parallel(self):
        """
        Apply DataParallel: replicate model, split batches.
        
        Best for: Models that fit on single GPU but benefit from batch parallelism
        """
        self.model = nn.DataParallel(
            self.original_model,
            device_ids=self.device_ids,
            output_device=self.output_device
        )
        
        # Move to first device
        self.model = self.model.to(f"cuda:{self.device_ids[0]}")
        
        logger.info(
            f"Model parallelized with DataParallel across GPUs: {self.device_ids}"
        )
    
    def _apply_pipeline_parallel(self):
        """
        Apply pipeline parallelism: split model layers across GPUs.
        
        Best for: Very large models that don't fit on single GPU
        Note: Requires manual layer assignment
        """
        # This is a simplified pipeline approach
        # Production would use torch.distributed.pipeline or DeepSpeed
        
        if not hasattr(self.original_model, 'split_for_pipeline'):
            logger.warning(
                "Model doesn't support pipeline parallelism, falling back to DataParallel"
            )
            self._apply_data_parallel()
            return
        
        # Model must implement split_for_pipeline method
        self.model = self.original_model.split_for_pipeline(self.device_ids)
        logger.info(
            f"Model split with pipeline parallelism across GPUs: {self.device_ids}"
        )
    
    def forward(self, *args, **kwargs):
        """Forward pass through parallelized model."""
        return self.model(*args, **kwargs)
    
    def __call__(self, *args, **kwargs):
        """Make wrapper callable."""
        return self.forward(*args, **kwargs)
    
    def get_memory_stats(self) -> Dict[str, Dict[str, int]]:
        """Get memory statistics for all devices."""
        stats = {}
        
        for device_id in self.device_ids:
            if torch.cuda.is_available():
                stats[f"cuda:{device_id}"] = {
                    "allocated_mb": torch.cuda.memory_allocated(device_id) // 1024 ** 2,
                    "reserved_mb": torch.cuda.memory_reserved(device_id) // 1024 ** 2,
                    "max_allocated_mb": torch.cuda.max_memory_allocated(device_id) // 1024 ** 2,
                }
        
        return stats


class MultiGPURouter:
    """
    Route different models to different GPUs for parallel inference.
    
    Use case: Running multiple LLM critics simultaneously on different GPUs
    """
    
    def __init__(self, device_ids: Optional[List[int]] = None):
        """Initialize multi-GPU router."""
        if device_ids is None:
            device_count = torch.cuda.device_count()
            device_ids = list(range(device_count)) if device_count > 0 else []
        
        self.device_ids = device_ids
        self.model_assignments: Dict[str, int] = {}
        self.current_device_idx = 0
        
        logger.info(f"MultiGPURouter initialized with devices: {device_ids}")
    
    def assign_model(self, model_name: str, device_id: Optional[int] = None) -> int:
        """
        Assign a model to a specific GPU (or auto-assign).
        
        Args:
            model_name: Identifier for the model/critic
            device_id: Specific GPU to use (None for auto-assignment)
        
        Returns:
            Device ID assigned
        """
        if device_id is not None:
            if device_id not in self.device_ids:
                raise ValueError(f"Device {device_id} not available")
            assigned_device = device_id
        else:
            # Round-robin assignment
            if not self.device_ids:
                return -1  # CPU
            
            assigned_device = self.device_ids[self.current_device_idx]
            self.current_device_idx = (self.current_device_idx + 1) % len(self.device_ids)
        
        self.model_assignments[model_name] = assigned_device
        logger.info(f"Assigned {model_name} to cuda:{assigned_device}")
        
        return assigned_device
    
    def get_device(self, model_name: str) -> torch.device:
        """
        Get the device for a specific model.
        
        Args:
            model_name: Model identifier
        
        Returns:
            torch.device for the model
        """
        device_id = self.model_assignments.get(model_name)
        
        if device_id is None:
            # Auto-assign if not already assigned
            device_id = self.assign_model(model_name)
        
        if device_id < 0:
            return torch.device("cpu")
        
        return torch.device(f"cuda:{device_id}")
    
    def get_load_balance_stats(self) -> Dict[int, int]:
        """
        Get load balance statistics (models per GPU).
        
        Returns:
            Dict mapping device_id to number of models assigned
        """
        stats = {device_id: 0 for device_id in self.device_ids}
        
        for device_id in self.model_assignments.values():
            if device_id in stats:
                stats[device_id] += 1
        
        return stats


__all__ = [
    "ModelParallelWrapper",
    "MultiGPURouter",
]
