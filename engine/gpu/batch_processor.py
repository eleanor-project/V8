"""
ELEANOR V8 — GPU Batch Processor

Efficient batched processing for critics and inference operations.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Callable
import asyncio

logger = logging.getLogger(__name__)


class GPUBatchProcessor:
    """
    Process operations in batches for GPU efficiency.
    
    Batching reduces GPU kernel launch overhead and improves throughput.
    """
    
    def __init__(
        self,
        gpu_manager: 'GPUManager',
        async_executor: 'AsyncGPUExecutor',
        default_batch_size: int = 8,
    ):
        """
        Initialize batch processor.
        
        Args:
            gpu_manager: GPU manager instance
            async_executor: Async GPU executor
            default_batch_size: Default batch size
        """
        self.gpu_manager = gpu_manager
        self.async_executor = async_executor
        self.default_batch_size = default_batch_size
        
        # Dynamic batch size based on config
        self.batch_size = gpu_manager.config.batch_size
        self.max_batch_size = gpu_manager.config.max_batch_size
        
        logger.info(
            "GPU batch processor initialized",
            batch_size=self.batch_size,
            max_batch_size=self.max_batch_size
        )
    
    async def process_batch(
        self,
        items: List[Any],
        process_fn: Callable,
        batch_size: Optional[int] = None,
    ) -> List[Any]:
        """
        Process items in batches.
        
        Args:
            items: Items to process
            process_fn: Function to apply to each batch
            batch_size: Batch size (uses default if None)
        
        Returns:
            Processed results
        """
        if not items:
            return []
        
        batch_size = batch_size or self.batch_size
        results = []
        
        start_time = time.time()
        num_batches = (len(items) + batch_size - 1) // batch_size
        
        logger.debug(
            f"Processing {len(items)} items in {num_batches} batches",
            batch_size=batch_size
        )
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            # Process batch on GPU
            batch_results = await self.async_executor.run_async(
                process_fn,
                batch
            )
            
            results.extend(batch_results)
        
        duration_ms = (time.time() - start_time) * 1000
        throughput = len(items) / (duration_ms / 1000) if duration_ms > 0 else 0
        
        logger.info(
            "Batch processing complete",
            total_items=len(items),
            num_batches=num_batches,
            duration_ms=duration_ms,
            throughput_per_sec=throughput
        )
        
        return results
    
    async def process_critics_batched(
        self,
        critics: Dict[str, Any],
        inputs: List[str],
        batch_size: Optional[int] = None,
    ) -> Dict[str, List[Any]]:
        """
        Execute multiple critics with batched inputs.
        
        Args:
            critics: Dictionary of critic name -> critic instance
            inputs: List of input texts
            batch_size: Batch size (uses default if None)
        
        Returns:
            Dictionary of critic name -> results list
        """
        results = {name: [] for name in critics.keys()}
        batch_size = batch_size or self.batch_size
        
        start_time = time.time()
        num_batches = (len(inputs) + batch_size - 1) // batch_size
        
        logger.info(
            "Starting batched critic execution",
            num_critics=len(critics),
            num_inputs=len(inputs),
            batch_size=batch_size,
            num_batches=num_batches
        )
        
        # Process each batch
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            
            # Prepare batch tensor if using GPU
            if self.gpu_manager.is_available():
                batch_tensor = await self._prepare_batch_tensor(batch)
            else:
                batch_tensor = batch
            
            # Run all critics on this batch in parallel
            critic_tasks = []
            for name, critic in critics.items():
                task = asyncio.create_task(
                    self._run_critic_on_batch(critic, batch_tensor, name)
                )
                critic_tasks.append((name, task))
            
            # Wait for all critics
            for name, task in critic_tasks:
                batch_results = await task
                results[name].extend(batch_results)
        
        duration_ms = (time.time() - start_time) * 1000
        total_ops = len(inputs) * len(critics)
        throughput = total_ops / (duration_ms / 1000) if duration_ms > 0 else 0
        
        logger.info(
            "Batched critic execution complete",
            total_operations=total_ops,
            duration_ms=duration_ms,
            throughput_per_sec=throughput
        )
        
        return results
    
    async def _prepare_batch_tensor(self, batch: List[str]) -> Any:
        """
        Prepare batch as GPU tensor.
        
        Args:
            batch: List of text inputs
        
        Returns:
            Batch tensor on GPU
        """
        # This is a placeholder - actual implementation depends on tokenizer
        # For now, just return the batch
        return batch
    
    async def _run_critic_on_batch(
        self,
        critic: Any,
        batch: Any,
        critic_name: str
    ) -> List[Any]:
        """
        Run single critic on batch.
        
        Args:
            critic: Critic instance
            batch: Batch data
            critic_name: Critic name for logging
        
        Returns:
            List of results
        """
        try:
            # Check if critic supports batched evaluation
            if hasattr(critic, 'evaluate_batch'):
                results = await self.async_executor.run_async(
                    critic.evaluate_batch,
                    batch
                )
            else:
                # Fall back to sequential evaluation
                results = []
                for item in batch:
                    result = await self.async_executor.run_async(
                        critic.evaluate,
                        item
                    )
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(
                f"Critic {critic_name} failed on batch",
                error=str(e)
            )
            return [None] * len(batch)
    
    def adjust_batch_size(self, success_rate: float, latency_ms: float) -> None:
        """
        Dynamically adjust batch size based on performance.
        
        Args:
            success_rate: Recent success rate (0-1)
            latency_ms: Average latency
        """
        target_latency = 500  # 500ms target
        
        if success_rate < 0.9:  # Too many failures
            self.batch_size = max(1, self.batch_size // 2)
            logger.info(f"Reduced batch size to {self.batch_size} due to failures")
        elif latency_ms > target_latency * 2:  # Too slow
            self.batch_size = max(1, self.batch_size - 2)
            logger.info(f"Reduced batch size to {self.batch_size} due to latency")
        elif latency_ms < target_latency * 0.5 and success_rate > 0.95:  # Can go faster
            self.batch_size = min(self.max_batch_size, self.batch_size + 2)
            logger.info(f"Increased batch size to {self.batch_size}")


__all__ = ["GPUBatchProcessor"]
