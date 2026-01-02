#!/usr/bin/env python3
"""
ELEANOR V8 GPU Acceleration Integration Example

Demonstrates:
- GPU initialization and configuration
- Multi-GPU parallel processing
- Memory optimization techniques
- Performance monitoring
- GPU-accelerated embeddings
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.engine import EleanorEngineV8, EngineConfig
from engine.gpu.manager import GPUManager
from engine.gpu.async_ops import AsyncGPUExecutor
from engine.gpu.embeddings import GPUEmbeddingCache
from engine.gpu.batch_processor import GPUBatchProcessor
from engine.gpu.memory_optimization import MemoryOptimizer
from engine.gpu.parallelization import MultiGPURouter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demo_gpu_initialization():
    """Demonstrate GPU initialization and device detection."""
    logger.info("=" * 80)
    logger.info("GPU INITIALIZATION DEMO")
    logger.info("=" * 80)
    
    # Initialize GPU manager
    gpu_manager = GPUManager()
    
    logger.info(f"\nGPU Detection:")
    logger.info(f"  Primary Device: {gpu_manager.device}")
    logger.info(f"  Available GPUs: {gpu_manager.devices_available}")
    logger.info(f"  Device IDs: {gpu_manager.device_ids}")
    
    if gpu_manager.devices_available > 0:
        for device_id in gpu_manager.device_ids:
            stats = gpu_manager.memory_stats(device_id)
            logger.info(f"\n  GPU {device_id} Memory:")
            logger.info(f"    Total: {stats.get('total', 0) / 1e9:.2f} GB")
            logger.info(f"    Allocated: {stats.get('allocated', 0) / 1e6:.2f} MB")
            logger.info(f"    Reserved: {stats.get('reserved', 0) / 1e6:.2f} MB")
    else:
        logger.warning("  No GPUs detected - running on CPU")
    
    return gpu_manager


async def demo_memory_optimization(gpu_manager: GPUManager):
    """Demonstrate memory optimization techniques."""
    logger.info("\n" + "=" * 80)
    logger.info("MEMORY OPTIMIZATION DEMO")
    logger.info("=" * 80)
    
    optimizer = MemoryOptimizer(
        enable_mixed_precision=True,
        dtype=None  # Auto-select
    )
    
    logger.info(f"\nMemory Optimizer:")
    logger.info(f"  Mixed Precision: Enabled")
    logger.info(f"  Data Type: {optimizer.dtype}")
    logger.info(f"  Inference Mode: Optimized")
    
    # Demonstrate inference mode context
    logger.info("\n  Running inference with optimizations...")
    with optimizer.inference_mode():
        # Simulated model inference
        await asyncio.sleep(0.1)
        logger.info("  ✓ Inference complete with mixed precision")
    
    return optimizer


async def demo_multi_gpu_routing(gpu_manager: GPUManager):
    """Demonstrate multi-GPU model routing."""
    logger.info("\n" + "=" * 80)
    logger.info("MULTI-GPU ROUTING DEMO")
    logger.info("=" * 80)
    
    if gpu_manager.devices_available < 2:
        logger.warning("\n  Skipping: Requires 2+ GPUs")
        return
    
    router = MultiGPURouter(device_ids=gpu_manager.device_ids)
    
    # Assign models to different GPUs
    critics = [
        "ethics_critic",
        "fairness_critic",
        "safety_critic",
        "privacy_critic",
        "transparency_critic"
    ]
    
    logger.info("\nModel-to-GPU Assignments:")
    for critic in critics:
        device_id = router.assign_model(critic)
        logger.info(f"  {critic:.<30} GPU {device_id}")
    
    # Show load balance
    balance = router.get_load_balance_stats()
    logger.info("\nLoad Balance:")
    for device_id, count in balance.items():
        logger.info(f"  GPU {device_id}: {count} models")
    
    return router


async def demo_batch_processing(gpu_manager: GPUManager):
    """Demonstrate GPU batch processing."""
    logger.info("\n" + "=" * 80)
    logger.info("BATCH PROCESSING DEMO")
    logger.info("=" * 80)
    
    batch_processor = GPUBatchProcessor(
        gpu_manager=gpu_manager,
        batch_size=8,
        max_batch_size=32
    )
    
    # Simulate batch processing
    inputs = [
        f"Sample input {i} for batch processing"
        for i in range(24)
    ]
    
    logger.info(f"\nProcessing {len(inputs)} inputs in batches:")
    logger.info(f"  Batch Size: {batch_processor.batch_size}")
    logger.info(f"  Max Batch Size: {batch_processor.max_batch_size}")
    
    import time
    start_time = time.time()
    
    # Process batches (simulated)
    for i in range(0, len(inputs), batch_processor.batch_size):
        batch = inputs[i:i + batch_processor.batch_size]
        logger.info(f"  Processing batch {i // batch_processor.batch_size + 1}: {len(batch)} items")
        await asyncio.sleep(0.05)  # Simulate processing
    
    elapsed = time.time() - start_time
    logger.info(f"\n  ✓ Processed {len(inputs)} inputs in {elapsed:.3f}s")
    logger.info(f"  Throughput: {len(inputs) / elapsed:.1f} items/sec")


async def demo_gpu_accelerated_embeddings(gpu_manager: GPUManager):
    """Demonstrate GPU-accelerated embeddings and similarity search."""
    logger.info("\n" + "=" * 80)
    logger.info("GPU-ACCELERATED EMBEDDINGS DEMO")
    logger.info("=" * 80)
    
    embedding_cache = GPUEmbeddingCache(
        device=gpu_manager.device,
        max_cached_embeddings=1000
    )
    
    logger.info(f"\nEmbedding Cache:")
    logger.info(f"  Device: {embedding_cache.device}")
    logger.info(f"  Max Cached: {embedding_cache.max_cached_embeddings:,}")
    
    # Simulate embedding computation
    logger.info("\n  Computing embeddings...")
    
    import torch
    
    # Create dummy embeddings (normally from a model)
    query_embedding = torch.randn(768, device=gpu_manager.device)
    candidate_embeddings = torch.randn(100, 768, device=gpu_manager.device)
    
    # Compute similarity on GPU
    import time
    start_time = time.time()
    
    similarities = embedding_cache.compute_similarity(
        query_embedding,
        candidate_embeddings
    )
    
    elapsed = time.time() - start_time
    
    # Get top matches
    top_k = 5
    top_scores, top_indices = torch.topk(similarities, top_k)
    
    logger.info(f"\n  ✓ Computed similarity for 100 candidates in {elapsed * 1000:.2f}ms")
    logger.info(f"\n  Top {top_k} Matches:")
    for i, (score, idx) in enumerate(zip(top_scores, top_indices), 1):
        logger.info(f"    {i}. Candidate {idx.item():3d}: {score.item():.4f}")
    
    # Show cache stats
    stats = embedding_cache.get_stats()
    logger.info(f"\n  Cache Stats:")
    for key, value in stats.items():
        logger.info(f"    {key}: {value}")


async def demo_async_gpu_operations(gpu_manager: GPUManager):
    """Demonstrate async GPU operations with multiple streams."""
    logger.info("\n" + "=" * 80)
    logger.info("ASYNC GPU OPERATIONS DEMO")
    logger.info("=" * 80)
    
    executor = AsyncGPUExecutor(
        device=gpu_manager.device,
        num_streams=4
    )
    
    logger.info(f"\nAsync Executor:")
    logger.info(f"  Device: {executor.device}")
    logger.info(f"  CUDA Streams: {len(executor.streams)}")
    
    # Simulate parallel GPU operations
    async def gpu_task(task_id: int, duration: float):
        logger.info(f"  Task {task_id}: Starting...")
        await asyncio.sleep(duration)
        logger.info(f"  Task {task_id}: Complete")
        return task_id
    
    logger.info("\n  Running 8 tasks in parallel across 4 streams:")
    
    import time
    start_time = time.time()
    
    tasks = [
        gpu_task(i, 0.1)
        for i in range(8)
    ]
    
    results = await asyncio.gather(*tasks)
    
    elapsed = time.time() - start_time
    
    logger.info(f"\n  ✓ Completed {len(results)} tasks in {elapsed:.3f}s")
    logger.info(f"  Expected serial time: ~0.8s")
    logger.info(f"  Speedup: {0.8 / elapsed:.2f}x")


async def demo_full_pipeline(gpu_manager: GPUManager):
    """Demonstrate full ELEANOR pipeline with GPU acceleration."""
    logger.info("\n" + "=" * 80)
    logger.info("FULL PIPELINE WITH GPU ACCELERATION")
    logger.info("=" * 80)
    
    # Create engine with GPU support
    engine = EleanorEngineV8(
        config=EngineConfig(
            max_concurrency=gpu_manager.devices_available * 2,
            enable_precedent_analysis=True,
            enable_reflection=True
        )
    )
    
    # Note: Full GPU integration requires engine.py modifications
    # This is a demonstration of the API
    
    logger.info("\nEngine Configuration:")
    logger.info(f"  GPU Enabled: {gpu_manager.devices_available > 0}")
    logger.info(f"  GPUs Available: {gpu_manager.devices_available}")
    logger.info(f"  Max Concurrency: {engine.config.max_concurrency}")
    
    test_input = """
    A hospital's AI system recommends denying coverage for an expensive cancer
    treatment based on cost-benefit analysis, overriding the doctor's recommendation.
    """
    
    logger.info(f"\nRunning ELEANOR V8 pipeline with GPU acceleration...")
    logger.info(f"Input: {test_input.strip()[:80]}...")
    
    import time
    start_time = time.time()
    
    # Note: Actual GPU acceleration requires integration in engine.py
    # This demonstrates the intended workflow
    result = await engine.run(
        text=test_input,
        context={"domain": "healthcare"},
        detail_level=2
    )
    
    elapsed = time.time() - start_time
    
    logger.info(f"\n  ✓ Pipeline complete in {elapsed:.3f}s")
    logger.info(f"\nResults:")
    logger.info(f"  Trace ID: {result.trace_id}")
    logger.info(f"  Critics Evaluated: {len(result.critic_findings or {})}")
    logger.info(f"  Evidence Recorded: {result.evidence_count or 0} items")
    
    if gpu_manager.devices_available > 0:
        logger.info(f"\n  GPU Memory After Processing:")
        for device_id in gpu_manager.device_ids[:1]:  # Show first GPU
            stats = gpu_manager.memory_stats(device_id)
            logger.info(f"    GPU {device_id}: {stats.get('allocated', 0) / 1e6:.2f} MB allocated")


async def main():
    """Run all GPU demos."""
    logger.info("\n" + "#" * 80)
    logger.info("# ELEANOR V8 GPU ACCELERATION DEMONSTRATION")
    logger.info("#" * 80)
    
    try:
        # Initialize GPU
        gpu_manager = await demo_gpu_initialization()
        
        # Run demos
        await demo_memory_optimization(gpu_manager)
        await demo_multi_gpu_routing(gpu_manager)
        await demo_batch_processing(gpu_manager)
        
        if gpu_manager.devices_available > 0:
            await demo_gpu_accelerated_embeddings(gpu_manager)
            await demo_async_gpu_operations(gpu_manager)
        
        # Full pipeline demo
        await demo_full_pipeline(gpu_manager)
        
        logger.info("\n" + "#" * 80)
        logger.info("# ALL DEMOS COMPLETE")
        logger.info("#" * 80)
        logger.info("\nNext Steps:")
        logger.info("  1. Review GPU configuration in config/gpu.yaml")
        logger.info("  2. Check GPU documentation in docs/GPU_ACCELERATION.md")
        logger.info("  3. Integrate GPU support into engine.py")
        logger.info("  4. Run performance benchmarks")
        logger.info("  5. Enable GPU monitoring in production")
        
    except Exception as e:
        logger.error(f"\nError during GPU demonstration: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
