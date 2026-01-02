#!/usr/bin/env python3
"""
ELEANOR V8 - GPU Integration Examples

Demonstrates GPU acceleration integration with the engine.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    print("⚠️  PyTorch not installed. Install with: pip install torch")
    print("    GPU examples will not run.")
    TORCH_AVAILABLE = False
    sys.exit(1)

from engine.gpu import GPUManager, AsyncGPUExecutor


# ============================================================================
# Example 1: Basic GPU Detection and Configuration
# ============================================================================

def example_basic_setup():
    """
    Demonstrate basic GPU manager setup and device detection.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic GPU Setup")
    print("=" * 70)
    
    # Initialize GPU manager
    gpu_manager = GPUManager()
    
    print(f"\n📊 GPU Manager initialized:")
    print(f"   Device: {gpu_manager.device}")
    print(f"   GPUs available: {gpu_manager.devices_available}")
    print(f"   PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        for i in range(gpu_manager.devices_available):
            print(f"\n   GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Check health
    health = gpu_manager.health_check()
    print(f"\n✅ GPU Health: {health['healthy']}")
    print(f"   Mode: {health['mode']}")
    
    return gpu_manager


# ============================================================================
# Example 2: Memory Monitoring
# ============================================================================

def example_memory_monitoring(gpu_manager: GPUManager):
    """
    Demonstrate GPU memory monitoring and statistics.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Memory Monitoring")
    print("=" * 70)
    
    if not gpu_manager.is_available():
        print("\n⚠️  CUDA not available, skipping memory monitoring example")
        return
    
    # Get initial stats
    stats = gpu_manager.memory_stats(device_id=0)
    
    print(f"\n📊 Initial GPU Memory:")
    print(f"   Allocated: {stats['allocated_mb']:.2f} MB")
    print(f"   Reserved: {stats['reserved_mb']:.2f} MB")
    print(f"   Total: {stats['total_mb']:.2f} MB")
    print(f"   Utilization: {stats['utilization_pct']:.1f}%")
    
    # Allocate some tensors
    print("\n🔄 Allocating tensors...")
    device = gpu_manager.get_device()
    tensors = []
    
    for i in range(5):
        tensor = torch.randn(1000, 1000).to(device)
        tensors.append(tensor)
        
        stats = gpu_manager.memory_stats(device_id=0)
        print(
            f"   After tensor {i+1}: "
            f"{stats['allocated_mb']:.2f} MB "
            f"({stats['utilization_pct']:.1f}%)"
        )
    
    # Clear tensors
    print("\n🧹 Clearing tensors...")
    tensors.clear()
    torch.cuda.empty_cache()
    
    stats = gpu_manager.memory_stats(device_id=0)
    print(f"   After cleanup: {stats['allocated_mb']:.2f} MB")


# ============================================================================
# Example 3: Async GPU Operations
# ============================================================================

async def example_async_operations(gpu_manager: GPUManager):
    """
    Demonstrate async GPU operations with CUDA streams.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Async GPU Operations")
    print("=" * 70)
    
    device = gpu_manager.get_device()
    executor = AsyncGPUExecutor(device=device, num_streams=4)
    
    print(f"\n📊 Executor initialized:")
    print(f"   Device: {device}")
    print(f"   Streams: {len(executor.streams)}")
    
    # Define some GPU operations
    def matrix_multiply(size: int):
        a = torch.randn(size, size).to(device)
        b = torch.randn(size, size).to(device)
        return torch.matmul(a, b)
    
    def relu_operation(size: int):
        x = torch.randn(size, size).to(device)
        return torch.nn.functional.relu(x)
    
    def softmax_operation(size: int):
        x = torch.randn(size, size).to(device)
        return torch.nn.functional.softmax(x, dim=-1)
    
    # Single async operation
    print("\n🔄 Single async operation...")
    start = time.time()
    result = await executor.execute_async(matrix_multiply, 1000)
    elapsed = time.time() - start
    print(f"   Completed in {elapsed*1000:.2f}ms")
    print(f"   Result shape: {result.shape}")
    
    # Batch parallel execution
    print("\n🔄 Batch parallel execution (3 operations)...")
    operations = [
        (matrix_multiply, (1000,), {}),
        (relu_operation, (1000,), {}),
        (softmax_operation, (1000,), {}),
    ]
    
    start = time.time()
    results = await executor.batch_execute(operations)
    elapsed = time.time() - start
    
    print(f"   Completed {len(results)} operations in {elapsed*1000:.2f}ms")
    print(f"   Average: {elapsed*1000/len(results):.2f}ms per operation")
    
    for i, result in enumerate(results):
        print(f"   Result {i+1} shape: {result.shape}")


# ============================================================================
# Example 4: Performance Comparison (CPU vs GPU)
# ============================================================================

async def example_performance_benchmark(gpu_manager: GPUManager):
    """
    Benchmark CPU vs GPU performance.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Performance Benchmark (CPU vs GPU)")
    print("=" * 70)
    
    if not gpu_manager.is_available():
        print("\n⚠️  CUDA not available, skipping benchmark")
        return
    
    # Test parameters
    batch_size = 32
    seq_length = 512
    hidden_size = 768
    iterations = 100
    
    print(f"\n📊 Benchmark parameters:")
    print(f"   Batch size: {batch_size}")
    print(f"   Sequence length: {seq_length}")
    print(f"   Hidden size: {hidden_size}")
    print(f"   Iterations: {iterations}")
    
    # Create test data
    x = torch.randn(batch_size, seq_length, hidden_size)
    
    # CPU benchmark
    print("\n🔄 Running CPU benchmark...")
    x_cpu = x.cpu()
    
    start = time.time()
    for _ in range(iterations):
        _ = torch.nn.functional.relu(x_cpu)
    cpu_time = time.time() - start
    
    print(f"   CPU time: {cpu_time:.3f}s ({cpu_time/iterations*1000:.2f}ms per iter)")
    
    # GPU benchmark
    print("\n🔄 Running GPU benchmark...")
    device = gpu_manager.get_device()
    x_gpu = x.to(device)
    
    # Warm up
    for _ in range(10):
        _ = torch.nn.functional.relu(x_gpu)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(iterations):
        _ = torch.nn.functional.relu(x_gpu)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gpu_time = time.time() - start
    
    print(f"   GPU time: {gpu_time:.3f}s ({gpu_time/iterations*1000:.2f}ms per iter)")
    
    # Results
    speedup = cpu_time / gpu_time
    print(f"\n✨ Speedup: {speedup:.2f}x faster on GPU")
    
    # Memory stats
    if torch.cuda.is_available():
        stats = gpu_manager.memory_stats(device_id=0)
        print(f"\n📊 GPU Memory after benchmark:")
        print(f"   Allocated: {stats['allocated_mb']:.2f} MB")
        print(f"   Utilization: {stats['utilization_pct']:.1f}%")


# ============================================================================
# Example 5: Optimal Batch Size Detection
# ============================================================================

def example_find_optimal_batch_size(gpu_manager: GPUManager):
    """
    Find optimal batch size for GPU memory.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Optimal Batch Size Detection")
    print("=" * 70)
    
    if not gpu_manager.is_available():
        print("\n⚠️  CUDA not available, skipping batch size optimization")
        return
    
    device = gpu_manager.get_device()
    seq_length = 512
    hidden_size = 768
    
    print(f"\n🔍 Finding optimal batch size...")
    print(f"   Input shape: (batch, {seq_length}, {hidden_size})")
    
    batch_size = 1
    max_batch = 256
    optimal_batch = 1
    
    while batch_size <= max_batch:
        try:
            # Test batch
            x = torch.randn(batch_size, seq_length, hidden_size).to(device)
            _ = torch.nn.functional.relu(x)
            
            # Check memory
            stats = gpu_manager.memory_stats()
            utilization = stats['utilization_pct']
            
            print(
                f"   Batch {batch_size:3d}: "
                f"{stats['allocated_mb']:6.0f} MB "
                f"({utilization:5.1f}%) "
                f"{'✓' if utilization < 85 else '⚠️'}"
            )
            
            if utilization > 85:
                optimal_batch = batch_size // 2
                print(f"\n✅ Optimal batch size: {optimal_batch} (with safety margin)")
                break
            
            optimal_batch = batch_size
            batch_size *= 2
            
            # Cleanup
            del x
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                optimal_batch = batch_size // 2
                print(f"\n✅ Optimal batch size: {optimal_batch} (OOM at {batch_size})")
                torch.cuda.empty_cache()
                break
            raise
    
    if batch_size > max_batch:
        print(f"\n✅ Optimal batch size: {max_batch}+ (GPU can handle more)")
    
    return optimal_batch


# ============================================================================
# Main Example Runner
# ============================================================================

async def main():
    """
    Run all GPU integration examples.
    """
    print("\n" + "=" * 70)
    print("ELEANOR V8 - GPU Integration Examples")
    print("=" * 70)
    
    if not TORCH_AVAILABLE:
        print("\n❌ PyTorch not available. Cannot run examples.")
        return
    
    try:
        # Example 1: Basic setup
        gpu_manager = example_basic_setup()
        
        # Example 2: Memory monitoring
        example_memory_monitoring(gpu_manager)
        
        # Example 3: Async operations
        await example_async_operations(gpu_manager)
        
        # Example 4: Performance benchmark
        await example_performance_benchmark(gpu_manager)
        
        # Example 5: Optimal batch size
        example_find_optimal_batch_size(gpu_manager)
        
        print("\n" + "=" * 70)
        print("✅ All examples completed successfully!")
        print("=" * 70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
