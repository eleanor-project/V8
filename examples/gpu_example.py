#!/usr/bin/env python3
"""
GPU Acceleration Example for ELEANOR V8

Demonstrates:
- GPU detection and initialization
- Async GPU operations
- Batch processing
- Memory monitoring
- Health checks
"""

import asyncio
import time
from engine.gpu import GPUManager, AsyncGPUExecutor


def gpu_intensive_operation(x: float) -> float:
    """
    Simulated GPU-intensive operation.
    
    In real usage, this would be:
    - Model inference
    - Embedding computation
    - Similarity search
    """
    # Simulate some work
    result = x ** 2
    return result


async def example_1_basic_usage():
    """Example 1: Basic GPU detection and usage"""
    print("\n" + "=" * 50)
    print("Example 1: Basic GPU Usage")
    print("=" * 50)
    
    # Initialize GPU manager
    gpu = GPUManager()
    
    print(f"\nGPU Available: {gpu.is_gpu_available()}")
    print(f"Device: {gpu.device}")
    print(f"Device Type: {gpu.device.type if gpu.device else 'none'}")
    print(f"Device Count: {gpu.devices_available}")
    
    if gpu.is_gpu_available():
        # Get memory stats
        stats = gpu.memory_stats(0)
        print(f"\nGPU Memory:")
        print(f"  Allocated: {stats['allocated_mb']:.1f} MB")
        print(f"  Total: {stats['total_mb']:.1f} MB")
        print(f"  Utilization: {stats['utilization_pct']:.1f}%")


async def example_2_async_operations():
    """Example 2: Async GPU operations"""
    print("\n" + "=" * 50)
    print("Example 2: Async GPU Operations")
    print("=" * 50)
    
    # Initialize
    gpu = GPUManager()
    executor = AsyncGPUExecutor(gpu.device, num_streams=4)
    
    print(f"\nExecutor: {executor}")
    
    # Single async operation
    print("\nExecuting single GPU operation...")
    start = time.time()
    result = await executor.execute_async(gpu_intensive_operation, 42)
    elapsed = time.time() - start
    
    print(f"Result: {result}")
    print(f"Time: {elapsed*1000:.2f}ms")


async def example_3_batch_operations():
    """Example 3: Batch GPU operations"""
    print("\n" + "=" * 50)
    print("Example 3: Batch GPU Operations")
    print("=" * 50)
    
    # Initialize
    gpu = GPUManager()
    executor = AsyncGPUExecutor(gpu.device, num_streams=4)
    
    # Prepare batch operations
    inputs = [1, 2, 3, 4, 5, 6, 7, 8]
    operations = [
        (gpu_intensive_operation, (x,), {})
        for x in inputs
    ]
    
    print(f"\nProcessing batch of {len(inputs)} operations...")
    start = time.time()
    results = await executor.batch_execute(operations)
    elapsed = time.time() - start
    
    print(f"Results: {results}")
    print(f"Time: {elapsed*1000:.2f}ms")
    print(f"Throughput: {len(inputs)/elapsed:.1f} ops/sec")


async def example_4_memory_monitoring():
    """Example 4: GPU memory monitoring"""
    print("\n" + "=" * 50)
    print("Example 4: Memory Monitoring")
    print("=" * 50)
    
    gpu = GPUManager()
    
    if not gpu.is_gpu_available():
        print("\nGPU not available for memory monitoring")
        return
    
    # Reset peak stats
    print("\nResetting peak memory stats...")
    gpu.reset_peak_stats()
    
    # Before operation
    stats_before = gpu.memory_stats(0)
    print(f"\nMemory Before:")
    print(f"  Allocated: {stats_before['allocated_mb']:.1f} MB")
    print(f"  Peak: {stats_before['max_allocated_mb']:.1f} MB")
    
    # Simulate GPU operations
    executor = AsyncGPUExecutor(gpu.device, num_streams=4)
    operations = [(gpu_intensive_operation, (i,), {}) for i in range(100)]
    await executor.batch_execute(operations)
    
    # After operation
    stats_after = gpu.memory_stats(0)
    print(f"\nMemory After:")
    print(f"  Allocated: {stats_after['allocated_mb']:.1f} MB")
    print(f"  Peak: {stats_after['max_allocated_mb']:.1f} MB")
    print(f"  Delta: {stats_after['max_allocated_mb'] - stats_before['max_allocated_mb']:.1f} MB")


async def example_5_health_check():
    """Example 5: GPU health monitoring"""
    print("\n" + "=" * 50)
    print("Example 5: Health Check")
    print("=" * 50)
    
    gpu = GPUManager()
    
    # Perform health check
    health = gpu.health_check()
    
    print(f"\nOverall Health: {'✅ Healthy' if health['healthy'] else '❌ Unhealthy'}")
    print(f"Mode: {health['mode']}")
    
    if 'devices' in health:
        print(f"\nDevice Details:")
        for device in health['devices']:
            device_id = device['device_id']
            status = '✅' if device['healthy'] else '❌'
            print(f"\n  GPU {device_id}: {status}")
            
            if 'utilization_pct' in device:
                print(f"    Utilization: {device['utilization_pct']:.1f}%")
                print(f"    Memory: {device['allocated_mb']:.0f} / {device['total_mb']:.0f} MB")
            
            if 'warning' in device:
                print(f"    ⚠️  Warning: {device['warning']}")
            
            if 'error' in device:
                print(f"    ❌ Error: {device['error']}")


async def example_6_multi_gpu():
    """Example 6: Multi-GPU detection"""
    print("\n" + "=" * 50)
    print("Example 6: Multi-GPU Detection")
    print("=" * 50)
    
    gpu = GPUManager()
    
    print(f"\nTotal GPUs: {gpu.devices_available}")
    
    if gpu.devices_available > 1:
        print(f"\nMulti-GPU Available!")
        
        # Show stats for each GPU
        for device_id in range(gpu.devices_available):
            stats = gpu.memory_stats(device_id)
            print(f"\nGPU {device_id}:")
            print(f"  Total Memory: {stats['total_mb']:.0f} MB")
            print(f"  Available: {(stats['total_mb'] - stats['allocated_mb']):.0f} MB")
            print(f"  Utilization: {stats['utilization_pct']:.1f}%")
    elif gpu.devices_available == 1:
        print("\nSingle GPU detected")
    else:
        print("\nNo GPUs detected (CPU mode)")


async def main():
    """Run all examples"""
    print("\n" + "#" * 50)
    print("# ELEANOR V8 - GPU Acceleration Examples")
    print("#" * 50)
    
    # Run all examples
    await example_1_basic_usage()
    await example_2_async_operations()
    await example_3_batch_operations()
    await example_4_memory_monitoring()
    await example_5_health_check()
    await example_6_multi_gpu()
    
    print("\n" + "#" * 50)
    print("# Examples Complete!")
    print("#" * 50)
    print("\nNext Steps:")
    print("- Read docs/GPU_QUICK_START.md for usage guide")
    print("- Read docs/GPU_ARCHITECTURE.md for technical details")
    print("- Check Issue #25 for roadmap and updates")
    print()


if __name__ == "__main__":
    asyncio.run(main())
