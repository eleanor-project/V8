# GPU Acceleration Guide - ELEANOR V8

Comprehensive guide for enabling and optimizing GPU acceleration in ELEANOR V8.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Integration](#integration)
- [Performance Tuning](#performance-tuning)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

## Overview

**GPU acceleration provides:**
- **5-10x faster** LLM inference
- **3-5x faster** embedding generation
- **2-3x higher** throughput for critic evaluation
- Lower latency for real-time applications

**Supported platforms:**
- ✅ NVIDIA GPUs (CUDA 11.8+)
- ✅ Apple Silicon (M1/M2/M3 with MPS)
- ✅ CPU fallback (automatic)

## Requirements

### Minimum Hardware

**NVIDIA GPUs:**
- GTX 1660 (6GB VRAM) or better
- CUDA Compute Capability 7.0+
- Driver version 525+ (for CUDA 12)

**Apple Silicon:**
- M1/M2/M3 series
- 8GB+ unified memory
- macOS 13.0+ (Ventura)

### Recommended Hardware

**For Production:**
- NVIDIA RTX 4090 (24GB VRAM)
- NVIDIA A100 (40GB or 80GB)
- Apple M3 Max (128GB unified memory)

**For Development:**
- NVIDIA RTX 3060 (12GB VRAM)
- Apple M1 Pro/Max

## Installation

### Step 1: Check GPU Availability

```bash
# NVIDIA: Check CUDA version
nvidia-smi

# Should show CUDA version (e.g., CUDA Version: 12.1)
# Note the CUDA version for PyTorch installation
```

### Step 2: Install PyTorch with GPU Support

**Option A: NVIDIA CUDA 11.8** (Most Compatible)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Option B: NVIDIA CUDA 12.1** (Latest)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Option C: Apple Silicon** (MPS)
```bash
pip install torch torchvision
# MPS support is automatic
```

**Option D: CPU Only** (No GPU)
```bash
pip install torch torchvision
```

### Step 3: Install GPU Dependencies

```bash
# Install GPU-specific requirements
pip install -r requirements-gpu.txt

# Verify installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}')"
```

**Expected output (NVIDIA):**
```
CUDA available: True
Device: NVIDIA GeForce RTX 4090
```

**Expected output (Apple):**
```
CUDA available: False
MPS available: True
Device: Apple M1 Pro
```

### Step 4: Install Optional Performance Packages

**For production with high throughput requirements:**

```bash
# FlashAttention (2-4x faster attention, requires CUDA 11.6+)
pip install flash-attn --no-build-isolation

# 8-bit quantization (reduces VRAM by 2x)
pip install bitsandbytes

# Multi-GPU support
pip install accelerate

# Optional: vLLM for optimized inference (Linux only)
pip install vllm
```

## Configuration

### Basic Configuration

Edit `config/gpu.yaml`:

```yaml
gpu:
  enabled: true
  device_preference: ["cuda", "mps", "cpu"]
  
  memory:
    mixed_precision: true
    precision: "fp16"
  
  batching:
    enabled: true
    default_batch_size: 8
    dynamic_batching: true
  
  async:
    num_streams: 4
```

### Platform-Specific Configurations

#### Configuration: NVIDIA RTX 4090 (24GB)

```yaml
gpu:
  enabled: true
  device_preference: ["cuda"]
  
  memory:
    mixed_precision: true
    precision: "fp16"
    max_memory_per_gpu: "22GB"  # Leave 2GB for system
    
  batching:
    enabled: true
    default_batch_size: 16
    max_batch_size: 32
    dynamic_batching: true
  
  async:
    num_streams: 8  # RTX 4090 has excellent async support

ollama:
  gpu_layers: -1  # Load all layers on GPU
  num_gpu: 1
  gpu_memory_utilization: 0.9
```

#### Configuration: Apple M1 Pro (16GB)

```yaml
gpu:
  enabled: true
  device_preference: ["mps", "cpu"]
  
  memory:
    mixed_precision: true
    precision: "fp16"
    max_memory_per_gpu: "12GB"  # Conservative for unified memory
  
  batching:
    enabled: true
    default_batch_size: 4  # Lower for unified memory
    max_batch_size: 8
    dynamic_batching: true
  
  async:
    num_streams: 2  # MPS has limited stream support

ollama:
  gpu_layers: 32  # Partial GPU offload
  num_gpu: 1
  gpu_memory_utilization: 0.7  # Conservative for unified memory
```

#### Configuration: Multi-GPU (4x A100)

```yaml
gpu:
  enabled: true
  device_preference: ["cuda"]
  
  multi_gpu:
    enabled: true
    device_ids: [0, 1, 2, 3]
    strategy: "data_parallel"
  
  memory:
    mixed_precision: true
    precision: "fp16"
    max_memory_per_gpu: "75GB"  # A100-80GB
  
  batching:
    enabled: true
    default_batch_size: 32
    max_batch_size: 64
    dynamic_batching: true
  
  async:
    num_streams: 16  # High parallelization
```

## Integration

### Basic Usage

```python
from engine.gpu import GPUManager, AsyncGPUExecutor
from engine.engine import EleanorEngineV8

# Initialize GPU manager
gpu_manager = GPUManager()
print(f"Using device: {gpu_manager.device}")

# Check GPU health
health = gpu_manager.health_check()
print(f"GPU healthy: {health['healthy']}")

# Initialize engine with GPU support
engine = EleanorEngineV8(
    config_path="config/gpu.yaml",
    gpu_manager=gpu_manager,
)

# Run inference (automatically uses GPU)
result = await engine.run(
    text="Evaluate this medical decision",
    context={"domain": "healthcare"}
)
```

### Async GPU Operations

```python
import torch
from engine.gpu import AsyncGPUExecutor

# Initialize executor
device = torch.device("cuda")
executor = AsyncGPUExecutor(device=device, num_streams=4)

# Single async operation
async def model_forward(inputs):
    return model(inputs)

result = await executor.execute_async(model_forward, inputs)

# Batch parallel execution
operations = [
    (critic1.evaluate, (text,), {}),
    (critic2.evaluate, (text,), {}),
    (critic3.evaluate, (text,), {}),
]

results = await executor.batch_execute(operations)
```

### GPU-Accelerated Embeddings

```python
from engine.gpu import GPUManager
import torch

gpu_manager = GPUManager()
device = gpu_manager.get_device()

# Load embedding model on GPU
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
model = model.to(device)

# Generate embeddings (batch processing)
texts = ["text1", "text2", "text3", ...]

# Automatic GPU batching
embeddings = model.encode(
    texts,
    batch_size=32,
    convert_to_tensor=True,
    device=device
)

print(f"Embeddings shape: {embeddings.shape}")
print(f"Device: {embeddings.device}")
```

### Memory Management

```python
# Check memory stats
stats = gpu_manager.memory_stats(device_id=0)
print(f"Allocated: {stats['allocated_mb']:.2f}MB")
print(f"Utilization: {stats['utilization_pct']:.1f}%")

# Clear GPU cache if needed
if stats['utilization_pct'] > 90:
    torch.cuda.empty_cache()
    print("GPU cache cleared")
```

## Performance Tuning

### Batch Size Optimization

```python
# Find optimal batch size for your GPU
from engine.gpu import GPUManager

def find_optimal_batch_size(model, input_size, gpu_manager):
    batch_size = 1
    max_batch = 128
    
    while batch_size <= max_batch:
        try:
            # Test batch
            dummy_input = torch.randn(batch_size, *input_size).to(gpu_manager.device)
            _ = model(dummy_input)
            
            # Check memory
            stats = gpu_manager.memory_stats()
            if stats['utilization_pct'] > 85:
                return batch_size // 2  # Safety margin
            
            batch_size *= 2
        except RuntimeError as e:
            if "out of memory" in str(e):
                return batch_size // 2
            raise
    
    return max_batch

optimal_batch = find_optimal_batch_size(model, (512,), gpu_manager)
print(f"Optimal batch size: {optimal_batch}")
```

### Mixed Precision Training

```python
import torch
from torch.cuda.amp import autocast

# Use automatic mixed precision
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)
```

### CUDA Stream Optimization

```python
from engine.gpu import AsyncGPUExecutor

# Use multiple streams for parallel operations
executor = AsyncGPUExecutor(device=device, num_streams=8)

# Execute critics in parallel on different streams
critic_ops = [
    (rights_critic.evaluate, (text,), {}),
    (risk_critic.evaluate, (text,), {}),
    (fairness_critic.evaluate, (text,), {}),
    (truth_critic.evaluate, (text,), {}),
]

results = await executor.batch_execute(critic_ops)
```

## Monitoring

### Real-time GPU Monitoring

```python
import asyncio
from engine.gpu import GPUManager

async def monitor_gpu(gpu_manager, interval=5):
    """Monitor GPU usage every interval seconds."""
    while True:
        health = gpu_manager.health_check()
        
        if not health['healthy']:
            print("⚠️ GPU health warning!")
        
        for device in health['devices']:
            stats = device['memory_stats']
            print(
                f"GPU {device['device_id']}: "
                f"{stats['allocated_mb']:.0f}MB / {stats['total_mb']:.0f}MB "
                f"({stats['utilization_pct']:.1f}%)"
            )
        
        await asyncio.sleep(interval)

# Run monitoring
gpu_manager = GPUManager()
await monitor_gpu(gpu_manager)
```

### Prometheus Metrics

```python
from prometheus_client import Gauge

# Define metrics
gpu_memory_used = Gauge('gpu_memory_used_mb', 'GPU memory used', ['device_id'])
gpu_memory_total = Gauge('gpu_memory_total_mb', 'GPU memory total', ['device_id'])
gpu_utilization = Gauge('gpu_utilization_pct', 'GPU utilization', ['device_id'])

# Update metrics
def update_gpu_metrics(gpu_manager):
    for device_id in range(gpu_manager.devices_available):
        stats = gpu_manager.memory_stats(device_id)
        
        gpu_memory_used.labels(device_id=device_id).set(stats['allocated_mb'])
        gpu_memory_total.labels(device_id=device_id).set(stats['total_mb'])
        gpu_utilization.labels(device_id=device_id).set(stats['utilization_pct'])
```

## Troubleshooting

### Common Issues

#### Issue: "CUDA out of memory"

**Solutions:**
1. Reduce batch size in `config/gpu.yaml`
2. Enable mixed precision: `precision: "fp16"`
3. Clear CUDA cache: `torch.cuda.empty_cache()`
4. Reduce model size with quantization

```yaml
gpu:
  batching:
    default_batch_size: 4  # Reduce from 8
  memory:
    mixed_precision: true
    quantization:
      enabled: true
      bits: 8
```

#### Issue: "No CUDA GPUs are available"

**Diagnostics:**
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA installation
nvcc --version

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

**Solutions:**
1. Install/update NVIDIA drivers
2. Reinstall PyTorch with correct CUDA version
3. Set `CUDA_VISIBLE_DEVICES` environment variable

#### Issue: Slow Performance on Apple Silicon

**Solutions:**
1. Ensure PyTorch is using MPS backend
2. Reduce batch size for unified memory
3. Use `precision: "fp16"` for M1/M2/M3

```python
# Verify MPS is being used
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
```

#### Issue: Multi-GPU Not Working

**Diagnostics:**
```python
import torch
print(f"GPU count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
```

**Solutions:**
1. Enable multi-GPU in config: `multi_gpu.enabled: true`
2. Specify device IDs: `multi_gpu.device_ids: [0, 1, 2, 3]`
3. Check `CUDA_VISIBLE_DEVICES` environment variable

### Performance Benchmarking

```python
import time
import torch
from engine.gpu import GPUManager, AsyncGPUExecutor

async def benchmark_gpu_performance():
    """Benchmark GPU vs CPU performance."""
    gpu_manager = GPUManager()
    
    # Test data
    batch_size = 16
    seq_length = 512
    hidden_size = 768
    
    x = torch.randn(batch_size, seq_length, hidden_size)
    
    # CPU benchmark
    x_cpu = x.cpu()
    start = time.time()
    for _ in range(100):
        _ = torch.nn.functional.relu(x_cpu)
    cpu_time = time.time() - start
    
    # GPU benchmark
    if gpu_manager.is_available():
        x_gpu = x.to(gpu_manager.device)
        torch.cuda.synchronize()  # Warm up
        
        start = time.time()
        for _ in range(100):
            _ = torch.nn.functional.relu(x_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        
        speedup = cpu_time / gpu_time
        print(f"CPU time: {cpu_time:.3f}s")
        print(f"GPU time: {gpu_time:.3f}s")
        print(f"Speedup: {speedup:.2f}x")
    else:
        print("GPU not available for benchmarking")

await benchmark_gpu_performance()
```

### Debugging GPU Issues

```python
import torch
from engine.gpu import GPUManager

def diagnose_gpu_setup():
    """Comprehensive GPU diagnostics."""
    print("=" * 50)
    print("GPU DIAGNOSTICS")
    print("=" * 50)
    
    # PyTorch info
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"GPU count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}:")
            print(f"  Name: {props.name}")
            print(f"  Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
    
    # MPS info (Apple)
    if hasattr(torch.backends, 'mps'):
        print(f"\nMPS available: {torch.backends.mps.is_available()}")
        print(f"MPS built: {torch.backends.mps.is_built()}")
    
    # GPUManager info
    print("\n" + "=" * 50)
    print("ELEANOR GPU MANAGER")
    print("=" * 50)
    
    gpu_manager = GPUManager()
    print(f"\n{gpu_manager}")
    
    health = gpu_manager.health_check()
    print(f"\nOverall health: {health['healthy']}")
    print(f"Mode: {health['mode']}")
    
    for device in health.get('devices', []):
        print(f"\nDevice {device['device_id']}:")
        print(f"  Healthy: {device['healthy']}")
        if 'memory_stats' in device:
            stats = device['memory_stats']
            print(f"  Memory: {stats['allocated_mb']:.0f}MB / {stats['total_mb']:.0f}MB")
            print(f"  Utilization: {stats['utilization_pct']:.1f}%")

diagnose_gpu_setup()
```

## Best Practices

### Production Deployment

1. **Always enable monitoring**:
   ```yaml
   gpu:
     monitoring:
       log_memory_stats: true
       memory_check_interval: 30
   ```

2. **Use mixed precision for memory efficiency**:
   ```yaml
   gpu:
     memory:
       mixed_precision: true
       precision: "fp16"
   ```

3. **Set conservative memory limits**:
   ```yaml
   gpu:
     memory:
       max_memory_per_gpu: "22GB"  # Leave headroom
   ```

4. **Enable dynamic batching for variable loads**:
   ```yaml
   gpu:
     batching:
       dynamic_batching: true
   ```

5. **Monitor GPU health regularly**:
   ```python
   # In your application
   async def periodic_health_check():
       while True:
           health = gpu_manager.health_check()
           if not health['healthy']:
               logger.warning("GPU health degraded", extra=health)
           await asyncio.sleep(60)
   ```

### Development Best Practices

1. **Test with CPU fallback**: Always ensure code works without GPU
2. **Clear CUDA cache**: Between experiments to avoid OOM
3. **Use profiling sparingly**: Only enable when debugging performance
4. **Start with small batches**: Increase gradually to find limits
5. **Monitor memory**: Watch for memory leaks during development

## Performance Expectations

### Typical Speedups

| Operation | CPU (baseline) | GPU (CUDA) | Speedup |
|-----------|----------------|------------|----------|
| LLM Inference (7B) | 1.0x | 8-12x | 8-12x |
| Embeddings (batch=32) | 1.0x | 5-8x | 5-8x |
| Critic Evaluation | 1.0x | 3-5x | 3-5x |
| Similarity Search | 1.0x | 10-15x | 10-15x |

### Expected Throughput

**RTX 4090 (24GB):**
- LLM inference: 150-200 tokens/sec
- Critic evaluation: 50-80 evaluations/sec
- End-to-end pipeline: 20-30 requests/sec

**Apple M1 Pro (16GB):**
- LLM inference: 80-120 tokens/sec
- Critic evaluation: 30-50 evaluations/sec
- End-to-end pipeline: 10-15 requests/sec

## Additional Resources

- [PyTorch CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html)
- [NVIDIA CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [Apple Metal Performance Shaders](https://developer.apple.com/metal/pytorch/)
- [Hugging Face GPU Optimization](https://huggingface.co/docs/transformers/perf_train_gpu_one)

## Support

For GPU-specific issues:
1. Check [Troubleshooting](#troubleshooting) section
2. Run diagnostics: `python -m engine.gpu.diagnostics`
3. Review logs for GPU warnings
4. Open issue with diagnostic output
