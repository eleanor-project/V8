# GPU Acceleration for ELEANOR V8

This guide explains how to enable and configure GPU acceleration in ELEANOR V8 for dramatically improved performance.

## Overview

ELEANOR V8 supports GPU acceleration for:
- **Model Inference**: Run LLMs on GPU for 2-10x speedup
- **Embeddings**: GPU-accelerated similarity search (10-50x faster)
- **Batch Processing**: Parallel critic evaluation
- **Multi-GPU**: Scale across multiple GPUs

## Quick Start

### 1. Install GPU Dependencies

**NVIDIA GPUs (CUDA):**
```bash
pip install -r requirements-gpu.txt
```

**Apple Silicon (M1/M2/M3):**
```bash
# PyTorch with MPS support is included in base requirements
pip install torch torchvision torchaudio
```

**Verify Installation:**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
```

### 2. Enable GPU in Configuration

Edit `config/gpu.yaml`:

```yaml
gpu:
  enabled: true
  device_preference: ["cuda", "mps", "cpu"]
  
  memory:
    mixed_precision: true
    dtype: "auto"  # BF16 on Ampere+, FP16 otherwise
```

### 3. Initialize Engine with GPU

```python
from engine.engine import EleanorEngineV8, EngineConfig
from engine.gpu.manager import GPUManager

# Initialize GPU manager
gpu_manager = GPUManager(
    device_ids=[0, 1],  # Use GPUs 0 and 1
    strategy="data_parallel"
)

# Create engine with GPU support
engine = EleanorEngineV8(
    config=EngineConfig(),
    gpu_manager=gpu_manager
)

# Run inference (automatically uses GPU)
result = await engine.run(
    text="Your input text",
    context={"domain": "healthcare"}
)
```

## Features

### Multi-GPU Support

**DataParallel (Recommended):**
```yaml
multi_gpu:
  device_ids: [0, 1, 2, 3]
  strategy: "data_parallel"
```

**Manual GPU Assignment:**
```yaml
model_assignments:
  ethics_critic: 0
  fairness_critic: 1
  safety_critic: 2
  precedent_retrieval: 3
```

### Memory Optimization

**Mixed Precision (2x faster, 2x less memory):**
```yaml
memory:
  mixed_precision: true
  dtype: "bfloat16"  # or "float16"
```

**Quantization (4x less memory):**
```yaml
memory:
  quantization:
    enabled: true
    bits: 8  # or 4 with bitsandbytes
```

### Batch Processing

**Dynamic Batching:**
```yaml
batch:
  batch_size: 8
  max_batch_size: 32
  dynamic_batching: true
```

**Usage:**
```python
from engine.gpu.batch_processor import GPUBatchProcessor

batch_processor = GPUBatchProcessor(gpu_manager)

# Process multiple inputs efficiently
inputs = ["text1", "text2", "text3", ...]
results = await batch_processor.process_batch(inputs)
```

### GPU-Accelerated Embeddings

```python
from engine.gpu.embeddings import GPUEmbeddingCache

embedding_cache = GPUEmbeddingCache(
    device=gpu_manager.device,
    max_cached_embeddings=100000
)

# Compute embeddings on GPU
embedding = await embedding_cache.compute_embedding("query text")

# Fast GPU-based similarity search
similarities = await embedding_cache.batch_similarity(
    query_embedding,
    candidate_embeddings
)
```

## Performance Benchmarks

| Operation | CPU | Single GPU | 4x GPU |
|-----------|-----|------------|--------|
| Model Inference (7B) | 5.2s | 0.8s | 0.3s |
| Embeddings (1000) | 2.1s | 0.04s | 0.01s |
| Critic Evaluation | 3.5s | 1.2s | 0.4s |
| Precedent Retrieval | 4.8s | 0.2s | 0.05s |

**Overall Throughput:**
- Single GPU: **5-10x** faster than CPU
- Multi-GPU: **15-30x** faster than CPU

## Advanced Configuration

### Async GPU Operations

```python
from engine.gpu.async_ops import AsyncGPUExecutor

executor = AsyncGPUExecutor(
    device=gpu_manager.device,
    num_streams=4  # Parallel CUDA streams
)

# Run multiple GPU operations in parallel
results = await asyncio.gather(
    executor.async_gpu_operation(op1),
    executor.async_gpu_operation(op2),
    executor.async_gpu_operation(op3)
)
```

### Memory Management

```python
# Check GPU memory
memory_stats = gpu_manager.memory_stats(device_id=0)
print(f"Allocated: {memory_stats['allocated_mb']} MB")
print(f"Reserved: {memory_stats['reserved_mb']} MB")

# Clear GPU cache
gpu_manager.clear_cache(device_id=0)

# Enable memory monitoring
gpu_manager.enable_monitoring(interval_seconds=60)
```

### Model Parallelization

```python
from engine.gpu.parallelization import ModelParallelWrapper

# Wrap model for multi-GPU
parallel_model = ModelParallelWrapper(
    model=critic_model,
    device_ids=[0, 1, 2, 3],
    strategy="data_parallel"
)

# Automatic parallelization
output = parallel_model(input_data)
```

## Ollama GPU Configuration

If using Ollama backend:

```yaml
ollama:
  gpu_layers: -1  # All layers on GPU
  num_gpu: 1      # Number of GPUs
  use_mmap: true  # Memory mapping
```

**Set environment variables:**
```bash
export OLLAMA_GPU_LAYERS=-1
export OLLAMA_NUM_GPU=1
```

## Troubleshooting

### CUDA Out of Memory (OOM)

**Solutions:**
1. Reduce batch size:
   ```yaml
   batch:
     batch_size: 4  # Reduce from 8
   ```

2. Enable quantization:
   ```yaml
   memory:
     quantization:
       enabled: true
       bits: 8
   ```

3. Enable CPU offloading:
   ```yaml
   memory:
     offload_to_cpu: true
   ```

### Slow GPU Performance

**Check:**
1. Mixed precision enabled?
2. TF32 enabled (Ampere+ GPUs)?
3. cuDNN benchmark mode enabled?

```yaml
cuda:
  benchmark: true
  allow_tf32: true
```

### Multi-GPU Not Working

**Verify:**
```python
import torch
print(torch.cuda.device_count())  # Should show > 1
print(torch.cuda.is_available())  # Should be True
```

**Check CUDA_VISIBLE_DEVICES:**
```bash
echo $CUDA_VISIBLE_DEVICES
# Should be "0,1,2,3" or empty
```

## Best Practices

1. **Use Mixed Precision**: 2x speedup with minimal accuracy impact
2. **Batch When Possible**: GPU efficiency scales with batch size
3. **Cache Embeddings**: Keep frequently used embeddings on GPU
4. **Monitor Memory**: Use `gpu_manager.memory_stats()` to track usage
5. **Profile First**: Enable profiling to identify bottlenecks

## Hardware Recommendations

| Use Case | Minimum | Recommended | Optimal |
|----------|---------|-------------|----------|
| Development | GTX 1660 (6GB) | RTX 3060 (12GB) | RTX 4090 (24GB) |
| Production | RTX 3090 (24GB) | A100 (40GB) | 4x A100 (40GB) |
| Embeddings Only | GTX 1650 (4GB) | RTX 3050 (8GB) | RTX 3060 (12GB) |
| Large Models (70B+) | A100 (80GB) | 2x A100 (80GB) | 4x H100 (80GB) |

## Cloud GPU Providers

- **AWS**: p4d.24xlarge (8x A100)
- **GCP**: a2-ultragpu-8g (8x A100)
- **Azure**: ND96asr_v4 (8x A100)
- **Lambda Labs**: 8x A100 instances
- **RunPod**: On-demand GPU rentals

## See Also

- [CACHING.md](CACHING.md) - Caching strategies
- [OBSERVABILITY.md](OBSERVABILITY.md) - Monitoring GPU performance
- [RESILIENCE.md](RESILIENCE.md) - Handling GPU failures

## Support

For GPU-related issues:
1. Check hardware compatibility
2. Verify CUDA/driver versions
3. Review logs in `logs/gpu_profiles/`
4. Open an issue with GPU specs and error logs
