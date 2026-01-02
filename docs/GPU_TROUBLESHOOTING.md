# GPU Acceleration - Troubleshooting Guide

## Quick Diagnostics

```python
from engine.gpu import GPUManager

gpu = GPUManager()

print(f"GPU Available: {gpu.is_gpu_available()}")
print(f"Device: {gpu.device}")
print(f"Device Count: {gpu.devices_available}")

if gpu.is_gpu_available():
    health = gpu.health_check()
    print(f"Health: {health}")
```

---

## Common Issues

### 1. "PyTorch not installed"

**Symptoms:**
```
WARNING: PyTorch not installed. GPU acceleration unavailable.
```

**Solution:**
```bash
# NVIDIA GPU (Linux/Windows)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Apple Silicon (Mac)
pip install torch torchvision

# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Verify:**
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

---

### 2. "CUDA not available"

**Symptoms:**
```
INFO: using_cpu_fallback - No GPU detected
```

**Diagnosis:**
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA installation
nvcc --version

# Check PyTorch CUDA
python -c "import torch; print(torch.version.cuda)"
```

**Solutions:**

**A. Update NVIDIA Drivers**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install nvidia-driver-525

# Check
nvidia-smi
```

**B. Reinstall PyTorch with CUDA**
```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**C. CUDA Version Mismatch**
```python
import torch
print(f"PyTorch CUDA: {torch.version.cuda}")
print(f"Driver supports: {torch.cuda.get_device_capability()}")
```

If mismatch, install matching PyTorch version:
```bash
# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

### 3. "CUDA out of memory" (OOM)

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Diagnosis:**
```python
from engine.gpu import GPUManager

gpu = GPUManager()
stats = gpu.memory_stats(0)

print(f"Allocated: {stats['allocated_mb']:.0f} MB")
print(f"Total: {stats['total_mb']:.0f} MB")
print(f"Utilization: {stats['utilization_pct']:.1f}%")
```

**Solutions:**

**A. Reduce Batch Size**
```yaml
# config/gpu.yaml
gpu:
  batching:
    default_batch_size: 4  # Was 8
    max_batch_size: 16     # Was 32
```

**B. Enable Mixed Precision**
```yaml
gpu:
  memory:
    mixed_precision: true
    precision: "fp16"  # 50% memory reduction
```

**C. Enable Quantization**
```yaml
gpu:
  memory:
    quantization:
      enabled: true
      bits: 8  # 75% memory reduction
```

**D. Offload to CPU**
```yaml
gpu:
  memory:
    offload_to_cpu: true  # Move inactive tensors to CPU
```

**E. Clear Cache**
```python
import torch
torch.cuda.empty_cache()  # Free unused memory
```

**F. Use Smaller Model**
```yaml
# Use 7B instead of 13B
ollama:
  model: "llama2:7b"  # Instead of "llama2:13b"
```

---

### 4. "Slow Performance on GPU"

**Symptoms:**
GPU slower than expected or similar to CPU

**Diagnosis:**
```python
import time
import torch

# Check GPU actually being used
device = torch.device("cuda")
tensor = torch.randn(1000, 1000, device=device)

start = time.time()
result = tensor @ tensor.T
torch.cuda.synchronize()
print(f"GPU Time: {time.time() - start:.4f}s")

# Compare to CPU
tensor_cpu = tensor.cpu()
start = time.time()
result_cpu = tensor_cpu @ tensor_cpu.T
print(f"CPU Time: {time.time() - start:.4f}s")
```

**Solutions:**

**A. Enable Mixed Precision**
```yaml
gpu:
  memory:
    mixed_precision: true
    precision: "fp16"  # 2x speedup
```

**B. Increase Batch Size**
```yaml
gpu:
  batching:
    default_batch_size: 16  # Larger batches = better GPU utilization
```

**C. Use More Streams**
```yaml
gpu:
  async:
    num_streams: 8  # Was 4
```

**D. Check GPU Utilization**
```bash
# Should see >80% utilization
watch -n 0.5 nvidia-smi
```

If low utilization:
- Increase batch size
- Check for CPU bottlenecks
- Profile with `torch.profiler`

---

### 5. "Apple MPS not available"

**Symptoms:**
```
False: torch.backends.mps.is_available()
```

**Requirements:**
- macOS 12.3+
- M1, M2, or M3 chip

**Solutions:**

**A. Update macOS**
```bash
# Check version
sw_vers

# Update to 12.3+
# System Settings → General → Software Update
```

**B. Reinstall PyTorch**
```bash
pip uninstall torch torchvision
pip install torch torchvision
```

**C. Verify MPS**
```python
import torch

print(f"MPS built: {torch.backends.mps.is_built()}")
print(f"MPS available: {torch.backends.mps.is_available()}")

if torch.backends.mps.is_available():
    device = torch.device("mps")
    x = torch.ones(1, device=device)
    print(f"MPS works: {x.device}")
```

---

### 6. "Multiple GPUs not detected"

**Symptoms:**
```python
gpu.devices_available == 1  # But you have 2+ GPUs
```

**Diagnosis:**
```bash
# Check all GPUs visible
nvidia-smi -L

# Check CUDA sees them
python -c "import torch; print(torch.cuda.device_count())"
```

**Solutions:**

**A. Set CUDA_VISIBLE_DEVICES**
```bash
# Make all GPUs visible
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Or in Python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
```

**B. Check GPU Modes**
```bash
# Some GPUs might be in compute mode
nvidia-smi -q -d COMPUTE
```

**C. Driver Issues**
```bash
# Update drivers
sudo apt update && sudo apt upgrade nvidia-driver-525
```

---

### 7. "GPU Health Check Fails"

**Symptoms:**
```python
health = gpu.health_check()
assert health['healthy'] == False
```

**Diagnosis:**
```python
from engine.gpu import GPUManager

gpu = GPUManager()
health = gpu.health_check()

print(f"Healthy: {health['healthy']}")
for device in health.get('devices', []):
    print(f"GPU {device['device_id']}: {device}")
```

**Common Issues:**

**A. High Memory Utilization (>95%)**
```python
# Clear memory
import torch
torch.cuda.empty_cache()
gpu.reset_peak_stats()
```

**B. GPU Hanging**
```bash
# Check processes
nvidia-smi

# Kill stuck processes
sudo kill -9 <PID>
```

**C. Temperature Issues**
```bash
# Check temperature
nvidia-smi --query-gpu=temperature.gpu --format=csv

# Should be <85°C
```

---

### 8. "Async Operations Not Working"

**Symptoms:**
Async GPU operations running sequentially

**Diagnosis:**
```python
import asyncio
import time
from engine.gpu import GPUManager, AsyncGPUExecutor

async def test():
    gpu = GPUManager()
    executor = AsyncGPUExecutor(gpu.device, num_streams=4)
    
    def slow_op():
        time.sleep(1)
        return 42
    
    start = time.time()
    
    # Should run in parallel (~1s total)
    results = await asyncio.gather(
        executor.execute_async(slow_op),
        executor.execute_async(slow_op),
        executor.execute_async(slow_op),
    )
    
    elapsed = time.time() - start
    print(f"Elapsed: {elapsed:.2f}s (should be ~1s, not 3s)")

asyncio.run(test())
```

**Solutions:**

**A. Check Event Loop**
```python
# Ensure running in async context
import asyncio

async def main():
    # Your async GPU code here
    pass

# Run
asyncio.run(main())
```

**B. Increase Streams**
```python
executor = AsyncGPUExecutor(device, num_streams=8)  # More parallelism
```

---

## Performance Profiling

### Basic Profiling

```python
import time
from engine.gpu import GPUManager

gpu = GPUManager()

# Reset stats
gpu.reset_peak_stats()

# Your GPU operation
start = time.time()
# ... your code ...
elapsed = time.time() - start

# Check stats
stats = gpu.memory_stats()
print(f"Time: {elapsed:.3f}s")
print(f"Peak Memory: {stats['max_allocated_mb']:.0f} MB")
```

### Detailed Profiling

```python
import torch.profiler

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
) as prof:
    # Your GPU operations
    pass

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

---
## Getting Help

### Collect Diagnostics

```bash
# Create diagnostic report
python -c "
import torch
from engine.gpu import GPUManager

print('=== PyTorch ===')
print(f'Version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Device count: {torch.cuda.device_count()}')

print('\n=== GPU Manager ===')
gpu = GPUManager()
print(f'Device: {gpu.device}')
print(f'GPU available: {gpu.is_gpu_available()}')

if gpu.is_gpu_available():
    print('\n=== Health Check ===')
    print(gpu.health_check())
    
    print('\n=== Memory Stats ===')
    print(gpu.memory_stats(0))
" > gpu_diagnostics.txt
```

### Report Issues

When reporting GPU issues, include:
1. Output of `gpu_diagnostics.txt`
2. Output of `nvidia-smi` (NVIDIA) or `system_profiler SPDisplaysDataType` (Mac)
3. Your `config/gpu.yaml`
4. Steps to reproduce

---

**Need more help?** Check:
- [GPU Architecture](GPU_ARCHITECTURE.md)
- [GPU Quick Start](GPU_QUICK_START.md)
- [Issue #25](https://github.com/eleanor-project/V8/issues/25) for updates
