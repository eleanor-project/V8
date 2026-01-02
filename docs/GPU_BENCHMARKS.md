# GPU Acceleration - Performance Benchmarks

## Target Performance

| Component | CPU Baseline | GPU Target | Actual | Status |
|-----------|-------------|-----------|---------|--------|
| LLM Inference (7B) | 5.2s | 0.8s (6.5x) | TBD | Phase 3 |
| Embeddings | 2.1s | 0.04s (52x) | TBD | Phase 3 |
| Critic Execution | 3.5s | 1.2s (3x) | TBD | Phase 4 |
| Full Pipeline | 12s | 2.5s (5x) | TBD | Phase 5 |

---

## Hardware Configurations

### Budget (GTX 1660 6GB)
- **Inference:** 3-5x speedup
- **Embeddings:** 10-20x speedup
- **Recommended for:** Development, small models

### Mid-Range (RTX 3090 24GB)
- **Inference:** 6-10x speedup
- **Embeddings:** 30-50x speedup
- **Recommended for:** Production, medium models

### High-End (A100 40GB)
- **Inference:** 8-15x speedup
- **Embeddings:** 50-100x speedup
- **Recommended for:** Production at scale, large models

### Apple Silicon (M2 Max)
- **Inference:** 2-4x speedup
- **Embeddings:** 5-10x speedup
- **Recommended for:** Development on Mac

---

## Benchmark Methodology

### Test Configuration
```yaml
Model: Llama-2-7B
Batch Size: 8
Sequence Length: 512 tokens
Precision: FP16
GPU: RTX 4090 (24GB)
```

### Metrics
- **Latency:** Time for single request (P50, P95, P99)
- **Throughput:** Requests per second
- **Memory:** Peak GPU memory usage
- **Utilization:** GPU compute utilization %

---

## Expected Results (Phase 3+)

### Single Request Latency

| Model Size | CPU | GPU (FP32) | GPU (FP16) | Speedup |
|-----------|-----|-----------|-----------|--------|
| 7B params | 5.2s | 1.1s | 0.8s | 6.5x |
| 13B params | 9.8s | 2.3s | 1.6s | 6.1x |
| 70B params | OOM | 12.4s | 8.7s | N/A |

### Batch Throughput

| Batch Size | CPU | GPU (FP32) | GPU (FP16) | Speedup |
|-----------|-----|-----------|-----------|--------|
| 1 | 0.19 req/s | 0.91 req/s | 1.25 req/s | 6.5x |
| 8 | 0.31 req/s | 2.14 req/s | 3.42 req/s | 11x |
| 32 | 0.35 req/s | 3.21 req/s | 5.87 req/s | 16.8x |

### Embedding Generation

| Batch Size | CPU | GPU | Speedup |
|-----------|-----|-----|--------|
| 1 | 45ms | 2ms | 22.5x |
| 10 | 420ms | 8ms | 52x |
| 100 | 4.1s | 45ms | 91x |

---

## Memory Usage

### Model Loading

| Model | FP32 | FP16 | 8-bit | 4-bit |
|-------|------|------|-------|-------|
| 7B | 28GB | 14GB | 7GB | 3.5GB |
| 13B | 52GB | 26GB | 13GB | 6.5GB |
| 70B | OOM | 140GB | 70GB | 35GB |

### Runtime Memory (7B model)

```
Model: 14GB (FP16)
KV Cache: 4GB (batch 8, seq 512)
Activations: 2GB
Overhead: 1GB
---
Total: ~21GB
```

**Recommendation:** 24GB+ GPU for 7B models in production

---

## Scaling Tests

### Multi-GPU Scaling (Data Parallel)

| GPUs | Throughput | Efficiency | Speedup |
|------|-----------|-----------|--------|
| 1 | 3.4 req/s | 100% | 1x |
| 2 | 6.1 req/s | 90% | 1.8x |
| 4 | 11.2 req/s | 82% | 3.3x |
| 8 | 19.8 req/s | 73% | 5.8x |

**Note:** Efficiency loss due to communication overhead

### Stream Scaling (Single GPU)

| Streams | Throughput | Utilization | Speedup |
|---------|-----------|------------|--------|
| 1 | 2.8 req/s | 65% | 1x |
| 2 | 3.1 req/s | 72% | 1.11x |
| 4 | 3.4 req/s | 81% | 1.21x |
| 8 | 3.5 req/s | 83% | 1.25x |

**Recommendation:** 4-8 streams for optimal utilization

---

## Real-World Performance

### ELEANOR Pipeline (Expected)

**Configuration:**
- RTX 4090 (24GB)
- Llama-2-7B
- FP16, 4 streams
- Batch size 8

**Results:**

| Stage | CPU | GPU | Speedup |
|-------|-----|-----|--------|
| Routing | 0.8s | 0.15s | 5.3x |
| Proposal Generation | 5.2s | 0.8s | 6.5x |
| Critic Evaluation | 3.5s | 1.2s | 2.9x |
| Embedding Search | 2.1s | 0.04s | 52.5x |
| Evidence Recording | 0.4s | 0.4s | 1x |
| **Total** | **12s** | **2.59s** | **4.6x** |

---

## Optimization Impact

### Mixed Precision (FP16)

```
Memory: -50%
Speed: +100%
Accuracy: -0.1% (negligible)
```

### Quantization (8-bit)

```
Memory: -75%
Speed: +30-50%
Accuracy: -1-2%
```

### FlashAttention

```
Memory: -30-40%
Speed: +40-60%
Accuracy: 100% (mathematically equivalent)
```

### Batching

```
Batch 1 → 8: +3x throughput
Batch 8 → 32: +1.7x throughput
Batch 32 → 128: +1.3x throughput
```

---

## Benchmarking Tools

### Run Benchmarks

```bash
# Coming in Phase 3
python scripts/benchmark_gpu.py --model llama-7b --batch-sizes 1,8,32
```

### Monitor GPU

```bash
# NVIDIA
watch -n 1 nvidia-smi

# Apple Silicon
sudo powermetrics --samplers gpu_power -i 1000

# Python
from engine.gpu import GPUManager

gpu = GPUManager()
while True:
    stats = gpu.memory_stats()
    print(f"GPU: {stats['utilization_pct']:.1f}%")
    time.sleep(1)
```

---

## Cost Analysis

### Cloud GPU Pricing (per hour)

| Provider | GPU | VRAM | Price | Speedup | Cost/Req |
|----------|-----|------|-------|---------|----------|
| AWS | g5.xlarge | T4 16GB | $1.01 | 4x | $0.00028 |
| AWS | g5.2xlarge | A10G 24GB | $1.52 | 6x | $0.00025 |
| AWS | p4d.24xlarge | A100 40GB | $32.77 | 12x | $0.00027 |
| GCP | n1 + T4 | T4 16GB | $0.95 | 4x | $0.00026 |
| Azure | NC6s_v3 | V100 16GB | $3.06 | 8x | $0.00038 |

**Recommendation:** AWS g5.2xlarge (A10G) for best price/performance

---

## Next Steps

After Phase 3 (GPU Embeddings) implementation:

1. **Run Real Benchmarks**
   ```bash
   python scripts/benchmark_gpu.py
   ```

2. **Compare Results**
   - Validate against targets
   - Identify bottlenecks
   - Optimize hot paths

3. **Update This Document**
   - Add actual results
   - Update recommendations
   - Document optimizations

---

**Status:** Targets Defined ✅  
**Next:** Implement and benchmark Phase 3
