# ELEANOR V8 - Caching Strategy

## Overview

ELEANOR implements a multi-level caching strategy to optimize performance:

- **L1 Cache**: In-memory TTL cache with LRU eviction (process-local)
- **L2 Cache**: Redis for shared caching across instances (optional)

## Architecture

```
Request → L1 (memory) → L2 (Redis) → Compute → Cache result
            ↓ hit          ↓ hit         ↓
          Return        Populate L1   Store L1+L2
```

## Cached Operations

### 1. Precedent Retrieval
- **Cache Key**: Hash of query text + critic results
- **L1 TTL**: 1 hour
- **L2 TTL**: 2 hours
- **Rationale**: Expensive vector search + database queries

### 2. Embeddings
- **Cache Key**: Hash of input text
- **L1 TTL**: 2 hours
- **L2 TTL**: 4 hours
- **Rationale**: Deterministic and expensive to compute

### 3. Router Selections
- **Cache Key**: Hash of text + context
- **L1 TTL**: 30 minutes
- **L2 TTL**: 1 hour
- **Rationale**: Similar queries should use same model

### 4. Critic Evaluations
- **Cache Key**: Hash of critic + input + model response
- **L1 TTL**: 30 minutes
- **L2 TTL**: 1 hour
- **Rationale**: Identical evaluations shouldn't be repeated

### 5. Detector Signals
- **Cache Key**: Hash of detector + input + context
- **L1 TTL**: 10 minutes
- **L2 TTL**: 20 minutes
- **Rationale**: Fast-changing, but expensive pattern matching

## Configuration

See `config/caching.yaml` for configuration options.

## Usage

```python
from engine.cache import CacheManager, CacheKey

# Initialize cache manager
cache = CacheManager(redis_client=redis_client)

# Get or compute value
key = CacheKey.from_data('precedent', query=query_text, critics=critics)
result = await cache.get_or_compute(
    key,
    compute_precedent_alignment,
    query=query_text,
    critics=critics
)

# Get statistics
stats = cache.get_stats()
print(f"Precedent cache hit rate: {stats['precedent']['hit_rate']:.2%}")
```

## Adaptive Concurrency

The system automatically adjusts concurrency based on observed latency:

```python
from engine.cache import AdaptiveConcurrencyManager

# Initialize adaptive concurrency
concurrency = AdaptiveConcurrencyManager(
    initial_limit=6,
    min_limit=2,
    max_limit=20,
    target_latency_ms=500
)

# Use as context manager
async with concurrency:
    result = await expensive_operation()
    
# Record latency for adjustment
concurrency.record_latency(latency_ms)
```

## Performance Targets

- **Cache Hit Rate**: >60% for precedent retrievals
- **Latency Reduction**: 40-50% for cached operations  
- **Throughput Increase**: 2-3x with adaptive concurrency
- **Memory Usage**: <500MB for L1 caches

## Monitoring

Cache statistics are exposed via structured logging:

```json
{
  "event": "cache_stats",
  "precedent": {
    "hits": 1234,
    "misses": 456,
    "hit_rate": 0.73,
    "sets": 456
  },
  "concurrency": {
    "current_limit": 8,
    "p95_latency_ms": 420,
    "target_latency_ms": 500
  }
}
```
