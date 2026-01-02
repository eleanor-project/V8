# ELEANOR V8 - Resource Management Guide

## Overview

ELEANOR V8 implements comprehensive async resource management to ensure:
- **Data Integrity**: Evidence is always flushed before shutdown
- **Reliability**: No resource leaks in production
- **Graceful Degradation**: Clean shutdown even under error conditions
- **Production Ready**: Signal handling for container orchestration

## Features

### 1. Context Manager Protocol

The engine implements Python's async context manager protocol:

```python
async with EleanorEngineV8(config=config) as engine:
    result = await engine.run(text="...")
    # Resources automatically cleaned up on exit
```

**Benefits:**
- Automatic resource initialization on `__aenter__`
- Guaranteed cleanup on `__aexit__`
- Works with exceptions and early returns
- Follows Python best practices

### 2. Async Evidence Recorder

The new `AsyncEvidenceRecorder` provides:
- **Buffered Writes**: Batch records for efficiency
- **Periodic Flushing**: Auto-flush every 5 seconds (configurable)
- **Graceful Shutdown**: Always flushes on close
- **Async I/O**: Non-blocking file operations with `aiofiles`

```python
from engine.evidence_recorder_async import AsyncEvidenceRecorder

async with AsyncEvidenceRecorder("evidence.jsonl") as recorder:
    await recorder.record(critic="rights", severity="HIGH", ...)
    # Auto-flushed periodically and on context exit
```

### 3. Graceful Shutdown

The engine handles shutdown gracefully:

```python
engine = EleanorEngineV8(config=config)
await engine._setup_resources()

try:
    # Process requests...
    pass
finally:
    await engine.shutdown(timeout=30.0)
```

**Shutdown Process:**
1. Signal all components to stop
2. Flush evidence buffer (critical - no data loss)
3. Close database/cache connections
4. Cancel pending background tasks
5. Log shutdown completion with timing

### 4. Signal Handling

For production deployments, use `ShutdownHandler`:

```python
from engine.resource_manager import ShutdownHandler

engine = EleanorEngineV8(config=config)

async def shutdown():
    await engine.shutdown()

handler = ShutdownHandler(shutdown)
handler.setup_handlers()  # Registers SIGTERM and SIGINT

# Run your application
# Shutdown will be triggered on Ctrl+C or container stop
```

### 5. Timeout Protection

Protect operations from hanging:

```python
from engine.resource_manager import TimeoutProtection

result = await TimeoutProtection.with_timeout(
    some_operation(),
    timeout=10.0,
    operation="critic_evaluation",
    raise_on_timeout=True
)
```

## Configuration

Configure resource management in `config/resource_management.yaml`:

```yaml
resource_management:
  shutdown:
    graceful_timeout: 30.0
    
  evidence_recorder:
    buffer_size: 1000
    flush_interval: 5.0
    
  timeouts:
    critic_evaluation: 10.0
    router_selection: 5.0
```

## Production Deployment

### Docker/Kubernetes

```python
# main.py
import asyncio
import signal
from engine.engine import EleanorEngineV8
from engine.resource_manager import ShutdownHandler

async def main():
    engine = EleanorEngineV8(config=...)
    
    # Setup signal handlers
    handler = ShutdownHandler(lambda: engine.shutdown())
    handler.setup_handlers()
    
    async with engine:
        # Run your API server
        await serve_api(engine)

if __name__ == "__main__":
    asyncio.run(main())
```

### Kubernetes Lifecycle

```yaml
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: eleanor-v8
    lifecycle:
      preStop:
        exec:
          command: ["/bin/sh", "-c", "sleep 5"]  # Allow time for graceful shutdown
    terminationGracePeriodSeconds: 35  # Match shutdown timeout + buffer
```

## Testing

Run resource management tests:

```bash
pytest tests/test_async_resource_management.py -v
```

**Test Coverage:**
- ✅ Context manager initialization and cleanup
- ✅ Evidence recorder buffering and flushing
- ✅ Auto-flush on buffer full
- ✅ Final flush on close
- ✅ Task tracking and cancellation
- ✅ Timeout protection
- ✅ Signal handler registration

## Troubleshooting

### Evidence Not Flushed

**Symptom**: Data missing from `evidence.jsonl`

**Solution**: Always use context manager or call `shutdown()` explicitly:

```python
# ✅ Good
async with engine:
    ...

# ❌ Bad - may lose buffered data
engine = EleanorEngineV8()
# ... use engine without cleanup
```

### Shutdown Timeout

**Symptom**: "Shutdown exceeded timeout" warning

**Solution**: Increase timeout or investigate slow operations:

```python
await engine.shutdown(timeout=60.0)  # Increase timeout
```

### Resource Leaks

**Symptom**: Open file handles or connections after shutdown

**Solution**: Verify all resources implement cleanup:

```python
# Check if resource has cleanup method
if hasattr(resource, 'close'):
    await resource.close()
```

## Migration Guide

### From Sync Evidence Recorder

**Before:**
```python
recorder = EvidenceRecorder("evidence.jsonl")
recorder.record(...)  # Sync, blocks event loop
```

**After:**
```python
recorder = AsyncEvidenceRecorder("evidence.jsonl")
await recorder.initialize()
await recorder.record(...)  # Async, non-blocking
await recorder.close()
```

### Adding Context Manager Support

**Before:**
```python
engine = EleanorEngineV8(config=config)
result = await engine.run(text="...")
# No cleanup!
```

**After:**
```python
async with EleanorEngineV8(config=config) as engine:
    result = await engine.run(text="...")
    # Automatic cleanup
```

## Best Practices

1. **Always Use Context Managers**: Ensures cleanup even on exceptions
2. **Configure Appropriate Timeouts**: Balance responsiveness vs. completion
3. **Monitor Flush Intervals**: Adjust based on evidence volume
4. **Test Shutdown Paths**: Verify data integrity under all exit conditions
5. **Log Resource Lifecycle**: Track initialization and cleanup for debugging

## Future Enhancements

- [ ] Redis connection pool management
- [ ] Database connection pool management
- [ ] Health check endpoint for resource status
- [ ] Metrics for buffer size and flush frequency
- [ ] Automatic resource leak detection
- [ ] Circuit breakers for external dependencies

## Related Issues

- #19: Async Resource Management (this implementation)
- #14: Circuit Breakers and Graceful Degradation
- #17: Observability and Structured Logging
- #18: Resilience Patterns
