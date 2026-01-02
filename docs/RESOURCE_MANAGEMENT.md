# ELEANOR V8 — Async Resource Management

## Overview

ELEANOR provides explicit lifecycle management for async resources to prevent
leaks during shutdown or unexpected interruptions. The engine implements an
async context manager and a `shutdown()` method that closes resources and
flushes buffers.

## Engine Lifecycle

```python
async with EleanorEngineV8(...) as engine:
    result = await engine.run("...")

# or, on shutdown:
await engine.shutdown()
```

### Resources Cleaned Up

- Evidence recorder: flush pending evidence and stop background flush task
- Cache manager: close Redis connections (if configured)
- Precedent retriever: close underlying store connection (if supported)

## API Integration

The FastAPI server uses its lifespan handler to call `engine.shutdown()` on
shutdown signals (SIGTERM/SIGINT). This ensures cleanup during deployments or
restarts.

## Evidence Recorder Flush

Evidence writes are buffered and flushed periodically. On shutdown, the engine
forces a final flush to prevent data loss.

## Configuration

Evidence flush interval and buffer size can be configured:

```bash
ELEANOR_EVIDENCE__BUFFER_SIZE=1000
ELEANOR_EVIDENCE__FLUSH_INTERVAL=5.0
```
