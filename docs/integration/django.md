# Django Integration

ELEANOR V8 can be used inside Django views by running the async engine within
an async view or by dispatching to an async task runner.

## Async view example (Django 3.1+)

```python
from django.http import JsonResponse
from engine.engine import EleanorEngineV8

engine = EleanorEngineV8()

async def evaluate(request):
    payload = request.GET.get("text", "")
    result = await engine.run(payload, detail_level=2)
    return JsonResponse({"output": result.output_text, "trace_id": result.trace_id})
```

## Sync view with task queue

For synchronous views, dispatch to a task queue (Celery, RQ) that can run async
code. This keeps request latency predictable and isolates model execution.

## Notes

- Use `EngineConfig` or `EleanorSettings` to configure timeouts and routing.
- For streaming responses, use Django Channels or ASGI.
- For shared setup patterns, see `docs/integration/fastapi.md`.
