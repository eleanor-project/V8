# Streaming Response

Use `run_stream` to stream events as the pipeline progresses.

```python
from engine.engine import EleanorEngineV8
import asyncio

async def main():
    engine = EleanorEngineV8()
    async for event in engine.run_stream(
        "Evaluate this input and stream events",
        detail_level=2,
    ):
        print(event["event"], event)

asyncio.run(main())
```

For a streaming governance example, see:

```bash
python examples/streaming_governance_demo.py
```
