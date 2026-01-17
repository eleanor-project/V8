# Basic Evaluation

This example shows a minimal use of `EleanorEngineV8`.

```python
import asyncio
from engine.engine import EleanorEngineV8

async def main():
    engine = EleanorEngineV8()
    result = await engine.run("Explain the governance gates.")
    print(result.output_text)

asyncio.run(main())
```

Notes:

- `engine.run` returns an `EngineResult` object defined in
  `engine/runtime/models.py`.
- `detail_level` controls verbosity; set it in `EngineConfig` or via settings.

For a complete demo including observability and resilience, run:

```bash
python examples/integration_example.py
```
