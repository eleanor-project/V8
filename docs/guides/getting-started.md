# Getting Started

## Install dependencies

```bash
pip install -r requirements.txt
```

Optional extras:

```bash
pip install -r requirements-observability.txt
pip install -r requirements-gpu.txt
```

## Create an engine

```python
from engine.engine import create_engine

engine = create_engine()
result = await engine.run("Evaluate this output", context={"domain": "general"})
print(result.output_text)
```

## Next steps

- `docs/guides/basic-usage.md`
- `docs/guides/configuration.md`
- `docs/guides/testing.md`
