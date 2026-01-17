# Custom Critic

Custom critics implement the `CriticProtocol` interface and can be passed to
`EleanorEngineV8` on initialization.

## Minimal example

```python
from typing import Dict, Any
from engine.engine import EleanorEngineV8
from engine.protocols import CriticProtocol

class DemoCritic(CriticProtocol):
    name = "demo_critic"

    async def evaluate(self, model_adapter, input_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "value": "allow",
            "severity": 0.0,
            "violations": [],
            "justification": "Demo critic always allows",
        }

engine = EleanorEngineV8(critics={"demo": DemoCritic()})
```

## Notes

- Use `engine/protocols.py` for the required method signatures.
- Ensure output matches the schema in `engine/schemas/pipeline_types.py`.
- For a full integration example, see `docs/integration/custom-critics.md`.
- `evaluate` may be sync or async; the runtime will handle either.
