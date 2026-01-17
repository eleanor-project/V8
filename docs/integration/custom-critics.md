# Custom Critics

Create a new critic by extending `BaseCriticV8`.

```python
from engine.critics.base import BaseCriticV8
from engine.schemas.pipeline_types import CriticResult

class CustomCritic(BaseCriticV8):
    async def evaluate(self, model, input_text: str, context: dict) -> CriticResult:
        return {
            "critic": "custom",
            "score": 0.1,
            "severity": 0.1,
            "violations": [],
            "justification": "custom check",
        }
```

Register the critic in your engine dependencies or configuration. See
`engine/factory.py` and `engine/runtime/critics.py`.
