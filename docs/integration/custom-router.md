# Custom Routers

Implement `RouterProtocol` from `engine/protocols.py` and return a response payload
matching the expected router output shape.

Example:

```python
class CustomRouter:
    async def route(self, text: str, context: dict):
        return {
            "response_text": "mock",
            "model_name": "custom",
            "model_version": "1.0",
            "reason": "custom",
            "health_score": 1.0,
            "cost": None,
            "diagnostics": {},
        }
```

Integrate via `EngineDependencies(router=CustomRouter())`.
