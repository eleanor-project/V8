import pytest
import asyncio
from engine.engine import EleanorEngineV8, EngineConfig, create_engine

@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

class FakeRouter:
    async def route(self, text, context):
        return {
            "model_name": "fake-model",
            "model_version": "1.0",
            "reason": "test-router",
            "response_text": "Hello world (fake model output)",
            "cost": {"prompt_tokens": 5, "completion_tokens": 5},
            "health_score": 1.0,
            "diagnostics": {"mock": True},
        }

@pytest.fixture
def fake_router():
    return FakeRouter()

@pytest.fixture
def engine(fake_router):
    return create_engine(
        config=EngineConfig(detail_level=2),
        router_backend=fake_router,
    )
