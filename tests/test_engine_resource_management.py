import pytest

from engine.engine import EleanorEngineV8, EngineConfig
from engine.factory import EngineDependencies
from engine.mocks import (
    MockRouter,
    MockDetectorEngine,
    MockAggregator,
    MockReviewTriggerEvaluator,
)


class DummyRecorder:
    def __init__(self):
        self.initialized = False
        self.closed = False

    async def initialize(self):
        self.initialized = True

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_engine_context_manager_runs_cleanup():
    recorder = DummyRecorder()
    deps = EngineDependencies(
        router=MockRouter(model_name="fake-model", response_text="mock response"),
        detector_engine=MockDetectorEngine(),
        evidence_recorder=recorder,
        critics={},
        precedent_engine=None,
        precedent_retriever=None,
        uncertainty_engine=None,
        aggregator=MockAggregator(),
        review_trigger_evaluator=MockReviewTriggerEvaluator(),
        critic_models=None,
    )

    engine = EleanorEngineV8(
        config=EngineConfig(enable_precedent_analysis=False, enable_reflection=False),
        dependencies=deps,
    )

    async with engine:
        pass

    assert recorder.initialized is True
    assert recorder.closed is True
