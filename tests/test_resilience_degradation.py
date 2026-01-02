import pytest

from engine.engine import EleanorEngineV8, EngineConfig
from engine.exceptions import RouterSelectionError
from engine.factory import EngineDependencies
from engine.mocks import (
    MockCritic,
    MockDetectorEngine,
    MockEvidenceRecorder,
    MockAggregator,
    MockReviewTriggerEvaluator,
)


class FailingRouter:
    def route(self, text, context):
        raise RouterSelectionError("router failure")


@pytest.mark.asyncio
async def test_router_failure_marks_degraded():
    deps = EngineDependencies(
        router=FailingRouter(),
        detector_engine=MockDetectorEngine(),
        evidence_recorder=MockEvidenceRecorder(),
        critics={"mock": MockCritic},
        precedent_engine=None,
        precedent_retriever=None,
        uncertainty_engine=None,
        aggregator=MockAggregator(),
        review_trigger_evaluator=MockReviewTriggerEvaluator(),
        critic_models=None,
    )

    engine = EleanorEngineV8(
        config=EngineConfig(
            enable_precedent_analysis=False,
            enable_reflection=False,
            enable_circuit_breakers=True,
            enable_graceful_degradation=True,
        ),
        dependencies=deps,
    )

    result = await engine.run("hello")

    assert result.is_degraded is True
    assert "router" in (result.degraded_components or [])
