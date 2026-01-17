"""Benchmarks for engine end-to-end execution using mocks."""

import asyncio

import pytest

from engine.engine import EngineConfig, create_engine
from engine.factory import EngineDependencies
from engine.mocks import (
    MockAggregator,
    MockCritic,
    MockDetectorEngine,
    MockEvidenceRecorder,
    MockPrecedentEngine,
    MockPrecedentRetriever,
    MockReviewTriggerEvaluator,
    MockRouter,
    MockUncertaintyEngine,
)


def _build_engine():
    critics = {
        "truth": MockCritic("truth", score=0.12),
        "fairness": MockCritic("fairness", score=0.18),
        "risk": MockCritic("risk", score=0.22),
        "pragmatics": MockCritic("pragmatics", score=0.1),
    }
    deps = EngineDependencies(
        router=MockRouter(response_text="mock response", model_name="bench-mock"),
        detector_engine=MockDetectorEngine(),
        evidence_recorder=MockEvidenceRecorder(),
        critics=critics,
        review_trigger_evaluator=MockReviewTriggerEvaluator(),
        precedent_engine=MockPrecedentEngine(),
        precedent_retriever=MockPrecedentRetriever(),
        uncertainty_engine=MockUncertaintyEngine(),
        aggregator=MockAggregator(),
        critic_models={},
    )
    config = EngineConfig(enable_precedent_analysis=False, enable_reflection=False)
    return create_engine(config=config, dependencies=deps)


@pytest.mark.performance
def test_engine_run_detail_level_1(benchmark):
    engine = _build_engine()

    def run_once():
        return asyncio.run(engine.run("benchmark input", context={"source": "bench"}, detail_level=1))

    result = benchmark(run_once)
    assert result is not None


@pytest.mark.performance
def test_engine_run_detail_level_3(benchmark):
    engine = _build_engine()

    def run_once():
        return asyncio.run(engine.run("benchmark input", context={"source": "bench"}, detail_level=3))

    result = benchmark(run_once)
    assert result is not None
