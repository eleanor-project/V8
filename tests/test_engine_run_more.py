import pytest

from engine.engine import EleanorEngineV8, EngineConfig
from engine.exceptions import CriticEvaluationError, InputValidationError
from engine.factory import EngineDependencies
from engine.mocks import (
    MockAggregator,
    MockCritic,
    MockDetectorEngine,
    MockReviewTriggerEvaluator,
    MockRouter,
)
from engine.utils.circuit_breaker import CircuitBreakerOpen


class FailingRecorder:
    async def record(self, **_kwargs):
        raise RuntimeError("record boom")


class DummyBreaker:
    async def call(self, _func, *args, **kwargs):
        raise CircuitBreakerOpen("open", 1.0)


def _build_engine(*, recorder=None, router=None, config=None):
    deps = EngineDependencies(
        router=router or MockRouter(model_name="m", response_text="ok"),
        detector_engine=MockDetectorEngine(),
        evidence_recorder=recorder or FailingRecorder(),
        critics={"mock": MockCritic},
        precedent_engine=None,
        precedent_retriever=None,
        uncertainty_engine=None,
        aggregator=MockAggregator(),
        review_trigger_evaluator=MockReviewTriggerEvaluator(),
        critic_models=None,
    )
    return EleanorEngineV8(
        config=config or EngineConfig(enable_precedent_analysis=False, enable_reflection=False),
        dependencies=deps,
    )


@pytest.mark.asyncio
async def test_run_skip_router_requires_output():
    engine = _build_engine()
    with pytest.raises(InputValidationError):
        await engine.run("hello", context={"skip_router": True})

    async def _consume():
        async for _event in engine.run_stream("hello", context={"skip_router": True}):
            pass

    with pytest.raises(InputValidationError):
        await _consume()


@pytest.mark.asyncio
async def test_run_degradation_disabled_raises():
    engine = _build_engine(
        config=EngineConfig(
            enable_precedent_analysis=False,
            enable_reflection=False,
            enable_graceful_degradation=False,
        )
    )
    engine._get_circuit_breaker = lambda name: DummyBreaker()
    with pytest.raises(CircuitBreakerOpen):
        await engine.run("hello")


@pytest.mark.asyncio
async def test_run_forensic_detail_level():
    engine = _build_engine()
    result = await engine.run(
        "hello",
        context={"skip_router": True, "model_output": "ok"},
        detail_level=3,
    )
    assert result.forensic is not None
    assert result.forensic.timings


@pytest.mark.asyncio
async def test_run_single_critic_records_error_path():
    engine = _build_engine()
    result = await engine._run_single_critic(
        "mock",
        MockCritic,
        "resp",
        "in",
        {},
        "trace",
    )
    assert result["justification"]

    engine = _build_engine(recorder=FailingRecorder())
    with pytest.raises(CriticEvaluationError):
        await engine._run_single_critic(
            "mock",
            type("FailingCritic", (), {"evaluate": lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom"))}),
            "resp",
            "in",
            {},
            "trace",
        )


@pytest.mark.asyncio
async def test_run_detectors_none():
    engine = _build_engine()
    engine.detector_engine = None
    assert await engine._run_detectors("text", {}, {}) is None
