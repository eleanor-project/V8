import asyncio
import sys
from types import SimpleNamespace

import pytest

from engine.engine import EleanorEngineV8, EngineConfig
from engine.factory import EngineDependencies
from engine.mocks import (
    MockAggregator,
    MockCritic,
    MockDetectorEngine,
    MockEvidenceRecorder,
    MockReviewTriggerEvaluator,
    MockRouter,
)


class DummyAsyncResource:
    def __init__(self):
        self.calls = []

    async def initialize(self):
        self.calls.append("initialize")

    async def connect(self):
        self.calls.append("connect")

    async def close(self):
        self.calls.append("close")


class DummySlowResource:
    def __init__(self, delay=0.05, should_raise=False):
        self.delay = delay
        self.should_raise = should_raise
        self.closed = False

    async def close(self):
        if self.should_raise:
            raise RuntimeError("boom")
        await asyncio.sleep(self.delay)
        self.closed = True


def _build_engine():
    deps = EngineDependencies(
        router=MockRouter(model_name="m", response_text="ok"),
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
    return EleanorEngineV8(
        config=EngineConfig(enable_precedent_analysis=False, enable_reflection=False),
        dependencies=deps,
    )


@pytest.mark.asyncio
async def test_setup_resources_calls_initialize_and_connect():
    engine = _build_engine()
    recorder = DummyAsyncResource()
    cache_manager = DummyAsyncResource()
    precedent_retriever = DummyAsyncResource()

    engine.recorder = recorder
    engine.cache_manager = cache_manager
    engine.precedent_retriever = precedent_retriever

    await engine._setup_resources()
    assert recorder.calls == ["initialize"]
    assert cache_manager.calls == ["connect"]
    assert precedent_retriever.calls == ["connect"]


@pytest.mark.asyncio
async def test_shutdown_timeout_and_cleanup_tasks(monkeypatch):
    engine = _build_engine()
    engine.recorder = DummySlowResource(delay=0.05)
    engine.precedent_retriever = DummySlowResource(delay=0.05)
    engine.cache_manager = DummySlowResource(delay=0.05, should_raise=True)

    async def _linger():
        await asyncio.sleep(1)

    engine._cleanup_tasks = [asyncio.create_task(_linger())]
    engine.gpu_embedding_cache = type("Cache", (), {"clear_cache": lambda self: None})()
    engine.gpu_manager = object()

    torch_stub = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda *_args, **_kwargs: False,
            empty_cache=lambda *_args, **_kwargs: None,
        )
    )
    monkeypatch.setitem(sys.modules, "torch", torch_stub)

    await engine.shutdown(timeout=0.01)
