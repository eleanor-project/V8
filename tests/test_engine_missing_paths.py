import builtins
import sys
from types import ModuleType, SimpleNamespace

import pytest

import engine.engine as engine_module
from engine.engine import EleanorEngineV8, EngineConfig
from engine.exceptions import (
    DetectorExecutionError,
    PrecedentRetrievalError,
    RouterSelectionError,
    UncertaintyComputationError,
)
from engine.factory import EngineDependencies
from engine.mocks import (
    MockAggregator,
    MockCritic,
    MockDetectorEngine,
    MockEvidenceRecorder,
    MockReviewTriggerEvaluator,
    MockRouter,
)
from engine.utils.circuit_breaker import CircuitBreakerOpen


class StaticBreaker:
    def __init__(self, result=None, exc=None):
        self._result = result
        self._exc = exc

    async def call(self, _func, *args, **kwargs):
        if self._exc:
            raise self._exc
        return self._result


def _make_deps(**overrides):
    return EngineDependencies(
        router=overrides.get("router") or MockRouter(model_name="m", response_text="ok"),
        detector_engine=overrides.get("detector_engine") or MockDetectorEngine(),
        evidence_recorder=overrides.get("evidence_recorder") or MockEvidenceRecorder(),
        critics=overrides.get("critics") or {"mock": MockCritic},
        precedent_engine=overrides.get("precedent_engine"),
        precedent_retriever=overrides.get("precedent_retriever"),
        uncertainty_engine=overrides.get("uncertainty_engine"),
        aggregator=overrides.get("aggregator") or MockAggregator(),
        review_trigger_evaluator=overrides.get("review_trigger_evaluator")
        or MockReviewTriggerEvaluator(),
        critic_models=None,
    )


def _build_engine(config=None, **overrides):
    return EleanorEngineV8(
        config=config
        or EngineConfig(
            enable_precedent_analysis=False,
            enable_reflection=False,
            enable_circuit_breakers=False,
        ),
        dependencies=_make_deps(**overrides),
    )


async def _collect_events(engine, text="hello", context=None, detail_level=1):
    events = []
    async for event in engine.run_stream(text, context=context, detail_level=detail_level):
        events.append(event)
    return events


def test_engine_init_config_manager_failure(monkeypatch):
    class DummyConfigManager:
        def __init__(self):
            raise RuntimeError("boom")

    import engine.config as config_module

    monkeypatch.setattr(config_module, "ConfigManager", DummyConfigManager)
    engine = EleanorEngineV8(config=None, dependencies=_make_deps())
    assert isinstance(engine.config, EngineConfig)


def test_engine_init_gpu_modules_unavailable(monkeypatch):
    class DummySettings:
        def __init__(self):
            self.environment = "test"
            self.cache = SimpleNamespace(enabled=False)
            self.gpu = SimpleNamespace(
                enabled=True,
                preferred_devices=[],
                device_preference="auto",
                memory=SimpleNamespace(
                    mixed_precision=False,
                    max_memory_per_gpu="1GB",
                    log_memory_stats=False,
                    memory_check_interval=1.0,
                ),
                async_ops=SimpleNamespace(num_streams=1),
                batching=SimpleNamespace(
                    default_batch_size=1,
                    max_batch_size=2,
                    dynamic_batching=False,
                ),
                embeddings=SimpleNamespace(
                    cache_on_gpu=False,
                    mixed_precision=False,
                    max_cache_size_gb=0.0,
                    embedding_dim=2,
                ),
                precedent=SimpleNamespace(cache_embeddings_on_gpu=False),
                multi_gpu=SimpleNamespace(enabled=False, device_ids=None),
                critics=SimpleNamespace(gpu_batching=False, use_gpu=False, batch_size=1),
            )

        def to_legacy_engine_config(self):
            return {
                "enable_precedent_analysis": False,
                "enable_reflection": False,
            }

    class DummyConfigManager:
        def __init__(self):
            self.settings = DummySettings()

    import engine.config as config_module

    monkeypatch.setattr(config_module, "ConfigManager", DummyConfigManager)
    original_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name.startswith("engine.gpu."):
            raise ImportError("boom")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    engine = EleanorEngineV8(config=None, dependencies=_make_deps())
    assert isinstance(engine.config, EngineConfig)


def test_engine_init_gpu_batch_processor_unavailable(monkeypatch):
    class DummySettings:
        def __init__(self):
            self.environment = "test"
            self.cache = SimpleNamespace(enabled=False)
            self.gpu = SimpleNamespace(
                enabled=True,
                preferred_devices=[],
                device_preference="auto",
                memory=SimpleNamespace(
                    mixed_precision=False,
                    max_memory_per_gpu="1GB",
                    log_memory_stats=False,
                    memory_check_interval=1.0,
                ),
                async_ops=SimpleNamespace(num_streams=1),
                batching=SimpleNamespace(
                    default_batch_size=1,
                    max_batch_size=2,
                    dynamic_batching=False,
                ),
                embeddings=SimpleNamespace(
                    cache_on_gpu=False,
                    mixed_precision=False,
                    max_cache_size_gb=0.0,
                    embedding_dim=2,
                ),
                precedent=SimpleNamespace(cache_embeddings_on_gpu=False),
                multi_gpu=SimpleNamespace(enabled=False, device_ids=None),
                critics=SimpleNamespace(gpu_batching=True, use_gpu=True, batch_size=2),
            )

        def to_legacy_engine_config(self):
            return {
                "enable_precedent_analysis": False,
                "enable_reflection": False,
            }

    class DummyConfigManager:
        def __init__(self):
            self.settings = DummySettings()

    class DummyGPUConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class DummyGPUManager:
        def __init__(self, *_args, **_kwargs):
            self.device = "cuda:0"

        def is_available(self):
            return True

        def get_device(self):
            return "cuda:0"

    class DummyAsyncGPUExecutor:
        def __init__(self, *_args, **_kwargs):
            self.ready = True

    class DummyGPUEmbeddingCache:
        def __init__(self, *_args, **_kwargs):
            self.ready = True

    class DummyMultiGPURouter:
        def __init__(self, *_args, **_kwargs):
            self.ready = True

    gpu_manager_module = ModuleType("engine.gpu.manager")
    gpu_manager_module.GPUManager = DummyGPUManager
    gpu_manager_module.GPUConfig = DummyGPUConfig

    async_ops_module = ModuleType("engine.gpu.async_ops")
    async_ops_module.AsyncGPUExecutor = DummyAsyncGPUExecutor

    embeddings_module = ModuleType("engine.gpu.embeddings")
    embeddings_module.GPUEmbeddingCache = DummyGPUEmbeddingCache

    parallel_module = ModuleType("engine.gpu.parallelization")
    parallel_module.MultiGPURouter = DummyMultiGPURouter

    monkeypatch.setitem(sys.modules, "engine.gpu.manager", gpu_manager_module)
    monkeypatch.setitem(sys.modules, "engine.gpu.async_ops", async_ops_module)
    monkeypatch.setitem(sys.modules, "engine.gpu.embeddings", embeddings_module)
    monkeypatch.setitem(sys.modules, "engine.gpu.parallelization", parallel_module)

    import engine.config as config_module

    monkeypatch.setattr(config_module, "ConfigManager", DummyConfigManager)
    original_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "engine.gpu.batch_processor":
            raise ImportError("boom")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    engine = EleanorEngineV8(config=None, dependencies=_make_deps())
    assert engine.critic_batcher is None


@pytest.mark.asyncio
async def test_setup_resources_skips_non_callable():
    engine = _build_engine()
    engine.recorder = SimpleNamespace(initialize=None)
    await engine._setup_resources()


@pytest.mark.asyncio
async def test_shutdown_handles_torch_cleanup_error(monkeypatch):
    engine = _build_engine()
    engine.gpu_manager = object()
    torch_stub = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")),
            empty_cache=lambda *_a, **_k: None,
        )
    )
    monkeypatch.setitem(sys.modules, "torch", torch_stub)

    async def _noop():
        return None

    import engine.utils.http_client as http_client

    monkeypatch.setattr(http_client, "aclose_async_client", _noop)
    await engine.shutdown(timeout=0)


@pytest.mark.asyncio
async def test_run_precedent_alignment_non_awaitable(monkeypatch):
    class SyncRetriever:
        def retrieve(self, *_args, **_kwargs):
            return {"precedent_cases": [{"id": 1}], "query_embedding": [0.1]}

    engine = _build_engine(
        config=EngineConfig(enable_precedent_analysis=True, enable_reflection=False),
        precedent_engine=SimpleNamespace(),
        precedent_retriever=SyncRetriever(),
    )
    monkeypatch.setattr(engine_module.inspect, "iscoroutinefunction", lambda _fn: True)
    result = await engine._run_precedent_alignment({"c": {"score": 0.1}}, "t", text="q")
    assert result["retrieval"]["query_embedding"] == [0.1]


@pytest.mark.asyncio
async def test_run_precedent_alignment_analyze_error():
    class BadEngine:
        def analyze(self, *_args, **_kwargs):
            raise RuntimeError("boom")

    engine = _build_engine(
        config=EngineConfig(enable_precedent_analysis=True, enable_reflection=False),
        precedent_engine=BadEngine(),
    )
    with pytest.raises(PrecedentRetrievalError):
        await engine._run_precedent_alignment({"c": {"score": 0.1}}, "t")


@pytest.mark.asyncio
async def test_run_detector_execution_error():
    engine = _build_engine()

    async def _raise(*_args, **_kwargs):
        raise DetectorExecutionError("boom")

    engine._run_detectors = _raise
    result = await engine.run(
        "hi",
        context={"skip_router": True, "model_output": "ok"},
        detail_level=1,
    )
    assert result.output_text == "ok"


@pytest.mark.asyncio
async def test_run_precedent_breaker_open_degrades():
    engine = _build_engine(
        config=EngineConfig(enable_precedent_analysis=True, enable_reflection=False),
        precedent_engine=SimpleNamespace(),
    )
    engine._get_circuit_breaker = lambda name: StaticBreaker(
        exc=CircuitBreakerOpen("open", 1.0)
    ) if name == "precedent" else None
    result = await engine.run(
        "hello",
        context={"skip_router": True, "model_output": "ok"},
        detail_level=1,
    )
    assert "precedent" in (result.degraded_components or [])


@pytest.mark.asyncio
async def test_run_precedent_error_no_degrade():
    engine = _build_engine(
        config=EngineConfig(
            enable_precedent_analysis=True,
            enable_reflection=False,
            enable_graceful_degradation=False,
        ),
        precedent_engine=SimpleNamespace(),
    )
    engine.degradation_enabled = False

    async def _raise(*_args, **_kwargs):
        raise PrecedentRetrievalError("boom")

    engine._run_precedent_alignment = _raise
    result = await engine.run(
        "hello",
        context={"skip_router": True, "model_output": "ok"},
        detail_level=1,
    )
    assert result.precedent_alignment is None


@pytest.mark.asyncio
async def test_run_uncertainty_breaker_open_degrades():
    engine = _build_engine(
        config=EngineConfig(enable_precedent_analysis=False, enable_reflection=True),
        uncertainty_engine=SimpleNamespace(),
    )
    engine._get_circuit_breaker = lambda name: StaticBreaker(
        exc=CircuitBreakerOpen("open", 1.0)
    ) if name == "uncertainty" else None
    result = await engine.run(
        "hello",
        context={"skip_router": True, "model_output": "ok"},
        detail_level=1,
    )
    assert "uncertainty" in (result.degraded_components or [])


@pytest.mark.asyncio
async def test_run_uncertainty_error_no_degrade():
    engine = _build_engine(
        config=EngineConfig(
            enable_precedent_analysis=False,
            enable_reflection=True,
            enable_graceful_degradation=False,
        ),
        uncertainty_engine=SimpleNamespace(),
    )
    engine.degradation_enabled = False

    async def _raise(*_args, **_kwargs):
        raise UncertaintyComputationError("boom")

    engine._run_uncertainty_engine = _raise
    result = await engine.run(
        "hello",
        context={"skip_router": True, "model_output": "ok"},
        detail_level=1,
    )
    assert result.uncertainty is None


@pytest.mark.asyncio
async def test_run_governance_generic_exception():
    engine = _build_engine()

    def _raise(*_args, **_kwargs):
        raise ValueError("boom")

    engine._run_governance_review_gate = _raise
    result = await engine.run(
        "hello",
        context={"skip_router": True, "model_output": "ok"},
        detail_level=1,
    )
    assert result.output_text == "ok"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc",
    [DetectorExecutionError("boom"), RuntimeError("boom")],
)
async def test_run_stream_detector_error_branches(exc):
    engine = _build_engine()

    async def _raise(*_args, **_kwargs):
        raise exc

    engine._run_detectors = _raise
    events = await _collect_events(
        engine,
        context={"skip_router": True, "model_output": "ok"},
        detail_level=1,
    )
    assert any(event["event"] == "final_output" for event in events)


@pytest.mark.asyncio
async def test_run_stream_skip_router_serializes_output():
    engine = _build_engine()
    events = await _collect_events(
        engine,
        context={"skip_router": True, "model_output": {"k": "v"}},
        detail_level=1,
    )
    final = [event for event in events if event["event"] == "final_output"][0]
    assert "\"k\"" in final["output_text"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["normal", "breaker_open", "selection_error"])
async def test_run_stream_router_branches(mode):
    engine = _build_engine()
    if mode == "breaker_open":
        engine._get_circuit_breaker = lambda name: StaticBreaker(
            exc=CircuitBreakerOpen("open", 1.0)
        ) if name == "router" else None
    elif mode == "selection_error":
        async def _fail(*_args, **_kwargs):
            raise RouterSelectionError("boom")

        engine._select_model = _fail

    events = await _collect_events(engine, detail_level=1)
    router_event = [event for event in events if event["event"] == "router_selected"][0]
    if mode == "selection_error":
        assert router_event["model_info"]["model_name"] == "router_error"
    else:
        assert router_event["model_info"]["model_name"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    ["normal", "breaker_call", "breaker_open", "error_no_degrade"],
)
async def test_run_stream_precedent_branches(mode):
    engine = _build_engine(
        config=EngineConfig(
            enable_precedent_analysis=True,
            enable_reflection=False,
            enable_circuit_breakers=False,
        ),
        precedent_engine=SimpleNamespace(),
    )
    if mode == "normal":
        async def _ok(*_args, **_kwargs):
            return {"alignment_score": 0.1}

        engine._run_precedent_alignment = _ok
    elif mode == "breaker_call":
        engine._get_circuit_breaker = lambda name: StaticBreaker(
            result={"alignment_score": 0.2}
        ) if name == "precedent" else None
    elif mode == "breaker_open":
        engine._get_circuit_breaker = lambda name: StaticBreaker(
            exc=CircuitBreakerOpen("open", 1.0)
        ) if name == "precedent" else None
    elif mode == "error_no_degrade":
        engine.degradation_enabled = False

        async def _raise(*_args, **_kwargs):
            raise PrecedentRetrievalError("boom")

        engine._run_precedent_alignment = _raise

    events = await _collect_events(
        engine,
        context={"skip_router": True, "model_output": "ok"},
        detail_level=1,
    )
    precedent_event = [event for event in events if event["event"] == "precedent_alignment"][0]
    assert precedent_event["data"] is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    ["breaker_call", "breaker_open", "error_no_degrade"],
)
async def test_run_stream_uncertainty_branches(mode):
    engine = _build_engine(
        config=EngineConfig(
            enable_precedent_analysis=False,
            enable_reflection=True,
            enable_circuit_breakers=False,
        ),
        uncertainty_engine=SimpleNamespace(),
    )
    if mode == "breaker_call":
        engine._get_circuit_breaker = lambda name: StaticBreaker(
            result={"overall_uncertainty": 0.2}
        ) if name == "uncertainty" else None
    elif mode == "breaker_open":
        engine._get_circuit_breaker = lambda name: StaticBreaker(
            exc=CircuitBreakerOpen("open", 1.0)
        ) if name == "uncertainty" else None
    elif mode == "error_no_degrade":
        engine.degradation_enabled = False

        async def _raise(*_args, **_kwargs):
            raise UncertaintyComputationError("boom")

        engine._run_uncertainty_engine = _raise

    events = await _collect_events(
        engine,
        context={"skip_router": True, "model_output": "ok"},
        detail_level=1,
    )
    uncertainty_event = [event for event in events if event["event"] == "uncertainty"][0]
    assert uncertainty_event["data"] is not None


@pytest.mark.asyncio
async def test_run_stream_governance_generic_exception():
    engine = _build_engine()

    def _raise(*_args, **_kwargs):
        raise ValueError("boom")

    engine._run_governance_review_gate = _raise
    events = await _collect_events(
        engine,
        context={"skip_router": True, "model_output": "ok"},
        detail_level=1,
    )
    assert any(event["event"] == "final_output" for event in events)
