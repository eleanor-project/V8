import asyncio
import builtins
import sys
from types import ModuleType, SimpleNamespace

import pytest

import engine.engine as engine_module
from engine.engine import EleanorEngineV8, EngineConfig, EngineForensicData
from engine.exceptions import (
    AggregationError,
    CriticEvaluationError,
    EleanorV8Exception,
    InputValidationError,
    PrecedentRetrievalError,
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


class DummyCache:
    def __init__(self):
        self.set_calls = []

    async def get(self, _key):
        return None

    async def set(self, key, value):
        self.set_calls.append((str(key), value))


class DummyRouterCache:
    def __init__(self):
        self.set_calls = []

    def get_similar(self, _text, _context):
        return None

    def set(self, _text, _context, selection):
        self.set_calls.append(selection)


class AsyncCritic:
    async def evaluate(self, _adapter, input_text, context):
        return {"score": 0.1, "violations": [], "justification": input_text}

    def severity(self, _score):
        return "INFO"


class SyncAwaitCritic:
    def evaluate(self, _adapter, input_text, context):
        async def _inner():
            return {"score": 0.2, "violations": [], "justification": input_text}

        return _inner()


class FailingCritic:
    async def evaluate(self, _adapter, input_text, context):
        raise RuntimeError("boom")


class DummyBreaker:
    async def call(self, _func, *args, **kwargs):
        raise CircuitBreakerOpen("open", 1.0)


def _build_engine(config=None, **overrides):
    deps = EngineDependencies(
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
    return EleanorEngineV8(
        config=config or EngineConfig(enable_precedent_analysis=False, enable_reflection=False),
        dependencies=deps,
    )


def test_engine_init_with_settings_gpu_cache(monkeypatch):
    class DummySettings:
        def __init__(self):
            self.environment = "test"
            self.cache = SimpleNamespace(
                enabled=True,
                redis_url="redis://local",
                precedent_ttl=1,
                embeddings_ttl=2,
                router_ttl=3,
                critics_ttl=4,
                detector_ttl=5,
            )
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
                async_ops=SimpleNamespace(num_streams=2),
                batching=SimpleNamespace(
                    default_batch_size=1, max_batch_size=4, dynamic_batching=True
                ),
                embeddings=SimpleNamespace(
                    cache_on_gpu=True,
                    mixed_precision=False,
                    max_cache_size_gb=0.000001,
                    embedding_dim=2,
                ),
                precedent=SimpleNamespace(cache_embeddings_on_gpu=True),
                multi_gpu=SimpleNamespace(enabled=True, device_ids=[0]),
                critics=SimpleNamespace(gpu_batching=True, use_gpu=True, batch_size=2),
            )

        def to_legacy_engine_config(self):
            return {
                "enable_precedent_analysis": False,
                "enable_reflection": False,
                "enable_adaptive_concurrency": True,
                "max_concurrency": 4,
                "target_latency_ms": 100.0,
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
            self.entries = []

    class DummyMultiGPURouter:
        def __init__(self, *_args, **_kwargs):
            self.ready = True

    class DummyBatchProcessor:
        def __init__(self, *_args, **_kwargs):
            self.ready = True

    redis_module = ModuleType("redis")

    class DummyRedis:
        @classmethod
        def from_url(cls, _url, decode_responses=True):
            return object()

    redis_module.Redis = DummyRedis

    gpu_manager_module = ModuleType("engine.gpu.manager")
    gpu_manager_module.GPUManager = DummyGPUManager
    gpu_manager_module.GPUConfig = DummyGPUConfig

    async_ops_module = ModuleType("engine.gpu.async_ops")
    async_ops_module.AsyncGPUExecutor = DummyAsyncGPUExecutor

    embeddings_module = ModuleType("engine.gpu.embeddings")
    embeddings_module.GPUEmbeddingCache = DummyGPUEmbeddingCache

    parallel_module = ModuleType("engine.gpu.parallelization")
    parallel_module.MultiGPURouter = DummyMultiGPURouter

    batch_module = ModuleType("engine.gpu.batch_processor")
    batch_module.BatchProcessor = DummyBatchProcessor

    monkeypatch.setitem(sys.modules, "redis", redis_module)
    monkeypatch.setitem(sys.modules, "engine.gpu.manager", gpu_manager_module)
    monkeypatch.setitem(sys.modules, "engine.gpu.async_ops", async_ops_module)
    monkeypatch.setitem(sys.modules, "engine.gpu.embeddings", embeddings_module)
    monkeypatch.setitem(sys.modules, "engine.gpu.parallelization", parallel_module)
    monkeypatch.setitem(sys.modules, "engine.gpu.batch_processor", batch_module)
    import engine.config as config_module

    monkeypatch.setattr(config_module, "ConfigManager", DummyConfigManager)

    class DummyRetriever:
        def __init__(self):
            self.embedding_cache = None

    deps = EngineDependencies(
        router=MockRouter(model_name="m", response_text="ok"),
        detector_engine=MockDetectorEngine(),
        evidence_recorder=MockEvidenceRecorder(),
        critics={"mock": MockCritic},
        precedent_engine=None,
        precedent_retriever=DummyRetriever(),
        uncertainty_engine=None,
        aggregator=MockAggregator(),
        review_trigger_evaluator=MockReviewTriggerEvaluator(),
        critic_models=None,
    )

    engine = EleanorEngineV8(config=None, dependencies=deps)
    assert engine.cache_manager is not None
    assert engine.router_cache is not None
    assert engine.gpu_manager is not None
    assert engine.gpu_executor is not None
    assert engine.gpu_embedding_cache is not None
    assert engine.gpu_multi_router is not None
    assert engine.precedent_retriever.embedding_cache is engine.gpu_embedding_cache
    assert engine.adaptive_concurrency is not None
    assert engine.critic_batcher is not None


def test_init_cache_redis_failure(monkeypatch):
    class DummyRedis:
        @classmethod
        def from_url(cls, _url, decode_responses=True):
            raise RuntimeError("boom")

    monkeypatch.setitem(sys.modules, "redis", SimpleNamespace(Redis=DummyRedis))
    assert engine_module._init_cache_redis("redis://local") is None
    assert engine_module._init_cache_redis(None) is None


def test_engine_config_helpers():
    settings = SimpleNamespace(to_legacy_engine_config=lambda: {"detail_level": 1})
    config = EngineConfig.from_settings(settings)
    assert config.detail_level == 1

    data = EngineForensicData()
    assert data.uncertainty_graph == {}
    assert data.precedent_alignment == {}


def test_resolve_dependencies_overrides(monkeypatch):
    deps = EngineDependencies(
        router="router",
        detector_engine="detector",
        evidence_recorder="recorder",
        critics={},
        precedent_engine="prec",
        precedent_retriever="retriever",
        uncertainty_engine="uncertainty",
        aggregator="agg",
        review_trigger_evaluator="review",
        critic_models=None,
    )

    monkeypatch.setattr(engine_module.DependencyFactory, "create_all_dependencies", lambda **_k: deps)
    out = engine_module._resolve_dependencies(
        config=EngineConfig(enable_precedent_analysis=False, enable_reflection=False),
        dependencies=None,
        evidence_recorder="override-recorder",
        detector_engine="override-detector",
        precedent_engine="override-precedent",
        precedent_retriever="override-retriever",
        uncertainty_engine="override-uncertainty",
        aggregator="override-agg",
        router_backend=object(),
        critics=None,
        critic_models=None,
        review_trigger_evaluator="override-review",
    )
    assert out.evidence_recorder == "override-recorder"
    assert out.review_trigger_evaluator == "override-review"


@pytest.mark.asyncio
async def test_shutdown_branches(monkeypatch):
    engine = _build_engine()
    engine.recorder = SimpleNamespace(close=None)
    engine.cache_manager = SimpleNamespace(close=lambda: None)

    class BadCache:
        def clear_cache(self):
            raise RuntimeError("boom")

    engine.gpu_embedding_cache = BadCache()
    engine.gpu_manager = object()

    torch_stub = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda *_a, **_k: True,
            empty_cache=lambda *_a, **_k: None,
        )
    )
    monkeypatch.setitem(sys.modules, "torch", torch_stub)

    async def _fail_close():
        raise RuntimeError("boom")

    import engine.utils.http_client as http_client

    monkeypatch.setattr(http_client, "aclose_async_client", _fail_close)
    await engine.shutdown(timeout=0)


def test_emit_error_and_validation_error(monkeypatch):
    engine = _build_engine()
    exc = EleanorV8Exception("boom", details={"message": "override"})
    engine._emit_error(exc, stage="test")

    monkeypatch.setattr(engine_module.json, "dumps", lambda *_a, **_k: (_ for _ in ()).throw(ValueError("bad")))
    val_exc = InputValidationError("bad", validation_type="x", field="f")
    engine._emit_validation_error(val_exc, text="text", context={"k": "v"}, trace_id="t")


def test_get_circuit_breaker_none():
    engine = _build_engine()
    engine.circuit_breakers = None
    assert engine._get_circuit_breaker("router") is None


@pytest.mark.asyncio
async def test_select_model_sets_caches():
    engine = _build_engine()
    engine.cache_manager = DummyCache()
    engine.router_cache = DummyRouterCache()
    result = await engine._select_model("text", {})
    assert result["response_text"] == "ok"
    assert engine.cache_manager.set_calls
    assert engine.router_cache.set_calls


@pytest.mark.asyncio
async def test_run_single_critic_timeout_zero_and_cache():
    config = EngineConfig(
        enable_precedent_analysis=False,
        enable_reflection=False,
        enable_adaptive_concurrency=True,
        timeout_seconds=0,
    )
    engine = _build_engine(config=config, critics={"async": AsyncCritic})
    engine.cache_manager = DummyCache()
    result = await engine._run_single_critic("async", AsyncCritic, "resp", "in", {}, "t")
    assert result["duration_ms"] >= 0
    assert engine.cache_manager.set_calls

    engine = _build_engine(config=config, critics={"sync": SyncAwaitCritic})
    result = await engine._run_single_critic("sync", SyncAwaitCritic, "resp", "in", {}, "t")
    assert result["justification"] == "in"


@pytest.mark.asyncio
async def test_run_single_critic_error_records_latency():
    config = EngineConfig(
        enable_precedent_analysis=False,
        enable_reflection=False,
        enable_adaptive_concurrency=True,
        timeout_seconds=0,
    )
    engine = _build_engine(config=config, critics={"fail": FailingCritic})
    with pytest.raises(CriticEvaluationError):
        await engine._run_single_critic("fail", FailingCritic, "resp", "in", {}, "t")


@pytest.mark.asyncio
async def test_run_single_critic_with_breaker_raises():
    engine = _build_engine()
    engine.degradation_enabled = False
    engine._get_circuit_breaker = lambda _name: DummyBreaker()
    with pytest.raises(CircuitBreakerOpen):
        await engine._run_single_critic_with_breaker("mock", MockCritic, "resp", "in", {}, "t")


@pytest.mark.asyncio
async def test_run_precedent_alignment_sync_paths(monkeypatch):
    class SyncRetriever:
        def retrieve(self, *_args, **_kwargs):
            async def _inner():
                return {"precedent_cases": [{"id": 1}], "query_embedding": [0.1]}

            return _inner()

    engine = _build_engine(
        config=EngineConfig(enable_precedent_analysis=True, enable_reflection=False),
        precedent_engine=SimpleNamespace(),
        precedent_retriever=SyncRetriever(),
    )
    out = await engine._run_precedent_alignment({"crit": {"score": 0.1}}, "t", text="q")
    assert out["retrieval"]["query_embedding"] == [0.1]


@pytest.mark.asyncio
async def test_run_precedent_alignment_cache_manager_path():
    class SyncRetriever:
        def retrieve(self, *_args, **_kwargs):
            async def _inner():
                return {"precedent_cases": [{"id": 1}], "query_embedding": [0.2]}

            return _inner()

    engine = _build_engine(
        config=EngineConfig(enable_precedent_analysis=True, enable_reflection=False),
        precedent_engine=SimpleNamespace(),
        precedent_retriever=SyncRetriever(),
    )
    engine.cache_manager = DummyCache()
    out = await engine._run_precedent_alignment({"crit": {"score": 0.1}}, "t", text="q")
    assert out["retrieval"]["query_embedding"] == [0.2]


def test_governance_helpers():
    engine = _build_engine()
    critic_outputs = {"a": {"severity": "bad"}, "b": "bad"}
    assert engine._calculate_critic_disagreement(critic_outputs) == 0.0
    assert engine._collect_uncertainty_flags({"overall_uncertainty": "bad"}) == []

    case = engine._build_case_for_review(
        trace_id="t",
        context={},
        aggregated={"score": {"average_severity": "bad"}},
        critic_results={"a": {"severity": "bad"}, "b": "bad"},
        precedent_data=None,
        uncertainty_data=None,
    )
    assert case.severity == 0.0


@pytest.mark.asyncio
async def test_run_detector_error_and_skip_router_value_error(monkeypatch):
    engine = _build_engine()

    async def _raise(*_args, **_kwargs):
        raise RuntimeError("boom")

    engine._run_detectors = _raise

    class DummyValidated:
        def __init__(self):
            self.text = "hi"
            self.context = {"skip_router": True}
            self.trace_id = "t"

    monkeypatch.setattr(engine_module, "validate_input", lambda *_a, **_k: DummyValidated())
    with pytest.raises(ValueError):
        await engine.run("hi")


@pytest.mark.asyncio
async def test_run_gpu_context_injection():
    engine = _build_engine()
    engine.gpu_manager = SimpleNamespace(device="cuda:0")
    engine.gpu_enabled = True
    result = await engine.run(
        "hello",
        context={"skip_router": True, "model_output": "ok"},
        detail_level=1,
    )
    assert result.output_text == "ok"


@pytest.mark.asyncio
async def test_run_invalid_detail_level(monkeypatch):
    engine = _build_engine()

    def _bad_validate(*_args, **_kwargs):
        return "hi", {"skip_router": True, "model_output": "ok"}, "t", 99

    monkeypatch.setattr(engine, "_validate_inputs", _bad_validate)
    with pytest.raises(ValueError):
        await engine.run("hi", detail_level=99)


@pytest.mark.asyncio
async def test_run_precedent_uncertainty_and_governance_errors():
    engine = _build_engine(
        config=EngineConfig(enable_precedent_analysis=True, enable_reflection=True),
        review_trigger_evaluator=SimpleNamespace(
            evaluate=lambda _c: (_ for _ in ()).throw(RuntimeError("boom"))
        ),
        uncertainty_engine=SimpleNamespace(),
    )
    engine.circuit_breakers = None

    async def _prec_fail(*_args, **_kwargs):
        raise PrecedentRetrievalError("boom")

    async def _uncertainty_fail(*_args, **_kwargs):
        raise UncertaintyComputationError("boom")

    engine._run_precedent_alignment = _prec_fail
    engine._run_uncertainty_engine = _uncertainty_fail
    result = await engine.run("hello")
    assert result.is_degraded is True


@pytest.mark.asyncio
async def test_run_precedent_uncertainty_success():
    engine = _build_engine(
        config=EngineConfig(enable_precedent_analysis=True, enable_reflection=True),
        uncertainty_engine=SimpleNamespace(),
    )
    engine.circuit_breakers = None

    async def _prec_ok(*_args, **_kwargs):
        return {"alignment_score": 0.1}

    async def _uncertainty_ok(*_args, **_kwargs):
        return {"overall_uncertainty": 0.1}

    engine._run_precedent_alignment = _prec_ok
    engine._run_uncertainty_engine = _uncertainty_ok
    result = await engine.run("hello")
    assert result.uncertainty is not None


@pytest.mark.asyncio
async def test_run_stream_batcher_and_error_paths(monkeypatch):
    engine = _build_engine(
        config=EngineConfig(enable_precedent_analysis=False, enable_reflection=False),
    )
    engine.critics = {"a": MockCritic, "b": MockCritic, "c": MockCritic}
    async def _process_batch(_items):
        return [
            CriticEvaluationError("a", "boom", "t", details={"result": {"critic": "a"}}),
            RuntimeError("oops"),
            {"critic": "c", "violations": [], "duration_ms": 1.0},
        ]

    engine.critic_batcher = SimpleNamespace(process_batch=_process_batch)

    def _validated(_text, _context, _trace_id, _detail_level):
        return (
            "hello",
            {"skip_router": True, "model_output": "ok", "input_text_override": 123},
            "t",
            2,
        )

    monkeypatch.setattr(engine, "_validate_inputs", _validated)

    events = []
    async for event in engine.run_stream(
        "hello",
        context={"skip_router": True, "model_output": "ok", "input_text_override": 123},
        detail_level=2,
    ):
        events.append(event)

    assert any(e["event"] == "critic_result" for e in events)


@pytest.mark.asyncio
async def test_run_stream_precedent_uncertainty_aggregation_errors(monkeypatch):
    engine = _build_engine(
        config=EngineConfig(enable_precedent_analysis=True, enable_reflection=True),
        review_trigger_evaluator=SimpleNamespace(
            evaluate=lambda _c: (_ for _ in ()).throw(RuntimeError("boom"))
        ),
        uncertainty_engine=SimpleNamespace(),
        precedent_engine=SimpleNamespace(),
    )
    engine.circuit_breakers = None

    async def _prec_fail(*_args, **_kwargs):
        raise PrecedentRetrievalError("boom")

    async def _uncertainty_fail(*_args, **_kwargs):
        raise UncertaintyComputationError("boom")

    async def _agg_fail(*_args, **_kwargs):
        raise AggregationError("boom")

    engine._run_precedent_alignment = _prec_fail
    engine._run_uncertainty_engine = _uncertainty_fail
    engine._aggregate_results = _agg_fail

    events = []
    async for event in engine.run_stream(
        "hello",
        context={"skip_router": True, "model_output": "ok"},
        detail_level=1,
    ):
        events.append(event)

    assert any(e["event"] == "aggregation" for e in events)


@pytest.mark.asyncio
async def test_run_stream_task_errors():
    engine = _build_engine()
    engine.critics = {"a": MockCritic, "b": MockCritic}

    async def _raise(name, *_args, **_kwargs):
        if name == "a":
            raise CriticEvaluationError("a", "boom", "t", details={})
        raise RuntimeError("oops")

    engine._run_single_critic_with_breaker = _raise

    events = []
    async for event in engine.run_stream(
        "hello",
        context={"skip_router": True, "model_output": "ok"},
        detail_level=1,
    ):
        events.append(event)

    assert any(e["event"] == "critic_result" for e in events)


@pytest.mark.asyncio
async def test_run_stream_final_output_fallback_and_invalid_level(monkeypatch):
    engine = _build_engine()

    async def _agg_empty(*_args, **_kwargs):
        return {"final_output": ""}

    engine._aggregate_results = _agg_empty
    events = []
    async for event in engine.run_stream(
        "hello",
        context={"skip_router": True, "model_output": "ok"},
        detail_level=1,
    ):
        events.append(event)
    final = [e for e in events if e["event"] == "final_output"][0]
    assert final["output_text"] == "ok"

    def _bad_validate(*_args, **_kwargs):
        return "hi", {"skip_router": True, "model_output": "ok"}, "t", 99

    monkeypatch.setattr(engine, "_validate_inputs", _bad_validate)
    with pytest.raises(ValueError):
        async for _ in engine.run_stream("hi", detail_level=99):
            pass


@pytest.mark.asyncio
async def test_run_stream_gpu_context():
    engine = _build_engine()
    engine.gpu_manager = SimpleNamespace(device="cuda:0")
    engine.gpu_enabled = True

    events = []
    async for event in engine.run_stream(
        "hello",
        context={"skip_router": True, "model_output": "ok"},
        detail_level=1,
    ):
        events.append(event)
    assert any(e["event"] == "final_output" for e in events)


@pytest.mark.asyncio
async def test_run_stream_missing_model_output(monkeypatch):
    engine = _build_engine()

    def _bad_validate(*_args, **_kwargs):
        return "hi", {"skip_router": True}, "t", 1

    monkeypatch.setattr(engine, "_validate_inputs", _bad_validate)
    with pytest.raises(ValueError):
        async for _ in engine.run_stream("hi"):
            pass


def test_load_config_from_yaml_import_error(monkeypatch, tmp_path):
    original_import = __import__

    def _fake_import(name, *args, **kwargs):
        if name == "yaml":
            raise ImportError("boom")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(ImportError):
        EleanorEngineV8.load_config_from_yaml(str(tmp_path / "config.yaml"))


def test_load_config_from_yaml_success(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("detail_level: 1\n", encoding="utf-8")

    yaml_stub = SimpleNamespace(safe_load=lambda _f: {"detail_level": 1})
    monkeypatch.setitem(sys.modules, "yaml", yaml_stub)
    config = EleanorEngineV8.load_config_from_yaml(str(config_path))
    assert config.detail_level == 1
