import asyncio

import pytest

from engine.engine import EleanorEngineV8, EngineConfig
from engine.exceptions import (
    AggregationError,
    CriticEvaluationError,
    DetectorExecutionError,
    GovernanceEvaluationError,
    InputValidationError,
    PrecedentRetrievalError,
    RouterSelectionError,
    UncertaintyComputationError,
)
from engine.factory import EngineDependencies
from engine.utils.circuit_breaker import CircuitBreakerOpen


class DummyCache:
    def __init__(self):
        self.values = {}
        self.set_calls = []

    async def get(self, key):
        return self.values.get(str(key))

    async def set(self, key, value):
        self.values[str(key)] = value
        self.set_calls.append(str(key))


class DummySignal:
    def model_dump(self):
        return {"severity": 0.1}


class DummyDetectorEngine:
    def __init__(self, should_raise=False):
        self.should_raise = should_raise
        self.calls = 0

    async def detect_all(self, text, context):
        self.calls += 1
        if self.should_raise:
            raise RuntimeError("detector boom")
        return {"det1": DummySignal()}

    def aggregate_signals(self, signals):
        return {"summary": "ok", "score": 0.1}


class DummyRouter:
    def __init__(self, result=None, should_raise=None):
        self.result = result
        self.should_raise = should_raise
        self.calls = 0

    def route(self, text, context=None):
        self.calls += 1
        if self.should_raise:
            raise self.should_raise
        return self.result


class DummyRecorder:
    def __init__(self, should_raise=False):
        self.should_raise = should_raise
        self.records = []

    async def record(self, **kwargs):
        if self.should_raise:
            raise RuntimeError("record boom")
        self.records.append(kwargs)
        return kwargs


class AdapterAwareCritic:
    async def evaluate(self, model_adapter, input_text, context):
        response = await model_adapter.generate("prompt", context=context)
        return {
            "score": 0.3,
            "severity": 0.3,
            "violations": [response],
            "justification": "ok",
        }

    def severity(self, score):
        return "WARNING"


class SyncCritic:
    def evaluate(self, model_adapter, input_text, context):
        return {"score": 0.1, "violations": [], "justification": "sync"}


class SyncAwaitableCritic:
    def evaluate(self, model_adapter, input_text, context):
        async def _inner():
            return {"score": 0.2, "violations": ["awaited"], "justification": "async"}

        return _inner()


class FailingCritic:
    async def evaluate(self, model_adapter, input_text, context):
        raise RuntimeError("critic boom")


class DummyAggregator:
    def __init__(self, should_raise=False):
        self.should_raise = should_raise

    def aggregate(self, critics, precedent, uncertainty, model_output=""):
        if self.should_raise:
            raise RuntimeError("agg boom")
        return {
            "decision": "allow",
            "final_output": model_output,
            "score": {"average_severity": 0.4, "total_severity": 0.4},
            "rights_impacted": ["r1"],
            "dissent": None,
            "precedent": precedent,
            "uncertainty": uncertainty,
        }


class DummyPrecedentEngine:
    def __init__(self, should_raise=False, with_analyze=True):
        self.should_raise = should_raise
        self.with_analyze = with_analyze

    def analyze(self, critics, precedent_cases, query_embedding):
        if self.should_raise:
            raise RuntimeError("analysis boom")
        return {"alignment_score": 0.2, "novel": False}


class DummyRetriever:
    def __init__(self, result=None, should_raise=False, async_mode=False):
        self.result = result or {"precedent_cases": [], "query_embedding": [0.1, 0.2]}
        self.should_raise = should_raise
        self.async_mode = async_mode
        self.calls = 0

    async def retrieve(self, text, critic_results):
        self.calls += 1
        if self.should_raise:
            raise RuntimeError("retrieve boom")
        return self.result


class DummyUncertaintyEngine:
    def __init__(self, should_raise=False, async_mode=False, has_compute=True):
        self.should_raise = should_raise
        self.async_mode = async_mode
        self.has_compute = has_compute

    async def compute(self, critics, model_used, precedent_alignment):
        if self.should_raise:
            raise RuntimeError("uncertainty boom")
        return {"overall_uncertainty": 0.2, "needs_escalation": False}

    def evaluate(self, critics, model_used, precedent_alignment):
        if self.should_raise:
            raise RuntimeError("uncertainty boom")
        return {"overall_uncertainty": 0.8, "needs_escalation": True}


class DummyReviewEvaluator:
    def __init__(self, decision=None, should_raise=False):
        self.decision = decision or {"review_required": False, "triggers": []}
        self.should_raise = should_raise

    def evaluate(self, case):
        if self.should_raise:
            raise RuntimeError("review boom")
        return self.decision


class DummyBreaker:
    async def call(self, func, *args, **kwargs):
        raise CircuitBreakerOpen("dummy", 1.0)


class DummyBatcher:
    def __init__(self, results):
        self.results = results

    async def process_batch(self, items):
        return self.results


class DummyAdapter:
    async def generate(self, prompt, context=None):
        return f"adapter:{prompt}"


def build_engine(
    *,
    router=None,
    detector_engine=None,
    recorder=None,
    critics=None,
    precedent_engine=None,
    precedent_retriever=None,
    uncertainty_engine=None,
    aggregator=None,
    review_trigger_evaluator=None,
    critic_models=None,
    error_monitor=None,
    config=None,
):
    deps = EngineDependencies(
        router=router or DummyRouter(
            result={"response_text": "ok", "model_name": "m", "model_version": "1"}
        ),
        detector_engine=detector_engine or DummyDetectorEngine(),
        evidence_recorder=recorder or DummyRecorder(),
        critics=critics or {"crit": AdapterAwareCritic},
        precedent_engine=precedent_engine,
        precedent_retriever=precedent_retriever,
        uncertainty_engine=uncertainty_engine,
        aggregator=aggregator or DummyAggregator(),
        review_trigger_evaluator=review_trigger_evaluator or DummyReviewEvaluator(),
        critic_models=critic_models,
    )
    return EleanorEngineV8(
        config=config or EngineConfig(enable_precedent_analysis=False, enable_reflection=False),
        dependencies=deps,
        error_monitor=error_monitor,
    )


@pytest.mark.asyncio
async def test_run_detectors_cache_and_error():
    engine = build_engine()
    engine.cache_manager = DummyCache()

    timings = {}
    out = await engine._run_detectors("text", {}, timings)
    assert out["summary"] == "ok"
    assert "detectors_ms" in timings
    assert engine.detector_engine.calls == 1

    cached = await engine._run_detectors("text", {}, {})
    assert cached["summary"] == "ok"
    assert engine.detector_engine.calls == 1

    engine_error = build_engine(detector_engine=DummyDetectorEngine(should_raise=True))
    with pytest.raises(DetectorExecutionError):
        await engine_error._run_detectors("text", {}, {})


@pytest.mark.asyncio
async def test_select_model_cache_similar_and_errors():
    selection = {"model_info": {"model_name": "a"}, "response_text": "cached"}
    router = DummyRouter(result={"response_text": "ok", "model_name": "a"})
    engine = build_engine(router=router)
    engine.cache_manager = DummyCache()
    engine.cache_manager.values["router:cached"] = selection

    async def fake_get(key):
        return selection

    engine.cache_manager.get = fake_get
    out = await engine._select_model("text", {})
    assert out["response_text"] == "cached"
    assert router.calls == 0

    class DummyRouterCache:
        def __init__(self, value):
            self.value = value
            self.set_called = False

        def get_similar(self, text, context):
            return self.value

        def set(self, text, context, value):
            self.set_called = True

    engine = build_engine(router=DummyRouter(result={"response_text": "ok", "model_name": "a"}))
    engine.cache_manager = DummyCache()
    engine.router_cache = DummyRouterCache(selection)
    out = await engine._select_model("text", {})
    assert out["response_text"] == "cached"
    assert engine.cache_manager.set_calls

    engine = build_engine(router=DummyRouter(result={"response_text": None}))
    with pytest.raises(RouterSelectionError):
        await engine._select_model("text", {})

    engine = build_engine(router=DummyRouter(should_raise=ValueError("boom")))
    with pytest.raises(RouterSelectionError):
        await engine._select_model("text", {})


@pytest.mark.asyncio
async def test_run_single_critic_cache_and_adapter_paths():
    engine = build_engine()
    engine.cache_manager = DummyCache()
    cache_key = "critic:cached"

    async def fake_get(key):
        return {"score": 0.1, "violations": []}

    engine.cache_manager.get = fake_get
    out = await engine._run_single_critic("crit", AdapterAwareCritic, "resp", "in", {}, "t")
    assert out["duration_ms"] == 0.0

    def bound_fn(prompt, context=None):
        return f"bound:{prompt}:{context.get('k')}"

    engine = build_engine(critic_models={"crit": bound_fn})
    result = await engine._run_single_critic(
        "crit",
        AdapterAwareCritic,
        "resp",
        "in",
        {"k": "v"},
        "t",
    )
    assert result["violations"][0].startswith("bound:prompt")

    engine = build_engine(critic_models={"crit": DummyAdapter()})
    result = await engine._run_single_critic(
        "crit",
        AdapterAwareCritic,
        "resp",
        "in",
        {},
        "t",
    )
    assert "adapter:prompt" in result["violations"][0]

    engine = build_engine(critic_models={"crit": object()})
    result = await engine._run_single_critic(
        "crit",
        AdapterAwareCritic,
        "resp",
        "in",
        {},
        "t",
    )
    assert result["violations"][0] == "resp"

    engine = build_engine(critic_models={"crit": DummyAdapter()})
    result = await engine._run_single_critic(
        "crit",
        AdapterAwareCritic,
        "resp",
        "in",
        {"force_model_output": True},
        "t",
    )
    assert result["violations"][0] == "resp"


@pytest.mark.asyncio
async def test_run_single_critic_sync_and_error_paths():
    engine = build_engine()
    result = await engine._run_single_critic("crit", SyncCritic, "resp", "in", {}, "t")
    assert result["justification"] == "sync"

    result = await engine._run_single_critic(
        "crit", SyncAwaitableCritic, "resp", "in", {}, "t"
    )
    assert "awaited" in result["violations"]

    errors = []

    def error_monitor(exc, payload):
        errors.append(payload)
        raise RuntimeError("monitor boom")

    engine = build_engine(recorder=DummyRecorder(should_raise=True), error_monitor=error_monitor)
    with pytest.raises(CriticEvaluationError):
        await engine._run_single_critic("crit", FailingCritic, "resp", "in", {}, "t")
    assert errors

    engine = build_engine(recorder=DummyRecorder(should_raise=True), error_monitor=error_monitor)
    result = await engine._run_single_critic("crit", SyncCritic, "resp", "in", {}, "t")
    assert result["justification"] == "sync"


@pytest.mark.asyncio
async def test_run_single_critic_with_breaker_open():
    engine = build_engine()

    def fake_breaker(name):
        return DummyBreaker()

    engine._get_circuit_breaker = fake_breaker
    degraded = []
    result = await engine._run_single_critic_with_breaker(
        "crit", SyncCritic, "resp", "in", {}, "t", degraded_components=degraded
    )
    assert result.get("degraded") is True
    assert "critic:crit" in degraded


@pytest.mark.asyncio
async def test_run_critics_parallel_with_batcher_and_errors():
    engine = build_engine(critics={"a": SyncCritic, "b": SyncCritic, "c": SyncCritic})

    crit_error = CriticEvaluationError(
        critic_name="a",
        message="boom",
        trace_id="t",
        details={"result": {"critic": "a", "violations": []}},
    )
    engine.critic_batcher = DummyBatcher([crit_error, RuntimeError("oops"), "bad"])
    out = await engine._run_critics_parallel("resp", {"input_text_override": 123}, "t")
    assert "a" in out
    assert "b" in out
    assert "c" in out


@pytest.mark.asyncio
async def test_run_precedent_alignment_cache_and_errors():
    retriever = DummyRetriever(result={"precedent_cases": [{"id": 1}], "query_embedding": [0.5]})
    engine = build_engine(
        precedent_engine=DummyPrecedentEngine(),
        precedent_retriever=retriever,
        config=EngineConfig(enable_precedent_analysis=True, enable_reflection=False),
    )
    engine.cache_manager = DummyCache()
    results = {"crit": {"score": 0.1}}
    out = await engine._run_precedent_alignment(results, "t", text="q", timings={})
    assert out["retrieval"]["precedent_cases"]
    assert retriever.calls == 1

    out_cached = await engine._run_precedent_alignment(results, "t", text="q", timings={})
    assert out_cached["retrieval"]["precedent_cases"]

    engine = build_engine(
        precedent_engine=DummyPrecedentEngine(),
        precedent_retriever=DummyRetriever(should_raise=True),
        config=EngineConfig(enable_precedent_analysis=True, enable_reflection=False),
    )
    with pytest.raises(PrecedentRetrievalError):
        await engine._run_precedent_alignment(results, "t", text="q")


@pytest.mark.asyncio
async def test_run_uncertainty_engine_variants():
    engine = build_engine(
        uncertainty_engine=DummyUncertaintyEngine(),
        config=EngineConfig(enable_precedent_analysis=False, enable_reflection=True),
    )
    out = await engine._run_uncertainty_engine({}, {"crit": {"score": 0.1}}, "m")
    assert out["overall_uncertainty"] == 0.2

    class EvalOnly:
        def evaluate(self, critics, model_used, precedent_alignment):
            return {"overall_uncertainty": 0.8, "needs_escalation": True}

    engine = build_engine(
        uncertainty_engine=EvalOnly(),
        config=EngineConfig(enable_precedent_analysis=False, enable_reflection=True),
    )
    out = await engine._run_uncertainty_engine({}, {"crit": {"score": 0.1}}, "m")
    assert out["needs_escalation"] is True

    engine = build_engine(
        uncertainty_engine=DummyUncertaintyEngine(should_raise=True),
        config=EngineConfig(enable_precedent_analysis=False, enable_reflection=True),
    )
    with pytest.raises(UncertaintyComputationError):
        await engine._run_uncertainty_engine({}, {"crit": {"score": 0.1}}, "m")

    class NoCompute:
        pass

    engine = build_engine(
        uncertainty_engine=NoCompute(),
        config=EngineConfig(enable_precedent_analysis=False, enable_reflection=True),
    )
    assert await engine._run_uncertainty_engine({}, {"crit": {"score": 0.1}}, "m") is None


@pytest.mark.asyncio
async def test_aggregate_results_errors():
    engine = build_engine()
    engine.aggregator = None
    with pytest.raises(AggregationError):
        await engine._aggregate_results({"crit": {}}, "out")

    engine = build_engine(aggregator=DummyAggregator(should_raise=True))
    with pytest.raises(AggregationError):
        await engine._aggregate_results({"crit": {}}, "out")


def test_case_building_and_governance_gate(monkeypatch):
    engine = build_engine()
    critic_results = {"a": {"severity": 0.8, "precedent_refs": ["p1"]}}
    aggregated = {"score": {"average_severity": "bad"}, "rights_impacted": []}
    uncertainty = {"needs_escalation": True, "overall_uncertainty": 0.9}
    case = engine._build_case_for_review("trace", {"domain": "d"}, aggregated, critic_results, {}, uncertainty)
    assert case.severity == 0.8
    assert "high_overall_uncertainty" in case.uncertainty_flags

    review_calls = []

    def fake_build(case_obj, decision):
        review_calls.append((case_obj, decision))
        return {"case_id": "trace"}

    def fake_store(packet):
        review_calls.append(packet)
        return "path"

    engine.review_trigger_evaluator = DummyReviewEvaluator(
        decision={"review_required": True, "triggers": ["t"]}
    )
    monkeypatch.setattr("engine.engine.build_review_packet", fake_build)
    monkeypatch.setattr("engine.engine.store_review_packet", fake_store)
    engine._run_governance_review_gate(case)
    assert case.governance_flags["human_review_required"] is True
    assert review_calls

    engine.review_trigger_evaluator = DummyReviewEvaluator(should_raise=True)
    with pytest.raises(GovernanceEvaluationError):
        engine._run_governance_review_gate(case)


def test_governance_flag_logging(monkeypatch):
    import engine.runtime.governance as governance
    from engine.runtime.governance import apply_governance_flags_to_aggregation

    aggregated = {"decision": "allow"}
    flags = {"human_review_required": True, "review_triggers": ["severity_threshold_exceeded"]}
    calls = []
    monkeypatch.setattr(governance.logger, "info", lambda msg, extra=None: calls.append((msg, extra)))
    result = apply_governance_flags_to_aggregation(aggregated, flags, trace_id="trace-logging")

    assert result["governance_flags"] == flags
    assert "human_review_required" in result
    assert any(call[0] == "governance_human_review_required" for call in calls)


@pytest.mark.asyncio
async def test_run_skip_router_and_aggregation_fallback():
    engine = build_engine(aggregator=DummyAggregator(should_raise=True))
    result = await engine.run(
        "hello",
        context={"skip_router": True, "model_output": {"k": "v"}},
        detail_level=2,
    )
    assert result.aggregated["decision"] == "requires_human_review"
    assert result.is_degraded is False


@pytest.mark.asyncio
async def test_run_router_error_and_breaker_open():
    router = DummyRouter(should_raise=RouterSelectionError("nope"))
    engine = build_engine(router=router)
    result = await engine.run("hello")
    assert result.model_info.model_name == "router_error"

    engine = build_engine(router=DummyRouter(result={"response_text": "ok", "model_name": "m"}))

    def breaker(name):
        return DummyBreaker() if name == "router" else None

    engine._get_circuit_breaker = breaker
    result = await engine.run("hello")
    assert result.is_degraded is True
    assert "router" in result.degraded_components


@pytest.mark.asyncio
async def test_run_stream_detail_levels_with_errors():
    engine = build_engine(critics={"good": SyncCritic, "bad": FailingCritic})
    events = []
    async for event in engine.run_stream(
        "hello",
        context={"skip_router": True, "model_output": "ok"},
        detail_level=3,
    ):
        events.append(event)

    assert any(e["event"] == "critic_result" for e in events)
    final_event = [e for e in events if e["event"] == "final_output"][0]
    assert "router_diagnostics" in final_event

    events = []
    async for event in engine.run_stream(
        "hello",
        context={"skip_router": True, "model_output": "ok"},
        detail_level=1,
    ):
        events.append(event)

    final_event = [e for e in events if e["event"] == "final_output"][0]
    assert "critic_findings" not in final_event


@pytest.mark.asyncio
async def test_run_rejects_invalid_detail_level():
    engine = build_engine()
    with pytest.raises(InputValidationError):
        await engine.run("hello", detail_level=99)
