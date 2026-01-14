"""  
Legacy critic execution pipeline.

NOTE: This module is retained for backward compatibility and targeted testing,
but the production runtime should execute critics via OrchestratorV2
(engine.runtime.critic_infrastructure.run_critics_with_orchestrator).

To prevent accidental use in production, run_critics_parallel is guarded by
ELEANOR_ENFORCE_SINGLE_CRITIC_PIPELINE (default: true). To explicitly allow
legacy execution (tests/dev only), set ELEANOR_ALLOW_LEGACY_CRITIC_RUNNER=1.
"""

import asyncio
import os
import warnings
import inspect
import logging
from typing import Any, Dict, List, Optional, cast, TYPE_CHECKING

if TYPE_CHECKING:
    from engine.types.engine_types import EngineType, CriticRef
    from engine.schemas.pipeline_types import CriticResult, CriticResultsMap

from engine.cache import CacheKey
from engine.exceptions import CriticEvaluationError, EvidenceRecordingError
from engine.resilience.degradation import DegradationStrategy
from engine.schemas.pipeline_types import CriticResult, CriticResultsMap
from engine.utils.circuit_breaker import CircuitBreakerOpen

# Enhanced observability
try:
    from engine.observability.tracing import run_critic_with_trace
    from engine.observability.correlation import get_correlation_id
    from engine.observability.business_metrics import record_critic_agreement
    TRACING_AVAILABLE = True
except ImportError:
    TRACING_AVAILABLE = False
    run_critic_with_trace = None
    get_correlation_id = None
    record_critic_agreement = None

# Event bus
try:
    from engine.events.event_bus import get_event_bus, CriticEvaluatedEvent, EventType
    EVENT_BUS_AVAILABLE = True
except ImportError:
    EVENT_BUS_AVAILABLE = False
    get_event_bus = None
    CriticEvaluatedEvent = None
    EventType = None

logger = logging.getLogger("engine.engine")


def _process_batch_with_engine(batcher: Any, batch_items: List[Any], engine: Any) -> Any:
    """Call process_batch, passing engine only if supported."""
    try:
        sig = inspect.signature(batcher.process_batch)
        if "engine" in sig.parameters:
            return batcher.process_batch(batch_items, engine=engine)
    except Exception:
        pass
    return batcher.process_batch(batch_items)


async def run_single_critic(
    engine: "EngineType",
    name: str,
    critic_ref: "CriticRef",
    model_response: str,
    input_text: str,
    context: dict,
    trace_id: str,
    evidence_records: Optional[List[Any]] = None,
) -> CriticResult:
    cache_key: Optional[CacheKey] = None
    if engine.cache_manager:
        cache_key = CacheKey.from_data(
            "critic",
            critic=name,
            input_text=input_text,
            model_response=model_response,
            context=context,
        )
        cached = await engine.cache_manager.get(cache_key)
        if cached is not None:
            cached_result = dict(cast(CriticResult, cached))
            cached_result.setdefault("critic", name)
            cached_result["duration_ms"] = 0.0
            return cast(CriticResult, cached_result)

    async with engine.semaphore:
        start = asyncio.get_event_loop().time()

        critic = critic_ref if not inspect.isclass(critic_ref) else critic_ref()

        bound_adapter = None if context.get("force_model_output") else engine.critic_models.get(name)
        model_adapter: Any = None

        if bound_adapter is None:

            class _StaticModelResponse:
                def __init__(self, response: str):
                    self.response = response

                async def generate(self, prompt: str, context: Optional[Dict[str, Any]] = None):
                    return self.response

            model_adapter = _StaticModelResponse(model_response)
        else:
            if hasattr(bound_adapter, "generate"):
                model_adapter = bound_adapter
            elif callable(bound_adapter):

                class _BoundCallable:
                    def __init__(self, fn):
                        self.fn = fn

                    async def generate(
                        self, prompt: str, context: Optional[Dict[str, Any]] = None
                    ):
                        res = (
                            self.fn(prompt, context=context)
                            if "context" in inspect.signature(self.fn).parameters
                            else self.fn(prompt)
                        )
                        return await res if inspect.isawaitable(res) else res

                model_adapter = _BoundCallable(bound_adapter)
            else:

                class _StaticModelFallback:
                    def __init__(self, response: str):
                        self.response = response

                    async def generate(
                        self, prompt: str, context: Optional[Dict[str, Any]] = None
                    ):
                        return self.response

                model_adapter = _StaticModelFallback(model_response)

        timeout = engine.config.timeout_seconds
        try:
            evaluate_fn = critic.evaluate
            if inspect.iscoroutinefunction(evaluate_fn):
                evaluation = evaluate_fn(model_adapter, input_text=input_text, context=context)
                if timeout and timeout > 0:
                    evaluation_result = await asyncio.wait_for(evaluation, timeout=timeout)
                else:
                    evaluation_result = await evaluation
            else:
                call = asyncio.to_thread(
                    evaluate_fn, model_adapter, input_text=input_text, context=context
                )
                if timeout and timeout > 0:
                    evaluation_result = await asyncio.wait_for(call, timeout=timeout)
                else:
                    evaluation_result = await call
                if inspect.isawaitable(evaluation_result):
                    if timeout and timeout > 0:
                        evaluation_result = await asyncio.wait_for(evaluation_result, timeout=timeout)
                    else:
                        evaluation_result = await evaluation_result
        except Exception as exc:
            end = asyncio.get_event_loop().time()
            duration_ms = (end - start) * 1000
            if engine.adaptive_concurrency:
                engine.adaptive_concurrency.record_latency(duration_ms)
            # Propagate timeouts directly for callers that expect them
            if isinstance(exc, asyncio.TimeoutError):
                raise
            failure_result = engine._build_critic_error_result(
                critic_name=name,
                error=exc,
                duration_ms=duration_ms,
            )
            record_failed = False
            try:
                record = await engine.recorder.record(
                    critic=name,
                    rule_id=str(name),
                    severity="INFO",
                    violation_description=str(failure_result.get("justification")),
                    confidence=0.0,
                    mitigation=None,
                    redundancy_group=None,
                    detector_metadata={"error": str(exc)},
                    context=context,
                    raw_text=model_response,
                    trace_id=trace_id,
                )
                if evidence_records is not None:
                    evidence_records.append(record)
            except Exception as record_exc:
                error = EvidenceRecordingError(
                    "Evidence recording failed",
                    details={"error": str(record_exc), "critic": name},
                )
                engine._emit_error(error, stage="evidence", trace_id=trace_id, critic=name)
                record_failed = True

            critic_error = CriticEvaluationError(
                critic_name=name,
                message=str(exc),
                trace_id=trace_id,
                details={"result": failure_result},
            )
            engine._emit_error(
                critic_error,
                stage="critic",
                trace_id=trace_id,
                critic=name,
                context=context,
            )
            if record_failed or engine.adaptive_concurrency is not None or getattr(engine, "error_monitor", None) is not None:
                raise critic_error
            return failure_result

        end = asyncio.get_event_loop().time()

        evaluation_result = cast(CriticResult, evaluation_result or {})
        evaluation_result["critic"] = name
        duration_ms = (end - start) * 1000
        evaluation_result["duration_ms"] = duration_ms
        if engine.adaptive_concurrency:
            engine.adaptive_concurrency.record_latency(duration_ms)
        
        # Publish event if event bus available
        if EVENT_BUS_AVAILABLE and get_event_bus and EventType is not None:
            try:
                event_bus = get_event_bus()
                severity = float(evaluation_result.get("severity", 0.0))
                violations = list(evaluation_result.get("violations", []))
                event = CriticEvaluatedEvent(
                    event_type=EventType.CRITIC_EVALUATED,
                    trace_id=trace_id,
                    data={"duration_ms": duration_ms},
                    critic_name=name,
                    severity=severity,
                    violations=violations,
                )
                await event_bus.publish(event)
            except Exception as exc:
                logger.debug(f"Failed to publish critic event: {exc}")

        try:
            severity_score = float(evaluation_result.get("score", 0.0))
            severity_label = (
                critic.severity(severity_score) if hasattr(critic, "severity") else "INFO"
            )
            violations_list = evaluation_result.get("violations") or []
            violation_description = evaluation_result.get("justification") or (
                violations_list[0] if violations_list else f"{name} check"
            )

            record = await engine.recorder.record(
                critic=name,
                rule_id=str(evaluation_result.get("principle") or name),
                severity=severity_label,
                violation_description=str(violation_description),
                confidence=float(evaluation_result.get("score", 0.0)),
                mitigation=None,
                redundancy_group=None,
                detector_metadata=evaluation_result.get("evidence") or {},
                context=context,
                raw_text=model_response,
                trace_id=trace_id,
            )
            if evidence_records is not None:
                evidence_records.append(record)
        except Exception:
            error = EvidenceRecordingError(
                "Evidence recording failed",
                details={"critic": name},
            )
            engine._emit_error(error, stage="evidence", trace_id=trace_id, critic=name)

        if engine.cache_manager and cache_key and not evaluation_result.get("error"):
            await engine.cache_manager.set(cache_key, dict(evaluation_result))
        return cast(CriticResult, evaluation_result)


async def run_single_critic_with_breaker(
    engine: "EngineType",
    name: str,
    critic_ref: "CriticRef",
    model_response: str,
    input_text: str,
    context: dict,
    trace_id: str,
    degraded_components: Optional[List[str]] = None,
    evidence_records: Optional[List[Any]] = None,
) -> CriticResult:
    breaker = engine._get_circuit_breaker(f"critic:{name}")
    if breaker is None:
        return await run_single_critic(
            engine,
            name,
            critic_ref,
            model_response,
            input_text,
            context,
            trace_id,
            evidence_records,
        )

    try:
        result = await breaker.call(
            run_single_critic,
            engine,
            name,
            critic_ref,
            model_response,
            input_text,
            context,
            trace_id,
            evidence_records,
        )
        return cast(CriticResult, result)
    except CircuitBreakerOpen as exc:
        if engine.degradation_enabled:
            if degraded_components is not None:
                degraded_components.append(f"critic:{name}")
            fallback = await DegradationStrategy.critic_fallback(
                critic_name=name,
                error=exc,
                context={"trace_id": trace_id},
            )
            return engine._build_critic_error_result(
                critic_name=name,
                error=exc,
                duration_ms=0.0,
                degraded=True,
                degradation_reason=fallback.get("degradation_reason"),
            )
        raise


async def process_critic_batch(
    engine: "EngineType",
    items: List[tuple[str, "CriticRef", str, str, dict, str, Optional[List[str]], Optional[List[Any]]]],
) -> List["CriticResult"]:
    tasks = [
        run_single_critic_with_breaker(
            engine,
            name,
            critic_ref,
            model_response,
            input_text,
            context,
            trace_id,
            degraded_components,
            evidence_records,
        )
        for (
            name,
            critic_ref,
            model_response,
            input_text,
            context,
            trace_id,
            degraded_components,
            evidence_records,
        ) in items
    ]
    return cast(List[Any], await asyncio.gather(*tasks, return_exceptions=True))


async def run_critics_parallel(
    engine: "EngineType",
    model_response: str,
    context: dict,
    trace_id: str,
    input_text: Optional[str] = None,
    degraded_components: Optional[List[str]] = None,
    evidence_records: Optional[List[Any]] = None,
) -> CriticResultsMap:

    # --- Legacy guardrail ---
    enforce = os.getenv("ELEANOR_ENFORCE_SINGLE_CRITIC_PIPELINE", "true").strip().lower() in ("1", "true", "yes", "on")
    allow = os.getenv("ELEANOR_ALLOW_LEGACY_CRITIC_RUNNER", "").strip().lower() in ("1", "true", "yes", "on")
    if enforce and not allow:
        raise RuntimeError(
            "Legacy critic runner invoked (engine.runtime.critics.run_critics_parallel). "
            "Production should use OrchestratorV2 via engine.runtime.critic_infrastructure.run_critics_with_orchestrator. "
            "Set ELEANOR_ALLOW_LEGACY_CRITIC_RUNNER=1 to override (tests/dev only), or set "
            "ELEANOR_ENFORCE_SINGLE_CRITIC_PIPELINE=false to disable this guard."
        )
    warnings.warn(
        "engine.runtime.critics.run_critics_parallel is legacy; use run_critics_with_orchestrator instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    input_text = context.get("input_text_override") or input_text or ""
    if not isinstance(input_text, str):
        input_text = str(input_text)
    critic_items = list(engine.critics.items())
    if engine.critic_batcher:
        batch_items = [
            (
                name,
                critic_ref,
                model_response,
                input_text,
                context,
                trace_id,
                degraded_components,
                evidence_records,
            )
            for name, critic_ref in critic_items
        ]
        results = await _process_batch_with_engine(engine.critic_batcher, batch_items, engine)
    else:
        tasks = [
            asyncio.create_task(
                run_single_critic_with_breaker(
                    engine,
                    name,
                    critic_ref,
                    model_response,
                    input_text,
                    context,
                    trace_id,
                    degraded_components,
                    evidence_records,
                )
            )
            for name, critic_ref in critic_items
        ]
        results = cast(List[Any], await asyncio.gather(*tasks, return_exceptions=True))
    output: CriticResultsMap = {}
    if isinstance(results, dict):
        iter_results = [(name, results.get(name)) for name, _ in critic_items]
    else:
        iter_results = list(zip((name for name, _ in critic_items), results))

    for critic_name, result in iter_results:
        if isinstance(result, CriticEvaluationError):
            engine._emit_error(
                result,
                stage="critic",
                trace_id=trace_id,
                critic=critic_name,
                context=context,
            )
            fallback = result.details.get("result") if isinstance(result.details, dict) else None
            output[critic_name] = fallback or engine._build_critic_error_result(
                critic_name, result
            )
            continue
        if isinstance(result, Exception):
            error = CriticEvaluationError(
                critic_name=critic_name,
                message=str(result),
                trace_id=trace_id,
                details={"error_type": type(result).__name__},
            )
            engine._emit_error(
                error,
                stage="critic",
                trace_id=trace_id,
                critic=critic_name,
                context=context,
            )
            output[critic_name] = engine._build_critic_error_result(critic_name, result)
            continue
        if not isinstance(result, dict):
            output[critic_name] = engine._build_critic_error_result(
                critic_name,
                Exception("critic_result_invalid"),
            )
            continue
        critic_result = cast(CriticResult, result)
        if "duration_ms" not in critic_result:
            critic_result["duration_ms"] = 0.0
        output[critic_result.get("critic", critic_name)] = critic_result
    return output
