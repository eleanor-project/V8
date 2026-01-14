import asyncio
import json
from typing import Any, AsyncGenerator, Dict, List, Optional, cast

from engine.exceptions import (
    AggregationError,
    CriticEvaluationError,
    DetectorExecutionError,
    EleanorV8Exception,
    GovernanceEvaluationError,
    InputValidationError,
    PrecedentRetrievalError,
    RouterSelectionError,
    UncertaintyComputationError,
)
from engine.resilience.degradation import DegradationStrategy
from engine.runtime.models import EngineModelInfo
from engine.schemas.pipeline_types import CriticResult, CriticResultsMap, PrecedentAlignmentResult, UncertaintyResult
from engine.utils.circuit_breaker import CircuitBreakerOpen
from engine.runtime.critics import _process_batch_with_engine
from engine.runtime.run import _track_precedent_evolution

try:
    from engine.observability.correlation import CorrelationContext
    from engine.events.event_bus import (
        get_event_bus,
        RouterSelectedEvent,
        Event,
        EventType,
    )
    STREAM_OBSERVABILITY_AVAILABLE = True
except ImportError:
    STREAM_OBSERVABILITY_AVAILABLE = False
    CorrelationContext = None  # type: ignore
    get_event_bus = None  # type: ignore
    RouterSelectedEvent = None  # type: ignore
    Event = None  # type: ignore
    EventType = None  # type: ignore


async def run_stream_engine(
    engine: Any,
    text: str,
    context: Optional[dict] = None,
    *,
    detail_level: Optional[int] = None,
    trace_id: Optional[str] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    raw_text = text
    raw_context = context
    raw_trace_id = trace_id
    try:
        text, context, trace_id, level = engine._validate_inputs(
            text,
            context,
            trace_id,
            detail_level,
        )
    except InputValidationError as exc:
        fallback_trace_id = str(raw_trace_id) if raw_trace_id else None
        engine._emit_validation_error(
            exc,
            text=raw_text,
            context=raw_context,
            trace_id=fallback_trace_id,
        )
        raise
    correlation_id: Optional[str] = None
    if STREAM_OBSERVABILITY_AVAILABLE and CorrelationContext:
        correlation_id = CorrelationContext.get_or_generate()
        context = {**context, "correlation_id": correlation_id}
    if engine.gpu_manager and engine.gpu_manager.device:
        context = {
            **context,
            "gpu_enabled": engine.gpu_enabled,
            "gpu_device": str(engine.gpu_manager.device),
        }
    pipeline_start = asyncio.get_event_loop().time()
    timings: Dict[str, float] = {}
    router_diagnostics: Dict[str, Any] = {}
    evidence_records: List[Any] = []
    degraded_components: List[str] = []
    try:
        detector_payload = await engine._run_detectors(text, context, timings)
    except DetectorExecutionError as exc:
        engine._emit_error(exc, stage="detectors", trace_id=trace_id, context=context)
        detector_payload = None
    except Exception as exc:
        detector_error = DetectorExecutionError(
            "Detector execution failed",
            details={"error": str(exc)},
        )
        engine._emit_error(
            detector_error,
            stage="detectors",
            trace_id=trace_id,
            context=context,
        )
        detector_payload = None
    if detector_payload:
        context = {**context, "detectors": detector_payload}
        yield {
            "event": "detectors_complete",
            "trace_id": trace_id,
            "data": {"summary": {k: v for k, v in detector_payload.items() if k != "signals"}},
        }
    if context.get("skip_router"):
        raw_output = context.get("model_output")
        if raw_output is None:
            raise ValueError("model_output is required when skip_router is true")
        if isinstance(raw_output, str):
            model_response = raw_output
        else:
            model_response = json.dumps(raw_output, ensure_ascii=True, default=str)
        meta = context.get("model_metadata") or {}
        model_info = {
            "model_name": meta.get("model_id") or meta.get("model_name") or "external",
            "model_version": meta.get("model_version") or "",
            "router_selection_reason": "model_output_override",
            "health_score": None,
            "cost_estimate": None,
        }
        router_diagnostics.update({"skipped": True, "reason": "model_output_override"})
    else:
        select_model_fn = getattr(engine, "_select_router", None) or getattr(engine, "_select_model", engine._select_model)
        try:
            breaker = engine._get_circuit_breaker("router")
            if breaker is not None:
                router_data = cast(
                    Dict[str, Any],
                    await breaker.call(
                        select_model_fn, text, context, timings, router_diagnostics
                    ),
                )
            else:
                router_data = await select_model_fn(text, context, timings, router_diagnostics)
            model_info = router_data["model_info"]
            model_response = router_data["response_text"]
            if STREAM_OBSERVABILITY_AVAILABLE and get_event_bus and EventType is not None:
                try:
                    event_bus = get_event_bus()
                    await event_bus.publish(
                        RouterSelectedEvent(
                            event_type=EventType.ROUTER_SELECTED,
                            trace_id=trace_id,
                            model_name=str(model_info.get("model_name") or "unknown"),
                            selection_reason=str(model_info.get("router_selection_reason") or ""),
                            cost_estimate=model_info.get("cost_estimate"),
                            data={"correlation_id": correlation_id} if correlation_id else {},
                        )
                    )
                except Exception:
                    pass
        except CircuitBreakerOpen as exc:
            if engine.degradation_enabled:
                degraded_components.append("router")
                fallback = await DegradationStrategy.router_fallback(
                    error=exc,
                    context={"trace_id": trace_id},
                )
                model_info = {
                    "model_name": fallback.get("model_name") or "router_fallback",
                    "model_version": fallback.get("model_version"),
                    "router_selection_reason": fallback.get("router_selection_reason"),
                    "health_score": None,
                    "cost_estimate": None,
                }
                model_response = ""
                router_diagnostics.update({"circuit_open": True, "error": str(exc)})
            else:
                raise
        except RouterSelectionError as exc:
            engine._emit_error(exc, stage="router", trace_id=trace_id, context=context)
            if engine.degradation_enabled:
                degraded_components.append("router")
            model_info = {
                "model_name": "router_error",
                "model_version": None,
                "router_selection_reason": "router_failure",
                "health_score": 0.0,
                "cost_estimate": None,
            }
            model_response = ""
            diagnostics = exc.details.get("diagnostics") if isinstance(exc, EleanorV8Exception) else None
            router_diagnostics.update(diagnostics or {"error": str(exc)})
    model_name_value = str(model_info.get("model_name") or "unknown-model")
    model_info["model_name"] = model_name_value
    engine_model_info = EngineModelInfo(
        model_name=model_name_value,
        model_version=cast(Optional[str], model_info.get("model_version")),
        router_selection_reason=cast(Optional[str], model_info.get("router_selection_reason")),
        cost_estimate=cast(Optional[Dict[str, Any]], model_info.get("cost_estimate")),
        health_score=cast(Optional[float], model_info.get("health_score")),
    )
    yield {
        "event": "router_selected",
        "trace_id": trace_id,
        "model_info": model_info,
        "timings": {"router_selection_ms": timings.get("router_selection_ms")},
    }
    yield {
        "event": "model_response",
        "trace_id": trace_id,
        "text": model_response,
    }
    yield {
        "event": "critics_start",
        "trace_id": trace_id,
        "critics": list(engine.critics.keys()),
    }
    critic_results: CriticResultsMap = {}
    critic_input_text = context.get("input_text_override") or text
    if not isinstance(critic_input_text, str):
        critic_input_text = str(critic_input_text)
    critic_items = list(engine.critics.items())
    if engine.critic_batcher:
        batch_items = [
            (
                name,
                critic_ref,
                model_response,
                critic_input_text,
                context,
                trace_id,
                degraded_components,
                evidence_records,
            )
            for name, critic_ref in critic_items
        ]
        results = await _process_batch_with_engine(engine.critic_batcher, batch_items, engine)
        result_items = results.items() if isinstance(results, dict) else zip(
            (name for name, _ in critic_items), results
        )
        for critic_name, res in result_items:
            if isinstance(res, CriticEvaluationError):
                crit_error = res
                engine._emit_error(
                    crit_error,
                    stage="critic",
                    trace_id=trace_id,
                    critic=critic_name,
                    context=context,
                )
                critic_fallback = crit_error.details.get("result") if isinstance(
                    crit_error.details, dict
                ) else None
                res = critic_fallback or engine._build_critic_error_result(critic_name, crit_error)
            elif isinstance(res, Exception):
                unknown_error = res
                critic_error = CriticEvaluationError(
                    critic_name=critic_name,
                    message=str(unknown_error),
                    trace_id=trace_id,
                    details={"error_type": type(unknown_error).__name__},
                )
                engine._emit_error(
                    critic_error,
                    stage="critic",
                    trace_id=trace_id,
                    critic=critic_name,
                    context=context,
                )
                res = engine._build_critic_error_result(critic_name, unknown_error)
            res = cast(CriticResult, res)
            critic_results[res.get("critic", critic_name)] = res
            yield {
                "event": "critic_result",
                "trace_id": trace_id,
                "critic": res.get("critic", critic_name),
                "violations": list(res.get("violations", [])),
                "duration_ms": res.get("duration_ms"),
                "evaluated_rules": res.get("evaluated_rules"),
            }
    else:

        async def _run_and_return(name, critic_ref):
            try:
                res = await engine._run_single_critic_with_breaker(
                    name,
                    critic_ref,
                    model_response,
                    critic_input_text,
                    context,
                    trace_id,
                    degraded_components,
                    evidence_records,
                )
                return name, res
            except Exception as exc:
                return name, exc

        tasks = [
            asyncio.create_task(_run_and_return(name, critic_ref))
            for name, critic_ref in critic_items
        ]
        for future in asyncio.as_completed(tasks):
            critic_name, res = await future
            if isinstance(res, CriticEvaluationError):
                crit_error = res
                engine._emit_error(
                    crit_error,
                    stage="critic",
                    trace_id=trace_id,
                    critic=critic_name,
                    context=context,
                )
                critic_fallback = (
                    crit_error.details.get("result")
                    if isinstance(crit_error.details, dict)
                    else None
                )
                res = critic_fallback or engine._build_critic_error_result(critic_name, crit_error)
            elif isinstance(res, Exception):
                unknown_error = res
                critic_error = CriticEvaluationError(
                    critic_name=critic_name,
                    message=str(unknown_error),
                    trace_id=trace_id,
                    details={"error_type": type(unknown_error).__name__},
                )
                engine._emit_error(
                    critic_error,
                    stage="critic",
                    trace_id=trace_id,
                    critic=critic_name,
                    context=context,
                )
                res = engine._build_critic_error_result(critic_name, unknown_error)
            res = cast(CriticResult, res)
            critic_results[res.get("critic", critic_name)] = res
            yield {
                "event": "critic_result",
                "trace_id": trace_id,
                "critic": res.get("critic", critic_name),
                "violations": list(res.get("violations", [])),
                "duration_ms": res.get("duration_ms"),
                "evaluated_rules": res.get("evaluated_rules"),
            }
    yield {
        "event": "critics_complete",
        "trace_id": trace_id,
    }
    precedent_data = None
    if engine.config.enable_precedent_analysis and engine.precedent_engine:

        async def _run_prec():
            return await engine._run_precedent_alignment(
                critic_results,
                trace_id,
                text,
                timings=timings,
            )

        async def _fallback_prec(exc: Exception) -> PrecedentAlignmentResult:
            return cast(
                PrecedentAlignmentResult,
                await DegradationStrategy.precedent_fallback(
                    error=exc,
                    context={"trace_id": trace_id},
                ),
            )

        precedent_data, _, prec_error = await engine._execute_with_degradation(
            stage="precedent",
            run_fn=_run_prec,
            fallback_fn=_fallback_prec,
            error_type=PrecedentRetrievalError,
            degraded_components=degraded_components,
            degrade_component="precedent",
            context=context,
            trace_id=trace_id,
            fallback_on_error=lambda exc: None,
        )
        event_payload = precedent_data
        if prec_error and not engine.degradation_enabled:
            event_payload = {"error": str(prec_error)}
        yield {
            "event": "precedent_alignment",
            "trace_id": trace_id,
            "data": event_payload,
        }
        if precedent_data and STREAM_OBSERVABILITY_AVAILABLE and get_event_bus and EventType is not None:
            try:
                event_bus = get_event_bus()
                await event_bus.publish(
                    Event(
                        event_type=EventType.PRECEDENT_RETRIEVED,
                        trace_id=trace_id,
                        data={
                            "alignment_score": precedent_data.get("alignment_score"),
                            "support_strength": precedent_data.get("support_strength"),
                            "correlation_id": correlation_id,
                        },
                    )
                )
            except Exception:
                pass
    uncertainty_data = None
    if engine.config.enable_reflection and engine.uncertainty_engine:

        async def _run_uncertainty():
            return await engine._run_uncertainty_engine(
                precedent_alignment=precedent_data,
                critic_results=critic_results,
                model_name=engine_model_info.model_name,
                timings=timings,
            )

        async def _fallback_uncertainty(exc: Exception) -> UncertaintyResult:
            return cast(
                UncertaintyResult,
                await DegradationStrategy.uncertainty_fallback(
                    error=exc,
                    context={"trace_id": trace_id},
                ),
            )

        uncertainty_data, _, uncertainty_error = await engine._execute_with_degradation(
            stage="uncertainty",
            run_fn=_run_uncertainty,
            fallback_fn=_fallback_uncertainty,
            error_type=UncertaintyComputationError,
            degraded_components=degraded_components,
            degrade_component="uncertainty",
            context=context,
            trace_id=trace_id,
            fallback_on_error=lambda exc: None,
        )
        event_payload = uncertainty_data
        if uncertainty_error and not engine.degradation_enabled:
            event_payload = {"error": str(uncertainty_error)}
        yield {
            "event": "uncertainty",
            "trace_id": trace_id,
            "data": event_payload,
        }
        if uncertainty_data and STREAM_OBSERVABILITY_AVAILABLE and get_event_bus and EventType is not None:
            try:
                event_bus = get_event_bus()
                await event_bus.publish(
                    Event(
                        event_type=EventType.UNCERTAINTY_COMPUTED,
                        trace_id=trace_id,
                        data={
                            "overall_uncertainty": uncertainty_data.get("overall_uncertainty"),
                            "needs_escalation": uncertainty_data.get("needs_escalation"),
                            "correlation_id": correlation_id,
                        },
                    )
                )
            except Exception:
                pass
    try:
        aggregated = await engine._aggregate_results(
            critic_results,
            model_response,
            precedent_data,
            uncertainty_data,
            timings=timings,
        )
    except AggregationError as exc:
        engine._emit_error(exc, stage="aggregation", trace_id=trace_id, context=context)
        aggregated = engine._build_aggregation_fallback(
            model_response,
            precedent_data,
            uncertainty_data,
            exc,
        )
    else:
        if STREAM_OBSERVABILITY_AVAILABLE and get_event_bus and EventType is not None:
            try:
                event_bus = get_event_bus()
                await event_bus.publish(
                    Event(
                        event_type=EventType.AGGREGATION_COMPLETE,
                        trace_id=trace_id,
                        data={
                            "decision": aggregated.get("decision") if isinstance(aggregated, dict) else None,
                            "correlation_id": correlation_id,
                        },
                    )
                )
            except Exception:
                pass
    degraded_components = sorted(set(degraded_components))
    is_degraded = bool(degraded_components)
    if is_degraded:
        aggregated = {
            **(aggregated or {}),
            "degraded_components": degraded_components,
            "is_degraded": True,
        }
    yield {
        "event": "aggregation",
        "trace_id": trace_id,
        "data": aggregated,
    }

    # Traffic Light governance (observer hook) — run_stream()
    governance_meta = None
    if getattr(engine, 'traffic_light_governance', None) is not None:
        try:
            governance_meta = await engine.traffic_light_governance.apply(
                trace_id=trace_id,
                text=text,
                context=context,
                aggregated=aggregated,
                precedent_data=precedent_data,
                uncertainty_data=uncertainty_data,
            )
        except Exception as exc:
            governance_meta = {'error': str(exc)}

        if isinstance(aggregated, dict) and governance_meta:
            aggregated['governance_meta'] = governance_meta

        yield {
            'event': 'governance',
            'trace_id': trace_id,
            'data': governance_meta,
        }

    try:
        case = engine._build_case_for_review(
            trace_id=trace_id,
            context=context,
            aggregated=aggregated,
            critic_results=critic_results,
            precedent_data=precedent_data,
            uncertainty_data=uncertainty_data,
        )
        engine._run_governance_review_gate(case)
        if isinstance(aggregated, dict):
            aggregated = {**aggregated, "governance_flags": getattr(case, "governance_flags", {})}
            if getattr(case, "governance_flags", {}).get("human_review_required"):
                aggregated["human_review_required"] = True
                aggregated.setdefault("decision", "requires_human_review")
        if STREAM_OBSERVABILITY_AVAILABLE and get_event_bus and EventType is not None and getattr(case, "governance_flags", None):
            try:
                event_bus = get_event_bus()
                await event_bus.publish(
                    Event(
                        event_type=EventType.ESCALATION_REQUIRED
                        if case.governance_flags.get("human_review_required")
                        else EventType.DECISION_MADE,
                        trace_id=trace_id,
                        data={
                            "governance_flags": case.governance_flags,
                            "correlation_id": correlation_id,
                        },
                    )
                )
            except Exception:
                pass
    except GovernanceEvaluationError as review_exc:
        engine._emit_error(review_exc, stage="governance", trace_id=trace_id, context=context)
    except Exception as review_exc:
        governance_error = GovernanceEvaluationError(
            "Governance review gate failed",
            details={"error": str(review_exc)},
        )
        engine._emit_error(
            governance_error,
            stage="governance",
            trace_id=trace_id,
            context=context,
        )

    _track_precedent_evolution(
        engine=engine,
        trace_id=trace_id,
        context=context,
        aggregated=cast(Dict[str, Any], aggregated or {}),
        critic_results=critic_results,
        model_name=engine_model_info.model_name,
        precedent_data=cast(Optional[Dict[str, Any]], precedent_data),
    )
    final_output = aggregated.get("final_output", "") if isinstance(aggregated, dict) else ""
    if not final_output:
        final_output = model_response
    pipeline_end = asyncio.get_event_loop().time()
    timings["total_pipeline_ms"] = (pipeline_end - pipeline_start) * 1000
    forensic_evidence = evidence_records[-200:] if evidence_records else []
    base_final = {
        "event": "final_output",
        "trace_id": trace_id,
        "output_text": final_output,
        "degraded_components": degraded_components,
        "is_degraded": is_degraded,
    }
    if level == 1:
        yield base_final
    elif level == 2:
        yield {
            **base_final,
            "critic_findings": critic_results,
            "precedent_alignment": precedent_data,
            "uncertainty": uncertainty_data,
        }
    elif level == 3:
        yield {
            **base_final,
            "critic_findings": critic_results,
            "precedent_alignment": precedent_data,
            "uncertainty": uncertainty_data,
            "router_diagnostics": router_diagnostics,
            "timings": timings,
            "forensic_evidence": [r.dict() if hasattr(r, "dict") else r for r in forensic_evidence],
        }
    else:
        raise ValueError(f"Invalid detail_level: {level}")
