import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, cast

from engine.exceptions import (
    AggregationError,
    DetectorExecutionError,
    EleanorV8Exception,
    GovernanceEvaluationError,
    InputValidationError,
    PrecedentRetrievalError,
    RouterSelectionError,
    UncertaintyComputationError,
)
from engine.resilience.degradation import DegradationStrategy
from engine.runtime.models import EngineCriticFinding, EngineForensicData, EngineModelInfo, EngineResult
from engine.schemas.pipeline_types import (
    PrecedentAlignmentResult,
    UncertaintyResult,
)
from engine.utils.circuit_breaker import CircuitBreakerOpen

logger = logging.getLogger("engine.engine")

# Enhanced observability
try:
    from engine.observability.business_metrics import (
        record_engine_result,
        set_degraded_components,
    )
    from engine.observability.correlation import CorrelationContext
    from engine.events.event_bus import get_event_bus, DecisionMadeEvent, EventType
    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False
    record_engine_result = None
    set_degraded_components = None
    CorrelationContext = None
    get_event_bus = None
    DecisionMadeEvent = None
    EventType = None


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _collect_precedent_values(critic_results: Dict[str, Any]) -> List[str]:
    values = []
    for critic_name, critic_data in critic_results.items():
        if isinstance(critic_data, dict):
            value = critic_data.get("value")
            if value:
                values.append(str(value))
                continue
        values.append(str(critic_name))
    deduped = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _track_precedent_evolution(
    *,
    engine: Any,
    trace_id: str,
    context: Dict[str, Any],
    aggregated: Dict[str, Any],
    critic_results: Dict[str, Any],
    model_name: str,
    precedent_data: Optional[Dict[str, Any]],
) -> None:
    tracker = getattr(engine, "temporal_evolution_tracker", None)
    if not tracker:
        return

    case_id = context.get("case_id") or context.get("precedent_case_id") or trace_id
    decision = aggregated.get("decision") or "allow"
    aggregate_score = _coerce_float(aggregated.get("score"))
    rationale_raw = (
        aggregated.get("final_output")
        or aggregated.get("summary")
        or aggregated.get("justification")
        or ""
    )
    rationale = str(rationale_raw)
    values = _collect_precedent_values(critic_results)

    top_case = (precedent_data or {}).get("top_case") if precedent_data else None
    top_case_id = None
    if isinstance(top_case, dict):
        top_case_id = top_case.get("case_id")

    metadata = {
        "trace_id": trace_id,
        "model_name": model_name,
        "policy_profile": context.get("policy_profile"),
        "precedent_top_case_id": top_case_id,
    }

    created_by = (
        context.get("actor_id")
        or context.get("user_id")
        or context.get("user")
        or context.get("requester")
    )

    try:
        tracker.track_precedent_update(
            case_id=str(case_id),
            decision=str(decision),
            aggregate_score=aggregate_score,
            values=values,
            rationale=rationale,
            critic_outputs=critic_results,
            metadata=metadata,
            created_by=str(created_by) if created_by is not None else None,
        )
    except Exception as exc:
        logger.warning(
            "precedent_evolution_track_failed",
            extra={"trace_id": trace_id, "error": str(exc)},
        )


async def run_engine(
    engine: Any,
    text: str,
    context: Optional[dict] = None,
    *,
    detail_level: Optional[int] = None,
    trace_id: Optional[str] = None,
) -> EngineResult:
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
        # Expected error - already properly typed
        engine._emit_error(exc, stage="detectors", trace_id=trace_id, context=context)
        detector_payload = None
    except (asyncio.TimeoutError, ConnectionError, OSError) as exc:
        # Network/system errors - convert to DetectorExecutionError
        detector_error = DetectorExecutionError(
            "Detector execution failed due to system error",
            details={"error": str(exc), "error_type": type(exc).__name__},
        )
        engine._emit_error(
            detector_error,
            stage="detectors",
            trace_id=trace_id,
            context=context,
        )
        detector_payload = None
    except Exception as exc:
        # Unexpected errors - log with full context
        detector_error = DetectorExecutionError(
            "Detector execution failed with unexpected error",
            details={"error": str(exc), "error_type": type(exc).__name__},
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
        try:
            breaker = engine._get_circuit_breaker("router")
            if breaker is not None:
                router_data = cast(
                    Dict[str, Any],
                    await breaker.call(
                        engine._select_model, text, context, timings, router_diagnostics
                    ),
                )
            else:
                router_data = await engine._select_model(text, context, timings, router_diagnostics)
            model_info = router_data["model_info"]
            model_response = router_data["response_text"]
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

    # Run critics using orchestrator with infrastructure integration
    from engine.runtime.critic_infrastructure import run_critics_with_orchestrator
    
    critic_results = await run_critics_with_orchestrator(
        engine=engine,
        model_response=model_response,
        input_text=text,
        context=context,
        trace_id=trace_id,
        degraded_components=degraded_components,
        evidence_records=evidence_records,
    )

    precedent_data = None
    if engine.config.enable_precedent_analysis:

        async def _run_prec():
            return await engine._run_precedent_alignment(
                critic_results=critic_results,
                trace_id=trace_id,
                text=text,
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

        precedent_data, _, _ = await engine._execute_with_degradation(
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

        uncertainty_data, _, _ = await engine._execute_with_degradation(
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

    try:
        aggregated = await engine._aggregate_results(
            critic_results=critic_results,
            model_response=model_response,
            precedent_data=precedent_data,
            uncertainty_data=uncertainty_data,
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

    # Traffic Light governance (observer hook) — run()
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

    pipeline_end = asyncio.get_event_loop().time()
    timings["total_pipeline_ms"] = (pipeline_end - pipeline_start) * 1000

    degraded_components = sorted(set(degraded_components))
    is_degraded = bool(degraded_components)
    if is_degraded:
        aggregated = {
            **(aggregated or {}),
            "degraded_components": degraded_components,
            "is_degraded": True,
        }
    
    # Record degraded components metric
    if OBSERVABILITY_AVAILABLE and set_degraded_components:
        set_degraded_components(len(degraded_components))

    evidence_count = len(evidence_records) if evidence_records else None

    critic_findings = {
        k: EngineCriticFinding(
            critic=k,
            violations=list(v.get("violations", [])),
            duration_ms=v.get("duration_ms"),
            evaluated_rules=cast(Optional[List[str]], v.get("evaluated_rules")),
        )
        for k, v in critic_results.items()
    }
    base_result = EngineResult(
        trace_id=trace_id,
        output_text=aggregated.get("final_output") or model_response,
        model_info=engine_model_info,
        critic_findings=critic_findings,
        aggregated=aggregated,
        uncertainty=uncertainty_data,
        precedent_alignment=precedent_data,
        evidence_count=evidence_count,
        degraded_components=degraded_components,
        is_degraded=is_degraded,
    )

    if level == 1:
        return EngineResult(
            trace_id=trace_id,
            output_text=aggregated.get("final_output") or model_response,
            model_info=engine_model_info,
            degraded_components=degraded_components,
            is_degraded=is_degraded,
        )

    if level == 2:
        return base_result

    forensic_data = None
    if level == 3:
        forensic_buffer = evidence_records[-200:] if evidence_records else []

        forensic_data = EngineForensicData(
            detector_metadata=detector_payload or {},
            uncertainty_graph=cast(UncertaintyResult, uncertainty_data or {}),
            precedent_alignment=cast(PrecedentAlignmentResult, precedent_data or {}),
            router_diagnostics=router_diagnostics,
            timings=timings,
            evidence_references=[r.dict() if hasattr(r, "dict") else r for r in forensic_buffer],
        )

        result = EngineResult(
            trace_id=base_result.trace_id,
            output_text=base_result.output_text,
            model_info=base_result.model_info,
            critic_findings=base_result.critic_findings,
            aggregated=base_result.aggregated,
            uncertainty=base_result.uncertainty,
            precedent_alignment=base_result.precedent_alignment,
            evidence_count=base_result.evidence_count,
            degraded_components=base_result.degraded_components,
            is_degraded=base_result.is_degraded,
            forensic=forensic_data,
        )
        
        # Record metrics and publish events
        if OBSERVABILITY_AVAILABLE:
            try:
                result_dict = result.model_dump() if hasattr(result, "model_dump") else result.dict()
                if record_engine_result:
                    record_engine_result(result_dict)
                
                # Publish decision event
                if get_event_bus and EventType is not None:
                    event_bus = get_event_bus()
                    decision = aggregated.get("decision", "unknown")
                    confidence = aggregated.get("confidence", 0.0)
                    escalated = bool(result_dict.get("human_review_required") or 
                                   any("escalate" in str(v).lower() for v in result_dict.get("critic_findings", {}).values()))
                    
                    event = DecisionMadeEvent(
                        event_type=EventType.DECISION_MADE,
                        timestamp=None,  # Will be set in __post_init__
                        trace_id=trace_id,
                        data={"degraded_components": degraded_components},
                        decision=decision,
                        confidence=float(confidence),
                        escalated=escalated,
                    )
                    await event_bus.publish(event)
            except Exception as exc:
                logger.debug(f"Failed to record metrics or publish event: {exc}")
        
        return result

    raise ValueError(f"Invalid detail_level: {level}")
