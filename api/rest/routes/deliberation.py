import asyncio
import time
import uuid
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Request, status

from api.schemas import (
    DeliberationRequest,
    EvaluateRequest,
    EvaluateResponse,
    EvaluateError,
    RoutingDecision,
    EvidenceBundle,
    EvidenceIntegrity,
    EvidenceProvenance,
    UncertaintyEnvelope,
)

from api.middleware.auth import require_authenticated_user
from api.middleware.rate_limit import check_rate_limit
from api.rest.request_guards import check_content_length
from api.rest.deps import get_engine, get_replay_store

from api.bootstrap import evaluate_opa, GOVERNANCE_SCHEMA_VERSION

from api.rest.metrics import DELIB_LATENCY, DELIB_REQUESTS, OPA_CALLS

from engine.logging_config import get_logger, log_deliberation
from engine.exceptions import InputValidationError
from engine.utils.validation import sanitize_for_logging
from api.middleware.fingerprinting import RequestFingerprinter

from engine.version import ELEANOR_VERSION

from api.rest.services.deliberation_utils import (
    resolve_final_decision,
    map_assessment_label,
    normalize_engine_result,
    resolve_execution_decision,
    apply_execution_gate,
    serialize_model_output,
    build_uncertainty_envelope,
    confidence_from_uncertainty,
    map_decision,
    build_constraints,
    build_evidence_bundle,
    _hash_payload,
)

logger = get_logger(__name__)
router = APIRouter(tags=["Deliberation"])


# Optional Observability (same pattern as main.py; safe if missing)
try:
    from engine.observability.business_metrics import record_engine_result, record_decision
    from engine.observability.correlation import CorrelationContext, get_correlation_id

    OBSERVABILITY_AVAILABLE = True
except Exception:
    OBSERVABILITY_AVAILABLE = False
    record_engine_result = None
    record_decision = None
    CorrelationContext = None
    get_correlation_id = None


@router.post("/deliberate")
async def deliberate(
    request: Request,
    payload: DeliberationRequest,
    user: str = Depends(require_authenticated_user),
    _size_guard: None = Depends(check_content_length),
    _rate_limit: None = Depends(check_rate_limit),
    include_explanation: bool = False,
    explanation_detail: str = "summary",
    engine=Depends(get_engine),
    replay_store=Depends(get_replay_store),
):
    start_time = time.time()

    correlation_id = None
    if CorrelationContext:
        correlation_id = get_correlation_id()
        CorrelationContext.set(correlation_id)

    trace_id = payload.trace_id or correlation_id or str(uuid.uuid4())

    fingerprinter = RequestFingerprinter()
    fingerprint = fingerprinter.fingerprint(request)
    fingerprint_components = fingerprinter.get_fingerprint_components(request)

    logger.info(
        "deliberation_started",
        extra={"trace_id": trace_id, "user_id": user, "input_length": len(payload.input)},
    )

    # Input validation with security hardening
    from engine.security.input_validation import InputValidator

    validator = InputValidator(strict_mode=True)
    try:
        validated_input = validator.validate_string(
            payload.input, field_name="input", allow_empty=False, sanitize=True
        )
    except ValueError as validation_error:
        logger.warning(
            "input_validation_failed",
            extra={"trace_id": trace_id, "user_id": user, "error": str(validation_error)},
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Input validation failed: {str(validation_error)}",
        )

    try:
        validated_context = validator.validate_dict(
            payload.context.model_dump(mode="json") if payload.context else {},
            field_name="context",
        )
    except ValueError as validation_error:
        logger.warning(
            "context_validation_failed",
            extra={"trace_id": trace_id, "user_id": user, "error": str(validation_error)},
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Context validation failed: {str(validation_error)}",
        )

    try:
        run_fn = getattr(engine, "run", None)
        deliberate_fn = getattr(engine, "deliberate", None)

        if run_fn is not None:
            result_obj = await run_fn(
                validated_input, context=validated_context, trace_id=trace_id, detail_level=3
            )
        elif asyncio.iscoroutinefunction(deliberate_fn):
            result_obj = await deliberate_fn(validated_input)
        elif deliberate_fn:
            result_obj = deliberate_fn(validated_input)
        else:
            raise RuntimeError("Engine does not expose run() or deliberate()")

        normalized = normalize_engine_result(result_obj, validated_input, trace_id, validated_context)
        model_used = normalized["model_used"]
        critic_outputs = normalized["critic_outputs"]
        aggregated = normalized["aggregator_output"]
        precedent_alignment = normalized["precedent_alignment"]
        uncertainty = normalized["uncertainty"]

        governance_payload: Dict[str, Any] = {
            "critics": critic_outputs,
            "aggregator": aggregated,
            "precedent": precedent_alignment,
            "uncertainty": uncertainty,
            "model_used": model_used,
            "input": validated_input,
            "trace_id": normalized["trace_id"],
            "timestamp": time.time(),
            "schema_version": GOVERNANCE_SCHEMA_VERSION,
            "policy_profile": payload.policy_profile,
            "proposed_action": payload.proposed_action.model_dump(mode="json"),
            "context": validated_context,
            "evidence_inputs": payload.evidence_inputs.model_dump(mode="json") if payload.evidence_inputs else {},
            "model_metadata": payload.model_metadata.model_dump(mode="json") if payload.model_metadata else {},
            "fingerprint": fingerprint,
            "fingerprint_components": fingerprint_components,
        }

        governance_result = await evaluate_opa(getattr(engine, "opa_callback", None), governance_payload)
        final_decision = resolve_final_decision(aggregated.get("decision"), governance_result)

        execution_decision = None
        try:
            execution_decision = resolve_execution_decision(aggregated, payload.human_action)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

        final_decision = apply_execution_gate(final_decision, execution_decision)
        final_assessment = map_assessment_label(final_decision)

        duration_ms = (time.time() - start_time) * 1000
        if DELIB_LATENCY:
            DELIB_LATENCY.observe(duration_ms / 1000.0)
        if DELIB_REQUESTS:
            DELIB_REQUESTS.labels(outcome=final_assessment).inc()
        if OPA_CALLS:
            opa_outcome = (
                "deny"
                if not governance_result.get("allow", True)
                else "escalate"
                if governance_result.get("escalate")
                else "allow"
            )
            OPA_CALLS.labels(result=opa_outcome).inc()

        log_deliberation(
            logger,
            trace_id=normalized.get("trace_id", trace_id),
            decision=final_assessment,
            model_used=model_used or "unknown",
            duration_ms=duration_ms,
            uncertainty=(uncertainty or {}).get("overall_uncertainty"),
            escalated=final_assessment == "requires_human_review",
            raw_decision=final_decision,
        )

        response_payload = {
            **normalized,
            "opa_governance": governance_result,
            "governance": governance_result,
            "final_decision": final_assessment,
            "execution_decision": execution_decision.model_dump(mode="json") if execution_decision else None,
        }

        if include_explanation and hasattr(engine, "explainable_governance"):
            try:
                from engine.core.feature_integration import get_explanation_for_result

                explanation = get_explanation_for_result(
                    engine,
                    {**normalized, "aggregated": aggregated, "decision": final_assessment},
                    detail_level=explanation_detail,
                )
                if explanation:
                    response_payload["explanation"] = explanation
            except Exception as exc:
                logger.warning(
                    "explanation_generation_failed",
                    extra={"trace_id": trace_id, "error": str(exc)},
                    exc_info=True,
                )

        if OBSERVABILITY_AVAILABLE and record_engine_result:
            try:
                record_engine_result(response_payload)
                if record_decision:
                    record_decision(final_decision, final_assessment)
            except Exception as exc:
                logger.debug(f"Failed to record metrics: {exc}")

        await replay_store.save_async(
            {
                "trace_id": response_payload["trace_id"],
                "input": validated_input,
                "context": validated_context,
                "response": response_payload,
                "timestamp": response_payload["timestamp"],
            }
        )

        return response_payload

    except InputValidationError as exc:
        duration_ms = (time.time() - start_time) * 1000
        safe_input = sanitize_for_logging(payload.input, max_length=300)
        logger.warning(
            "input_validation_failed",
            extra={
                "trace_id": trace_id,
                "error": exc.message,
                "validation_type": exc.details.get("validation_type"),
                "field": exc.details.get("field"),
                "input_excerpt": safe_input,
                "duration_ms": duration_ms,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "invalid_input", "message": exc.message, "details": exc.details},
        )
    except HTTPException:
        raise
    except (ValueError, TypeError, KeyError, AttributeError) as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.warning(
            "deliberation_validation_error",
            extra={"trace_id": trace_id, "error": str(e), "error_type": type(e).__name__, "duration_ms": duration_ms},
        )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid request: {str(e)}")
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.error(
            "deliberation_failed",
            extra={"trace_id": trace_id, "error": str(e), "error_type": type(e).__name__, "duration_ms": duration_ms},
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Deliberation failed")


@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(
    request: Request,
    payload: EvaluateRequest,
    user: str = Depends(require_authenticated_user),
    _rate_limit: None = Depends(check_rate_limit),
    include_explanation: bool = False,
    explanation_detail: str = "summary",
    engine=Depends(get_engine),
):
    from engine.security.input_validation import InputValidator

    validator = InputValidator(strict_mode=True)

    try:
        context = payload.context.model_dump(mode="json") if payload.context else {}
        validated_context = validator.validate_dict(context, field_name="context")
    except ValueError as validation_error:
        logger.warning(
            "context_validation_failed",
            extra={"user_id": user, "endpoint": "/evaluate", "error": str(validation_error)},
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Context validation failed: {str(validation_error)}",
        )

    model_output_text = serialize_model_output(payload.model_output)
    user_intent = validated_context.get("user_intent")
    input_override = user_intent if isinstance(user_intent, str) else ""

    validated_context.update(
        {
            "policy_profile": payload.policy_profile,
            "proposed_action": payload.proposed_action.model_dump(mode="json"),
            "evidence_inputs": payload.evidence_inputs.model_dump(mode="json") if payload.evidence_inputs else {},
            "model_metadata": payload.model_metadata.model_dump(mode="json") if payload.model_metadata else {},
            "model_output": model_output_text,
            "skip_router": True,
            "force_model_output": True,
            "input_text_override": input_override,
        }
    )

    try:
        run_fn = getattr(engine, "run", None)
        if run_fn is None:
            raise RuntimeError("Engine does not expose run()")

        result_obj = await run_fn(
            model_output_text,
            context=validated_context,
            trace_id=payload.request_id,
            detail_level=3,
        )

        normalized = normalize_engine_result(result_obj, model_output_text, payload.request_id, validated_context)
        aggregated = normalized.get("aggregator_output") or {}
        precedent_alignment = normalized.get("precedent_alignment") or {}
        uncertainty = normalized.get("uncertainty") or {}

        governance_payload: Dict[str, Any] = {
            "critics": aggregated.get("critics", {}),
            "aggregator": aggregated,
            "precedent": precedent_alignment,
            "uncertainty": uncertainty,
            "model_used": normalized.get("model_used"),
            "input": model_output_text,
            "trace_id": payload.request_id,
            "timestamp": payload.timestamp.isoformat(),
            "schema_version": GOVERNANCE_SCHEMA_VERSION,
            "policy_profile": payload.policy_profile,
            "proposed_action": payload.proposed_action.model_dump(mode="json"),
            "context": validated_context,
            "evidence_inputs": payload.evidence_inputs.model_dump(mode="json") if payload.evidence_inputs else {},
        }

        governance_result = await evaluate_opa(getattr(engine, "opa_callback", None), governance_payload)
        final_decision = resolve_final_decision(aggregated.get("decision"), governance_result)

        execution_decision = None
        try:
            execution_decision = resolve_execution_decision(aggregated, None)
        except Exception as exc:
            logger.warning("Execution gate evaluation failed", extra={"error": str(exc)})

        final_decision = apply_execution_gate(final_decision, execution_decision)

        decision = map_decision(final_decision)
        uncertainty_envelope = build_uncertainty_envelope(uncertainty)
        confidence = confidence_from_uncertainty(uncertainty)
        critic_details = aggregated.get("critics") or {}
        constraints = build_constraints(critic_details) if decision == "ALLOW_WITH_CONSTRAINTS" else None

        routing_notes = execution_decision.execution_reason if (execution_decision and not execution_decision.executable) else None
        if decision in ("ESCALATE", "ABSTAIN"):
            routing = RoutingDecision(next_step="human_required", notes=routing_notes)
        elif decision == "DENY":
            routing = RoutingDecision(next_step="review_queue", notes=routing_notes)
        elif decision == "ALLOW_WITH_CONSTRAINTS":
            routing = RoutingDecision(next_step="policy_gate_required", notes=routing_notes)
        else:
            routing = RoutingDecision(next_step="safe_to_execute", notes=routing_notes)

        provenance_inputs = {
            "model_output_hash": f"sha256:{_hash_payload(model_output_text)}",
            "context_hash": f"sha256:{_hash_payload(context)}",
            "proposed_action_hash": f"sha256:{_hash_payload(payload.proposed_action.model_dump(mode='json'))}",
            "policy_profile": payload.policy_profile,
        }
        if payload.evidence_inputs:
            provenance_inputs["evidence_inputs_hash"] = f"sha256:{_hash_payload(payload.evidence_inputs.model_dump(mode='json'))}"
        if payload.model_metadata:
            provenance_inputs["model_metadata_hash"] = f"sha256:{_hash_payload(payload.model_metadata.model_dump(mode='json'))}"

        evidence_bundle = build_evidence_bundle(
            decision=decision,
            confidence=confidence,
            critic_details=critic_details,
            precedent_alignment=precedent_alignment,
            governance_result=governance_result,
            provenance_inputs=provenance_inputs,
        )

        return EvaluateResponse(
            request_id=payload.request_id,
            engine_version=ELEANOR_VERSION,
            decision=decision,
            confidence=confidence,
            uncertainty=uncertainty_envelope,
            constraints=constraints,
            routing=routing,
            evidence_bundle=evidence_bundle,
            errors=[],
        )

    except InputValidationError as exc:
        logger.warning(
            "evaluation_input_validation_failed",
            extra={
                "request_id": payload.request_id,
                "error": exc.message,
                "validation_type": exc.details.get("validation_type"),
                "field": exc.details.get("field"),
            },
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "invalid_input", "message": exc.message, "details": exc.details},
        )
    except HTTPException:
        raise
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        logger.warning(
            "evaluation_validation_error",
            extra={"request_id": payload.request_id, "error": str(exc), "error_type": type(exc).__name__},
        )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid request: {str(exc)}")
    except Exception as exc:
        logger.error(
            "evaluation_failed",
            extra={"request_id": payload.request_id, "error": str(exc), "error_type": type(exc).__name__},
        )

        error = EvaluateError(code="E_ENGINE_ERROR", message=str(exc))
        uncertainty_env = UncertaintyEnvelope(level="HIGH", reasons=[str(exc)])
        routing = RoutingDecision(next_step="human_required", notes="Engine error; manual review required.")

        bundle_payload = {
            "summary": "Engine error; abstained from decision.",
            "critic_outputs": [],
            "precedent_trace": [],
            "policy_trace": [],
            "provenance": {"inputs": {}},
        }
        bundle_hash = f"sha256:{_hash_payload(bundle_payload)}"
        evidence_bundle = EvidenceBundle(
            summary="Engine error; abstained from decision.",
            critic_outputs=[],
            precedent_trace=[],
            policy_trace=[],
            provenance=EvidenceProvenance(inputs={}),
            integrity=EvidenceIntegrity(hash=bundle_hash),
        )

        return EvaluateResponse(
            request_id=payload.request_id,
            engine_version=ELEANOR_VERSION,
            decision="ABSTAIN",
            confidence=0.0,
            uncertainty=uncertainty_env,
            constraints=None,
            routing=routing,
            evidence_bundle=evidence_bundle,
            errors=[error],
        )
