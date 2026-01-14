import json as _json
import logging
from importlib import import_module
from typing import Any, Dict, Optional

from engine.exceptions import EleanorV8Exception, InputValidationError
from engine.schemas.pipeline_types import AggregationOutput, PrecedentAlignmentResult, UncertaintyResult
from engine.utils.validation import sanitize_for_logging
from engine.validation import validate_input

logger = logging.getLogger("engine.engine")


def emit_error(
    engine: Any,
    exc: Exception,
    *,
    stage: str,
    trace_id: Optional[str] = None,
    critic: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "stage": stage,
        "trace_id": trace_id,
        "critic": critic,
        "exception_type": type(exc).__name__,
        "error_message": str(exc),
    }
    if context is not None:
        payload["context_keys"] = list(context.keys())
    if isinstance(exc, EleanorV8Exception):
        payload.update(exc.details)
    if extra:
        payload.update(extra)

    if "message" in payload:
        payload["error_message"] = payload.pop("message")

    logger.error("engine_error", extra=payload, exc_info=True)

    if engine.error_monitor:
        try:
            engine.error_monitor(exc, payload)
        except Exception:
            logger.debug("Error monitor hook failed", exc_info=True)


def emit_validation_error(
    engine: Any,
    exc: InputValidationError,
    *,
    text: Any,
    context: Any,
    trace_id: Optional[str],
    json_module: Any = _json,
) -> None:
    safe_text = sanitize_for_logging(str(text), max_length=500)
    if isinstance(context, dict):
        safe_context_keys = context
    else:
        safe_context_keys = None
    try:
        context_payload = json_module.dumps(context, default=str, ensure_ascii=True)
    except Exception:
        context_payload = str(context)
    safe_context_excerpt = sanitize_for_logging(context_payload, max_length=500)
    emit_error(
        engine,
        exc,
        stage="validation",
        trace_id=trace_id,
        context=safe_context_keys,
        extra={
            "input_excerpt": safe_text,
            "context_excerpt": safe_context_excerpt,
        },
    )


def validate_inputs(
    engine: Any,
    text: str,
    context: Optional[dict],
    trace_id: Optional[str],
    detail_level: Optional[int],
) -> tuple[str, Dict[str, Any], str, int]:
    validate_fn = None
    try:
        import engine.engine as engine_module

        validate_fn = getattr(engine_module, "validate_input", None)
    except Exception:
        validate_fn = None

    if validate_fn is None:
        try:
            validation_module = import_module("engine.validation")
            validate_fn = getattr(validation_module, "validate_input", validate_input)
        except Exception:
            validate_fn = validate_input

    validated = validate_fn(text, context=context, trace_id=trace_id)
    level = detail_level or engine.config.detail_level
    if level not in (1, 2, 3):
        raise InputValidationError(
            "detail_level must be between 1 and 3",
            validation_type="range_error",
            field="detail_level",
            context={"detail_level": level},
        )
    return validated.text, validated.context, validated.trace_id, level


def build_critic_error_result(
    critic_name: str,
    error: Exception,
    duration_ms: Optional[float] = None,
    *,
    degraded: bool = False,
    degradation_reason: Optional[str] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "critic": critic_name,
        "severity": 0.0,
        "score": 0.0,
        "violations": [],
        "justification": f"critic_error:{type(error).__name__}",
        "duration_ms": duration_ms if duration_ms is not None else 0.0,
        "error": str(error),
    }
    if degraded:
        payload["degraded"] = True
        if degradation_reason:
            payload["degradation_reason"] = degradation_reason
    return payload


def build_aggregation_fallback(
    model_response: str,
    precedent_data: Optional[PrecedentAlignmentResult],
    uncertainty_data: Optional[UncertaintyResult],
    error: Exception,
) -> AggregationOutput:
    return {
        "decision": "requires_human_review",
        "final_output": model_response,
        "score": {"average_severity": 0.0, "total_severity": 0.0},
        "rights_impacted": [],
        "dissent": None,
        "precedent": precedent_data or {},
        "uncertainty": uncertainty_data or {},
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
    }
