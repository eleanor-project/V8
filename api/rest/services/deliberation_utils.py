import hashlib
import json
import time
from typing import Any, Dict, List, Literal, Optional

from engine.logging_config import get_logger
from engine.utils.critic_names import canonicalize_critic_map

from api.schemas import (
    EvidenceBundle,
    EvidenceIntegrity,
    EvidenceProvenance,
    CriticEvidence,
    PrecedentTrace,
    PolicyTrace,
    UncertaintyEnvelope,
)

from engine.execution.human_review import enforce_human_review
from engine.schemas.escalation import AggregationResult, HumanAction, ExecutableDecision

logger = get_logger(__name__)


def resolve_final_decision(aggregator_decision: Optional[str], opa_result: Dict[str, Any]) -> str:
    """Combine aggregator decision with OPA governance outcome."""
    if opa_result.get("allow") is False:
        if opa_result.get("escalate") is True:
            return "escalate"
        return "deny"
    if opa_result.get("escalate") is True:
        return "escalate"
    return aggregator_decision or "allow"


def map_assessment_label(decision: Optional[str]) -> str:
    """Map internal decision labels to non-coercive assessment language."""
    if not decision:
        return "requires_human_review"
    mapping = {
        "allow": "aligned",
        "constrained_allow": "aligned_with_constraints",
        "deny": "misaligned",
        "escalate": "requires_human_review",
    }
    normalized = str(decision).lower()
    return mapping.get(normalized, normalized)


def normalize_engine_result(
    result_obj: Any, input_text: str, trace_id: str, context: Dict[str, Any]
) -> Dict[str, Any]:
    """Normalize EngineResult or dict into a common shape used by the API."""
    result = (
        result_obj.model_dump()
        if hasattr(result_obj, "model_dump")
        else result_obj.dict()
        if hasattr(result_obj, "dict")
        else result_obj
    )

    model_info = result.get("model_info") or {}
    model_used = (
        model_info.get("model_name")
        if isinstance(model_info, dict)
        else getattr(model_info, "model_name", "unknown")
    )

    aggregated = result.get("aggregated") or result.get("aggregator_output") or {}
    aggregated = aggregated if isinstance(aggregated, dict) else {}

    critic_findings = result.get("critic_findings") or result.get("critics") or {}
    critic_outputs = {
        k: (v.model_dump() if hasattr(v, "model_dump") else v) for k, v in critic_findings.items()
    }
    critic_outputs = canonicalize_critic_map(critic_outputs)

    precedent_alignment = result.get("precedent_alignment") or result.get("precedent") or {}
    uncertainty = result.get("uncertainty") or {}

    model_output = aggregated.get("final_output") if aggregated else None
    model_output = model_output or result.get("output_text")

    degraded_components = (
        result.get("degraded_components") or aggregated.get("degraded_components") or []
    )
    is_degraded = result.get("is_degraded")
    if is_degraded is None:
        is_degraded = aggregated.get("is_degraded", False)

    return {
        "trace_id": result.get("trace_id", trace_id),
        "timestamp": time.time(),
        "model_used": model_used,
        "model_output": model_output,
        "critic_outputs": critic_outputs,
        "precedent": precedent_alignment,
        "precedent_alignment": precedent_alignment,
        "uncertainty": uncertainty,
        "aggregator_output": aggregated,
        "input": input_text,
        "context": context,
        "degraded_components": degraded_components,
        "is_degraded": bool(is_degraded),
    }


def _safe_json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def _hash_payload(payload: Any) -> str:
    raw = _safe_json_dumps(payload).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def serialize_model_output(model_output: Any) -> str:
    if isinstance(model_output, str):
        return model_output
    return json.dumps(model_output, ensure_ascii=True, default=str)


def _severity_score(data: Dict[str, Any]) -> float:
    for key in ("final_severity", "severity"):
        if data.get(key) is not None:
            try:
                return float(data[key])
            except (TypeError, ValueError):
                continue
    if data.get("score") is not None:
        try:
            return float(data["score"]) * 3.0
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _critic_verdict(severity: float) -> Literal["PASS", "WARN", "FAIL"]:
    if severity >= 2.0:
        return "FAIL"
    if severity >= 1.0:
        return "WARN"
    return "PASS"


def build_uncertainty_envelope(uncertainty: Dict[str, Any]) -> UncertaintyEnvelope:
    overall = 0.0
    try:
        overall = float(uncertainty.get("overall_uncertainty", 0.0))
    except (TypeError, ValueError):
        overall = 0.0

    if overall >= 0.6:
        level: Literal["LOW", "MEDIUM", "HIGH"] = "HIGH"
    elif overall >= 0.3:
        level = "MEDIUM"
    else:
        level = "LOW"

    reasons: List[str] = []
    explanation = uncertainty.get("explanation")
    if explanation:
        reasons.append(str(explanation))
    if uncertainty.get("needs_escalation"):
        reasons.append("Uncertainty threshold exceeded")

    return UncertaintyEnvelope(level=level, reasons=reasons)


def confidence_from_uncertainty(uncertainty: Dict[str, Any]) -> float:
    try:
        overall = float(uncertainty.get("overall_uncertainty", 0.0))
    except (TypeError, ValueError):
        overall = 0.0
    return max(0.0, min(1.0, 1.0 - overall))


def map_decision(
    final_decision: Optional[str],
) -> Literal["ALLOW", "ALLOW_WITH_CONSTRAINTS", "ABSTAIN", "ESCALATE", "DENY"]:
    if not final_decision:
        return "ABSTAIN"
    decision = str(final_decision).lower()
    if decision == "allow":
        return "ALLOW"
    if decision == "constrained_allow":
        return "ALLOW_WITH_CONSTRAINTS"
    if decision == "deny":
        return "DENY"
    if decision == "escalate":
        return "ESCALATE"
    return "ABSTAIN"


def build_constraints(critic_details: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    advisories = []
    for name, data in critic_details.items():
        severity = _severity_score(data)
        if severity < 1.0:
            continue
        rationale = data.get("justification") or data.get("rationale") or ""
        if not rationale and data.get("violations"):
            rationale = str(data.get("violations")[0])
        advisories.append({"critic": name, "severity": severity, "note": rationale})

    if not advisories:
        return None
    return {"advisories": advisories}


def build_precedent_trace(precedent_alignment: Dict[str, Any]) -> List[PrecedentTrace]:
    retrieval = precedent_alignment.get("retrieval") or {}
    cases = retrieval.get("precedent_cases") or retrieval.get("cases") or []
    traces: List[PrecedentTrace] = []
    for case in cases:
        meta = case.get("metadata") or {}
        case_id = (
            meta.get("id")
            or meta.get("case_id")
            or case.get("id")
            or case.get("case_id")
            or meta.get("title")
        )
        if not case_id:
            case_id = "precedent"
        traces.append(
            PrecedentTrace(
                id=str(case_id),
                type=meta.get("type") or meta.get("category") or "internal_precedent",
                applied_as=meta.get("applied_as") or "supporting",
                note=meta.get("summary") or meta.get("note"),
            )
        )
    return traces


def build_policy_trace(governance_result: Dict[str, Any]) -> List[PolicyTrace]:
    failures = governance_result.get("failures") or []
    traces: List[PolicyTrace] = []
    for failure in failures:
        traces.append(
            PolicyTrace(
                policy=str(failure.get("policy", "unknown")),
                result="FAIL",
                reason=failure.get("reason"),
            )
        )
    allow = governance_result.get("allow")
    if allow is True:
        traces.append(PolicyTrace(policy="opa", result="PASS", reason=None))
    elif allow is False and governance_result.get("escalate"):
        traces.append(PolicyTrace(policy="opa", result="WARN", reason="Escalation required"))
    elif allow is False:
        traces.append(PolicyTrace(policy="opa", result="FAIL", reason="Denied"))
    return traces


def build_critic_evidence(critic_details: Dict[str, Any]) -> List[CriticEvidence]:
    items: List[CriticEvidence] = []
    for name, data in critic_details.items():
        severity = _severity_score(data)
        verdict = _critic_verdict(severity)
        items.append(
            CriticEvidence(
                critic=name,
                verdict=verdict,
                severity=severity,
                note=data.get("justification") or data.get("rationale"),
            )
        )
    return items


def build_evidence_bundle(
    *,
    decision: str,
    confidence: float,
    critic_details: Dict[str, Any],
    precedent_alignment: Dict[str, Any],
    governance_result: Dict[str, Any],
    provenance_inputs: Dict[str, Any],
) -> EvidenceBundle:
    critic_outputs = build_critic_evidence(critic_details)
    precedent_trace = build_precedent_trace(precedent_alignment)
    policy_trace = build_policy_trace(governance_result)
    summary = f"Decision {decision} with confidence {confidence:.2f}."

    provenance = EvidenceProvenance(inputs=provenance_inputs)
    bundle_payload = {
        "summary": summary,
        "critic_outputs": [c.model_dump(mode="json") for c in critic_outputs],
        "precedent_trace": [p.model_dump(mode="json") for p in precedent_trace],
        "policy_trace": [p.model_dump(mode="json") for p in policy_trace],
        "provenance": provenance.model_dump(mode="json"),
    }
    bundle_hash = f"sha256:{_hash_payload(bundle_payload)}"
    integrity = EvidenceIntegrity(hash=bundle_hash)

    return EvidenceBundle(
        summary=summary,
        critic_outputs=critic_outputs,
        precedent_trace=precedent_trace,
        policy_trace=policy_trace,
        provenance=provenance,
        integrity=integrity,
    )


def resolve_execution_decision(
    aggregated: Dict[str, Any],
    human_action: Optional[HumanAction],
) -> Optional[ExecutableDecision]:
    aggregation_payload = aggregated.get("aggregation_result") if isinstance(aggregated, dict) else None
    if not aggregation_payload:
        return None
    try:
        aggregation_result = AggregationResult.model_validate(aggregation_payload)
    except Exception as exc:
        logger.error("Invalid aggregation_result payload", extra={"error": str(exc)})
        raise RuntimeError("Invalid aggregation_result payload") from exc
    return enforce_human_review(aggregation_result=aggregation_result, human_action=human_action)


def apply_execution_gate(final_decision: str, execution_decision: Optional[ExecutableDecision]) -> str:
    if execution_decision and not execution_decision.executable and final_decision != "deny":
        return "escalate"
    return final_decision


__all__ = [
    "resolve_final_decision",
    "map_assessment_label",
    "normalize_engine_result",
    "serialize_model_output",
    "build_uncertainty_envelope",
    "confidence_from_uncertainty",
    "map_decision",
    "build_constraints",
    "build_evidence_bundle",
    "resolve_execution_decision",
    "apply_execution_gate",
    "_hash_payload",
]
