from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from governance.review_triggers import Case
from engine.schemas.pipeline_types import AggregationOutput, CriticResultsMap, PrecedentAlignmentResult, UncertaintyResult
from engine.exceptions import GovernanceEvaluationError


def calculate_critic_disagreement(critic_outputs: CriticResultsMap) -> float:
    severities: List[float] = []
    for critic_data in critic_outputs.values():
        if not isinstance(critic_data, dict):
            continue
        val: Any = critic_data.get("severity")
        if val is None:
            val = critic_data.get("score")
        try:
            severities.append(float(val))
        except (TypeError, ValueError):
            continue

    if len(severities) < 2:
        return 0.0

    mean = sum(severities) / len(severities)
    variance = sum((s - mean) ** 2 for s in severities) / len(severities)
    return min(1.0, variance / 1.56)


def collect_citations(critic_outputs: CriticResultsMap) -> Dict[str, Any]:
    citations = {}
    for critic_name, critic_data in critic_outputs.items():
        if isinstance(critic_data, dict) and "precedent_refs" in critic_data:
            citations[critic_name] = critic_data["precedent_refs"]
    return citations


def collect_uncertainty_flags(uncertainty_data: Optional[UncertaintyResult]) -> List[str]:
    flags: List[str] = []
    if not uncertainty_data:
        return flags

    if uncertainty_data.get("needs_escalation"):
        flags.append("needs_escalation")

    overall = uncertainty_data.get("overall_uncertainty")
    try:
        if overall is not None and float(overall) >= 0.65:
            flags.append("high_overall_uncertainty")
    except (TypeError, ValueError):
        pass

    return flags


def build_case_for_review(
    trace_id: str,
    context: Dict[str, Any],
    aggregated: AggregationOutput,
    critic_results: CriticResultsMap,
    precedent_data: Optional[PrecedentAlignmentResult],
    uncertainty_data: Optional[UncertaintyResult],
) -> Case:
    aggregated = aggregated or {}
    critic_results = critic_results or {}

    severity_raw: Any = (aggregated.get("score") or {}).get("average_severity", 0.0)
    try:
        severity = float(severity_raw)
    except (TypeError, ValueError):
        severity = 0.0

    critic_severities: List[float] = []
    for critic_data in critic_results.values():
        if not isinstance(critic_data, dict):
            continue
        val: Any = critic_data.get("severity")
        if val is None:
            val = critic_data.get("score")
        try:
            critic_severities.append(float(val))
        except (TypeError, ValueError):
            continue
    if not severity and critic_severities:
        severity = max(critic_severities)

    uncertainty_flags = collect_uncertainty_flags(uncertainty_data)
    case_uncertainty = SimpleNamespace(flags=uncertainty_flags)

    case_obj = Case(
        severity=severity,
        critic_disagreement=calculate_critic_disagreement(critic_results),
        novel_precedent=bool((precedent_data or {}).get("novel", False)),
        rights_impacted=aggregated.get("rights_impacted", []),
        uncertainty_flags=uncertainty_flags,
        uncertainty=case_uncertainty,
    )

    for key, value in {
        "id": trace_id,
        "domain": context.get("domain", "general"),
        "critic_outputs": critic_results,
        "aggregator_summary": aggregated.get("final_output", "") or "",
        "dissent": aggregated.get("dissent"),
        "citations": collect_citations(critic_results),
    }.items():
        setattr(case_obj, key, value)

    return case_obj


def run_governance_review_gate(
    engine: Any,
    case: Case,
    *,
    build_review_packet: Any,
    store_review_packet: Any,
) -> None:
    try:
        review_decision = engine.review_trigger_evaluator.evaluate(case)

        if review_decision.get("review_required"):
            review_packet = build_review_packet(case, review_decision)
            store_review_packet(review_packet)

            setattr(
                case,
                "governance_flags",
                {
                    "human_review_required": True,
                    "review_triggers": review_decision.get("triggers", []),
                },
            )
        else:
            setattr(
                case,
                "governance_flags",
                {
                    "human_review_required": False,
                },
            )
    except Exception as review_exc:
        setattr(
            case,
            "governance_flags",
            {
                "human_review_required": False,
                "error": str(review_exc),
            },
        )
        raise GovernanceEvaluationError(
            "Governance review gate failed",
            details={"error": str(review_exc)},
        ) from review_exc
