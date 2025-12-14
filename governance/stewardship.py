"""
Stewardship functions for Eleanor governance system.

Coordinates the human review process and manages review packet emission.
This is the bridge between automated reasoning and human oversight.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import json
from datetime import datetime
import uuid
from types import SimpleNamespace

from .review_packets import ReviewPacket, build_review_packet
from .review_triggers import ReviewTriggerEvaluator, Case


# Storage for pending reviews
PENDING_REVIEWS_PATH = Path("governance/pending_reviews")
PENDING_REVIEWS_PATH.mkdir(parents=True, exist_ok=True)


def emit_review(packet: ReviewPacket) -> Dict[str, Any]:
    """
    Emit a review packet for human review.

    This is async and non-blocking. The system does not wait for human review
    to complete before responding to the user. It only blocks PROMOTION.

    Args:
        packet: Immutable ReviewPacket to emit

    Returns:
        dict with status and packet metadata
    """
    # Store packet for pending review
    packet_file = PENDING_REVIEWS_PATH / f"{packet.case_id}.json"

    with open(packet_file, "w") as f:
        json.dump(packet.dict(), f, indent=2, default=str)

    # In production, this would also:
    # - Send to review queue
    # - Notify reviewers
    # - Trigger UI updates
    # - Log to audit trail

    return {
        "status": "review_pending",
        "case_id": packet.case_id,
        "triggers": packet.triggers,
        "severity": packet.severity,
    }


def should_review(
    case_data: Dict[str, Any],
    evaluator: Optional[ReviewTriggerEvaluator] = None
) -> Dict[str, Any]:
    """
    Determine if a case requires human review.

    Args:
        case_data: Dict with case information (severity, critic_outputs, etc.)
        evaluator: Optional custom trigger evaluator

    Returns:
        dict with review_required flag and triggers
    """
    if evaluator is None:
        evaluator = ReviewTriggerEvaluator()

    # Build Case object from data
    case = Case(
        severity=case_data.get("severity", 0.0),
        critic_disagreement=_calculate_critic_disagreement(
            case_data.get("critic_outputs", {})
        ),
        novel_precedent=case_data.get("novel_precedent", False),
        rights_impacted=case_data.get("rights_impacted", []),
        uncertainty_flags=case_data.get("uncertainty_flags", []),
    )

    return evaluator.evaluate(case)


def create_and_emit_review_packet(
    case_id: str,
    domain: str,
    severity: float,
    uncertainty_flags: list,
    critic_outputs: Dict[str, Any],
    aggregator_summary: str,
    dissent: Optional[str],
    citations: Dict[str, list],
    triggers: list,
) -> Dict[str, Any]:
    """
    Create and emit a review packet in one operation.

    Args:
        case_id: Unique case identifier
        domain: Domain/context
        severity: Impact magnitude
        uncertainty_flags: Uncertainty indicators
        critic_outputs: Raw critic outputs
        aggregator_summary: Synthesized summary
        dissent: Preserved dissent if present
        citations: Precedent citations
        triggers: Triggers that fired

    Returns:
        dict with emission status
    """
    case = Case(
        severity=severity,
        critic_disagreement=_calculate_critic_disagreement(critic_outputs),
        novel_precedent=False,
        rights_impacted=[],
        uncertainty_flags=uncertainty_flags,
        uncertainty=SimpleNamespace(flags=uncertainty_flags),
    )

    # Attach additional fields used by packet builder
    for key, value in {
        "id": case_id,
        "domain": domain,
        "critic_outputs": critic_outputs,
        "aggregator_summary": aggregator_summary,
        "dissent": dissent,
        "citations": citations,
    }.items():
        setattr(case, key, value)

    # build_review_packet expects a case-like object and review_decision dict
    packet = build_review_packet(
        case,
        {"triggers": triggers, "review_required": True},
    )

    return emit_review(packet)


def get_pending_reviews() -> list[Dict[str, Any]]:
    """
    Get all pending review packets.

    Returns:
        List of pending review packets
    """
    pending = []

    for packet_file in PENDING_REVIEWS_PATH.glob("*.json"):
        with open(packet_file, "r") as f:
            data = json.load(f)
        pending.append(data)

    return pending


def resolve_review(case_id: str) -> bool:
    """
    Mark a review as resolved (after human review is submitted).

    Args:
        case_id: Case identifier

    Returns:
        bool indicating if review was found and resolved
    """
    packet_file = PENDING_REVIEWS_PATH / f"{case_id}.json"

    if not packet_file.exists():
        return False

    # Move to resolved
    resolved_path = Path("governance/resolved_reviews")
    resolved_path.mkdir(parents=True, exist_ok=True)

    packet_file.rename(resolved_path / f"{case_id}.json")
    return True


def _calculate_critic_disagreement(critic_outputs: Dict[str, Any]) -> float:
    """
    Calculate disagreement level among critics.

    Uses severity variance as a proxy for disagreement.

    Args:
        critic_outputs: Dict of critic outputs

    Returns:
        float between 0.0 and 1.0 representing disagreement level
    """
    if not critic_outputs:
        return 0.0

    severities = []
    for critic_data in critic_outputs.values():
        if isinstance(critic_data, dict) and "severity" in critic_data:
            severities.append(critic_data["severity"])

    if len(severities) < 2:
        return 0.0

    # Calculate variance
    mean_severity = sum(severities) / len(severities)
    variance = sum((s - mean_severity) ** 2 for s in severities) / len(severities)

    # Normalize to 0-1 range (max variance is ~1.56 for 0-2.5 range)
    normalized_disagreement = min(variance / 1.56, 1.0)

    return normalized_disagreement
