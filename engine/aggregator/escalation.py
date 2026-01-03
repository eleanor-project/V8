from __future__ import annotations

import hashlib
import json
from typing import List

from engine.schemas.escalation import (
    CriticEvaluation,
    EscalationSignal,
    EscalationTier,
    EscalationSummary,
    ExecutionGate,
    HumanActionType,
    AggregationResult,
)


# ---------------------------------------------------------
# Core Escalation Resolution Logic
# ---------------------------------------------------------


def resolve_escalation(
    *,
    critic_evaluations: List[CriticEvaluation],
    synthesis: str,
) -> AggregationResult:
    """
    Resolve escalation across critic evaluations.

    - Collects all escalation signals
    - Determines highest escalation tier
    - Constructs a binding execution gate
    - Preserves dissent verbatim
    """

    escalation_signals: List[EscalationSignal] = [
        ev.escalation for ev in critic_evaluations if ev.escalation is not None
    ]

    # -----------------------------------------------------
    # Determine highest escalation tier (if any)
    # -----------------------------------------------------

    highest_tier: EscalationTier | None = None

    if escalation_signals:
        highest_tier = max(
            (signal.tier for signal in escalation_signals),
            key=lambda t: _tier_rank(t),
        )

    # -----------------------------------------------------
    # Build Execution Gate
    # -----------------------------------------------------

    execution_gate = _build_execution_gate(
        highest_tier=highest_tier,
        escalation_signals=escalation_signals,
    )

    # -----------------------------------------------------
    # Escalation Summary
    # -----------------------------------------------------

    escalation_summary = EscalationSummary(
        highest_tier=highest_tier,
        triggering_signals=escalation_signals,
        critics_triggered=list({signal.critic_id for signal in escalation_signals}),
        explanation=_build_escalation_explanation(
            highest_tier=highest_tier,
            signals=escalation_signals,
        ),
    )

    # -----------------------------------------------------
    # Dissent detection
    # -----------------------------------------------------

    dissent_present = _detect_dissent(critic_evaluations)

    # -----------------------------------------------------
    # Audit Hash (immutable linkage)
    # -----------------------------------------------------

    audit_hash = _compute_audit_hash(
        synthesis=synthesis,
        critic_evaluations=critic_evaluations,
        escalation_summary=escalation_summary,
    )

    return AggregationResult(
        synthesis=synthesis,
        critic_evaluations=critic_evaluations,
        escalation_summary=escalation_summary,
        execution_gate=execution_gate,
        dissent_present=dissent_present,
        audit_hash=audit_hash,
    )


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------


def _tier_rank(tier: EscalationTier) -> int:
    """
    Higher number = higher authority boundary.
    """
    return {
        EscalationTier.TIER_2: 2,
        EscalationTier.TIER_3: 3,
    }[tier]


def _build_execution_gate(
    *,
    highest_tier: EscalationTier | None,
    escalation_signals: List[EscalationSignal],
) -> ExecutionGate:
    if highest_tier is None:
        return ExecutionGate(
            gated=False,
            required_action=None,
            reason="No escalation triggered.",
            escalation_tier=None,
        )

    if highest_tier == EscalationTier.TIER_2:
        return ExecutionGate(
            gated=True,
            required_action=HumanActionType.HUMAN_ACK,
            reason=_gate_reason(escalation_signals),
            escalation_tier=highest_tier,
        )

    if highest_tier == EscalationTier.TIER_3:
        return ExecutionGate(
            gated=True,
            required_action=HumanActionType.HUMAN_DETERMINATION,
            reason=_gate_reason(escalation_signals),
            escalation_tier=highest_tier,
        )

    raise RuntimeError(f"Unhandled escalation tier: {highest_tier}")


def _gate_reason(signals: List[EscalationSignal]) -> str:
    clauses = ", ".join(f"{s.critic_id}:{s.clause_id}" for s in signals)
    return f"Escalation triggered by constitutional clauses: {clauses}"


def _build_escalation_explanation(
    *,
    highest_tier: EscalationTier | None,
    signals: List[EscalationSignal],
) -> str:
    if highest_tier is None:
        return "No constitutional escalation detected."

    lines = [
        f"Highest escalation tier: {highest_tier.value}",
        "Triggering clauses:",
    ]

    for s in signals:
        lines.append(f"- [{s.critic_id} / {s.clause_id}] {s.clause_description}")

    return "\n".join(lines)


def _detect_dissent(
    critic_evaluations: List[CriticEvaluation],
) -> bool:
    """
    Dissent exists if at least one critic escalates
    while at least one other does not.
    """
    escalated = [ev for ev in critic_evaluations if ev.escalation]
    non_escalated = [ev for ev in critic_evaluations if not ev.escalation]
    return bool(escalated and non_escalated)


def _compute_audit_hash(
    *,
    synthesis: str,
    critic_evaluations: List[CriticEvaluation],
    escalation_summary: EscalationSummary,
) -> str:
    """
    Produces a stable, content-addressable audit hash.
    """

    def _strip_timestamps(value):
        if isinstance(value, dict):
            return {
                k: _strip_timestamps(v)
                for k, v in value.items()
                if k not in ("timestamp", "created_at")
            }
        if isinstance(value, list):
            return [_strip_timestamps(v) for v in value]
        return value

    payload = {
        "synthesis": synthesis,
        "critics": _strip_timestamps([ev.model_dump(mode="json") for ev in critic_evaluations]),
        "escalation": _strip_timestamps(escalation_summary.model_dump(mode="json")),
    }

    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
