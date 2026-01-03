from __future__ import annotations

from typing import Optional
from datetime import datetime, timezone
import uuid

from engine.schemas.escalation import (
    AggregationResult,
    ExecutableDecision,
    HumanAction,
    HumanActionType,
    EscalationTier,
    AuditRecord,
)
from engine.execution.audit_store import store_audit_record


# ---------------------------------------------------------
# Canonical Statements (Rubber-stamp resistant)
# ---------------------------------------------------------

TIER_2_ACK_STATEMENT = (
    "I acknowledge the identified constitutional risks and accept responsibility for proceeding."
)

TIER_3_DETERMINATION_STATEMENT = "I affirmatively determine the appropriate course of action in light of the identified constitutional risks."


# ---------------------------------------------------------
# Human Review Enforcement
# ---------------------------------------------------------


def enforce_human_review(
    *,
    aggregation_result: AggregationResult,
    human_action: Optional[HumanAction],
) -> ExecutableDecision:
    """
    Enforces human review requirements based on escalation tier.

    - Blocks execution if required human action is missing or invalid
    - Validates action type and statement
    - Produces an ExecutableDecision with immutable audit linkage
    """

    gate = aggregation_result.execution_gate

    # No escalation → executable
    if not gate.gated:
        return ExecutableDecision(
            aggregation_result=aggregation_result,
            human_action=None,
            executable=True,
            execution_reason="No escalation present. Automatic execution permitted.",
            audit_record_id=_create_audit_record(
                aggregation_result=aggregation_result,
                human_action=None,
            ).record_id,
        )

    # Escalation present → human action required
    if human_action is None:
        return _blocked_decision(
            aggregation_result,
            "Execution blocked: required human review not provided.",
        )

    if gate.required_action is None or gate.escalation_tier is None:
        return _blocked_decision(
            aggregation_result,
            "Execution blocked: escalation gate is missing required action metadata.",
        )

    # Validate human action
    _validate_human_action(
        required_action=gate.required_action,
        escalation_tier=gate.escalation_tier,
        human_action=human_action,
        aggregation_result=aggregation_result,
    )

    # Success: execution permitted with human ownership
    audit_record = _create_audit_record(
        aggregation_result=aggregation_result,
        human_action=human_action,
    )

    return ExecutableDecision(
        aggregation_result=aggregation_result,
        human_action=human_action,
        executable=True,
        execution_reason="Required human review satisfied. Execution permitted.",
        audit_record_id=audit_record.record_id,
    )


# ---------------------------------------------------------
# Validation Logic
# ---------------------------------------------------------


def _validate_human_action(
    *,
    required_action: HumanActionType,
    escalation_tier: EscalationTier,
    human_action: HumanAction,
    aggregation_result: AggregationResult,
) -> None:
    """
    Enforces strict matching between escalation tier and human action.
    """

    if human_action.action_type != required_action:
        raise ValueError(
            f"Invalid human action type. "
            f"Required: {required_action.value}, "
            f"Provided: {human_action.action_type.value}"
        )

    # Statement enforcement
    if escalation_tier == EscalationTier.TIER_2:
        if human_action.statement.strip() != TIER_2_ACK_STATEMENT:
            raise ValueError("Invalid acknowledgment statement for Tier 2 escalation.")

    if escalation_tier == EscalationTier.TIER_3:
        if human_action.statement.strip() != TIER_3_DETERMINATION_STATEMENT:
            raise ValueError("Invalid determination statement for Tier 3 escalation.")

    # Ensure linkage to escalation signals
    required_ids = {
        f"{s.critic_id}:{s.clause_id}"
        for s in aggregation_result.escalation_summary.triggering_signals
    }

    provided_ids = {f"{s.critic_id}:{s.clause_id}" for s in human_action.linked_escalations}

    if required_ids - provided_ids:
        raise ValueError("Human action does not reference all triggering escalation signals.")


# ---------------------------------------------------------
# Blocking Helper
# ---------------------------------------------------------


def _blocked_decision(
    aggregation_result: AggregationResult,
    reason: str,
) -> ExecutableDecision:
    return ExecutableDecision(
        aggregation_result=aggregation_result,
        human_action=None,
        executable=False,
        execution_reason=reason,
        audit_record_id=_create_audit_record(
            aggregation_result=aggregation_result,
            human_action=None,
        ).record_id,
    )


# ---------------------------------------------------------
# Audit Record Creation (Immutable)
# ---------------------------------------------------------


def _create_audit_record(
    *,
    aggregation_result: AggregationResult,
    human_action: Optional[HumanAction],
) -> AuditRecord:
    """
    Creates an immutable audit record binding aggregation + human action.
    """

    record = AuditRecord(
        record_id=str(uuid.uuid4()),
        aggregation_hash=aggregation_result.audit_hash,
        escalation_signals=aggregation_result.escalation_summary.triggering_signals,
        human_action=human_action,
        created_at=datetime.now(timezone.utc),
    )

    store_audit_record(record)

    return record
