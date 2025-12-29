from datetime import datetime, timedelta

from engine.aggregator.escalation import _compute_audit_hash
from engine.schemas.escalation import (
    CriticEvaluation,
    EscalationSummary,
    EscalationSignal,
    EscalationTier,
)


def _signal(ts: datetime) -> EscalationSignal:
    return EscalationSignal(
        tier=EscalationTier.TIER_2,
        critic_id="rights",
        clause_id="A1",
        clause_description="Dignity violation",
        doctrine_ref="UDHR-1",
        rationale="Test rationale",
        blocking=False,
        timestamp=ts,
    )


def _evaluation(ts: datetime, signal: EscalationSignal) -> CriticEvaluation:
    return CriticEvaluation(
        critic_id="rights",
        charter_version="8.0",
        concerns=[],
        escalation=signal,
        severity_score=0.7,
        citations=["UDHR-1"],
        uncertainty=None,
        revision=0,
        timestamp=ts,
    )


def _summary(signal: EscalationSignal) -> EscalationSummary:
    return EscalationSummary(
        highest_tier=EscalationTier.TIER_2,
        triggering_signals=[signal],
        critics_triggered=["rights"],
        explanation="Test escalation summary",
    )


def test_audit_hash_is_deterministic_across_timestamps():
    base = datetime(2025, 1, 1, 0, 0, 0)
    later = base + timedelta(seconds=30)

    signal_a = _signal(base)
    signal_b = _signal(later)

    eval_a = _evaluation(base, signal_a)
    eval_b = _evaluation(later, signal_b)

    summary_a = _summary(signal_a)
    summary_b = _summary(signal_b)

    hash_a = _compute_audit_hash(
        synthesis="Test synthesis",
        critic_evaluations=[eval_a],
        escalation_summary=summary_a,
    )
    hash_b = _compute_audit_hash(
        synthesis="Test synthesis",
        critic_evaluations=[eval_b],
        escalation_summary=summary_b,
    )

    assert hash_a == hash_b


def test_audit_hash_changes_when_content_changes():
    ts = datetime(2025, 1, 1, 0, 0, 0)
    signal = _signal(ts)
    evaluation = _evaluation(ts, signal)
    summary = _summary(signal)

    hash_a = _compute_audit_hash(
        synthesis="Test synthesis",
        critic_evaluations=[evaluation],
        escalation_summary=summary,
    )
    hash_b = _compute_audit_hash(
        synthesis="Different synthesis",
        critic_evaluations=[evaluation],
        escalation_summary=summary,
    )

    assert hash_a != hash_b
