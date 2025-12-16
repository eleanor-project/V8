from engine.aggregator.escalation import resolve_escalation
from engine.schemas.escalation import EscalationTier


def test_no_escalation_allows_execution(critic_eval_no_escalation):
    result = resolve_escalation(
        critic_evaluations=[critic_eval_no_escalation],
        synthesis="All good",
    )

    assert result.execution_gate.gated is False
    assert result.execution_gate.escalation_tier is None


def test_tier2_escalation_blocks_execution(critic_eval_tier2):
    result = resolve_escalation(
        critic_evaluations=[critic_eval_tier2],
        synthesis="Consent unclear",
    )

    assert result.execution_gate.gated is True
    assert result.execution_gate.escalation_tier == EscalationTier.TIER_2


def test_highest_tier_wins(critic_eval_tier2, critic_eval_tier3):
    result = resolve_escalation(
        critic_evaluations=[critic_eval_tier2, critic_eval_tier3],
        synthesis="Multiple concerns",
    )

    assert result.execution_gate.escalation_tier == EscalationTier.TIER_3
    assert result.execution_gate.gated is True
