import pytest

from engine.aggregator.escalation import resolve_escalation
from engine.execution.human_review import enforce_human_review
from engine.schemas.escalation import HumanActionType


def test_tier2_blocks_without_human_ack(critic_eval_tier2):
    aggregation = resolve_escalation(
        critic_evaluations=[critic_eval_tier2],
        synthesis="Consent issue",
    )

    decision = enforce_human_review(
        aggregation_result=aggregation,
        human_action=None,
    )

    assert decision.executable is False


def test_tier2_allows_with_valid_ack(
    critic_eval_tier2,
    valid_tier2_human_action,
):
    aggregation = resolve_escalation(
        critic_evaluations=[critic_eval_tier2],
        synthesis="Consent issue",
    )

    decision = enforce_human_review(
        aggregation_result=aggregation,
        human_action=valid_tier2_human_action,
    )

    assert decision.executable is True


def test_tier3_blocks_without_determination(critic_eval_tier3):
    aggregation = resolve_escalation(
        critic_evaluations=[critic_eval_tier3],
        synthesis="Privacy violation",
    )

    decision = enforce_human_review(
        aggregation_result=aggregation,
        human_action=None,
    )

    assert decision.executable is False


def test_tier3_allows_with_valid_determination(
    critic_eval_tier3,
    valid_tier3_human_action,
):
    aggregation = resolve_escalation(
        critic_evaluations=[critic_eval_tier3],
        synthesis="Privacy violation",
    )

    decision = enforce_human_review(
        aggregation_result=aggregation,
        human_action=valid_tier3_human_action,
    )

    assert decision.executable is True


def test_wrong_action_type_rejected(
    critic_eval_tier3,
    valid_tier2_human_action,
):
    aggregation = resolve_escalation(
        critic_evaluations=[critic_eval_tier3],
        synthesis="Privacy violation",
    )

    with pytest.raises(ValueError):
        enforce_human_review(
            aggregation_result=aggregation,
            human_action=valid_tier2_human_action,
        )
