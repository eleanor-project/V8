import pytest

from engine.aggregator.escalation import resolve_escalation
from engine.execution.human_review import enforce_human_review
from engine.execution.runner import ensure_executable, ExecutionBlocked
from engine.execution.executor import execute_decision
from engine.schemas.escalation import ExecutableDecision


def test_ensure_executable_blocks_when_not_executable(critic_eval_tier2):
    aggregation = resolve_escalation(
        critic_evaluations=[critic_eval_tier2],
        synthesis="Consent issue",
    )
    decision = enforce_human_review(
        aggregation_result=aggregation,
        human_action=None,
    )

    with pytest.raises(ExecutionBlocked):
        ensure_executable(decision)


def test_ensure_executable_allows_when_valid(
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

    ensure_executable(decision)  # should not raise


def test_execute_requires_executable_decision(aggregation_result_fixture):
    """
    Guard invariant:
    Raw AggregationResult must never be executable.
    """
    with pytest.raises(ExecutionBlocked):
        execute_decision(aggregation_result_fixture)  # type: ignore[arg-type]


def test_non_executable_decision_cannot_run(blocked_decision_fixture):
    with pytest.raises(ExecutionBlocked):
        # blocked_decision_fixture is ExecutableDecision but executable=False
        execute_decision(blocked_decision_fixture)  # type: ignore[arg-type]
