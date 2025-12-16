"""
Execution entrypoint that only accepts constitutionally cleared decisions.
"""

from engine.execution.runner import ensure_executable
from engine.schemas.escalation import ExecutableDecision


def execute_decision(decision: ExecutableDecision) -> None:
    """
    Execute a decision that has already passed human review enforcement.

    Only ExecutableDecision objects are accepted; AggregationResult or raw critic
    output must never reach this layer.
    """
    ensure_executable(decision)

    _perform_execution(decision)


def _perform_execution(decision: ExecutableDecision) -> None:
    """
    Placeholder for the real side-effecting execution.
    """
    # Side effects (API calls, state mutation, etc.) would live here.
    pass
