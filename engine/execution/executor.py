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
    Side-effecting execution hook.

    Replace or extend this with the actual invocation of downstream systems
    (e.g., workflow engines, actuators, or API calls). The default behavior
    intentionally raises to prevent silent no-ops.
    """
    raise RuntimeError(
        "Execution backend is not configured. Implement _perform_execution to call your "
        "side-effecting systems."
    )
