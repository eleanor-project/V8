"""
Execution guardrails: ensure that only ExecutableDecision objects (from enforce_human_review)
may proceed to execution.
"""

from engine.schemas.escalation import ExecutableDecision


class ExecutionBlocked(Exception):
    """Raised when execution is attempted without satisfying governance gates."""


def ensure_executable(decision: ExecutableDecision) -> None:
    """
    Hard guard: execution MUST only proceed with a valid, executable decision.
    """
    if not isinstance(decision, ExecutableDecision):
        raise ExecutionBlocked(
            "Execution requires an ExecutableDecision produced by enforce_human_review."
        )

    if not decision.executable:
        raise ExecutionBlocked(f"Execution blocked: {decision.execution_reason}")

    if not decision.audit_record_id:
        raise ExecutionBlocked("Execution blocked: missing audit_record_id on decision.")
