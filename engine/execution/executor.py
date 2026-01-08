"""engine.execution.executor

Execution entrypoint that only accepts constitutionally cleared decisions.

Production posture:
- Hard gate execution on local invariants (ExecutableDecision)
- Hard gate execution on OPA policy (fail-closed by default)

This keeps "real system" guarantees even if an upstream caller bypasses the API layer.
"""

from __future__ import annotations

from engine.execution.runner import ExecutionBlocked, ensure_executable
from engine.schemas.escalation import ExecutableDecision


def execute_decision(decision: ExecutableDecision) -> None:
    """Execute a decision that has already passed governance + human review enforcement."""
    ensure_executable(decision)

    # Defense-in-depth: re-check the execution gate in OPA right before side effects.
    _enforce_opa_execution_gate(decision)

    _perform_execution(decision)


def _enforce_opa_execution_gate(decision: ExecutableDecision) -> None:
    """Fail-closed execution gate via OPA.

    If OPA is disabled, this is a no-op.
    If OPA is unavailable and fail_open is False, this blocks execution.
    """
    try:
        from engine.governance.opa_enforcer import OPAEnforcer

        enforcer = OPAEnforcer()
        check = enforcer.check_execution(decision.model_dump(mode="json"), action="execute")

        if not check.get("allowed", False):
            reasons = check.get("deny_reasons") or []
            err = check.get("error")
            suffix = f" (error={err})" if err else ""
            if reasons:
                raise ExecutionBlocked(f"Execution blocked by OPA: {', '.join(reasons)}{suffix}")
            raise ExecutionBlocked(f"Execution blocked by OPA{suffix}")

    except ExecutionBlocked:
        raise
    except Exception as exc:
        # Never fail-open here; if OPA is misconfigured, we want a hard stop.
        raise ExecutionBlocked(f"Execution blocked: OPA enforcement error: {exc}")


def _perform_execution(decision: ExecutableDecision) -> None:
    """Side-effecting execution hook.

    Replace or extend this with the actual invocation of downstream systems
    (e.g., workflow engines, actuators, or API calls). The default behavior
    intentionally raises to prevent silent no-ops.
    """
    raise RuntimeError(
        "Execution backend is not configured. Implement _perform_execution to call your "
        "side-effecting systems."
    )
