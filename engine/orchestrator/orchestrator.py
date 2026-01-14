"""
ELEANOR V8 — Orchestrator (LEGACY - See orchestrator_v2.py)
------------------------------------

⚠️ DEPRECATION NOTICE:
This is the original orchestrator implementation (V1).
For production use, see orchestrator_v2.py which provides:
- Hook-based infrastructure integration
- Better error handling and observability
- Structured input/output schemas
- Full backward compatibility

This V1 implementation remains for reference and potential rollback.
Current runtime uses orchestrator_v2.py (integrated via critic_infrastructure.py)

The Orchestrator manages the critical execution path:

1. Load and run all critics in parallel
2. Enforce timeouts per critic
3. Ensure one critic failure does not collapse the engine
4. Normalize outputs into the schema the Aggregator expects
5. Return a complete `critic_outputs` dictionary

The Orchestrator does NOT:
- aggregate values (Aggregator handles that)
- compute uncertainty (Uncertainty Engine handles that)
- perform governance checks (OPA handles that)
- classify or resolve precedent (Precedent Engine handles that)

This module performs ONLY:
  Parallelized, robust critic execution.

This is the backbone of the deliberation pipeline.
"""

import asyncio
from typing import Awaitable, Dict, Any, Callable


class OrchestratorV8:
    def __init__(
        self,
        critics: Dict[str, Callable[[Any], Awaitable[Dict[str, Any]]]],
        timeout_seconds: float = 3.0,
    ):
        """
        critics: dict of {critic_name: critic_callable(input) -> dict}
        timeout_seconds: max time allowed for each critic
        """
        self.critics = critics
        self.timeout = timeout_seconds

    # ---------------------------------------------------------------
    #  Default critic failure output (non-fatal)
    # ---------------------------------------------------------------
    def _critic_failure_template(self, name: str, error: str) -> Dict[str, Any]:
        return {
            "value": None,
            "score": 0,
            "violation": False,
            "details": {"error": error, "critic_failed": True},
        }

    # ---------------------------------------------------------------
    #  Run a single critic safely with timeout + error isolation
    # ---------------------------------------------------------------
    async def _run_critic(
        self,
        name: str,
        critic_fn: Callable[[Any], Awaitable[Dict[str, Any]]],
        input_snapshot: Any,
    ) -> Dict[str, Any]:
        """
        Executes a critic with timeout and error handling.
        """
        try:
            result = await asyncio.wait_for(critic_fn(input_snapshot), timeout=self.timeout)
            return result

        except Exception as e:
            return self._critic_failure_template(name, str(e))

    # ---------------------------------------------------------------
    #  Public entry point: run all critics
    # ---------------------------------------------------------------
    async def run_all(self, input_snapshot: Any) -> Dict[str, Dict[str, Any]]:
        """
        Executes all critics concurrently and returns structure:

        {
            "rights": {...},
            "fairness": {...},
            ...
        }
        """
        tasks = []
        for name, critic_fn in self.critics.items():
            task = asyncio.create_task(self._run_critic(name, critic_fn, input_snapshot))
            tasks.append((name, task))

        critic_outputs = {}

        for name, task in tasks:
            try:
                critic_outputs[name] = await task
            except Exception as e:
                # Should never hit here because _run_critic handles errors,
                # but we defensively guard the pipeline.
                critic_outputs[name] = self._critic_failure_template(
                    name, f"Unhandled orchestrator exception: {str(e)}"
                )

        return critic_outputs

    # ---------------------------------------------------------------
    #  Sync wrapper for ease of use in non-async contexts
    # ---------------------------------------------------------------
    def run(self, input_snapshot: Any) -> Dict[str, Dict[str, Any]]:
        """
        Sync wrapper around async execution — allows the engine to run
        in environments that may not be async.
        """
        return asyncio.run(self.run_all(input_snapshot))
