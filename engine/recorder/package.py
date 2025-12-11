"""
ELEANOR V8 — Evidence Package Builder
----------------------------------------

Responsible for constructing a structured, immutable "evidence bundle"
representing a single deliberation event.

The evidence bundle is passed to:
    - OPA governance layer
    - audit storage (Append-only)
    - downstream decision builder
    - external oversight tools

Evidence Bundle Contains:
    - input snapshot
    - critic outputs
    - constitutional violations
    - dissent metrics
    - uncertainty profile
    - precedent summary
    - router metadata
    - orchestrator metadata
    - timestamps and trace IDs
"""

from typing import Dict, Any
import time
import uuid


class EvidencePackageV8:

    def __init__(self):
        pass

    # ---------------------------------------------------------------
    # Generate unique trace IDs
    # ---------------------------------------------------------------
    def _trace_id(self) -> str:
        return f"trace_{uuid.uuid4().hex[:16]}"

    # ---------------------------------------------------------------
    # Main packaging function
    # ---------------------------------------------------------------
    def build(
        self,
        input_snapshot: Any,
        router_result: Dict[str, Any],
        critic_outputs: Dict[str, Any],
        deliberation_state: Dict[str, Any],
        uncertainty_state: Dict[str, Any],
        precedent_result: Dict[str, Any]
    ) -> Dict[str, Any]:

        timestamp = time.time()

        evidence = {
            "timestamp": timestamp,
            "trace_id": self._trace_id(),

            # --- Input ---
            "input_snapshot": input_snapshot,

            # --- Model ---
            "model_used": router_result.get("used_adapter"),
            "model_attempts": router_result.get("attempts", []),

            # --- Critics ---
            "critic_outputs": critic_outputs,

            # --- Aggregation ---
            "priority_violations": deliberation_state.get("priority_violations", []),
            "values_respected": deliberation_state.get("values_respected", []),
            "values_violated": deliberation_state.get("values_violated", []),

            # --- Uncertainty ---
            "uncertainty": {
                "uncertainty_score": uncertainty_state.get("uncertainty_score"),
                "dissent": uncertainty_state.get("dissent_score"),
                "entropy": uncertainty_state.get("entropy_estimate"),
                "stability": uncertainty_state.get("stability"),
                "requires_escalation": uncertainty_state.get("requires_escalation"),
                "escalation_reasons": uncertainty_state.get("escalation_reasons", []),
            },

            # --- Precedent ---
            "precedent": {
                "alignment_score": precedent_result.get("alignment_score", 1.0),
                "top_case": precedent_result.get("top_case"),
                "precedent_cases": precedent_result.get("precedent_cases"),
            },

            # --- Final governance state (OPA evaluated externally) ---
            "governance_ready_payload": {
                "critic_outputs": critic_outputs,
                "priority_violations": deliberation_state.get("priority_violations", []),
                "uncertainty": uncertainty_state,
                "precedent_alignment": precedent_result.get("alignment_score", 1.0)
            }
        }

        return evidence

