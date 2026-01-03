"""
ELEANOR V8 — Precedent Conflict Detection
---------------------------------------------

Detects when:
    - A new decision contradicts stable precedent
    - Value priority hierarchy is violated
    - Critic reasoning is inconsistent with prior cases

This module flags jurisprudential conflicts that may require:
    - escalation
    - constitutional review
    - refinement of critic logic
"""

from typing import Dict, Any


class PrecedentConflictV8:
    def detect(
        self, precedent_case: Dict[str, Any], deliberation_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        if precedent_case is None:
            return {"conflict_detected": False, "reasons": []}

        reasons = []

        # Compare violated values
        prev_violations = set(precedent_case.get("violated_values", []))
        curr_violations = set(deliberation_state.get("values_violated", []))

        # If current decision violates a value the precedent upheld
        newly_violated = curr_violations - prev_violations
        if newly_violated:
            reasons.append(f"Newly violated values compared to precedent: {list(newly_violated)}")

        # If current decision *fails to* violate something precedent did
        reversed_violations = prev_violations - curr_violations
        if reversed_violations:
            reasons.append(f"Reversal of previous violations: {list(reversed_violations)}")

        # Priority hierarchies
        if precedent_case.get("priority_order") != deliberation_state.get(
            "constitutional_priority_order"
        ):
            reasons.append("Priority hierarchy mismatch detected.")

        return {"conflict_detected": len(reasons) > 0, "reasons": reasons}
