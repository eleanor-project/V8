"""
Deterministic evaluation of whether a case requires human constitutional review.

This module MUST remain non-ML, non-heuristic.
It defines explicit constitutional triggers only.
"""

class ReviewTriggerEvaluator:
    def __init__(
        self,
        severity_threshold: float = 1.5,
        disagreement_threshold: float = 0.6
    ):
        self.severity_threshold = severity_threshold
        self.disagreement_threshold = disagreement_threshold

    def evaluate(self, case) -> dict:
        triggers = []

        if getattr(case, "severity", 0) >= self.severity_threshold:
            triggers.append("severity_threshold_exceeded")

        if getattr(case, "critic_disagreement", 0) >= self.disagreement_threshold:
            triggers.append("critic_disagreement_high")

        if getattr(case, "novel_precedent", False):
            triggers.append("novel_precedent")

        rights = getattr(case, "rights_impacted", [])
        if "dignity" in rights or "autonomy" in rights:
            triggers.append("fundamental_rights_implicated")

        uncertainty = getattr(case, "uncertainty", None)
        if uncertainty and getattr(uncertainty, "flags", []):
            triggers.append("uncertainty_present")

        return {
            "review_required": len(triggers) > 0,
            "triggers": triggers
        }
