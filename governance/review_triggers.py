"""
Deterministic evaluation of whether a case requires human constitutional review.

This module MUST remain non-ML, non-heuristic.
It defines explicit constitutional triggers only.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Case:
    severity: float = 0.0
    critic_disagreement: float = 0.0
    novel_precedent: bool = False
    rights_impacted: List[str] = field(default_factory=list)
    uncertainty_flags: List[str] = field(default_factory=list)
    uncertainty: any = None


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

        # Accept both nested uncertainty.flags and direct uncertainty_flags
        uncertainty = getattr(case, "uncertainty", None)
        uncertainty_flags = []
        if uncertainty and getattr(uncertainty, "flags", []):
            uncertainty_flags = list(getattr(uncertainty, "flags"))
        if hasattr(case, "uncertainty_flags") and getattr(case, "uncertainty_flags"):
            uncertainty_flags.extend(getattr(case, "uncertainty_flags"))
        if uncertainty_flags:
            triggers.append("uncertainty_present")

        return {
            "review_required": len(triggers) > 0,
            "triggers": triggers
        }
