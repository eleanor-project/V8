"""
ELEANOR V8 — Full Constitutional Aggregator
-------------------------------------------

This module fuses:

• Critic evidence bundles
• Precedent retrieval + alignment scoring
• Uncertainty engine output
• Constitutional lexicographic rules
• Conflict detection
• Proportionality & dignity safeguards
• Final allow / constrained-allow / deny / escalate decision
• Auditor-friendly justification package

This is the heart of the V8 constitutional engine.
"""

from typing import Dict, Any, List
import statistics

from engine.schemas.escalation import CriticEvaluation
from engine.schemas.pipeline_types import (
    AggregationOutput,
    AggregationScore,
    CriticResult,
    PrecedentAlignmentResult,
    UncertaintyResult,
)
from engine.utils.critic_names import canonicalize_critic_map
from engine.aggregator.escalation import resolve_escalation


# ============================================================
# CONSTITUTIONAL PRIORITY ORDER (Lexicographic)
# ============================================================

PRIORITY_ORDER = [
    "rights",
    "autonomy",
    "fairness",
    "truth",
    "risk",
    "operations",
]

# Severity scale:
#   0.0 = no violation
#   1.0 = minor concern
#   2.0 = moderate concern
#   3.0 = severe / constitutional concern


# ============================================================
#  AggregatorV8
# ============================================================


class AggregatorV8:
    def __init__(self):
        pass

    # ----------------------------------------------------------
    # MAIN ENTRYPOINT
    # ----------------------------------------------------------
    def aggregate(
        self,
        critics: Dict[str, CriticResult],
        precedent: PrecedentAlignmentResult,
        uncertainty: UncertaintyResult,
        model_output: str = "",
    ) -> AggregationOutput:
        """
        critics: dict critic_name -> critic_output
        precedent: { "cases": [...], "alignment_score": float }
        uncertainty: { "overall_uncertainty": float, ... }

        Returns structured constitutional decision.
        """

        # 1. Normalize critic outputs
        norm = self._normalize_critics(critics)

        # 2. Compute lexicographic violation structure
        lex = self._compute_lexicographic(norm)

        # 3. Apply precedent alignment weighting
        adjusted = self._apply_precedent(norm, precedent)

        # 4. Apply uncertainty weighting
        adjusted = self._apply_uncertainty(adjusted, uncertainty)

        # 5. Compute aggregate constitutional score
        score = self._final_score(adjusted, lex)

        # 6. Determine outcome
        decision = self._decision_logic(score, lex, uncertainty)

        # 7. Escalation resolution (doctrine-compliant)
        critic_evals: List[CriticEvaluation] = []
        for name, data in adjusted.items():
            severity_score = self._build_severity_score(data)

            critic_evals.append(
                CriticEvaluation(
                    critic_id=name,
                    charter_version=str(data.get("charter_version", "")),
                    concerns=[],
                    escalation=data.get("escalation"),
                    severity_score=severity_score,
                    citations=data.get("precedent_refs", []) or [],
                    uncertainty=data.get("uncertainty"),
                )
            )

        escalation_result = resolve_escalation(
            critic_evaluations=critic_evals,
            synthesis=model_output,
        )

        # 8. Build auditor package
        return {
            "decision": decision,
            "score": score,
            "critics": adjusted,
            "lexicographic_violations": lex.get("violations", []),
            "precedent": precedent,
            "uncertainty": uncertainty,
            "final_output": model_output,
            "escalation_summary": escalation_result.escalation_summary.model_dump(),
            "execution_gate": escalation_result.execution_gate.model_dump(),
            "dissent_present": escalation_result.dissent_present,
            "audit_hash": escalation_result.audit_hash,
            "aggregation_result": escalation_result.model_dump(mode="json"),
        }

    # ----------------------------------------------------------
    # STEP 1: Normalize critic outputs
    # ----------------------------------------------------------
    def _normalize_critics(self, critics: Dict[str, CriticResult]) -> Dict[str, Dict[str, Any]]:
        """
        Expected critic schema:
           {
               "severity": float 0-3,
               "violations": [...],
               "justification": str,
               ...
           }
        """

        normalized = {}
        critics = canonicalize_critic_map(critics)
        for name, data in critics.items():
            severity = self._coerce_numeric(data.get("severity", 0.0))
            violations = data.get("violations", [])
            justification = data.get("justification", "")

            normalized[name] = {
                "severity": severity,
                "violations": violations,
                "justification": justification,
            "raw": data,
        }

        return normalized

    def _coerce_numeric(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _build_severity_score(self, data: Dict[str, Any]) -> float:
        severity = data.get("severity")
        if severity is not None:
            raw = self._coerce_numeric(severity) / 3.0
        else:
            raw = self._coerce_numeric(data.get("score", 0.0))
        return max(0.0, min(1.0, raw))

    # ----------------------------------------------------------
    # STEP 2: Lexicographic analysis
    # ----------------------------------------------------------
    def _compute_lexicographic(self, critics_norm) -> Dict[str, Any]:
        """
        Lexicographic rule:
           A violation in a higher-order critic (e.g., rights)
           cannot be overridden by lower-order critics.

        Detect:
         • highest-priority active violations
         • minimum constitutional requirement failing
         • conflict patterns
        """

        violations = []

        for critic in PRIORITY_ORDER:
            entry = critics_norm.get(critic, {})
            sev = entry.get("severity", 0.0)
            if sev >= 2.0:  # threshold for constitutional relevance
                violations.append(
                    {"critic": critic, "severity": sev, "type": "constitutional_violation"}
                )

        highest = None
        if violations:
            # highest priority by lexicographic order
            highest = sorted(violations, key=lambda v: PRIORITY_ORDER.index(v["critic"]))[0]

        return {"violations": violations, "highest_priority_violation": highest}

    # ----------------------------------------------------------
    # STEP 3: Apply precedent weighting
    # ----------------------------------------------------------
    def _apply_precedent(self, critics_norm, precedent) -> Dict[str, Any]:
        """
        Precedent modifies severity downward when supportive,
        upward when contradictory.

        precedent = {
            "cases": [...],
            "alignment_score": float between -1 and 1
        }

        alignment_score:
            -1.0 strongly contradicts critic findings
             0.0 neutral / no precedent
             1.0 strongly supports critic findings
        """

        align = (precedent or {}).get("alignment_score", 0.0)
        adjusted = {}

        for name, entry in critics_norm.items():
            sev = entry["severity"]

            # Apply precedent weight
            # If preceding cases contradict critic finding strongly,
            # severity increases.
            if align < 0:
                sev_adj = sev * (1 + abs(align))
            else:
                sev_adj = sev * (1 - (align * 0.3))

            adjusted[name] = {**entry, "precedent_adjusted_severity": max(0.0, min(3.0, sev_adj))}

        return adjusted

    # ----------------------------------------------------------
    # STEP 4: Apply uncertainty weighting
    # ----------------------------------------------------------
    def _apply_uncertainty(self, critics_adj, uncertainty) -> Dict[str, Any]:
        """
        If uncertainty is high, amplifier increases severity,
        because the system becomes more cautious.

        overall_uncertainty: 0-1

        Weighting:
            sev_final = sev * (1 + 0.5 * uncertainty)
        """

        u = float((uncertainty or {}).get("overall_uncertainty", 0.0))
        adjusted = {}

        for name, entry in critics_adj.items():
            sev = entry["precedent_adjusted_severity"]
            sev_u = sev * (1 + 0.5 * u)

            adjusted[name] = {**entry, "final_severity": max(0.0, min(3.0, sev_u))}

        return adjusted

    # ----------------------------------------------------------
    # STEP 5: Compute final constitutional score
    # ----------------------------------------------------------
    def _final_score(self, critics_final, lex_info) -> AggregationScore:
        """
        Produces:
            • weighted_total
            • average_severity
        """

        severities = [c["final_severity"] for c in critics_final.values()]

        avg = statistics.mean(severities) if severities else 0.0
        total = sum(severities)

        return {"average_severity": avg, "total_severity": total}

    # ----------------------------------------------------------
    # STEP 6: Decision Logic
    # ----------------------------------------------------------
    def _decision_logic(self, final_score, lex_info, uncertainty):
        """
        Returns one of:

           "allow"
           "constrained_allow"
           "deny"
           "escalate"

        Based on:
          • lexicographic violations
          • severity thresholds
          • uncertainty triggers
        """

        highest = lex_info.get("highest_priority_violation")
        avg = final_score["average_severity"]
        u = float(uncertainty.get("overall_uncertainty", 0.0))

        # 1) Lexicographic hard block
        if highest and highest["severity"] >= 2.5:
            return "deny"

        # 2) Escalation on high uncertainty + moderate severity
        if u >= 0.6 and avg >= 1.0:
            return "escalate"

        # 3) Constrained allow for medium constitutional tension
        if avg >= 1.0:
            return "constrained_allow"

        # 4) All clear
        return "allow"
