"""
ELEANOR V8 — Full Uncertainty Engine (Epistemic + Aleatoric)
-------------------------------------------------------------

This module provides a structured system for computing:

• Epistemic uncertainty (model disagreement, critic divergence)
• Aleatoric uncertainty (input ambiguity, unclear precedent signals)
• Precedent consistency & drift-based uncertainty
• Model stability uncertainty
• Composite uncertainty score
• Escalation triggers

Output schema:
{
    "epistemic_uncertainty": float,
    "aleatoric_uncertainty": float,
    "critic_divergence": float,
    "precedent_conflict_uncertainty": float,
    "model_stability_uncertainty": float,
    "overall_uncertainty": float,
    "needs_escalation": bool,
    "explanation": "text summary"
}
"""

from typing import Dict
import statistics

from engine.schemas.pipeline_types import UncertaintyResult, PrecedentAlignmentResult, CriticResult


class UncertaintyEngineV8:
    def __init__(self):
        pass

    # ============================================================
    # MAIN ENTRYPOINT
    # ============================================================
    def compute(
        self,
        critics: Dict[str, CriticResult],
        model_used: str,
        precedent_alignment: PrecedentAlignmentResult,
    ) -> UncertaintyResult:
        """
        Compute all uncertainty dimensions.
        """

        # ----------------------------------------------------------
        # 1. Critic divergence (epistemic)
        # ----------------------------------------------------------
        critic_div = self._critic_divergence(critics)

        # ----------------------------------------------------------
        # 2. Precedent conflict + drift (aleatoric + epistemic mix)
        # ----------------------------------------------------------
        conflict_u = self._precedent_conflict_uncertainty(precedent_alignment)
        drift_u = precedent_alignment.get("drift_score", 0.0)

        # ----------------------------------------------------------
        # 3. Model stability uncertainty
        # ----------------------------------------------------------
        model_u = self._model_stability(model_used)

        # ----------------------------------------------------------
        # 4. Composite uncertainty
        # ----------------------------------------------------------
        epistemic = self._epistemic_uncertainty(critic_divergence=critic_div, drift=drift_u)

        aleatoric = self._aleatoric_uncertainty(
            precedent_conflict=conflict_u,
        )

        overall = self._combine(epistemic, aleatoric, model_u)

        # ----------------------------------------------------------
        # 5. Escalation determination
        # ----------------------------------------------------------
        needs_escalation = overall >= 0.65 or conflict_u >= 0.75

        # ----------------------------------------------------------
        # 6. Explanation for auditors
        # ----------------------------------------------------------
        explain = self._summary(
            epistemic=epistemic,
            aleatoric=aleatoric,
            critic_div=critic_div,
            conflict=conflict_u,
            drift=drift_u,
            model_u=model_u,
            overall=overall,
        )

        return {
            "epistemic_uncertainty": epistemic,
            "aleatoric_uncertainty": aleatoric,
            "critic_divergence": critic_div,
            "precedent_conflict_uncertainty": conflict_u,
            "model_stability_uncertainty": model_u,
            "overall_uncertainty": overall,
            "needs_escalation": needs_escalation,
            "explanation": explain,
        }

    # ============================================================
    # Critic Divergence (Epistemic)
    # ============================================================
    def _critic_divergence(self, critics):
        """
        Measures how much critics disagree in severity.

        High divergence → model uncertainty about moral dimension.

        Divergence = normalized variance of critic severities.
        """

        values = [c.get("severity", 0.0) for c in critics.values()]
        if len(values) <= 1:
            return 0.0

        variance = statistics.pvariance(values)

        # Normalize to 0–1 range (max meaningful var ~ 2)
        return max(0.0, min(1.0, variance / 2.0))

    # ============================================================
    # Precedent Conflict Uncertainty
    # ============================================================
    def _precedent_conflict_uncertainty(self, alignment):
        """
        Precedent conflict reflects aleatoric ambiguity:
        the world itself contains contradictory historical examples.

        Input: alignment bundle from PrecedentAlignmentEngineV8
        """

        return float(alignment.get("conflict_level", 0.0))

    # ============================================================
    # Model Stability Uncertainty
    # ============================================================
    def _model_stability(self, model_name):
        """
        Placeholder for future statistical model drift detection.
        For now, produce stable but expandable measure:

        - Models with name patterns indicating smaller or finetuned variants
          are considered less stable than frontier models.
        """

        name = model_name.lower()

        if "llama" in name or "mistral" in name or "phi" in name:
            return 0.15

        if "gpt" in name or "claude" in name or "grok" in name:
            return 0.05

        return 0.2

    # ============================================================
    # Epistemic Uncertainty (Critic Divergence + Drift)
    # ============================================================
    def _epistemic_uncertainty(self, critic_divergence, drift):
        """
        Epistemic = confusion inside the reasoning system.

        critic_divergence = disagreement across critics
        drift = deviation from historical precedent patterns
        """

        # Weighted sum
        return max(0.0, min(1.0, (critic_divergence * 0.6) + (drift * 0.4)))

    # ============================================================
    # Aleatoric Uncertainty (Conflict)
    # ============================================================
    def _aleatoric_uncertainty(self, precedent_conflict):
        """
        Aleatoric = uncertainty from the input environment itself.
        """

        return max(0.0, min(1.0, precedent_conflict))

    # ============================================================
    # Combine All Uncertainties
    # ============================================================
    def _combine(self, epistemic, aleatoric, model_u):
        """
        Final uncertainty score is a weighted combination:

            epistemic (50%)
            aleatoric (40%)
            model instability (10%)
        """

        return max(0.0, min(1.0, (epistemic * 0.5) + (aleatoric * 0.4) + (model_u * 0.1)))

    # ============================================================
    # Natural Language Summary
    # ============================================================
    def _summary(self, epistemic, aleatoric, critic_div, conflict, drift, model_u, overall):
        return (
            f"Epistemic uncertainty: {epistemic:.2f} "
            f"(critic divergence={critic_div:.2f}, drift={drift:.2f}). "
            f"Aleatoric uncertainty: {aleatoric:.2f} "
            f"(precedent conflict={conflict:.2f}). "
            f"Model stability uncertainty: {model_u:.2f}. "
            f"Overall uncertainty: {overall:.2f}."
        )
