from __future__ import annotations

from typing import List, Optional

from engine.critics.base import ConstitutionalCritic
from engine.schemas.escalation import (
    Concern,
    CriticEvaluation,
    EscalationSignal,
    EscalationTier,
)


class PrivacyIdentityCritic(ConstitutionalCritic):
    """Privacy & Identity critic with explicit clause helpers (P1–P4)."""

    critic_id = "privacy_identity"

    def __init__(self, charter_version: str = "v1"):
        super().__init__(charter_version=charter_version)

    # -----------------------------------------------------
    # Clause helpers (single source of truth)
    # -----------------------------------------------------
    def _escalate_P1(self, rationale: str) -> EscalationSignal:
        return EscalationSignal.for_tier(
            tier=EscalationTier.TIER_3,
            critic_id=self.critic_id,
            clause_id="P1",
            clause_description="Non-consensual identity inference",
            doctrine_ref="Escalation Doctrine §4",
            rationale=rationale,
        )

    def _escalate_P2(self, rationale: str) -> EscalationSignal:
        return EscalationSignal.for_tier(
            tier=EscalationTier.TIER_3,
            critic_id=self.critic_id,
            clause_id="P2",
            clause_description="Persistent or linked identity construction",
            doctrine_ref="Escalation Doctrine §4",
            rationale=rationale,
        )

    def _escalate_P3(self, rationale: str) -> EscalationSignal:
        return EscalationSignal.for_tier(
            tier=EscalationTier.TIER_2,
            critic_id=self.critic_id,
            clause_id="P3",
            clause_description="Context collapse across domains",
            doctrine_ref="Escalation Doctrine §4",
            rationale=rationale,
        )

    def _escalate_P4(self, rationale: str) -> EscalationSignal:
        return EscalationSignal.for_tier(
            tier=EscalationTier.TIER_2,
            critic_id=self.critic_id,
            clause_id="P4",
            clause_description="Secondary use expansion without renewed consent",
            doctrine_ref="Escalation Doctrine §4",
            rationale=rationale,
        )

    # -----------------------------------------------------
    # Evaluation Logic
    # -----------------------------------------------------
    def evaluate(self, **kwargs) -> CriticEvaluation:
        identity_inference: bool = bool(kwargs.get("identity_inference", False))
        persistent_identity: bool = bool(kwargs.get("persistent_identity", False))
        context_mismatch: bool = bool(kwargs.get("context_mismatch", False))
        secondary_use: bool = bool(kwargs.get("secondary_use", False))
        concerns: List[Concern] = []
        escalation: Optional[EscalationSignal] = None
        severity = 0.0

        if identity_inference:
            concerns.append(
                Concern(
                    summary="Identity inferred without explicit consent",
                    rationale="Sensitive identity attributes inferred implicitly.",
                    impacted_rights=["UDHR Article 12"],
                    confidence=0.9,
                )
            )
            escalation = self._escalate_P1("Identity inference occurred without explicit consent.")
            severity = max(severity, 0.9)

        if persistent_identity and escalation is None:
            concerns.append(
                Concern(
                    summary="Persistent identity linkage detected",
                    rationale="Identity persists across sessions or contexts.",
                    impacted_rights=["UDHR Article 12"],
                    confidence=0.85,
                )
            )
            escalation = self._escalate_P2("Identity persistence exceeds contextual expectations.")
            severity = max(severity, 0.85)

        if context_mismatch and escalation is None:
            concerns.append(
                Concern(
                    summary="Context collapse risk",
                    rationale="Data used outside original contextual boundary.",
                    impacted_rights=["UDHR Article 12"],
                    confidence=0.7,
                )
            )
            escalation = self._escalate_P3("Contextual integrity violated across domains.")
            severity = max(severity, 0.7)

        if secondary_use and escalation is None:
            concerns.append(
                Concern(
                    summary="Secondary use without renewed consent",
                    rationale="Data reused beyond original authorization scope.",
                    impacted_rights=["UDHR Article 12"],
                    confidence=0.75,
                )
            )
            escalation = self._escalate_P4("Secondary use occurred without renewed consent.")
            severity = max(severity, 0.75)

        return self._build_evaluation(
            concerns=concerns,
            severity_score=severity,
            citations=["UDHR Article 12"],
            escalation=escalation,
            uncertainty=None,
        )
