from __future__ import annotations

from enum import Enum
from typing import List, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict


class EscalationTier(str, Enum):
    TIER_2 = "TIER_2"  # Mandatory Human Acknowledgment
    TIER_3 = "TIER_3"  # Mandatory Human Determination


class HumanActionType(str, Enum):
    HUMAN_ACK = "HUMAN_ACK"
    HUMAN_DETERMINATION = "HUMAN_DETERMINATION"


class Concern(BaseModel):
    model_config = ConfigDict(frozen=True)

    summary: str = Field(..., description="Short description of the concern")
    rationale: str = Field(..., description="Reasoned explanation of the concern")
    impacted_rights: List[str] = Field(
        default_factory=list,
        description="UDHR / constitutional references"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Critic confidence in this concern"
    )


class EscalationSignal(BaseModel):
    model_config = ConfigDict(frozen=True)

    tier: EscalationTier
    critic_id: str
    clause_id: str
    clause_description: str
    doctrine_ref: str
    rationale: str
    blocking: bool
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @staticmethod
    def for_tier(
        *,
        tier: EscalationTier,
        critic_id: str,
        clause_id: str,
        clause_description: str,
        doctrine_ref: str,
        rationale: str,
    ) -> "EscalationSignal":
        return EscalationSignal(
            tier=tier,
            critic_id=critic_id,
            clause_id=clause_id,
            clause_description=clause_description,
            doctrine_ref=doctrine_ref,
            rationale=rationale,
            blocking=(tier == EscalationTier.TIER_3),
        )


class CriticEvaluation(BaseModel):
    model_config = ConfigDict(frozen=True)

    critic_id: str
    charter_version: str
    concerns: List[Concern] = Field(default_factory=list)
    escalation: Optional[EscalationSignal] = None
    severity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Impact magnitude, not probability"
    )
    citations: List[str] = Field(default_factory=list)
    uncertainty: Optional[str] = None
    revision: int = Field(default=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EscalationSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    highest_tier: Optional[EscalationTier]
    triggering_signals: List[EscalationSignal]
    critics_triggered: List[str]
    explanation: str


class ExecutionGate(BaseModel):
    model_config = ConfigDict(frozen=True)

    gated: bool
    required_action: Optional[HumanActionType]
    reason: str
    escalation_tier: Optional[EscalationTier]


class AggregationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    synthesis: str
    critic_evaluations: List[CriticEvaluation]
    escalation_summary: EscalationSummary
    execution_gate: ExecutionGate
    dissent_present: bool
    audit_hash: str


class HumanAction(BaseModel):
    model_config = ConfigDict(frozen=True)

    action_type: HumanActionType
    actor_id: str
    statement: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    linked_escalations: List[EscalationSignal]


class ExecutableDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    aggregation_result: AggregationResult
    human_action: Optional[HumanAction]
    executable: bool
    execution_reason: str
    audit_record_id: str


class AuditRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    record_id: str
    aggregation_hash: str
    escalation_signals: List[EscalationSignal]
    human_action: Optional[HumanAction]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    immutable: bool = Field(default=True)
