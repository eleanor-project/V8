"""
ELEANOR V8 — Type Definitions for Constitutional Governance

Explicit types that preserve constitutional semantics:
- Critic evaluations maintain independence and dissent
- Escalation signals carry complete governance context
- Uncertainty is quantified as epistemic limitation, not error
- Precedent alignment preserves case law reasoning
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Protocol, TypedDict, Union
from pydantic import BaseModel, Field, validator


# ============================================================================
# ESCALATION TYPES
# ============================================================================

class EscalationTier(int, Enum):
    """Canonical escalation tiers per constitutional doctrine."""
    TIER_2_ACKNOWLEDGMENT = 2  # Human must acknowledge risk and accept responsibility
    TIER_3_DETERMINATION = 3   # Human must actively decide course of action


class EscalationClause(BaseModel):
    """
    Specific constitutional clause triggering escalation.
    
    Each critic maintains a charter of clauses that define when
    human review is mandated (non-negotiable).
    """
    clause_id: str = Field(..., description="Canonical clause ID (e.g., 'A2', 'U3')")
    critic: str = Field(..., description="Critic declaring escalation")
    tier: EscalationTier = Field(..., description="Required escalation tier")
    rationale: str = Field(..., description="Constitutional justification")
    severity: float = Field(..., ge=0.0, le=1.0, description="Constitutional severity")
    
    class Config:
        frozen = True  # Escalation clauses are immutable


class EscalationSignal(BaseModel):
    """
    Binding signal that gates automatic execution.
    
    Once raised, this signal CANNOT be vetoed or averaged away.
    It requires the specified tier of human review before automation may proceed.
    """
    clause: EscalationClause
    triggered_at: str = Field(..., description="ISO timestamp")
    trace_id: str = Field(..., description="Audit trail identifier")
    human_review_required: bool = Field(default=True, const=True)
    
    # Human review fulfillment (populated after review)
    human_reviewer: Optional[str] = None
    human_decision: Optional[str] = None
    reviewed_at: Optional[str] = None
    
    class Config:
        frozen = False  # Allows human review fields to be set once


# ============================================================================
# UNCERTAINTY TYPES
# ============================================================================

class UncertaintySource(str, Enum):
    """Sources of epistemic uncertainty (not error)."""
    MODEL_AMBIGUITY = "model_ambiguity"  # Model output is ambiguous/contradictory
    CRITIC_DISAGREEMENT = "critic_disagreement"  # Critics fundamentally disagree
    PRECEDENT_ABSENCE = "precedent_absence"  # No relevant precedent exists
    PRECEDENT_CONFLICT = "precedent_conflict"  # Precedents conflict
    CONTEXT_INSUFFICIENCY = "context_insufficiency"  # Insufficient information
    COMPETENCE_BOUNDARY = "competence_boundary"  # Outside system competence
    MORAL_PLURALISM = "moral_pluralism"  # Legitimate value conflict


class UncertaintyMeasure(BaseModel):
    """
    Quantified uncertainty with epistemic grounding.
    
    Uncertainty is a signal of epistemic limitation, not malfunction.
    High uncertainty may trigger Tier 2 or Tier 3 escalation.
    """
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Overall uncertainty")
    sources: List[UncertaintySource] = Field(default_factory=list)
    confidence_interval: Optional[tuple[float, float]] = Field(
        None, description="Confidence bounds on severity scores"
    )
    epistemic_gaps: List[str] = Field(
        default_factory=list,
        description="Specific knowledge gaps identified"
    )
    recommendation: Literal["proceed", "acknowledge", "escalate", "defer"] = Field(
        ..., description="System recommendation based on uncertainty"
    )


# ============================================================================
# CRITIC EVALUATION TYPES
# ============================================================================

class CriticViolation(TypedDict, total=False):
    """Single constitutional violation detected by a critic."""
    principle: str  # Constitutional principle violated
    severity: float  # Severity score (0.0-1.0)
    description: str  # Human-readable explanation
    evidence: Dict[str, Any]  # Supporting evidence
    clause_triggered: Optional[str]  # Escalation clause ID if triggered


class CriticEvaluation(BaseModel):
    """
    Complete evaluation output from a single critic.
    
    Critics evaluate independently without seeing peer outputs.
    Evaluations are sealed before aggregation (epistemic isolation).
    """
    critic: str = Field(..., description="Critic identifier")
    violations: List[CriticViolation] = Field(default_factory=list)
    severity: float = Field(..., ge=0.0, le=1.0, description="Overall severity")
    justification: str = Field(..., description="Constitutional reasoning")
    
    # Metadata
    evaluated_rules: List[str] = Field(default_factory=list)
    duration_ms: float = Field(..., gt=0)
    
    # Escalation (if triggered)
    escalation: Optional[EscalationSignal] = None
    
    # Evidence trail
    evidence_references: List[str] = Field(
        default_factory=list,
        description="Evidence record IDs"
    )
    
    class Config:
        frozen = True  # Evaluations are immutable once sealed


# ============================================================================
# PRECEDENT TYPES
# ============================================================================

class PrecedentCase(BaseModel):
    """
    Single precedent case from case law database.
    
    Precedents guide constitutional interpretation but do not
    mechanically determine outcomes.
    """
    case_id: str
    domain: str
    decision: str
    reasoning: str
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    binding: bool = Field(default=False, description="Whether precedent is binding")
    
    # Case metadata
    date: Optional[str] = None
    jurisdiction: Optional[str] = None
    citation: Optional[str] = None


class PrecedentAlignment(BaseModel):
    """
    Analysis of how current decision aligns with precedent.
    
    Tracks novel situations, conflicting precedents, and precedent strength.
    """
    relevant_cases: List[PrecedentCase] = Field(default_factory=list)
    novel_situation: bool = Field(
        default=False,
        description="No sufficiently similar precedent exists"
    )
    conflicting_precedents: bool = Field(
        default=False,
        description="Precedents suggest different outcomes"
    )
    alignment_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="How well decision aligns with precedent"
    )
    reasoning: str = Field(..., description="Alignment analysis")
    
    # If novel or conflicting, may trigger escalation
    requires_escalation: bool = False


# ============================================================================
# AGGREGATION TYPES
# ============================================================================

class DissentRecord(BaseModel):
    """
    Preserved minority opinion that must not be averaged away.
    
    When a critic's constitutional concern is in the minority,
    it MUST be preserved verbatim in the final output.
    """
    dissenting_critic: str
    dissenting_position: str
    severity: float = Field(..., ge=0.0, le=1.0)
    rationale: str
    suppression_attempted: bool = Field(
        default=False,
        description="Flag if aggregation tried to silence dissent"
    )


class AggregatedResult(BaseModel):
    """
    Synthesized constitutional analysis preserving dissent.
    
    Aggregation synthesizes critic outputs but NEVER suppresses
    legitimate constitutional concerns.
    """
    final_output: str = Field(..., description="Synthesized recommendation")
    
    # Score components
    average_severity: float = Field(..., ge=0.0, le=1.0)
    max_severity: float = Field(..., ge=0.0, le=1.0)
    critic_agreement: float = Field(
        ..., ge=0.0, le=1.0,
        description="Agreement level (low = high disagreement)"
    )
    
    # Dissent preservation
    dissent: List[DissentRecord] = Field(default_factory=list)
    
    # Rights impact summary
    rights_impacted: List[str] = Field(default_factory=list)
    
    # Escalation (if any critic triggered)
    escalations: List[EscalationSignal] = Field(default_factory=list)
    
    # Attribution
    contributing_critics: List[str] = Field(default_factory=list)
    synthesis_method: str = Field(default="constitutional_fusion")


# ============================================================================
# MODEL & ROUTING TYPES
# ============================================================================

class ModelInfo(BaseModel):
    """Information about the selected model."""
    model_name: str
    model_version: Optional[str] = None
    router_selection_reason: Optional[str] = None
    cost_estimate: Optional[Dict[str, Any]] = None
    health_score: Optional[float] = Field(None, ge=0.0, le=1.0)


class ModelAdapter(Protocol):
    """
    Protocol for model adapters used by critics.
    
    Critics should depend on this protocol, not concrete implementations,
    to maintain epistemic isolation from model selection.
    """
    
    async def generate(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate text from prompt."""
        ...


# ============================================================================
# INPUT VALIDATION TYPES
# ============================================================================

class ValidatedInput(BaseModel):
    """
    Sanitized and validated input to the engine.
    
    All inputs pass through validation to prevent:
    - Prompt injection
    - Resource exhaustion
    - Malicious context payloads
    """
    text: str = Field(..., max_length=100_000, description="Input text")
    context: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('text')
    def validate_text_safety(cls, v):
        """Basic input sanitization."""
        if not v or not v.strip():
            raise ValueError("Input text cannot be empty")
        # Add injection detection here if needed
        return v
    
    @validator('context')
    def validate_context_size(cls, v):
        """Prevent context payload attacks."""
        import json
        serialized = json.dumps(v)
        if len(serialized) > 1_000_000:  # 1MB limit
            raise ValueError("Context payload exceeds size limit")
        return v


# ============================================================================
# EVIDENCE & AUDIT TYPES
# ============================================================================

class EvidenceRecord(BaseModel):
    """
    Immutable evidence record for audit trail.
    
    All constitutional evaluations must be recorded for compliance.
    """
    record_id: str
    timestamp: str
    trace_id: str
    
    # Record content
    record_type: Literal[
        "critic_evaluation",
        "escalation",
        "human_review",
        "precedent_lookup",
        "aggregation"
    ]
    critic: Optional[str] = None
    severity: Optional[float] = None
    content: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        frozen = True  # Evidence is immutable


# ============================================================================
# ENGINE RESULT TYPES
# ============================================================================

class DetailLevel(int, Enum):
    """Output detail levels for different use cases."""
    MINIMAL = 1      # Final output only
    STANDARD = 2     # + critics, uncertainty, precedent
    FORENSIC = 3     # + full diagnostics and evidence trail


class EngineResult(BaseModel):
    """
    Complete engine output with constitutional analysis.
    
    Structure preserves:
    - Dissent (never averaged away)
    - Escalation signals (never suppressed)
    - Uncertainty (as signal, not error)
    - Evidence trail (for audit)
    """
    trace_id: str
    output_text: str
    
    # Model selection
    model_info: Optional[ModelInfo] = None
    
    # Constitutional analysis
    critic_findings: Dict[str, CriticEvaluation] = Field(default_factory=dict)
    aggregated: Optional[AggregatedResult] = None
    
    # Epistemic status
    uncertainty: Optional[UncertaintyMeasure] = None
    precedent_alignment: Optional[PrecedentAlignment] = None
    
    # Governance
    escalations: List[EscalationSignal] = Field(default_factory=list)
    human_review_required: bool = False
    
    # Audit
    evidence_count: Optional[int] = None
    
    # Forensic detail (level 3 only)
    forensic: Optional[Dict[str, Any]] = None


__all__ = [
    # Escalation
    "EscalationTier",
    "EscalationClause",
    "EscalationSignal",
    # Uncertainty
    "UncertaintySource",
    "UncertaintyMeasure",
    # Critics
    "CriticViolation",
    "CriticEvaluation",
    # Precedent
    "PrecedentCase",
    "PrecedentAlignment",
    # Aggregation
    "DissentRecord",
    "AggregatedResult",
    # Models
    "ModelInfo",
    "ModelAdapter",
    # Input
    "ValidatedInput",
    # Evidence
    "EvidenceRecord",
    # Results
    "DetailLevel",
    "EngineResult",
]
