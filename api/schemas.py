"""
ELEANOR V8 — API Schemas
------------------------

Pydantic models for request/response validation.
"""

from typing import Optional, Dict, Any, List, Literal
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict, field_validator
import re

from engine.schemas.escalation import HumanAction

# Input sanitization pattern - remove potential injection attempts
SANITIZE_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


class DeliberationRequest(BaseModel):
    """Request schema for the /deliberate endpoint."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "input": "Should I approve this loan application?",
                "context": {"user_id": "12345", "category": "finance"},
            }
        }
    )

    input: str = Field(
        ..., min_length=1, max_length=100000, description="The input text to deliberate on"
    )
    context: Dict[str, Any] = Field(
        default_factory=dict, description="Optional context for the deliberation"
    )
    trace_id: Optional[str] = Field(None, description="Optional trace ID for correlation")
    human_action: Optional[HumanAction] = Field(
        None, description="Optional human action to satisfy escalation gate"
    )

    @field_validator("input")
    @classmethod
    def sanitize_input(cls, v: str) -> str:
        """Remove control characters and trim whitespace."""
        v = SANITIZE_PATTERN.sub("", v)
        return v.strip()

    @field_validator("context")
    @classmethod
    def validate_context(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure context doesn't contain overly nested structures."""

        def check_depth(obj: Any, depth: int = 0, max_depth: int = 5) -> bool:
            if depth > max_depth:
                return False
            if isinstance(obj, dict):
                return all(check_depth(val, depth + 1, max_depth) for val in obj.values())
            if isinstance(obj, list):
                return all(check_depth(item, depth + 1, max_depth) for item in obj)
            return True

        if not check_depth(v):
            raise ValueError("Context structure is too deeply nested (max depth: 5)")
        return v


class GovernancePreviewRequest(BaseModel):
    """Request schema for the /governance/preview endpoint."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "critics": {"rights": {"severity": 0.5}},
                "aggregator": {"decision": "allow"},
                "precedent": {"alignment_score": 0.8},
                "uncertainty": {"overall_uncertainty": 0.2},
            }
        }
    )

    critics: Dict[str, Any] = Field(default_factory=dict, description="Mock critic outputs")
    aggregator: Dict[str, Any] = Field(default_factory=dict, description="Mock aggregator output")
    precedent: Dict[str, Any] = Field(default_factory=dict, description="Mock precedent data")
    uncertainty: Dict[str, Any] = Field(default_factory=dict, description="Mock uncertainty data")


class CriticOutput(BaseModel):
    """Schema for individual critic output."""

    severity: float = Field(ge=0.0, le=3.0)
    violations: List[str] = Field(default_factory=list)
    justification: str = ""
    evidence: Dict[str, Any] = Field(default_factory=dict)


class UncertaintyOutput(BaseModel):
    """Schema for uncertainty engine output."""

    epistemic_uncertainty: float = Field(ge=0.0, le=1.0)
    aleatoric_uncertainty: float = Field(ge=0.0, le=1.0)
    overall_uncertainty: float = Field(ge=0.0, le=1.0)
    needs_escalation: bool = False
    explanation: str = ""


class DeliberationResponse(BaseModel):
    """Response schema for the /deliberate endpoint."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "trace_id": "550e8400-e29b-41d4-a716-446655440000",
                "timestamp": 1702300000.0,
                "model_used": "gpt-4",
                "final_decision": "aligned",
                "critics": {},
                "precedent_alignment": {"alignment_score": 0.8},
                "uncertainty": {"overall_uncertainty": 0.2},
                "aggregator_output": {"decision": "allow"},
                "opa_governance": {"allow": True},
            }
        }
    )

    trace_id: str
    timestamp: float
    model_used: str
    final_decision: str = Field(
        description="One of: aligned, aligned_with_constraints, misaligned, requires_human_review"
    )
    critics: Dict[str, Any]
    precedent_alignment: Dict[str, Any]
    uncertainty: Dict[str, Any]
    aggregator_output: Dict[str, Any]
    opa_governance: Dict[str, Any]
    execution_decision: Optional[Dict[str, Any]] = None


class ProposedAction(BaseModel):
    """Proposed downstream action to evaluate."""

    type: str = Field(..., description="Action type identifier")
    params: Dict[str, Any] = Field(default_factory=dict)


class EvidenceInputs(BaseModel):
    """Optional evidence inputs referenced by the caller."""

    precedent_keys: List[str] = Field(default_factory=list)
    citations: List[str] = Field(default_factory=list)
    logs: List[str] = Field(default_factory=list)
    attestations: List[str] = Field(default_factory=list)


class ModelMetadata(BaseModel):
    """Optional model metadata for auditability."""

    model_id: Optional[str] = None
    temperature: Optional[float] = None
    tool_trace_ref: Optional[str] = None


class EvaluateContext(BaseModel):
    """Contextual information required by the Engine contract."""

    model_config = ConfigDict(extra="allow")

    domain: str
    jurisdiction: Optional[str] = None
    sensitivity: Optional[str] = None
    user_intent: Optional[Any] = None
    constraints: Optional[Dict[str, Any]] = None
    case_refs: Optional[Dict[str, Any]] = None


class EvaluateRequest(BaseModel):
    """Request schema for the /evaluate endpoint."""

    request_id: str
    timestamp: datetime
    policy_profile: str
    model_output: Any
    proposed_action: ProposedAction
    context: EvaluateContext
    evidence_inputs: Optional[EvidenceInputs] = None
    model_metadata: Optional[ModelMetadata] = None


class CriticEvidence(BaseModel):
    critic: str
    verdict: Literal["PASS", "WARN", "FAIL"]
    score: Optional[float] = None
    rationale: str
    precedents: List[str] = Field(default_factory=list)
    policy_rules: List[str] = Field(default_factory=list)
    signals: Optional[Dict[str, Any]] = None


class PrecedentTrace(BaseModel):
    id: str
    type: Optional[str] = None
    applied_as: Optional[str] = None
    note: Optional[str] = None


class PolicyTrace(BaseModel):
    rule_id: str
    effect: str
    matched_on: Optional[Dict[str, Any]] = None
    note: Optional[str] = None


class EvidenceProvenance(BaseModel):
    inputs: Dict[str, Any]
    nondeterminism: Optional[str] = None


class EvidenceIntegrity(BaseModel):
    hash: str
    signature: Optional[str] = None


class EvidenceBundle(BaseModel):
    summary: str
    critic_outputs: List[CriticEvidence]
    precedent_trace: List[PrecedentTrace] = Field(default_factory=list)
    policy_trace: List[PolicyTrace] = Field(default_factory=list)
    provenance: EvidenceProvenance
    integrity: EvidenceIntegrity


class RoutingDecision(BaseModel):
    next_step: str
    queue: Optional[str] = None
    notes: Optional[str] = None


class UncertaintyEnvelope(BaseModel):
    level: Literal["LOW", "MEDIUM", "HIGH"]
    reasons: List[str] = Field(default_factory=list)


class EvaluateError(BaseModel):
    code: Optional[str] = None
    message: Optional[str] = None


class EvaluateResponse(BaseModel):
    """Response schema for the /evaluate endpoint."""

    request_id: str
    engine_version: str
    decision: Literal["ALLOW", "ALLOW_WITH_CONSTRAINTS", "ABSTAIN", "ESCALATE", "DENY"]
    confidence: float = Field(ge=0.0, le=1.0)
    uncertainty: UncertaintyEnvelope
    constraints: Optional[Dict[str, Any]] = None
    routing: RoutingDecision
    evidence_bundle: EvidenceBundle
    errors: List[EvaluateError] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Response schema for the /health endpoint."""

    status: str = Field(description="healthy, degraded, or unhealthy")
    version: str
    checks: Dict[str, str]
    timestamp: str


class ErrorResponse(BaseModel):
    """Standard error response schema."""

    error: str
    detail: Optional[str] = None
    trace_id: Optional[str] = None


class ConfigProposalRequest(BaseModel):
    schema_version: int = 1
    proposal_type: str
    title: str
    changes: Dict[str, Any]
    notes: Optional[str] = None


class ConfigProposalResponse(BaseModel):
    proposal_id: str
    status: str
    submitted_at: str


class ConfigProposalSummary(BaseModel):
    proposal_id: str
    title: Optional[str] = None
    status: str
    submitted_at: Optional[str] = None
    last_preview: Optional[str] = None
    last_apply: Optional[str] = None


class ConfigProposalListResponse(BaseModel):
    schema_version: int = 1
    environment: str
    items: List[ConfigProposalSummary]


class PreviewWindow(BaseModel):
    type: Literal["time", "count"] = "time"
    duration: Optional[str] = None
    limit: Optional[int] = None


class PreviewLimits(BaseModel):
    max_traces: Optional[int] = Field(default=None, ge=1)
    max_changed_traces: Optional[int] = Field(default=None, ge=1)


class ConfigProposalPreviewRequest(BaseModel):
    schema_version: int = 1
    mode: Literal["full_replay", "policy_only"] = "policy_only"
    window: Optional[PreviewWindow] = None
    limits: Optional[PreviewLimits] = None


class ConfigProposalApplyRequest(BaseModel):
    schema_version: int = 1
    artifact_id: Optional[str] = None
    artifact_hash: Optional[str] = None
    apply_mode: Optional[str] = None
    expected: Optional[Dict[str, Any]] = None


class ConfigProposalApplyResponse(BaseModel):
    schema_version: int = 1
    proposal_id: str
    status: str
    applied_at: str
    fingerprints: Dict[str, Any]
    ledger: Dict[str, Any]
