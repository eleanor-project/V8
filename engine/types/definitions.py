"""
ELEANOR V8 — Type Definitions
------------------------------

Comprehensive type definitions to replace Any types.
"""

from typing import TypedDict, List, Optional, Dict, Any, Union
from datetime import datetime


# ============================================================================
# Critic Types
# ============================================================================

class CriticResultDict(TypedDict, total=False):
    """Type definition for critic results."""
    critic: str
    severity: float
    violations: List[str]
    confidence: float
    duration_ms: Optional[float]
    justification: Optional[str]
    evaluated_rules: Optional[List[str]]
    precedent_refs: Optional[List[str]]


class CriticEvaluationInput(TypedDict):
    """Input for critic evaluation."""
    model_response: str
    input_text: str
    context: Dict[str, Any]
    trace_id: str


# ============================================================================
# Router Types
# ============================================================================

class ModelInfoDict(TypedDict, total=False):
    """Type definition for model information."""
    model_name: str
    model_version: Optional[str]
    router_selection_reason: Optional[str]
    health_score: Optional[float]
    cost_estimate: Optional[Dict[str, Any]]


class RouterResponse(TypedDict):
    """Type definition for router responses."""
    response_text: str
    model_info: ModelInfoDict
    adapter: Optional[str]
    success: bool
    duration_ms: Optional[float]
    diagnostics: Dict[str, Any]


class RouterDiagnostics(TypedDict, total=False):
    """Router diagnostic information."""
    skipped: Optional[bool]
    reason: Optional[str]
    circuit_open: Optional[bool]
    error: Optional[str]
    fallback_used: Optional[bool]


# ============================================================================
# Precedent Types
# ============================================================================

class PrecedentCaseDict(TypedDict, total=False):
    """Type definition for precedent case."""
    id: str
    title: Optional[str]
    summary: Optional[str]
    category: Optional[str]
    similarity_score: Optional[float]
    metadata: Dict[str, Any]


class PrecedentAlignmentDict(TypedDict, total=False):
    """Type definition for precedent alignment."""
    retrieval: Dict[str, Any]
    precedent_cases: List[PrecedentCaseDict]
    alignment_score: Optional[float]
    drift_detected: Optional[bool]


# ============================================================================
# Uncertainty Types
# ============================================================================

class UncertaintyAnalysisDict(TypedDict, total=False):
    """Type definition for uncertainty analysis."""
    overall_uncertainty: float
    epistemic_uncertainty: Optional[float]
    aleatoric_uncertainty: Optional[float]
    sources: List[str]
    explanation: Optional[str]
    needs_escalation: Optional[bool]


# ============================================================================
# Aggregation Types
# ============================================================================

class AggregationOutputDict(TypedDict, total=False):
    """Type definition for aggregation output."""
    decision: str
    final_output: Optional[str]
    severity: Optional[float]
    confidence: Optional[float]
    dissent: Optional[Dict[str, Any]]
    degraded_components: Optional[List[str]]
    is_degraded: Optional[bool]


# ============================================================================
# Evidence Types
# ============================================================================

class EvidenceRecordDict(TypedDict, total=False):
    """Type definition for evidence record."""
    timestamp: str
    trace_id: str
    request_id: Optional[str]
    model_name: Optional[str]
    model_version: Optional[str]
    critic: Optional[str]
    rule_id: Optional[str]
    severity: Optional[str]
    confidence: Optional[float]
    violation_description: Optional[str]
    mitigation: Optional[str]
    detector_metadata: Dict[str, Any]
    context: Dict[str, Any]
    uncertainty_flags: Dict[str, Any]
    precedent_sources: List[str]
    precedent_candidates: List[str]
    raw_text: Optional[str]


# ============================================================================
# Governance Types
# ============================================================================

class GovernanceResultDict(TypedDict, total=False):
    """Type definition for governance evaluation result."""
    allow: bool
    escalate: bool
    failures: List[Dict[str, Any]]
    matched_policies: List[str]
    rationale: Optional[str]


# ============================================================================
# Engine Result Types
# ============================================================================

class EngineResultDict(TypedDict, total=False):
    """Type definition for engine result."""
    trace_id: str
    output_text: str
    model_info: ModelInfoDict
    critic_findings: Dict[str, CriticResultDict]
    aggregated: AggregationOutputDict
    uncertainty: Optional[UncertaintyAnalysisDict]
    precedent_alignment: Optional[PrecedentAlignmentDict]
    evidence_count: Optional[int]
    degraded_components: List[str]
    is_degraded: bool
    forensic: Optional[Dict[str, Any]]


# ============================================================================
# Configuration Types
# ============================================================================

class CacheConfigDict(TypedDict, total=False):
    """Type definition for cache configuration."""
    enabled: bool
    redis_url: Optional[str]
    precedent_ttl: int
    embeddings_ttl: int
    router_ttl: int
    critics_ttl: int
    detector_ttl: int


class CircuitBreakerConfigDict(TypedDict, total=False):
    """Type definition for circuit breaker configuration."""
    enabled: bool
    failure_threshold: int
    recovery_timeout: float
    success_threshold: int


# ============================================================================
# API Types
# ============================================================================

class DeliberationRequestDict(TypedDict):
    """Type definition for deliberation request."""
    input: str
    context: Dict[str, Any]
    trace_id: Optional[str]
    policy_profile: Optional[str]
    proposed_action: Optional[Dict[str, Any]]
    evidence_inputs: Optional[Dict[str, Any]]
    model_metadata: Optional[Dict[str, Any]]
    human_action: Optional[Dict[str, Any]]


class DeliberationResponseDict(TypedDict, total=False):
    """Type definition for deliberation response."""
    trace_id: str
    timestamp: float
    model_used: str
    model_output: Optional[str]
    critic_outputs: Dict[str, CriticResultDict]
    precedent: Optional[PrecedentAlignmentDict]
    uncertainty: Optional[UncertaintyAnalysisDict]
    aggregator_output: AggregationOutputDict
    governance: Optional[GovernanceResultDict]
    final_decision: str
    execution_decision: Optional[Dict[str, Any]]
    degraded_components: List[str]
    is_degraded: bool


__all__ = [
    "CriticResultDict",
    "CriticEvaluationInput",
    "ModelInfoDict",
    "RouterResponse",
    "RouterDiagnostics",
    "PrecedentCaseDict",
    "PrecedentAlignmentDict",
    "UncertaintyAnalysisDict",
    "AggregationOutputDict",
    "EvidenceRecordDict",
    "GovernanceResultDict",
    "EngineResultDict",
    "CacheConfigDict",
    "CircuitBreakerConfigDict",
    "DeliberationRequestDict",
    "DeliberationResponseDict",
]
