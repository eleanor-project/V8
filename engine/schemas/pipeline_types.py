"""TypedDict models for core pipeline outputs.

These types provide explicit shapes for critic, precedent, uncertainty,
and aggregation outputs without enforcing runtime validation.
"""

from typing import Dict, List, Optional, TypedDict, Union

from engine.types import CriticViolation


ViolationEntry = Union[CriticViolation, Dict[str, object], str]


class CriticResult(TypedDict, total=False):
    critic: str
    value: str
    severity: float
    score: float
    violations: List[ViolationEntry]
    justification: str
    principle: str
    evaluated_rules: List[str]
    duration_ms: float
    evidence: Dict[str, object]
    precedent_refs: List[str]
    error: str
    critic_id: str
    timestamp: float
    rationale: str
    flags: List[str]
    uuid: str


CriticResultsMap = Dict[str, CriticResult]


class PrecedentAlignmentResult(TypedDict, total=False):
    alignment_score: float
    support_strength: float
    conflict_level: float
    drift_score: float
    clusters: Dict[str, List[Dict[str, object]]]
    is_novel: bool
    analysis: str
    retrieval: Dict[str, object]


class PrecedentCaseResult(TypedDict, total=False):
    text: str
    metadata: Dict[str, object]
    embedding: List[float]
    values: List[str]
    aggregate_score: float


class PrecedentRetrievalResult(TypedDict, total=False):
    precedent_cases: List[PrecedentCaseResult]
    alignment_score: float
    top_case: Optional[PrecedentCaseResult]
    query_embedding: List[float]


class UncertaintyResult(TypedDict, total=False):
    epistemic_uncertainty: float
    aleatoric_uncertainty: float
    critic_divergence: float
    precedent_conflict_uncertainty: float
    model_stability_uncertainty: float
    overall_uncertainty: float
    needs_escalation: bool
    explanation: str


class AggregationScore(TypedDict, total=False):
    average_severity: float
    total_severity: float


class AggregationOutput(TypedDict, total=False):
    decision: str
    score: AggregationScore
    critics: Dict[str, CriticResult]
    lexicographic_violations: List[Dict[str, object]]
    precedent: PrecedentAlignmentResult
    uncertainty: UncertaintyResult
    final_output: str
    escalation_summary: Dict[str, object]
    execution_gate: Dict[str, object]
    dissent_present: bool
    audit_hash: str
    aggregation_result: Dict[str, object]
    rights_impacted: List[str]
    dissent: Optional[Dict[str, object]]
    error: Dict[str, object]


__all__ = [
    "CriticResult",
    "CriticResultsMap",
    "ViolationEntry",
    "PrecedentAlignmentResult",
    "PrecedentCaseResult",
    "PrecedentRetrievalResult",
    "UncertaintyResult",
    "AggregationScore",
    "AggregationOutput",
]
