"""Typed schemas for critic evaluation results.

Provides strong typing for critic outputs to replace Dict[str, Any].
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class SeverityLevel(str, Enum):
    """Standardized severity levels across all critics."""
    NONE = "none"
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Violation(BaseModel):
    """Individual violation detected by a critic."""
    rule_id: str
    description: str
    severity: SeverityLevel
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: Optional[Dict[str, Any]] = None
    mitigation: Optional[str] = None
    
    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence must be between 0.0 and 1.0')
        return v


class CriticEvaluationResult(BaseModel):
    """Structured result from a critic evaluation."""
    critic: str
    score: float = Field(ge=0.0, le=1.0, description="Overall severity score")
    severity: SeverityLevel
    violations: List[Violation] = Field(default_factory=list)
    justification: str = Field(description="Human-readable explanation")
    principle: Optional[str] = None
    evaluated_rules: List[str] = Field(default_factory=list)
    duration_ms: Optional[float] = None
    evidence: Optional[Dict[str, Any]] = None
    precedent_refs: List[str] = Field(default_factory=list)
    
    @field_validator('score')
    @classmethod
    def validate_score(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError('Score must be between 0.0 and 1.0')
        return v


class CriticDissentRecord(BaseModel):
    """Records when critics disagree on severity."""
    critics: List[str]
    disagreement_score: float = Field(ge=0.0, le=1.0)
    primary_concern: str
    dissenting_views: List[Dict[str, Any]]


class AggregatedCriticResult(BaseModel):
    """Aggregated results from all critics."""
    average_severity: float = Field(ge=0.0, le=1.0)
    max_severity: float = Field(ge=0.0, le=1.0)
    total_violations: int = Field(ge=0)
    critics_flagged: List[str]
    dissent: Optional[CriticDissentRecord] = None
    rights_impacted: List[str] = Field(default_factory=list)
    final_output: str
    constitutional_verdict: str
