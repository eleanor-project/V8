from pydantic import BaseModel
from typing import List, Optional
from enum import Enum


class CoverageIssue(BaseModel):
    critic: str
    missing_reference: str
    notes: str


class SeverityAssessment(BaseModel):
    original: float
    adjusted: Optional[float]
    justification: str


class DissentEvaluation(BaseModel):
    present: bool
    preserved: bool
    notes: Optional[str] = None


class ReviewOutcome(str, Enum):
    AFFIRMED = "affirmed"
    CLARIFIED = "clarified"
    ADJUSTED = "adjusted"
    CONTESTED = "contested"
    REJECTED = "rejected"


class HumanReviewRecord(BaseModel):
    review_id: str
    case_id: str
    reviewer_role: str
    timestamp: str

    coverage_issues: List[CoverageIssue]
    severity_assessment: SeverityAssessment
    dissent_evaluation: DissentEvaluation

    uncertainty_adequate: bool
    outcome: ReviewOutcome

    reviewer_justification: str
