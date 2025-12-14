"""
Human Review Module for Eleanor Governance System.

This module defines the human-in-the-loop review process for high-stakes
ethical deliberations that require human oversight before precedent promotion.
"""

from .schemas import (
    CoverageIssue,
    SeverityAssessment,
    DissentEvaluation,
    ReviewOutcome,
    HumanReviewRecord,
)
from .prompts import REVIEWER_SYSTEM_PROMPT
from .service import submit_review, get_review
from .audit import audit_review, get_review_stats

__all__ = [
    "CoverageIssue",
    "SeverityAssessment",
    "DissentEvaluation",
    "ReviewOutcome",
    "HumanReviewRecord",
    "REVIEWER_SYSTEM_PROMPT",
    "submit_review",
    "get_review",
    "audit_review",
    "get_review_stats",
]
