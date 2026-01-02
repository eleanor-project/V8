"""
Audit functions for human review records.

Provides quality gates and traceability for human review process.
"""

from typing import Dict, Any, Optional
import json

from .schemas import HumanReviewRecord, ReviewOutcome


def audit_review(review: HumanReviewRecord) -> Dict[str, Any]:
    """
    Audit a review record for quality and completeness.

    Quality gates:
    - Justification must not be empty
    - Severity adjustments require justification
    - Rejected outcomes require coverage issues or severity problems
    - Affirmed outcomes should not have major coverage issues

    Args:
        review: HumanReviewRecord to audit

    Returns:
        dict with:
            - valid: bool
            - issues: List[str] of any problems found
            - warnings: List[str] of non-blocking concerns
    """
    issues = []
    warnings = []

    # Check justification is meaningful
    if not review.reviewer_justification or len(review.reviewer_justification.strip()) < 10:
        issues.append("Reviewer justification is empty or too short")

    # Check severity adjustment has justification
    if review.severity_assessment.adjusted is not None:
        if not review.severity_assessment.justification or \
           len(review.severity_assessment.justification.strip()) < 10:
            issues.append("Severity was adjusted but justification is insufficient")

    # Check rejected outcomes have supporting evidence
    if review.outcome == ReviewOutcome.REJECTED:
        if not review.coverage_issues and \
           review.severity_assessment.adjusted is None:
            warnings.append("Rejected outcome has no coverage issues or severity adjustments")

    # Check affirmed outcomes don't have major issues
    if review.outcome == ReviewOutcome.AFFIRMED:
        if len(review.coverage_issues) > 2:
            warnings.append("Affirmed outcome has multiple coverage issues")

    # Check dissent evaluation consistency
    if review.dissent_evaluation.present and not review.dissent_evaluation.preserved:
        if review.outcome == ReviewOutcome.AFFIRMED:
            issues.append("Cannot affirm when dissent was present but not preserved")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
    }


def get_review_stats(case_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Get statistics about reviews.

    Args:
        case_id: Optional case ID to filter by

    Returns:
        dict with review statistics
    """
    from .service import REVIEW_STORE_PATH

    reviews = []
    for review_file in REVIEW_STORE_PATH.glob("*.json"):
        with open(review_file, "r") as f:
            data = json.load(f)

        if case_id is None or data.get("case_id") == case_id:
            reviews.append(data)

    if not reviews:
        return {
            "total_reviews": 0,
            "outcomes": {},
            "avg_coverage_issues": 0,
            "severity_adjusted_pct": 0,
        }

    # Calculate stats
    outcome_counts: dict[str, int] = {}
    total_coverage_issues = 0
    severity_adjusted_count = 0

    for review in reviews:
        outcome = review.get("outcome", "unknown")
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1

        total_coverage_issues += len(review.get("coverage_issues", []))

        if review.get("severity_assessment", {}).get("adjusted") is not None:
            severity_adjusted_count += 1

    return {
        "total_reviews": len(reviews),
        "outcomes": outcome_counts,
        "avg_coverage_issues": total_coverage_issues / len(reviews),
        "severity_adjusted_pct": (severity_adjusted_count / len(reviews)) * 100,
    }


def validate_review_chain(case_id: str) -> Dict[str, Any]:
    """
    Validate the complete review chain for a case.

    Checks:
    - Are there conflicting reviews?
    - Is there a clear final outcome?
    - Are there unresolved issues?

    Args:
        case_id: Case identifier

    Returns:
        dict with validation results
    """
    from .service import get_reviews_for_case

    reviews = get_reviews_for_case(case_id)

    if not reviews:
        return {
            "valid": False,
            "reason": "No reviews found for case",
        }

    # Check for conflicting outcomes
    outcomes = [r.outcome for r in reviews]
    if len(set(outcomes)) > 1:
        return {
            "valid": True,  # Conflicting outcomes are legitimate
            "warning": "Multiple reviewers gave different outcomes",
            "outcomes": outcomes,
        }

    # Get most recent review
    latest_review = max(reviews, key=lambda r: r.timestamp)

    return {
        "valid": True,
        "final_outcome": latest_review.outcome,
        "review_count": len(reviews),
        "latest_review_id": latest_review.review_id,
    }
