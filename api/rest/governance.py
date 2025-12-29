"""
Governance-related REST endpoints for human review observability.
These are read-only, audit-safe views for metrics and quarantine status.
"""

from fastapi import APIRouter, HTTPException
import os

from governance.human_review.metrics import review_metrics
from governance.human_review.analytics import severity_drift, dissent_suppression_rate
from precedent.quarantine import list_quarantined_cases
from api.middleware.auth import require_role

router = APIRouter(prefix="/governance", tags=["governance"])
REVIEWER_ROLE = os.getenv("REVIEWER_ROLE", "reviewer")


@router.get("/review/metrics/{case_id}")
@require_role(REVIEWER_ROLE)
def get_review_metrics(case_id: str):
    metrics = review_metrics(case_id)
    drift = severity_drift(case_id)
    dissent = dissent_suppression_rate(case_id)

    if metrics == {} and drift is None and dissent is None:
        raise HTTPException(status_code=404, detail="No reviews found for case")

    return {
        "case_id": case_id,
        "metrics": metrics,
        "severity_drift": drift,
        "dissent_suppression": dissent,
    }


@router.get("/review/quarantine")
@require_role(REVIEWER_ROLE)
def get_quarantined_cases():
    return list_quarantined_cases()
