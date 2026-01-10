"""
Governance-related REST endpoints for human review observability.
These are read-only, audit-safe views for metrics and quarantine status.
"""

import os

from fastapi import APIRouter, HTTPException, Depends, status

from governance.human_review.metrics import review_metrics
from governance.human_review.analytics import severity_drift, dissent_suppression_rate
from precedent.quarantine import list_quarantined_cases
from api.middleware.auth import require_role, require_authenticated_user
from api.schemas import GovernancePreviewRequest
from api.bootstrap import evaluate_opa, GOVERNANCE_SCHEMA_VERSION
from api.rest.deps import get_engine
from api.rest.metrics import OPA_CALLS
from engine.logging_config import get_logger

router = APIRouter(prefix="/governance", tags=["Governance"])
REVIEWER_ROLE = os.getenv("REVIEWER_ROLE", "reviewer")
logger = get_logger(__name__)


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


@router.post("/preview", tags=["Governance"])
async def governance_preview(
    payload: GovernancePreviewRequest,
    user: str = Depends(require_authenticated_user),
    engine=Depends(get_engine),
):
    """
    Run governance evaluation on a mock evidence bundle.
    Useful for testing OPA policies without running the full deliberation pipeline.
    """
    try:
        opa_callback = getattr(engine, "opa_callback", None) or getattr(engine, "opa", None)
        if opa_callback is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="OPA callback not configured",
            )

        payload_dict = payload.model_dump()
        payload_dict["schema_version"] = GOVERNANCE_SCHEMA_VERSION
        result = await evaluate_opa(opa_callback, payload_dict)
        if OPA_CALLS:
            opa_outcome = (
                "deny"
                if not result.get("allow", True)
                else "escalate"
                if result.get("escalate")
                else "allow"
            )
            OPA_CALLS.labels(result=opa_outcome).inc()
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Governance preview failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Governance evaluation failed",
        )
