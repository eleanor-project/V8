"""
Human review API endpoints for Eleanor V8.

Provides REST endpoints for submitting and retrieving human reviews.
This fits into the existing API structure without creating a new server.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
import os
import uuid

from governance.human_review.schemas import HumanReviewRecord
from governance.human_review.service import submit_review, get_review, get_reviews_for_case
from governance.human_review.audit import get_review_stats, validate_review_chain
from governance.stewardship import get_pending_reviews, resolve_review
from precedent.promotion_router import PromotionRouter
from api.middleware.auth import require_role


router = APIRouter(prefix="/review", tags=["human-review"])
promotion_router = PromotionRouter()
REVIEWER_ROLE = os.getenv("REVIEWER_ROLE", "reviewer")


@router.post("/submit")
@require_role(REVIEWER_ROLE)
def submit_review_endpoint(review: HumanReviewRecord):
    """
    Submit a human review record.

    The review must conform to the structured HumanReviewRecord schema.
    No free-form reviews allowed.

    Returns:
        dict with status and review details
    """
    try:
        result = submit_review(review)

        # Route to appropriate promotion lane
        lane = promotion_router.route(review)
        result["promotion_lane"] = lane

        # Resolve pending review
        resolve_review(review.case_id)

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/get/{review_id}")
@require_role(REVIEWER_ROLE)
def get_review_endpoint(review_id: str):
    """
    Get a specific review by ID.

    Args:
        review_id: Unique review identifier

    Returns:
        HumanReviewRecord if found

    Raises:
        404 if review not found
    """
    review = get_review(review_id)

    if review is None:
        raise HTTPException(status_code=404, detail="Review not found")

    return review


@router.get("/case/{case_id}")
@require_role(REVIEWER_ROLE)
def get_case_reviews_endpoint(case_id: str):
    """
    Get all reviews for a specific case.

    Args:
        case_id: Case identifier

    Returns:
        List of HumanReviewRecord for this case
    """
    reviews = get_reviews_for_case(case_id)
    return {"case_id": case_id, "reviews": reviews, "count": len(reviews)}


@router.get("/pending")
@require_role(REVIEWER_ROLE)
def get_pending_reviews_endpoint():
    """
    Get all pending review packets awaiting human review.

    Returns:
        List of pending ReviewPackets
    """
    pending = get_pending_reviews()
    return {"pending_reviews": pending, "count": len(pending)}


@router.get("/stats")
@require_role(REVIEWER_ROLE)
def get_stats_endpoint(case_id: Optional[str] = Query(None)):
    """
    Get review statistics.

    Args:
        case_id: Optional case ID to filter by

    Returns:
        dict with review statistics
    """
    stats = get_review_stats(case_id)
    return stats


@router.get("/validate/{case_id}")
@require_role(REVIEWER_ROLE)
def validate_case_endpoint(case_id: str):
    """
    Validate the review chain for a case.

    Checks for conflicting reviews, final outcome, etc.

    Args:
        case_id: Case identifier

    Returns:
        dict with validation results
    """
    validation = validate_review_chain(case_id)
    return validation


@router.get("/lane/{lane_name}")
@require_role(REVIEWER_ROLE)
def get_lane_contents_endpoint(lane_name: str):
    """
    Get all cases in a specific promotion lane.

    Valid lanes:
    - precedent_candidate
    - training_calibration
    - policy_insight
    - quarantine

    Args:
        lane_name: Name of the lane

    Returns:
        List of cases in that lane
    """
    valid_lanes = ["precedent_candidate", "training_calibration", "policy_insight", "quarantine"]

    if lane_name not in valid_lanes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid lane. Must be one of: {', '.join(valid_lanes)}"
        )

    contents = promotion_router.get_lane_contents(lane_name)
    return {"lane": lane_name, "cases": contents, "count": len(contents)}
