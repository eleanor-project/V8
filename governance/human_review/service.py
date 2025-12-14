"""
Human review service for Eleanor governance system.

Handles storage, retrieval, and validation of human review records.
This is governance infrastructure, not UX.
"""

from typing import Optional, Dict, Any
from pathlib import Path
import json
from datetime import datetime

from .schemas import HumanReviewRecord
from .audit import audit_review


# Storage location (could be database, file system, or external service)
REVIEW_STORE_PATH = Path("governance/human_review/reviews")
REVIEW_STORE_PATH.mkdir(parents=True, exist_ok=True)


def submit_review(review: HumanReviewRecord) -> Dict[str, Any]:
    """
    Submit a human review record.

    Args:
        review: Validated HumanReviewRecord

    Returns:
        dict with status and review_id

    Raises:
        ValueError: If review validation fails
    """
    # Validate schema automatically via Pydantic
    # This ensures NO free-form reviews get through

    # Audit the review for quality gates
    audit_result = audit_review(review)
    if not audit_result["valid"]:
        raise ValueError(f"Review failed audit: {audit_result['issues']}")

    # Store review
    review_file = REVIEW_STORE_PATH / f"{review.review_id}.json"
    with open(review_file, "w") as f:
        json.dump(review.dict(), f, indent=2, default=str)

    return {
        "status": "accepted",
        "review_id": review.review_id,
        "case_id": review.case_id,
        "outcome": review.outcome,
        "timestamp": review.timestamp.isoformat(),
    }


def get_review(review_id: str) -> Optional[HumanReviewRecord]:
    """
    Retrieve a review record by ID.

    Args:
        review_id: Unique review identifier

    Returns:
        HumanReviewRecord if found, None otherwise
    """
    review_file = REVIEW_STORE_PATH / f"{review_id}.json"

    if not review_file.exists():
        return None

    with open(review_file, "r") as f:
        data = json.load(f)

    # Reconstruct datetime from ISO string
    if "timestamp" in data and isinstance(data["timestamp"], str):
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])

    return HumanReviewRecord(**data)


def get_reviews_for_case(case_id: str) -> list[HumanReviewRecord]:
    """
    Get all reviews for a specific case.

    Args:
        case_id: Case identifier

    Returns:
        List of HumanReviewRecord for this case
    """
    reviews = []

    for review_file in REVIEW_STORE_PATH.glob("*.json"):
        with open(review_file, "r") as f:
            data = json.load(f)

        if data.get("case_id") == case_id:
            # Reconstruct datetime
            if "timestamp" in data and isinstance(data["timestamp"], str):
                data["timestamp"] = datetime.fromisoformat(data["timestamp"])
            reviews.append(HumanReviewRecord(**data))

    return reviews


def store_review(review: HumanReviewRecord) -> None:
    """
    Internal function to store review (used by submit_review).

    Args:
        review: HumanReviewRecord to store
    """
    review_file = REVIEW_STORE_PATH / f"{review.review_id}.json"
    with open(review_file, "w") as f:
        json.dump(review.dict(), f, indent=2, default=str)
