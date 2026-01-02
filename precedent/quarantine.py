"""
Quarantined cases:
- rejected
- contested
- explicitly excluded from automation
"""

from datetime import datetime
from typing import Any
from replay_store import load_human_reviews


def is_quarantined(case_id: str) -> bool:
    reviews = load_human_reviews(case_id)
    if not reviews:
        return False

    latest = sorted(
        reviews,
        key=lambda r: r.get("timestamp") or "",
        reverse=True,
    )[0]

    return latest.get("outcome") in ("contested", "rejected")


def list_quarantined_cases() -> list[dict]:
    """
    Enumerate cases whose latest review outcome is contested or rejected.
    Returns lightweight metadata for UI / reporting.
    """
    from os import listdir
    from replay_store import REVIEW_RECORD_DIR
    import json
    import os

    results: list[dict[str, Any]] = []
    if not os.path.exists(REVIEW_RECORD_DIR):
        return results

    for fname in listdir(REVIEW_RECORD_DIR):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(REVIEW_RECORD_DIR, fname)
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception:
            continue

        case_id = data.get("case_id")
        outcome = data.get("outcome")
        if outcome not in ("contested", "rejected"):
            continue

        # Gather all reviews for this case to count and check dissent
        reviews = load_human_reviews(case_id)
        latest = sorted(
            reviews,
            key=lambda r: r.get("timestamp") or "",
            reverse=True,
        )[0]

        results.append(
            {
                "case_id": case_id,
                "latest_outcome": latest.get("outcome"),
                "dissent_present": bool(
                    latest.get("dissent_evaluation", {}).get("present")
                ),
                "review_count": len(reviews),
                "last_reviewed": latest.get("timestamp"),
            }
        )

    # Sort by most recently reviewed
    results.sort(
        key=lambda r: (
            datetime.fromisoformat(r["last_reviewed"])
            if r.get("last_reviewed")
            else datetime.min
        ),
        reverse=True,
    )

    return results
