"""
Post-hoc analytics for human review integrity.
"""

from engine.replay_store import load_human_reviews


def severity_drift(case_id: str):
    reviews = load_human_reviews(case_id)
    if not reviews:
        return None

    deltas = []
    for r in reviews:
        sa = r.get("severity_assessment", {})
        original = sa.get("original")
        adjusted = sa.get("adjusted")
        if adjusted is not None and original is not None:
            try:
                deltas.append(float(adjusted) - float(original))
            except (TypeError, ValueError):
                continue

    return {
        "count": len(deltas),
        "mean_delta": (sum(deltas) / len(deltas)) if deltas else 0.0,
        "deltas": deltas,
    }


def dissent_suppression_rate(case_id: str):
    reviews = load_human_reviews(case_id)
    if not reviews:
        return None

    suppressed = [
        r
        for r in reviews
        if r.get("dissent_evaluation", {}).get("present")
        and not r.get("dissent_evaluation", {}).get("preserved")
    ]

    return {
        "total_reviews": len(reviews),
        "suppressed": len(suppressed),
        "rate": (len(suppressed) / len(reviews)) if reviews else 0.0,
    }
