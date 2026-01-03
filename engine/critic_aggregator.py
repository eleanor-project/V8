from typing import List
from critics.schema import CriticOutput


def aggregate(critic_outputs: List[CriticOutput]) -> dict:
    """
    Aggregate critic outputs into deliberation and explanation.

    - Preserves dissent
    - Weights by severity (impact magnitude)
    - Produces explanation, not advice
    """

    # Sort by severity descending (moral gravity)
    ordered = sorted(critic_outputs, key=lambda c: c.severity, reverse=True)

    # Build deliberation lines for UI
    deliberation = []
    for c in ordered:
        line = f"{c.critic}: {c.concern} " f"[severity {c.severity}] — {c.principle}"
        if c.precedent:
            line += f"\n  ⚖️  {c.precedent}"
        deliberation.append(line)

    # Use highest-severity concerns to shape explanation
    dominant = ordered[0]

    explanation = (
        f"{dominant.critic} considerations carried the greatest moral weight "
        f"(severity {dominant.severity}, principle: {dominant.principle}). "
        f"{dominant.rationale} "
    )

    if dominant.precedent:
        explanation += f"Grounded in: {dominant.precedent} "

    if dominant.uncertainty:
        explanation += f"Uncertainty noted: {dominant.uncertainty} "

    explanation += "Other perspectives were considered, but did not outweigh the primary concern."

    return {"deliberation": deliberation, "final_answer": explanation}
