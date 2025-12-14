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
    ordered = sorted(
        critic_outputs,
        key=lambda c: c.severity,
        reverse=True
    )

    # Build deliberation lines for UI
    deliberation = []
    for c in ordered:
        line = f"{c.critic}: {c.concern}"
        if c.severity >= 1.5:
            line += f" (severity {c.severity})"
        deliberation.append(line)

    # Use highest-severity concerns to shape explanation
    dominant = ordered[0]

    explanation = (
        f"{dominant.critic} considerations carried the greatest moral weight. "
        f"{dominant.rationale} "
        "Other perspectives were considered, but did not outweigh the primary concern. "
        "Uncertainty was acknowledged where context was incomplete."
    )

    return {
        "deliberation": deliberation,
        "final_answer": explanation
    }
