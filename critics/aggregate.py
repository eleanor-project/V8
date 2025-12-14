from typing import List, Dict, Any
from .base import CriticOutput
from .llm import LLMClient


def _compose_aggregation_prompt(critic_outputs: List[CriticOutput]) -> str:
    """
    Convert critic outputs into a text prompt for the aggregator LLM.
    """
    lines = []
    for c in critic_outputs:
        precedent_str = ', '.join(c.precedent_refs or []) if c.precedent_refs else "none"
        lines.append(
            f"Critic: {c.critic}\n"
            f"Concern: {c.concern}\n"
            f"Severity: {c.severity}\n"
            f"Principle: {c.principle}\n"
            f"Uncertainty: {c.uncertainty if c.uncertainty else 'none'}\n"
            f"Rationale: {c.rationale}\n"
            f"Precedent: {precedent_str}\n"
            "----\n"
        )
    return "\n".join(lines)


def aggregate(critic_outputs: List[CriticOutput], llm: LLMClient) -> Dict[str, Any]:
    """
    Aggregate critic outputs into:
    - deliberation: list of short per-critic lines.
    - final_answer: reasoned explanation that preserves dissent and highlights severity.
    - max_severity: highest severity score among all critics.
    """
    max_severity = max((c.severity for c in critic_outputs), default=0.0)

    # Build deliberation list
    deliberation = []
    for c in critic_outputs:
        sev_part = f" (severity {c.severity:.1f})" if c.severity > 0 else ""
        deliberation.append(f"{c.critic}: {c.concern}{sev_part}")

    # Compose aggregation context for LLM
    aggregation_context = _compose_aggregation_prompt(critic_outputs)

    system_prompt = """
You are the Eleanor Aggregator.

Your role:
- Integrate multiple critic outputs into a single, coherent explanation.
- Preserve dissent and minority views instead of averaging them away.
- Give weight to higher-severity concerns (impact magnitude).
- Explicitly mention when critics disagree or when uncertainty is significant.
- Do not override critic judgments; instead, explain their relationships.
- Write in a style closer to reasoned justification or jurisprudence, not casual advice.
- Do not directly tell the user what to do; describe the ethical landscape.

Respond with a single field:
{
  "final_answer": "one to three paragraphs of structured, reasoned explanation"
}
"""

    raw = llm.invoke(system_prompt=system_prompt, user_prompt=aggregation_context)
    final_answer = str(raw.get("final_answer", "")).strip()

    # Fallback if LLM doesn't return properly
    if not final_answer:
        final_answer = f"The most significant ethical concern comes from critics with severity {max_severity:.1f}. Multiple perspectives were considered in this deliberation."

    return {
        "deliberation": deliberation,
        "final_answer": final_answer,
        "max_severity": max_severity,
    }
