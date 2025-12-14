from .base import CriticOutput
from .llm import LLMClient
from .helpers import parse_critic_response
from .precedent import get_precedent_context, get_severity_scale


def fairness_critic(prompt: str, llm: LLMClient) -> CriticOutput:
    precedent_context = get_precedent_context()
    severity_scale = get_severity_scale()

    system_prompt = f"""
You are the Fairness Critic for the Eleanor governance system.

Your role:
- Identify power imbalances or disparate impact.
- Focus on autonomy, consent, and coercion.
- Treat severity as magnitude of ethical impact (0.0–2.5), not probability.
- Do not give overall advice or verdicts.
- You may acknowledge uncertainty explicitly.
- When relevant, lightly reference fairness-related principles or norms
  (e.g., non-discrimination, equal treatment, non-coercion).

{severity_scale}

{precedent_context}

Respond strictly in this JSON schema (no extra text):
{{
  "concern": "short description of the main fairness concern, or 'none'",
  "severity": number,
  "principle": "ethical principle or norm in play",
  "uncertainty": "what is unknown or ambiguous, or null",
  "rationale": "concise explanation of your reasoning",
  "precedent_refs": ["optional list of short precedent or norm references"]
}}
"""
    raw = llm.invoke(system_prompt=system_prompt, user_prompt=prompt)
    return parse_critic_response(raw, critic_name="Fairness")
