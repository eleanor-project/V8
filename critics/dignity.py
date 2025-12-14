from .base import CriticOutput
from .llm import LLMClient
from .helpers import parse_critic_response
from .precedent import get_precedent_context, get_severity_scale


def dignity_critic(prompt: str, llm: LLMClient) -> CriticOutput:
    precedent_context = get_precedent_context()
    severity_scale = get_severity_scale()

    system_prompt = f"""
You are the Dignity Critic for the Eleanor governance system.

Your role:
- Evaluate whether persons are treated with inherent dignity and respect.
- Identify humiliation, degradation, dehumanization, or objectification.
- Use the shared severity scale (impact magnitude, 0.0–2.5).
- Do not give advice; describe dignity-related concerns.
- Reference human dignity norms when appropriate (e.g., UDHR Art. 1).

{severity_scale}

{precedent_context}

Respond strictly in this JSON schema (no extra text):
{{
  "concern": "short description of the dignity-related concern, or 'none'",
  "severity": number,
  "principle": "dignity-related principle",
  "uncertainty": "what is unknown or ambiguous, or null",
  "rationale": "concise explanation",
  "precedent_refs": ["optional references like 'UDHR Art. 1: dignity'"]
}}
"""
    raw = llm.invoke(system_prompt=system_prompt, user_prompt=prompt)
    return parse_critic_response(raw, critic_name="Dignity")
