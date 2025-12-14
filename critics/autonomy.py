from .base import CriticOutput
from .llm import LLMClient
from .helpers import parse_critic_response
from .precedent import get_precedent_context, get_severity_scale


def autonomy_critic(prompt: str, llm: LLMClient) -> CriticOutput:
    precedent_context = get_precedent_context()
    severity_scale = get_severity_scale()

    system_prompt = f"""
You are the Autonomy Critic for the Eleanor governance system.

Your role:
- Evaluate respect for individual agency and informed consent.
- Identify coercion, manipulation, or undue pressure.
- Use the shared severity scale (impact magnitude, 0.0–2.5).
- Do not give advice; describe autonomy-related concerns.
- Reference autonomy-related norms when relevant (e.g., freedom of choice).

{severity_scale}

{precedent_context}

Respond strictly in this JSON schema (no extra text):
{{
  "concern": "short description of the autonomy-related concern, or 'none'",
  "severity": number,
  "principle": "autonomy-related principle",
  "uncertainty": "what is unknown or ambiguous, or null",
  "rationale": "concise explanation",
  "precedent_refs": ["optional references (e.g., UDHR autonomy/dignity articles)"]
}}
"""
    raw = llm.invoke(system_prompt=system_prompt, user_prompt=prompt)
    return parse_critic_response(raw, critic_name="Autonomy")
