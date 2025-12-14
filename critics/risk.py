from .base import CriticOutput
from .llm import LLMClient
from .helpers import parse_critic_response
from .precedent import get_precedent_context, get_severity_scale


def risk_critic(prompt: str, llm: LLMClient) -> CriticOutput:
    precedent_context = get_precedent_context()
    severity_scale = get_severity_scale()

    system_prompt = f"""
You are the Risk Critic for the Eleanor governance system.

Your role:
- Identify plausible harms (physical, psychological, social, economic, rights-based).
- Focus on severity of impact, not likelihood.
- Treat severity as impact magnitude (0.0–2.5).
- Pay attention to irreversible or large-scale harms.
- Do not provide generic safety advice; focus on structured risk reasoning.

{severity_scale}

{precedent_context}

Respond strictly in this JSON schema (no extra text):
{{
  "concern": "short description of the main risk/harms concern, or 'none'",
  "severity": number,
  "principle": "risk-related principle (e.g., non-maleficence, precaution)",
  "uncertainty": "what is unknown or ambiguous, or null",
  "rationale": "concise explanation of your reasoning",
  "precedent_refs": ["optional norms or frameworks referenced"]
}}
"""
    raw = llm.invoke(system_prompt=system_prompt, user_prompt=prompt)
    return parse_critic_response(raw, critic_name="Risk")
