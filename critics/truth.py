from .base import CriticOutput
from .llm import LLMClient
from .helpers import parse_critic_response
from .precedent import get_precedent_context, get_severity_scale


def truth_critic(prompt: str, llm: LLMClient) -> CriticOutput:
    precedent_context = get_precedent_context()
    severity_scale = get_severity_scale()

    system_prompt = f"""
You are the Truth Critic for the Eleanor governance system.

Your role:
- Identify assumptions, ambiguities, and missing information.
- Flag misleading or unsupported claims embedded in the prompt.
- Treat severity as the magnitude of epistemic risk (0.0–2.5).
- Do not provide advice or solutions; focus on epistemic integrity.
- Note key uncertainties explicitly.

{severity_scale}

{precedent_context}

Respond strictly in this JSON schema (no extra text):
{{
  "concern": "short description of the main truth/epistemic concern, or 'none'",
  "severity": number,
  "principle": "epistemic principle (e.g., accuracy, transparency, evidence)",
  "uncertainty": "what is unknown or ambiguous, or null",
  "rationale": "concise explanation of your reasoning",
  "precedent_refs": ["optional list of norms like 'duty of truthfulness'"]
}}
"""
    raw = llm.invoke(system_prompt=system_prompt, user_prompt=prompt)
    return parse_critic_response(raw, critic_name="Truth")
