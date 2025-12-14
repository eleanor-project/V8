from .base import CriticOutput
from .llm import LLMClient
from .helpers import parse_critic_response
from .precedent import get_precedent_context, get_severity_scale


def pragmatics_critic(prompt: str, llm: LLMClient) -> CriticOutput:
    precedent_context = get_precedent_context()
    severity_scale = get_severity_scale()

    system_prompt = f"""
You are the Pragmatics Critic for the Eleanor governance system.

Your role:
- Identify real-world constraints (legal, organizational, resource, feasibility).
- Evaluate whether actions implied by the prompt are realistically implementable.
- Treat severity as the magnitude of practical misalignment or infeasibility.
- Do not provide detailed plans; focus on constraints and friction.

{severity_scale}

{precedent_context}

Respond strictly in this JSON schema (no extra text):
{{
  "concern": "short description of the main pragmatic concern, or 'none'",
  "severity": number,
  "principle": "pragmatic/value principle (e.g., feasibility, institutional fit)",
  "uncertainty": "what is unknown or ambiguous, or null",
  "rationale": "concise explanation of your reasoning",
  "precedent_refs": ["optional references (e.g., organizational norms, policies)"]
}}
"""
    raw = llm.invoke(system_prompt=system_prompt, user_prompt=prompt)
    return parse_critic_response(raw, critic_name="Pragmatics")
