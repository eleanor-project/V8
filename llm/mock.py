from typing import Any, Dict
from llm.base import LLMClient


class MockLLM(LLMClient):
    """
    Deterministic mock LLM used for development, testing, and demos.

    This implementation does NOT generate language probabilistically.
    It returns structured, plausible ethical reasoning outputs that
    allow Eleanor's critic and aggregation logic to be exercised safely.
    """

    def invoke(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        # NOTE:
        # We intentionally ignore the specific text of the system prompt here.
        # Real implementations will not.
        # This mock exists to validate governance flow, not linguistic creativity.

        lower_prompt = user_prompt.lower()

        # Very light heuristic branching to feel "alive"
        if "boss" in lower_prompt or "manager" in lower_prompt:
            return {
                "concern": "Potential power imbalance in an employment relationship",
                "severity": 1.8,
                "principle": "Autonomy / Non-coercion",
                "uncertainty": "Specific workplace policies are unknown",
                "rationale": (
                    "Relationships involving supervisors can introduce implicit pressure, "
                    "even when no explicit coercion is intended."
                ),
                "precedent": "UDHR Article 23(1): Everyone has the right to work, to free choice of employment, to just and favourable conditions of work.",
            }

        return {
            "concern": "No significant ethical concern identified",
            "severity": 0.4,
            "principle": "Personal autonomy",
            "uncertainty": None,
            "rationale": (
                "The scenario does not appear to involve coercion, harm, "
                "or rights violations based on the available information."
            ),
            "precedent": "UDHR Article 1: All human beings are born free and equal in dignity and rights.",
        }
