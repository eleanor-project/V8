import os
from typing import Dict

from llm.base import LLMClient


class AnthropicClient(LLMClient):
    """
    Minimal Anthropic client placeholder.

    Intentionally conservative: raises a clear runtime error if the
    required API key is not present instead of silently falling back.
    """

    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_KEY")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_KEY is required for the Anthropic backend.")

    def invoke(self, system_prompt: str, user_prompt: str) -> Dict:
        # Real implementation would call Anthropic Claude models.
        # Kept lean here to avoid hidden network calls in tests.
        raise NotImplementedError("AnthropicClient.invoke is not implemented yet.")
