import os
from typing import Dict

from llm.base import LLMClient


class GeminiClient(LLMClient):
    """
    Minimal Google Gemini client placeholder.

    We fail fast if the API key is missing; call sites should only enable
    this backend when the integration is wired.
    """

    def __init__(self):
        self.api_key = os.getenv("GEMINI_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise RuntimeError("GEMINI_KEY/GOOGLE_API_KEY is required for the Gemini backend.")

    def invoke(self, system_prompt: str, user_prompt: str) -> Dict:
        raise NotImplementedError("GeminiClient.invoke is not implemented yet.")
