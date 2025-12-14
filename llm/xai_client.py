import os
from typing import Dict

from llm.base import LLMClient


class XAIClient(LLMClient):
    """
    Minimal xAI (Grok) client placeholder.
    """

    def __init__(self):
        self.api_key = os.getenv("XAI_KEY")
        if not self.api_key:
            raise RuntimeError("XAI_KEY is required for the xAI backend.")

    def invoke(self, system_prompt: str, user_prompt: str) -> Dict:
        raise NotImplementedError("XAIClient.invoke is not implemented yet.")
