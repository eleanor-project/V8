import os
from typing import Dict

from llm.base import LLMClient


class LlamaClient(LLMClient):
    """
    Minimal local Llama client placeholder.

    Designed to be wired to an on-prem/ollama/gguf runtime by the
    operator. Fails fast if no model is configured.
    """

    def __init__(self):
        self.model = os.getenv("OLLAMA_MODEL") or os.getenv("LLAMA_MODEL")
        if not self.model:
            raise RuntimeError("OLLAMA_MODEL or LLAMA_MODEL is required for the Llama backend.")

    def invoke(self, system_prompt: str, user_prompt: str) -> Dict:
        raise NotImplementedError("LlamaClient.invoke is not implemented yet.")
