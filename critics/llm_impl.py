from typing import Dict, Any
from .llm import LLMClient
from llm.factory import get_llm as get_base_llm


class LLMClientImpl(LLMClient):
    """
    Concrete implementation of LLMClient that wraps the base LLM factory.
    """

    def __init__(self):
        self._llm = get_base_llm()

    def invoke(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """
        Invoke the underlying LLM and return the result.
        """
        return self._llm.invoke(system_prompt, user_prompt)


def get_llm() -> LLMClient:
    """
    Factory function to get an LLMClient implementation.
    """
    return LLMClientImpl()
