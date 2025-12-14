import os
from llm.base import LLMClient
from llm.mock import MockLLM
from llm.openai_client import OpenAIClient


def get_llm() -> LLMClient:
    """
    Select LLM backend based on environment configuration.

    Default: MockLLM (safe, deterministic)
    """

    backend = os.getenv("ELEANOR_LLM_BACKEND", "mock").lower()

    if backend == "openai":
        return OpenAIClient()

    if backend == "mock":
        return MockLLM()

    raise ValueError(f"Unknown LLM backend: {backend}")
