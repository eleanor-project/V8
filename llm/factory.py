import os
from typing import Dict, Type
from llm.base import LLMClient
from llm.mock import MockLLM
from llm.openai_client import OpenAIClient
from llm.anthropic_client import AnthropicClient
from llm.gemini_client import GeminiClient
from llm.xai_client import XAIClient
from llm.llama_client import LlamaClient


def get_llm() -> LLMClient:
    """
    Select LLM backend based on environment configuration.

    Default: MockLLM (safe, deterministic)
    """

    backend = os.getenv("ELEANOR_LLM_BACKEND", "mock").lower()

    adapters: Dict[str, Type[LLMClient]] = {
        "openai": OpenAIClient,
        "mock": MockLLM,
        "anthropic": AnthropicClient,
        "gemini": GeminiClient,
        "xai": XAIClient,
        "llama": LlamaClient,
    }

    if backend not in adapters:
        raise ValueError(f"Unknown LLM backend: {backend}")

    return adapters[backend]()
