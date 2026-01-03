"""
ELEANOR V8 — Model Adapter Registry
------------------------------------

This module provides a unified interface for attaching ANY LLM backend
to ELEANOR’s critics, router, orchestrator, and runtime engine.

Every adapter exposes a simple call signature:

    adapter(prompt: str) -> str

This keeps the engine backend-agnostic.

Included adapters:
    - GPTAdapter (OpenAI)
    - ClaudeAdapter (Anthropic)
    - GrokAdapter (xAI)
    - LlamaHFAdapter (HuggingFace Transformers; supports Llama/Mistral/Phi3/etc.)
    - OllamaAdapter (local models; supports any Ollama model name)

You may add new adapters without modifying the engine.
"""

import os
import logging
from typing import Any, Optional, Awaitable, TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from openai import OpenAI as OpenAIClient
else:
    OpenAIClient = None  # type: ignore[assignment]

try:
    import anthropic
except Exception:
    anthropic = None  # type: ignore[assignment]

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore[import-not-found]
    import torch
except Exception:
    AutoTokenizer = AutoModelForCausalLM = torch = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


def _get_timeout() -> float:
    raw = os.getenv("ELEANOR_HTTP_TIMEOUT", "10")
    try:
        return float(raw)
    except ValueError:
        return 10.0


DEFAULT_HTTP_TIMEOUT = _get_timeout()


async def _post_json(url: str, payload: dict, headers: Optional[dict] = None) -> Any:
    timeout = DEFAULT_HTTP_TIMEOUT
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()


# ============================================================
#  Base Adapter Pattern
# ============================================================


class BaseLLMAdapter:
    """All adapters must subclass this and implement __call__."""

    def __call__(self, prompt: str) -> str | Awaitable[str]:
        raise NotImplementedError


# ============================================================
#  GPT Adapter (OpenAI)
# ============================================================


class GPTAdapter(BaseLLMAdapter):
    """Adapter for GPT-4.1, GPT-5, etc. via OpenAI API."""

    def __init__(self, model="gpt-4.1", api_key=None):
        if OpenAIClient is None:
            raise ImportError("OpenAI SDK not installed")
        self.client = OpenAIClient(api_key=api_key)
        self.model = model

    def __call__(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}]
        )
        content: Any = response.choices[0].message.content
        return str(content).strip()


# ============================================================
#  Claude Adapter (Anthropic)
# ============================================================


class ClaudeAdapter(BaseLLMAdapter):
    """Adapter for Claude 3.x models."""

    def __init__(self, model="claude-3-opus-20240229", api_key=None):
        if anthropic is None:
            raise ImportError("Anthropic SDK not installed")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def __call__(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model, max_tokens=2048, messages=[{"role": "user", "content": prompt}]
        )
        return str(response.content[0].text).strip()


# ============================================================
#  Grok Adapter (xAI)
# ============================================================


class GrokAdapter(BaseLLMAdapter):
    """Adapter for Grok 1.5 / Grok 3 via xAI’s HTTP API."""

    def __init__(self, model: str = "grok-beta", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key

    async def __call__(self, prompt: str) -> str:
        url = "https://api.x.ai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        payload = {"model": self.model, "messages": [{"role": "user", "content": prompt}]}

        data: Any = await _post_json(url, payload, headers=headers)
        return str(data["choices"][0]["message"]["content"]).strip()


# ============================================================
#  Llama HF Adapter (Transformers)
# ============================================================


class LlamaHFAdapter(BaseLLMAdapter):
    """Adapter for running Llama locally via HuggingFace Transformers."""

    def __init__(self, model_path: str = "meta-llama/Llama-3-8b", device: str = "cpu"):
        if AutoTokenizer is None:
            raise ImportError("Transformers library not installed")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        self.device = device

    def __call__(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=300)
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return str(text).strip()


# ============================================================
#  Ollama Local Adapter
# ============================================================


class OllamaAdapter(BaseLLMAdapter):
    """Adapter for running local models via Ollama HTTP endpoints."""

    def __init__(self, model: str = "llama3"):
        self.model = model

    async def __call__(self, prompt: str) -> str:
        data: Any = await _post_json(
            "http://localhost:11434/api/generate",
            {"model": self.model, "prompt": prompt, "stream": False},
        )
        return str(data.get("response", "")).strip()


# ============================================================
#  Unified Adapter Registry
# ============================================================


class AdapterRegistry:
    """
    Holds all available LLM adapters and exposes them
    through a simple lookup interface.
    """

    def __init__(self):
        self.adapters = {}

    def register(self, name: str, adapter: BaseLLMAdapter):
        self.adapters[name] = adapter

    def get(self, name: str):
        if name not in self.adapters:
            raise ValueError(f"Adapter '{name}' not found")
        return self.adapters[name]

    def list(self):
        return list(self.adapters.keys())


# ============================================================
#  Helper: Bootstrap Common Adapters
# ============================================================


def bootstrap_default_registry(
    openai_key=None,
    anthropic_key=None,
    xai_key=None,
    hf_device: Optional[str] = None,
) -> AdapterRegistry:
    """
    Build a registry of available adapters.
    Only registers backends whose SDKs + keys are present to avoid import errors.
    """

    reg = AdapterRegistry()

    def _register_ollama_models():
        models = []
        env_single = os.getenv("OLLAMA_MODEL")
        if env_single:
            models.append(env_single)
        env_many = os.getenv("OLLAMA_MODELS")
        if env_many:
            models.extend([m.strip() for m in env_many.split(",") if m.strip()])
        if not models:
            models = ["llama3"]
        for m in models:
            name = f"ollama-{m}" if m != "llama3" else "ollama"
            try:
                reg.register(name, OllamaAdapter(model=m))
            except Exception:
                continue

    def _register_hf_models():
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            return
        models = []
        env_many = os.getenv("HF_MODELS")
        if env_many:
            models.extend([m.strip() for m in env_many.split(",") if m.strip()])
        if not models:
            models = ["meta-llama/Llama-3-8b"]
        device = hf_device or os.getenv("HF_DEVICE", "cpu")
        for m in models:
            # derive a short name
            short = m.split("/")[-1].lower().replace(" ", "-")
            name = f"hf-{short}"
            try:
                reg.register(name, LlamaHFAdapter(model_path=m, device=device))
            except Exception:
                continue

    # Cloud Models (guarded)
    if OpenAIClient is not None and openai_key:
        try:
            reg.register("gpt", GPTAdapter(api_key=openai_key))
        except Exception as exc:
            logger.warning("Failed to register GPT adapter: %s", exc)
    if anthropic is not None and anthropic_key:
        try:
            reg.register("claude", ClaudeAdapter(api_key=anthropic_key))
        except Exception as exc:
            logger.warning("Failed to register Claude adapter: %s", exc)
    if xai_key:
        try:
            reg.register("grok", GrokAdapter(api_key=xai_key))
        except Exception as exc:
            logger.warning("Failed to register Grok adapter: %s", exc)

    # Local Models (guarded)
    _register_hf_models()
    _register_ollama_models()

    return reg
