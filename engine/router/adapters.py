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
import json
import requests

# Optional imports (only load if available)
try:
    from openai import OpenAI
except:
    OpenAI = None

try:
    import anthropic
except:
    anthropic = None

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
except:
    AutoTokenizer = AutoModelForCausalLM = torch = None


# ============================================================
#  Base Adapter Pattern
# ============================================================

class BaseLLMAdapter:
    """All adapters must subclass this and implement __call__."""

    def __call__(self, prompt: str) -> str:
        raise NotImplementedError


# ============================================================
#  GPT Adapter (OpenAI)
# ============================================================

class GPTAdapter(BaseLLMAdapter):
    """Adapter for GPT-4.1, GPT-5, etc. via OpenAI API."""

    def __init__(self, model="gpt-4.1", api_key=None):
        if OpenAI is None:
            raise ImportError("OpenAI SDK not installed")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def __call__(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()


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
            model=self.model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()


# ============================================================
#  Grok Adapter (xAI)
# ============================================================

class GrokAdapter(BaseLLMAdapter):
    """Adapter for Grok 1.5 / Grok 3 via xAI’s HTTP API."""

    def __init__(self, model="grok-beta", api_key=None):
        self.model = model
        self.api_key = api_key

    def __call__(self, prompt: str) -> str:
        url = "https://api.x.ai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}]
        }

        resp = requests.post(url, headers=headers, json=payload)
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()


# ============================================================
#  Llama HF Adapter (Transformers)
# ============================================================

class LlamaHFAdapter(BaseLLMAdapter):
    """Adapter for running Llama locally via HuggingFace Transformers."""

    def __init__(self, model_path="meta-llama/Llama-3-8b", device="cpu"):
        if AutoTokenizer is None:
            raise ImportError("Transformers library not installed")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        self.device = device

    def __call__(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=300)
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text.strip()


# ============================================================
#  Ollama Local Adapter
# ============================================================

class OllamaAdapter(BaseLLMAdapter):
    """Adapter for running local models via Ollama HTTP endpoints."""

    def __init__(self, model="llama3"):
        self.model = model

    def __call__(self, prompt: str) -> str:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": self.model, "prompt": prompt}
        )
        data = response.json()
        return data.get("response", "").strip()


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
    xai_key=None
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
        device = os.getenv("HF_DEVICE", "cpu")
        for m in models:
            # derive a short name
            short = m.split("/")[-1].lower().replace(" ", "-")
            name = f"hf-{short}"
            try:
                reg.register(name, LlamaHFAdapter(model_path=m, device=device))
            except Exception:
                continue

    # Cloud Models (guarded)
    if OpenAI is not None and openai_key:
        try:
            reg.register("gpt", GPTAdapter(api_key=openai_key))
        except Exception:
            pass
    if anthropic is not None and anthropic_key:
        try:
            reg.register("claude", ClaudeAdapter(api_key=anthropic_key))
        except Exception:
            pass
    if xai_key:
        try:
            reg.register("grok", GrokAdapter(api_key=xai_key))
        except Exception:
            pass

    # Local Models (guarded)
    _register_hf_models()
    _register_ollama_models()

    return reg
