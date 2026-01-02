import json
import os
from typing import Any, Dict, cast

import requests

from llm.base import LLMClient


class LlamaClient(LLMClient):
    """
    Local Llama/Ollama backend.

    Assumes an Ollama-compatible endpoint; defaults to localhost.
    """

    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
    ):
        self.model = model or os.getenv("OLLAMA_MODEL") or os.getenv("LLAMA_MODEL")
        if not self.model:
            raise RuntimeError("OLLAMA_MODEL or LLAMA_MODEL is required for the Llama backend.")
        self.host: str = host or os.getenv("OLLAMA_HOST") or "http://localhost:11434"
        self.url = f"{self.host.rstrip('/')}/api/generate"

    def invoke(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        prompt = f"{system_prompt}\n\n{user_prompt}"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "options": {"temperature": 0.2},
            "stream": False,
        }

        resp = requests.post(self.url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("response") or ""

        try:
            return cast(Dict[str, Any], json.loads(content))
        except json.JSONDecodeError as exc:
            raise RuntimeError("Llama/Ollama response was not valid JSON") from exc
