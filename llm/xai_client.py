import json
import os
from typing import Any, Dict, cast

import requests

from llm.base import LLMClient


class XAIClient(LLMClient):
    """
    xAI Grok backend using the OpenAI-compatible chat completion API.
    """

    def __init__(
        self,
        model: str = "grok-beta",
        api_key: str | None = None,
        base_url: str = "https://api.x.ai/v1/chat/completions",
    ):
        self.model = os.getenv("XAI_MODEL", model)
        self.api_key = api_key or os.getenv("XAI_KEY")
        if not self.api_key:
            raise RuntimeError("XAI_KEY is required for the xAI backend.")
        self.base_url = os.getenv("XAI_BASE_URL", base_url)

    def invoke(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
        }

        resp = requests.post(self.base_url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()

        content = resp.json()["choices"][0]["message"]["content"]
        try:
            return cast(Dict[str, Any], json.loads(content))
        except json.JSONDecodeError as exc:
            raise RuntimeError("xAI response was not valid JSON") from exc
