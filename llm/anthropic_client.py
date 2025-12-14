import json
import os
from typing import Dict

import requests

from llm.base import LLMClient


class AnthropicClient(LLMClient):
    """
    Anthropic Claude backend (messages API).

    Uses the official HTTP interface and requests a JSON response to keep
    downstream critic parsing deterministic.
    """

    def __init__(
        self,
        model: str = "claude-3-haiku-20240307",
        api_key: str | None = None,
        base_url: str = "https://api.anthropic.com/v1/messages",
        api_version: str = "2023-06-01",
    ):
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_KEY")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_KEY is required for the Anthropic backend.")
        self.base_url = os.getenv("ANTHROPIC_BASE_URL", base_url)
        self.api_version = api_version

    def invoke(self, system_prompt: str, user_prompt: str) -> Dict:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.api_version,
            "content-type": "application/json",
        }

        payload = {
            "model": self.model,
            "max_tokens": 1024,
            "temperature": 0.2,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
        }

        resp = requests.post(self.base_url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        content = data["content"][0]["text"]
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Anthropic response was not valid JSON") from exc
