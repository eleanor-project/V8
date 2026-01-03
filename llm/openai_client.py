import json
import os
from typing import Any, Dict, cast
import requests
from llm.base import LLMClient


class OpenAIClient(LLMClient):
    """
    OpenAI-compatible LLM backend for ELEANOR.

    This client:
    - obeys Eleanor's abstraction contract
    - returns structured JSON only
    - does NOT perform aggregation or moral decisions
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1/chat/completions",
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

        self.base_url = base_url

    def invoke(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "temperature": 0.2,  # low creativity, high discipline
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
        }

        response = requests.post(
            self.base_url,
            headers=headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()

        content = response.json()["choices"][0]["message"]["content"]

        try:
            return cast(Dict[str, Any], json.loads(content))
        except json.JSONDecodeError as e:
            raise RuntimeError("LLM did not return valid JSON required by critic schema") from e
