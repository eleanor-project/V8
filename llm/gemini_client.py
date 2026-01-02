import json
import os
from typing import Any, Dict, cast

import requests

from llm.base import LLMClient


class GeminiClient(LLMClient):
    """
    Google Gemini backend via the REST Generative Language API.
    """

    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        api_key: str | None = None,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
    ):
        self.model = os.getenv("GEMINI_MODEL", model)
        self.api_key = api_key or os.getenv("GEMINI_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise RuntimeError("GEMINI_KEY/GOOGLE_API_KEY is required for the Gemini backend.")
        self.base_url = os.getenv("GEMINI_BASE_URL", base_url)

    def invoke(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "system_instruction": {
                "parts": [{"text": system_prompt}],
            },
            "contents": [
                {"role": "user", "parts": [{"text": user_prompt}]},
            ],
            "generationConfig": {
                "temperature": 0.2,
                "response_mime_type": "application/json",
            },
        }

        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Unexpected Gemini response structure: {data}") from exc

        try:
            return cast(Dict[str, Any], json.loads(text))
        except json.JSONDecodeError as exc:
            raise RuntimeError("Gemini response was not valid JSON") from exc
