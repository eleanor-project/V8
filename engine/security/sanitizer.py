"""
ELEANOR V8 — Credential Sanitization

Redacts credentials from logs and evidence records.
"""

import re
from typing import Any, Dict


class CredentialSanitizer:
    """Sanitize text and nested data structures to remove credentials."""

    SECRET_KEY_HINTS = ("key", "token", "password", "secret", "authorization")

    PATTERNS = [
        (re.compile(r"sk-[a-zA-Z0-9]{20,}"), "[OPENAI_API_KEY_REDACTED]"),
        (re.compile(r"AIza[0-9A-Za-z-_]{30,}"), "[GOOGLE_API_KEY_REDACTED]"),
        (re.compile(r"Bearer\\s+[A-Za-z0-9._-]{20,}", re.IGNORECASE), "[BEARER_TOKEN_REDACTED]"),
        (re.compile(r"api[_-]?key\\s*[:=]\\s*\\S+", re.IGNORECASE), "api_key=[REDACTED]"),
        (re.compile(r"token\\s*[:=]\\s*\\S+", re.IGNORECASE), "token=[REDACTED]"),
        (re.compile(r"password\\s*[:=]\\s*\\S+", re.IGNORECASE), "password=[REDACTED]"),
    ]

    @classmethod
    def sanitize_text(cls, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
        sanitized = text
        for pattern, replacement in cls.PATTERNS:
            sanitized = pattern.sub(replacement, sanitized)
        return sanitized

    @classmethod
    def sanitize_value(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return cls.sanitize_dict(value)
        if isinstance(value, list):
            return [cls.sanitize_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls.sanitize_value(item) for item in value)
        if isinstance(value, str):
            return cls.sanitize_text(value)
        return value

    @classmethod
    def sanitize_dict(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in data.items():
            key_str = str(key)
            key_lower = key_str.lower()
            if any(hint in key_lower for hint in cls.SECRET_KEY_HINTS):
                result[key_str] = "[REDACTED]"
            else:
                result[key_str] = cls.sanitize_value(value)
        return result


__all__ = ["CredentialSanitizer"]
