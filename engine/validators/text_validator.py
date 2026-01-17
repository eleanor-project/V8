"""Text validation and sanitization logic."""

from __future__ import annotations

import re
import unicodedata
from typing import List, Tuple

from engine.exceptions import ValidationError as InputValidationError

from .config import ValidationConfig


class TextValidator:
    """Validate and sanitize input text."""

    _CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
    _HOMOGLYPH_MAP = str.maketrans(
        {
            "\u0430": "a",  # Cyrillic small a
            "\u0410": "a",  # Cyrillic capital A
        }
    )

    _XSS_PATTERNS = [
        re.compile(r"<\s*script", re.IGNORECASE),
        re.compile(r"javascript\s*:", re.IGNORECASE),
        re.compile(r"onerror\s*=", re.IGNORECASE),
        re.compile(r"<\s*iframe", re.IGNORECASE),
        re.compile(r"<\s*svg", re.IGNORECASE),
    ]
    _SQL_PATTERNS = [
        re.compile(r"'\s*;\s*drop\s+table", re.IGNORECASE),
        re.compile(r"\bunion\s+select\b", re.IGNORECASE),
        re.compile(r"\bor\s+'?1'?\s*=\s*'?1'?", re.IGNORECASE),
        re.compile(r"'\s*--", re.IGNORECASE),
    ]

    def __init__(self, config: ValidationConfig):
        self.config = config
        self.injection_regexes = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.config.injection_patterns
        ]

    def validate(self, text: str) -> tuple[str, List[str], bool]:
        """
        Validate and sanitize text input.

        Returns:
            (sanitized_text, warnings, was_sanitized)
        """
        warnings: List[str] = []
        sanitized = False

        if not isinstance(text, str):
            raise InputValidationError(
                "Input text must be a string",
                validation_type="type_error",
                field="text",
                context={"received_type": type(text).__name__},
            )

        if not text or not text.strip():
            raise InputValidationError(
                "Input text cannot be empty",
                validation_type="empty_input",
                field="text",
            )

        control_chars = self._CONTROL_CHAR_PATTERN.findall(text)
        if len(control_chars) > 1:
            raise InputValidationError(
                "Control characters detected in input",
                validation_type="control_chars",
                field="text",
                context={"count": len(control_chars)},
            )

        sanitized_text, text_sanitized = self._sanitize_string(text)
        if text_sanitized:
            sanitized = True
            warnings.append("Input text normalized and control characters removed")

        if len(sanitized_text) > self.config.max_text_length:
            raise InputValidationError(
                f"Input text exceeds maximum length ({self.config.max_text_length} characters)",
                validation_type="size_limit",
                field="text",
                context={
                    "length": len(sanitized_text),
                    "max_length": self.config.max_text_length,
                },
            )

        traversal_patterns = (
            "../",
            "..\\",
            "/etc/passwd",
            "\\windows\\system32",
        )
        lowered = sanitized_text.lower()
        if any(pat in lowered for pat in traversal_patterns) or re.match(
            r"^[a-z]:\\\\", sanitized_text, re.IGNORECASE
        ):
            raise InputValidationError(
                "Potential path traversal or filesystem access attempt detected",
                validation_type="path_traversal",
                field="text",
            )

        match_text = self._normalize_for_matching(sanitized_text)

        if self.config.enable_injection_detection:
            for pattern in self.injection_regexes:
                if pattern.search(match_text):
                    if self.config.reject_on_injection:
                        raise InputValidationError(
                            "Potential prompt injection detected",
                            validation_type="prompt_injection",
                            field="text",
                            context={"pattern": pattern.pattern},
                        )
                    warnings.append(
                        f"Potential prompt injection detected: pattern '{pattern.pattern}'"
                    )

        if self.config.enable_malicious_pattern_detection:
            if self._has_excessive_repetition(sanitized_text):
                raise InputValidationError(
                    "Excessive character repetition detected",
                    validation_type="repetition",
                    field="text",
                )

            if any(pattern.search(match_text) for pattern in self._XSS_PATTERNS):
                raise InputValidationError(
                    "XSS pattern detected",
                    validation_type="xss",
                    field="text",
                )

            if any(pattern.search(match_text) for pattern in self._SQL_PATTERNS):
                raise InputValidationError(
                    "SQL injection pattern detected",
                    validation_type="sql_injection",
                    field="text",
                )

        return sanitized_text, warnings, sanitized

    def _sanitize_string(self, value: str) -> Tuple[str, bool]:
        normalized = unicodedata.normalize("NFKC", value)
        sanitized = self._CONTROL_CHAR_PATTERN.sub("", normalized)
        return sanitized, sanitized != value

    def _normalize_for_matching(self, value: str) -> str:
        normalized = unicodedata.normalize("NFKC", value)
        return normalized.translate(self._HOMOGLYPH_MAP).lower()

    @staticmethod
    def _has_excessive_repetition(text: str, threshold: float = 0.3) -> bool:
        """
        Check for excessive character repetition (potential DoS).

        Returns True if more than threshold of characters are repetitions.
        """
        if len(text) < 10:
            return False

        max_run = 1
        current_run = 1

        for i in range(1, len(text)):
            if text[i] == text[i - 1]:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1

        if max_run / len(text) > threshold:
            return True

        tokens = text.split()
        if len(tokens) >= 100:
            unique_ratio = len(set(tokens)) / max(len(tokens), 1)
            if unique_ratio < 0.1:
                return True

        return False


__all__ = ["TextValidator"]
