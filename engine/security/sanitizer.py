"""
Secrets Sanitization for ELEANOR V8

Prevents credential leakage in:
- Logs and debug output
- Error messages and stack traces
- Evidence recordings
- Audit trails
"""

import logging
import re
from typing import Any, Dict, List, Set, Union

logger = logging.getLogger(__name__)


class SecretsSanitizer:
    """
    Sanitize logs and outputs to prevent credential leakage.

    Features:
    - Pattern-based secret detection
    - Recursive dictionary sanitization
    - Configurable sensitive key names
    - Support for custom patterns
    """

    # Patterns for common secret formats
    DEFAULT_PATTERNS = [
        # OpenAI API keys
        (r"sk-[a-zA-Z0-9]{48}", "[OPENAI_API_KEY]"),
        (r"sk-proj-[a-zA-Z0-9_-]{48,}", "[OPENAI_PROJECT_KEY]"),
        # Anthropic API keys
        (r"sk-ant-[a-zA-Z0-9_-]{95,}", "[ANTHROPIC_API_KEY]"),
        # Google API keys
        (r"AIza[0-9A-Za-z\-_]{35}", "[GOOGLE_API_KEY]"),
        # AWS credentials
        (r"AKIA[0-9A-Z]{16}", "[AWS_ACCESS_KEY]"),
        (
            r"(?i)aws[_-]?secret[_-]?access[_-]?key[\s:=]+['\"]?([a-zA-Z0-9/+=]{40})['\"]?",
            "[AWS_SECRET_KEY]",
        ),
        # Bearer tokens
        (r"Bearer [a-zA-Z0-9\-._~+/]+=*", "[BEARER_TOKEN]"),
        # GitHub tokens
        (r"ghp_[a-zA-Z0-9]{36}", "[GITHUB_TOKEN]"),
        (r"gho_[a-zA-Z0-9]{36}", "[GITHUB_OAUTH_TOKEN]"),
        (r"github_pat_[a-zA-Z0-9_]{82}", "[GITHUB_PAT]"),
        # Generic patterns
        (r"(?i)password[\s:=]+['\"]?([^'\"\s,]+)['\"]?", "password=[REDACTED]"),
        (r"(?i)api[_-]?key[\s:=]+['\"]?([^'\"\s,]+)['\"]?", "api_key=[REDACTED]"),
        (r"(?i)token[\s:=]+['\"]?([^'\"\s,]+)['\"]?", "token=[REDACTED]"),
        (r"(?i)secret[\s:=]+['\"]?([^'\"\s,]+)['\"]?", "secret=[REDACTED]"),
        # JWT tokens
        (r"eyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*", "[JWT_TOKEN]"),
        # Private keys
        (
            r"-----BEGIN [A-Z ]+ PRIVATE KEY-----[\s\S]+?-----END [A-Z ]+ PRIVATE KEY-----",
            "[PRIVATE_KEY]",
        ),
    ]

    # Keys that should always be redacted
    SENSITIVE_KEYS = {
        "api_key",
        "apikey",
        "token",
        "password",
        "secret",
        "credential",
        "credentials",
        "auth",
        "authorization",
        "private_key",
        "access_key",
        "secret_key",
        "session_token",
        "bearer",
        "oauth",
    }

    def __init__(
        self,
        custom_patterns: List[tuple[str, str]] = None,
        additional_sensitive_keys: Set[str] = None,
    ):
        """
        Args:
            custom_patterns: Additional (regex, replacement) patterns
            additional_sensitive_keys: Additional key names to redact
        """
        self.patterns = self.DEFAULT_PATTERNS.copy()
        if custom_patterns:
            self.patterns.extend(custom_patterns)

        self.sensitive_keys = self.SENSITIVE_KEYS.copy()
        if additional_sensitive_keys:
            self.sensitive_keys.update(additional_sensitive_keys)

        # Compile regex patterns for performance
        self.compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE | re.MULTILINE), replacement)
            for pattern, replacement in self.patterns
        ]

        logger.debug(
            "secrets_sanitizer_initialized",
            extra={
                "pattern_count": len(self.compiled_patterns),
                "sensitive_key_count": len(self.sensitive_keys),
            },
        )

    def sanitize_string(self, text: str) -> str:
        """
        Remove secrets from string using pattern matching.

        Args:
            text: String to sanitize

        Returns:
            Sanitized string with secrets replaced
        """
        if not isinstance(text, str):
            return text

        result = text
        for pattern, replacement in self.compiled_patterns:
            result = pattern.sub(replacement, result)

        return result

    def sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively sanitize dictionary, redacting sensitive keys.

        Args:
            data: Dictionary to sanitize

        Returns:
            Sanitized dictionary
        """
        if not isinstance(data, dict):
            return data

        sanitized = {}

        for key, value in data.items():
            # Check if key is sensitive
            if self._is_sensitive_key(key):
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, dict):
                sanitized[key] = self.sanitize_dict(value)
            elif isinstance(value, list):
                sanitized[key] = self.sanitize_list(value)
            elif isinstance(value, str):
                sanitized[key] = self.sanitize_string(value)
            else:
                sanitized[key] = value

        return sanitized

    def sanitize_list(self, data: List[Any]) -> List[Any]:
        """
        Recursively sanitize list items.

        Args:
            data: List to sanitize

        Returns:
            Sanitized list
        """
        if not isinstance(data, list):
            return data

        return [
            self.sanitize_dict(item)
            if isinstance(item, dict)
            else self.sanitize_list(item)
            if isinstance(item, list)
            else self.sanitize_string(item)
            if isinstance(item, str)
            else item
            for item in data
        ]

    def sanitize(self, data: Union[str, Dict, List]) -> Union[str, Dict, List]:
        """
        Universal sanitization method.

        Args:
            data: Data to sanitize (string, dict, or list)

        Returns:
            Sanitized data of same type
        """
        if isinstance(data, str):
            return self.sanitize_string(data)
        elif isinstance(data, dict):
            return self.sanitize_dict(data)
        elif isinstance(data, list):
            return self.sanitize_list(data)
        else:
            return data

    def _is_sensitive_key(self, key: str) -> bool:
        """
        Check if key name indicates sensitive data.

        Args:
            key: Key name to check

        Returns:
            True if key should be redacted
        """
        key_lower = key.lower()
        return any(sensitive in key_lower for sensitive in self.sensitive_keys)

    def add_pattern(self, pattern: str, replacement: str):
        """Add a custom pattern at runtime"""
        compiled = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
        self.compiled_patterns.append((compiled, replacement))
        logger.debug("custom_pattern_added", extra={"pattern": pattern, "replacement": replacement})

    def add_sensitive_key(self, key: str):
        """Add a sensitive key name at runtime"""
        self.sensitive_keys.add(key.lower())
        logger.debug("sensitive_key_added", extra={"key": key})


class CredentialSanitizer:
    """
    Lightweight static sanitizer used by logging and validation helpers.

    Uses explicit redaction markers for credential-like tokens.
    """

    DEFAULT_PATTERNS = [
        (r"sk-[a-zA-Z0-9]{20,}", "[OPENAI_API_KEY_REDACTED]"),
        (r"sk-proj-[a-zA-Z0-9_-]{48,}", "[OPENAI_PROJECT_KEY_REDACTED]"),
        (r"sk-ant-[a-zA-Z0-9_-]{95,}", "[ANTHROPIC_API_KEY_REDACTED]"),
        (r"AIza[0-9A-Za-z\\-_]{35}", "[GOOGLE_API_KEY_REDACTED]"),
        (r"AKIA[0-9A-Z]{16}", "[AWS_ACCESS_KEY_REDACTED]"),
        (
            r"(?i)aws[_-]?secret[_-]?access[_-]?key[\\s:=]+['\"]?([a-zA-Z0-9/+=]{40})['\"]?",
            "[AWS_SECRET_KEY_REDACTED]",
        ),
        (r"Bearer [a-zA-Z0-9._~+/=-]*", "[BEARER_TOKEN_REDACTED]"),
        (r"ghp_[a-zA-Z0-9]{36}", "[GITHUB_TOKEN_REDACTED]"),
        (r"gho_[a-zA-Z0-9]{36}", "[GITHUB_OAUTH_TOKEN_REDACTED]"),
        (r"github_pat_[a-zA-Z0-9_]{82}", "[GITHUB_PAT_REDACTED]"),
        (r"(?i)password[\\s:=]+['\"]?([^'\"\\s,]+)['\"]?", "password=[REDACTED]"),
        (r"(?i)api[_-]?key[\\s:=]+['\"]?([^'\"\\s,]+)['\"]?", "api_key=[REDACTED]"),
        (r"(?i)token[\\s:=]+['\"]?([^'\"\\s,]+)['\"]?", "token=[REDACTED]"),
        (r"(?i)secret[\\s:=]+['\"]?([^'\"\\s,]+)['\"]?", "secret=[REDACTED]"),
        (r"eyJ[a-zA-Z0-9_-]*\\.eyJ[a-zA-Z0-9_-]*\\.[a-zA-Z0-9_-]*", "[JWT_TOKEN_REDACTED]"),
        (
            r"-----BEGIN [A-Z ]+ PRIVATE KEY-----[\\s\\S]+?-----END [A-Z ]+ PRIVATE KEY-----",
            "[PRIVATE_KEY_REDACTED]",
        ),
    ]

    SENSITIVE_KEYS = SecretsSanitizer.SENSITIVE_KEYS

    _compiled_patterns = [
        (re.compile(pattern, re.IGNORECASE | re.MULTILINE), replacement)
        for pattern, replacement in DEFAULT_PATTERNS
    ]

    @classmethod
    def sanitize_text(cls, text: str) -> str:
        if not isinstance(text, str):
            return text
        result = text
        for pattern, replacement in cls._compiled_patterns:
            result = pattern.sub(replacement, result)
        return result

    @classmethod
    def sanitize_dict(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(data, dict):
            return data

        sanitized = {}
        for key, value in data.items():
            key_lower = key.lower()
            if any(sensitive in key_lower for sensitive in cls.SENSITIVE_KEYS):
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, dict):
                sanitized[key] = cls.sanitize_dict(value)
            elif isinstance(value, list):
                sanitized[key] = cls.sanitize_list(value)
            elif isinstance(value, str):
                sanitized[key] = cls.sanitize_text(value)
            else:
                sanitized[key] = value
        return sanitized

    @classmethod
    def sanitize_list(cls, data: List[Any]) -> List[Any]:
        if not isinstance(data, list):
            return data

        sanitized_list = []
        for item in data:
            if isinstance(item, dict):
                sanitized_list.append(cls.sanitize_dict(item))
            elif isinstance(item, list):
                sanitized_list.append(cls.sanitize_list(item))
            elif isinstance(item, str):
                sanitized_list.append(cls.sanitize_text(item))
            else:
                sanitized_list.append(item)
        return sanitized_list

    @classmethod
    def sanitize_value(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return cls.sanitize_dict(value)
        if isinstance(value, list):
            return cls.sanitize_list(value)
        if isinstance(value, str):
            return cls.sanitize_text(value)
        return value
