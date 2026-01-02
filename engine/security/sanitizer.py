"""
Secrets Sanitizer for ELEANOR V8

Prevents credential leakage in:
- Log messages
- Evidence recordings
- Error messages
- Audit trails
- Debug output
"""

import logging
import re
from typing import Any, Dict, List, Set

logger = logging.getLogger(__name__)


class SecretsSanitizer:
    """
    Sanitize logs and outputs to prevent credential leakage.

    Features:
    - Pattern-based secret detection
    - Recursive dictionary sanitization
    - Key-based sensitive field detection
    - Configurable redaction
    """

    # Regex patterns for common API keys and tokens
    SECRET_PATTERNS = [
        # OpenAI
        (r"sk-[a-zA-Z0-9]{48}", "[OPENAI_API_KEY]"),
        (r"sk-proj-[a-zA-Z0-9_-]{48,}", "[OPENAI_PROJECT_KEY]"),
        
        # Anthropic
        (r"sk-ant-[a-zA-Z0-9_-]{95,}", "[ANTHROPIC_API_KEY]"),
        
        # Google
        (r"AIza[0-9A-Za-z\-_]{35}", "[GOOGLE_API_KEY]"),
        
        # AWS
        (r"AKIA[0-9A-Z]{16}", "[AWS_ACCESS_KEY]"),
        (r"aws_secret_access_key\s*=\s*[A-Za-z0-9/+=]{40}", "[AWS_SECRET_KEY]"),
        
        # GitHub
        (r"ghp_[a-zA-Z0-9]{36}", "[GITHUB_TOKEN]"),
        (r"gho_[a-zA-Z0-9]{36}", "[GITHUB_OAUTH_TOKEN]"),
        
        # OAuth/Bearer tokens
        (r"Bearer [a-zA-Z0-9\-._~+/]+=*", "Bearer [REDACTED]"),
        
        # Generic patterns
        (r'password["\']?\s*[:=]\s*["\']?([^"\',\s]+)', 'password="[REDACTED]"'),
        (r'api[_-]?key["\']?\s*[:=]\s*["\']?([^"\',\s]+)', 'api_key="[REDACTED]"'),
        (r'secret["\']?\s*[:=]\s*["\']?([^"\',\s]+)', 'secret="[REDACTED]"'),
        (r'token["\']?\s*[:=]\s*["\']?([^"\',\s]+)', 'token="[REDACTED]"'),
    ]

    # Field names that should always be redacted
    SENSITIVE_KEYS = {
        "api_key",
        "apikey",
        "api-key",
        "token",
        "access_token",
        "refresh_token",
        "bearer",
        "password",
        "passwd",
        "pwd",
        "secret",
        "secret_key",
        "private_key",
        "credential",
        "credentials",
        "auth",
        "authorization",
        "session",
        "cookie",
    }

    @classmethod
    def sanitize_string(cls, text: str) -> str:
        """
        Remove secrets from string using pattern matching.

        Args:
            text: Input string potentially containing secrets

        Returns:
            Sanitized string with secrets replaced
        """
        if not isinstance(text, str):
            return text

        sanitized = text
        for pattern, replacement in cls.SECRET_PATTERNS:
            sanitized = re.sub(
                pattern,
                replacement,
                sanitized,
                flags=re.IGNORECASE,
            )

        return sanitized

    @classmethod
    def sanitize_dict(
        cls,
        data: Dict[str, Any],
        sensitive_keys: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """
        Recursively sanitize dictionary, redacting sensitive fields.

        Args:
            data: Dictionary to sanitize
            sensitive_keys: Additional sensitive key names

        Returns:
            Sanitized dictionary
        """
        if not isinstance(data, dict):
            return data

        # Merge default and custom sensitive keys
        all_sensitive_keys = cls.SENSITIVE_KEYS.copy()
        if sensitive_keys:
            all_sensitive_keys.update(
                k.lower() for k in sensitive_keys
            )

        sanitized = {}

        for key, value in data.items():
            # Check if key is sensitive
            if any(
                sensitive in key.lower()
                for sensitive in all_sensitive_keys
            ):
                sanitized[key] = "[REDACTED]"
                continue

            # Recursively sanitize based on value type
            if isinstance(value, dict):
                sanitized[key] = cls.sanitize_dict(value, sensitive_keys)
            elif isinstance(value, list):
                sanitized[key] = cls.sanitize_list(value, sensitive_keys)
            elif isinstance(value, str):
                sanitized[key] = cls.sanitize_string(value)
            else:
                sanitized[key] = value

        return sanitized

    @classmethod
    def sanitize_list(
        cls,
        data: List[Any],
        sensitive_keys: Optional[Set[str]] = None,
    ) -> List[Any]:
        """
        Recursively sanitize list elements.

        Args:
            data: List to sanitize
            sensitive_keys: Additional sensitive key names

        Returns:
            Sanitized list
        """
        if not isinstance(data, list):
            return data

        sanitized = []
        for item in data:
            if isinstance(item, dict):
                sanitized.append(cls.sanitize_dict(item, sensitive_keys))
            elif isinstance(item, list):
                sanitized.append(cls.sanitize_list(item, sensitive_keys))
            elif isinstance(item, str):
                sanitized.append(cls.sanitize_string(item))
            else:
                sanitized.append(item)

        return sanitized

    @classmethod
    def sanitize(cls, data: Any, sensitive_keys: Optional[Set[str]] = None) -> Any:
        """
        Sanitize any data type.

        Args:
            data: Data to sanitize (dict, list, str, or other)
            sensitive_keys: Additional sensitive key names

        Returns:
            Sanitized data
        """
        if isinstance(data, dict):
            return cls.sanitize_dict(data, sensitive_keys)
        elif isinstance(data, list):
            return cls.sanitize_list(data, sensitive_keys)
        elif isinstance(data, str):
            return cls.sanitize_string(data)
        else:
            return data
