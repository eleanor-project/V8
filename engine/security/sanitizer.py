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
import time
from typing import Any, Dict, List, Optional, Set, Union
from collections import defaultdict

logger = logging.getLogger(__name__)

# ML-based anomaly detection (optional)
try:
    from engine.security.anomaly_detection import get_anomaly_detector, AnomalyScore
    ANOMALY_DETECTION_AVAILABLE = True
except ImportError:
    ANOMALY_DETECTION_AVAILABLE = False
    get_anomaly_detector = None
    AnomalyScore = None

# Sanitization performance metrics
_sanitization_metrics = {
    "total_operations": 0,
    "total_time_ms": 0.0,
    "operations_by_type": defaultdict(int),
    "time_by_type": defaultdict(float),
}


class SecretsSanitizer:
    """
    Sanitize logs and outputs to prevent credential leakage.

    Features:
    - Pattern-based secret detection
    - Recursive dictionary sanitization
    - Configurable sensitive key names
    - Support for custom patterns
    """

    _default_instance: Optional["SecretsSanitizer"] = None

    # Patterns for common secret formats
    DEFAULT_PATTERNS = [
        # OpenAI API keys
        (r"sk-[a-zA-Z0-9]{20,}", "[OPENAI_API_KEY]"),
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
        custom_patterns: Optional[List[tuple[str, str]]] = None,
        additional_sensitive_keys: Optional[Set[str]] = None,
        track_metrics: bool = True,
        enable_anomaly_detection: bool = True,
    ):
        """
        Args:
            custom_patterns: Additional (regex, replacement) patterns
            additional_sensitive_keys: Additional key names to redact
            track_metrics: Track performance metrics for sanitization operations
            enable_anomaly_detection: Enable ML-based anomaly detection for unusual patterns
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
        
        self.track_metrics = track_metrics
        self.enable_anomaly_detection = enable_anomaly_detection and ANOMALY_DETECTION_AVAILABLE
        
        # Initialize anomaly detector if available
        self._anomaly_detector = None
        if self.enable_anomaly_detection and get_anomaly_detector:
            try:
                self._anomaly_detector = get_anomaly_detector(enable_ml=True)
            except Exception as e:
                logger.warning(
                    "anomaly_detector_init_failed",
                    extra={"error": str(e)},
                )
                self._anomaly_detector = None

        logger.debug(
            "secrets_sanitizer_initialized",
            extra={
                "pattern_count": len(self.compiled_patterns),
                "sensitive_key_count": len(self.sensitive_keys),
                "anomaly_detection_enabled": self.enable_anomaly_detection,
            },
        )

        # Bind instance methods to preserve instance-specific behavior.
        self.sanitize_string = self._sanitize_string  # type: ignore[assignment]
        self.sanitize_dict = self._sanitize_dict  # type: ignore[assignment]
        self.sanitize_list = self._sanitize_list  # type: ignore[assignment]
        self.sanitize = self._sanitize  # type: ignore[assignment]

    @classmethod
    def _default(cls) -> "SecretsSanitizer":
        if cls._default_instance is None:
            cls._default_instance = cls()
        return cls._default_instance

    @classmethod
    def sanitize_string(cls, text: str) -> str:
        return cls._default()._sanitize_string(text)

    @classmethod
    def sanitize_dict(
        cls, data: Dict[str, Any], sensitive_keys: Optional[Set[str]] = None
    ) -> Dict[str, Any]:
        return cls._default()._sanitize_dict(data, sensitive_keys=sensitive_keys)

    @classmethod
    def sanitize_list(
        cls, data: List[Any], sensitive_keys: Optional[Set[str]] = None
    ) -> List[Any]:
        return cls._default()._sanitize_list(data, sensitive_keys=sensitive_keys)

    @classmethod
    def sanitize(
        cls, data: Union[str, Dict, List], sensitive_keys: Optional[Set[str]] = None
    ) -> Union[str, Dict, List]:
        return cls._default()._sanitize(data, sensitive_keys=sensitive_keys)

    def _sanitize_string(self, text: str) -> str:
        """
        Remove secrets from string using pattern matching.

        Args:
            text: String to sanitize

        Returns:
            Sanitized string with secrets replaced
        """
        if not isinstance(text, str):
            return text

        start_time = time.time() if self.track_metrics else None
        
        # ML-based anomaly detection
        anomaly_detected = False
        if self._anomaly_detector:
            try:
                anomaly_score = self._anomaly_detector.analyze(text)
                if anomaly_score.score >= 0.7:  # High anomaly threshold
                    anomaly_detected = True
                    logger.warning(
                        "anomaly_detected_in_sanitization",
                        extra={
                            "anomaly_score": round(anomaly_score.score, 3),
                            "confidence": round(anomaly_score.confidence, 3),
                            "pattern_type": anomaly_score.pattern_type,
                            "reasons": anomaly_score.reasons,
                        },
                    )
            except Exception as e:
                logger.debug(f"Anomaly detection failed: {e}")
        
        result = text
        for pattern, replacement in self.compiled_patterns:
            result = pattern.sub(replacement, result)

        if self.track_metrics and start_time:
            duration_ms = (time.time() - start_time) * 1000
            _sanitization_metrics["total_operations"] += 1
            _sanitization_metrics["total_time_ms"] += duration_ms
            _sanitization_metrics["operations_by_type"]["string"] += 1
            _sanitization_metrics["time_by_type"]["string"] += duration_ms

        return result

    def _sanitize_dict(
        self, data: Dict[str, Any], sensitive_keys: Optional[Set[str]] = None
    ) -> Dict[str, Any]:
        """
        Recursively sanitize dictionary, redacting sensitive keys.

        Args:
            data: Dictionary to sanitize

        Returns:
            Sanitized dictionary
        """
        if not isinstance(data, dict):
            return data

        start_time = time.time() if self.track_metrics else None

        sanitized: Dict[str, Any] = {}

        for key, value in data.items():
            # Check if key is sensitive
            if self._is_sensitive_key(key, sensitive_keys=sensitive_keys):
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_dict(value, sensitive_keys=sensitive_keys)
            elif isinstance(value, list):
                sanitized[key] = self._sanitize_list(value, sensitive_keys=sensitive_keys)
            elif isinstance(value, str):
                sanitized[key] = self._sanitize_string(value)
            else:
                sanitized[key] = value

        if self.track_metrics and start_time:
            duration_ms = (time.time() - start_time) * 1000
            _sanitization_metrics["total_operations"] += 1
            _sanitization_metrics["total_time_ms"] += duration_ms
            _sanitization_metrics["operations_by_type"]["dict"] += 1
            _sanitization_metrics["time_by_type"]["dict"] += duration_ms

        return sanitized

    def _sanitize_list(
        self, data: List[Any], sensitive_keys: Optional[Set[str]] = None
    ) -> List[Any]:
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
            self._sanitize_dict(item, sensitive_keys=sensitive_keys)
            if isinstance(item, dict)
            else self._sanitize_list(item, sensitive_keys=sensitive_keys)
            if isinstance(item, list)
            else self._sanitize_string(item)
            if isinstance(item, str)
            else item
            for item in data
        ]

    def _sanitize(
        self, data: Union[str, Dict, List], sensitive_keys: Optional[Set[str]] = None
    ) -> Union[str, Dict, List]:
        """
        Universal sanitization method.

        Args:
            data: Data to sanitize (string, dict, or list)

        Returns:
            Sanitized data of same type
        """
        if isinstance(data, str):
            return self._sanitize_string(data)
        elif isinstance(data, dict):
            return self._sanitize_dict(data, sensitive_keys=sensitive_keys)
        elif isinstance(data, list):
            return self._sanitize_list(data, sensitive_keys=sensitive_keys)
        else:
            return data

    def _is_sensitive_key(self, key: str, sensitive_keys: Optional[Set[str]] = None) -> bool:
        """
        Check if key name indicates sensitive data.

        Args:
            key: Key name to check

        Returns:
            True if key should be redacted
        """
        key_lower = key.lower()
        active_keys = self.sensitive_keys
        if sensitive_keys:
            active_keys = active_keys | {key.lower() for key in sensitive_keys}
        return any(sensitive in key_lower for sensitive in active_keys)

    def add_pattern(self, pattern: str, replacement: str):
        """Add a custom pattern at runtime"""
        compiled = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
        self.compiled_patterns.append((compiled, replacement))
        logger.debug("custom_pattern_added", extra={"pattern": pattern, "replacement": replacement})

    def add_sensitive_key(self, key: str):
        """Add a sensitive key name at runtime"""
        self.sensitive_keys.add(key.lower())
        logger.debug("sensitive_key_added", extra={"key": key})
    
    @staticmethod
    def get_sanitization_metrics() -> Dict[str, Any]:
        """
        Get sanitization performance metrics.
        
        Returns:
            Dictionary with sanitization statistics:
            - total_operations: Total number of sanitization operations
            - total_time_ms: Total time spent on sanitization (ms)
            - avg_time_ms: Average time per operation (ms)
            - operations_by_type: Count of operations by type (string, dict, list)
            - time_by_type: Time spent by operation type (ms)
        """
        total_ops = _sanitization_metrics["total_operations"]
        total_time = _sanitization_metrics["total_time_ms"]
        avg_time = (total_time / total_ops) if total_ops > 0 else 0.0
        
        return {
            "total_operations": total_ops,
            "total_time_ms": round(total_time, 2),
            "avg_time_ms": round(avg_time, 2),
            "operations_by_type": dict(_sanitization_metrics["operations_by_type"]),
            "time_by_type": {k: round(v, 2) for k, v in _sanitization_metrics["time_by_type"].items()},
        }
    
    @staticmethod
    def reset_sanitization_metrics() -> None:
        """Reset sanitization metrics (useful for testing)."""
        global _sanitization_metrics
        _sanitization_metrics = {
            "total_operations": 0,
            "total_time_ms": 0.0,
            "operations_by_type": defaultdict(int),
            "time_by_type": defaultdict(float),
        }


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
    _anomaly_detector = None
    _anomaly_enabled = ANOMALY_DETECTION_AVAILABLE

    @classmethod
    def _get_anomaly_detector(cls):
        if cls._anomaly_detector is None and cls._anomaly_enabled and get_anomaly_detector:
            try:
                cls._anomaly_detector = get_anomaly_detector(enable_ml=True)
            except Exception:
                logger.debug("credential_anomaly_detector_init_failed", exc_info=True)
                cls._anomaly_detector = None
        return cls._anomaly_detector

    @classmethod
    def sanitize_text(cls, text: str) -> str:
        if not isinstance(text, str):
            return text
        detector = cls._get_anomaly_detector()
        if detector:
            try:
                score = detector.analyze(text)
                if score.score >= 0.7:
                    logger.warning(
                        "anomaly_detected_in_credential_sanitization",
                        extra={
                            "anomaly_score": round(score.score, 3),
                            "confidence": round(score.confidence, 3),
                            "pattern_type": score.pattern_type,
                        },
                    )
            except Exception:
                logger.debug("credential_anomaly_detection_failed", exc_info=True)
        result = text
        for pattern, replacement in cls._compiled_patterns:
            result = pattern.sub(replacement, result)
        return result

    @classmethod
    def sanitize_dict(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(data, dict):
            return data

        sanitized: Dict[str, Any] = {}
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

        sanitized_list: List[Any] = []
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
