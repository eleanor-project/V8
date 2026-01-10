"""
Simple audit logger helper.

Provides a dedicated audit logger so sensitive operations can emit structured
events without changing application logic.
"""

import logging
from typing import Dict, Any, Optional

from engine.security.sanitizer import CredentialSanitizer

logger = logging.getLogger(__name__)


class SecureAuditLogger:
    """Audit logger with secret sanitization."""

    def __init__(
        self,
        *,
        sanitizer: Optional[CredentialSanitizer] = None,
        logger_name: str = "audit",
        logger: Optional[logging.Logger] = None,
    ):
        # Defer logger lookup so monkeypatching engine.security.audit.logger works in tests
        self._logger_override = logger
        self.logger_name = logger_name
        self.sanitizer = sanitizer or CredentialSanitizer()

    def _get_logger(self) -> logging.Logger:
        if self._logger_override:
            return self._logger_override
        # Prefer module-level logger so patch(\"engine.security.audit.logger\") is honored
        module_logger = globals().get("logger")
        if module_logger:
            return module_logger
        return logging.getLogger(self.logger_name)

    def _sanitize(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            sanitized = self.sanitizer.sanitize_dict(payload)
            # Preserve secret identifiers for auditability while still sanitizing nested values
            if (
                isinstance(payload, dict)
                and "details" in payload
                and isinstance(payload.get("details"), dict)
            ):
                original_secret = payload["details"].get("secret_key")
                if original_secret is not None and isinstance(sanitized, dict):
                    details = sanitized.get("details")
                    if isinstance(details, dict):
                        details["secret_key"] = original_secret
            return sanitized
        except Exception:
            return payload

    def log(self, event: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Generic audit log entry."""
        sanitized = self._sanitize(extra or {})
        try:
            self._get_logger().info(event, extra=sanitized)
        except Exception:
            pass

    def log_audit_event(self, event: str, details: Optional[Dict[str, Any]] = None) -> None:
        sanitized = self._sanitize({"details": details or {}})
        try:
            self._get_logger().info(event, extra=sanitized)
        except Exception:
            pass

    def log_access(
        self,
        *,
        user: str,
        resource: str,
        action: str,
        allowed: bool,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = {
            "user": user,
            "resource": resource,
            "action": action,
            "allowed": allowed,
            "details": details or {},
        }
        try:
            self._get_logger().info("access_log", extra=self._sanitize({"details": payload}))
        except Exception:
            pass

    def log_secret_access(
        self,
        *,
        secret_key: str,
        accessor: str,
        success: bool,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = {
            "secret_key": secret_key,
            "accessor": accessor,
            "success": success,
            **(details or {}),
        }
        try:
            self._get_logger().info("secret_access_log", extra=self._sanitize({"details": payload}))
        except Exception:
            pass


_audit_logger = SecureAuditLogger()


def audit_log(event: str, extra: Optional[Dict[str, Any]] = None) -> None:
    """
    Emit an audit log event (generic helper).

    Args:
        event: Event name/key.
        extra: Optional structured data to attach.
    """
    _audit_logger.log(event, extra=extra)


__all__ = ["audit_log", "SecureAuditLogger"]
