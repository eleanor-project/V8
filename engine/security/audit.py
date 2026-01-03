"""
Secure Audit Logging for ELEANOR V8

Provides audit logging with automatic sanitization.
"""

import logging
from typing import Any, Dict, Optional

from engine.security.sanitizer import SecretsSanitizer

logger = logging.getLogger(__name__)


class SecureAuditLogger:
    """
    Audit logger with automatic credential sanitization.

    Ensures no secrets leak into audit trails.
    """

    def __init__(self, sanitizer: Optional[SecretsSanitizer] = None):
        """
        Args:
            sanitizer: SecretsSanitizer instance (creates default if None)
        """
        self.sanitizer = sanitizer or SecretsSanitizer()
        logger.info("secure_audit_logger_initialized")

    def log_audit_event(
        self,
        event_type: str,
        details: Dict[str, Any],
        severity: str = "INFO",
        *,
        sanitize: bool = True,
    ):
        """
        Log audit event with sanitization.

        Args:
            event_type: Type of audit event
            details: Event details (will be sanitized)
            severity: Log severity (INFO, WARNING, ERROR)
        """
        # Sanitize details unless explicitly skipped.
        sanitized_details = (
            self.sanitizer.sanitize_dict(details) if sanitize else details
        )

        # Log to audit trail
        log_entry = {
            "event_type": event_type,
            "severity": severity,
            "details": sanitized_details,
        }

        # Use appropriate log level
        if severity == "ERROR":
            logger.error("audit_event", extra=log_entry)
        elif severity == "WARNING":
            logger.warning("audit_event", extra=log_entry)
        else:
            logger.info("audit_event", extra=log_entry)

    def log_access(
        self,
        user: str,
        resource: str,
        action: str,
        allowed: bool,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Log access control event.

        Args:
            user: User identifier
            resource: Resource being accessed
            action: Action attempted
            allowed: Whether access was granted
            metadata: Additional context (will be sanitized)
        """
        details = {
            "user": user,
            "resource": resource,
            "action": action,
            "allowed": allowed,
        }

        if metadata:
            details["metadata"] = metadata

        severity = "INFO" if allowed else "WARNING"
        self.log_audit_event("access_control", details, severity)

    def log_configuration_change(
        self,
        user: str,
        config_key: str,
        old_value: Any,
        new_value: Any,
    ):
        """
        Log configuration change with sanitization.

        Args:
            user: User who made the change
            config_key: Configuration key
            old_value: Previous value (will be sanitized)
            new_value: New value (will be sanitized)
        """
        details = {
            "user": user,
            "config_key": config_key,
            "old_value": old_value,
            "new_value": new_value,
        }

        self.log_audit_event("configuration_change", details, "WARNING")

    def log_secret_access(
        self,
        secret_key: str,
        accessor: str,
        success: bool,
    ):
        """
        Log secret access (never logs actual secret value).

        Args:
            secret_key: Key of secret accessed
            accessor: Who accessed the secret
            success: Whether access succeeded
        """
        details = {
            "secret_key": secret_key,
            "accessor": accessor,
            "success": success,
        }

        severity = "WARNING" if not success else "INFO"
        sanitized_details = self.sanitizer.sanitize_dict(details)
        sanitized_details["secret_key"] = secret_key
        self.log_audit_event("secret_access", sanitized_details, severity, sanitize=False)
