"""
Core audit logging functionality with enterprise features.

Provides the main AuditLogger class and foundational audit infrastructure.
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from engine.security.ledger import LedgerRecord, get_ledger_writer
from engine.security.sanitizer import CredentialSanitizer

logger = logging.getLogger(__name__)


class AuditLevel(str, Enum):
    """Audit event severity levels."""

    DEBUG = "debug"  # Detailed diagnostic information
    INFO = "info"  # Informational events
    WARNING = "warning"  # Warning conditions
    ERROR = "error"  # Error conditions
    CRITICAL = "critical"  # Critical security events requiring immediate attention


class ComplianceFramework(str, Enum):
    """Supported compliance frameworks."""

    GDPR = "gdpr"  # General Data Protection Regulation
    HIPAA = "hipaa"  # Health Insurance Portability and Accountability Act
    SOC2 = "soc2"  # Service Organization Control 2
    ISO27001 = "iso27001"  # ISO/IEC 27001 Information Security Management
    PCI_DSS = "pci_dss"  # Payment Card Industry Data Security Standard
    NIST = "nist"  # NIST Cybersecurity Framework
    CCPA = "ccpa"  # California Consumer Privacy Act
    FedRAMP = "fedramp"  # Federal Risk and Authorization Management Program


class EventCategory(str, Enum):
    """High-level audit event categories."""

    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    CONFIGURATION = "configuration"
    SECURITY = "security"
    GOVERNANCE = "governance"
    MODEL_INFERENCE = "model_inference"
    SYSTEM = "system"
    COMPLIANCE = "compliance"


@dataclass
class AuditEvent:
    """Base audit event with comprehensive metadata."""

    # Core identification
    event_id: str = field(default_factory=lambda: uuid4().hex)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )
    event_type: str = "generic"
    category: EventCategory = EventCategory.SYSTEM
    level: AuditLevel = AuditLevel.INFO

    # Actor information
    actor_id: Optional[str] = None
    actor_type: Optional[str] = None  # user, service, system, api_key
    actor_ip: Optional[str] = None
    actor_user_agent: Optional[str] = None

    # Request context
    trace_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    correlation_id: Optional[str] = None

    # Resource information
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    resource_name: Optional[str] = None

    # Action details
    action: Optional[str] = None
    outcome: str = "success"  # success, failure, partial, unknown
    error_code: Optional[str] = None
    error_message: Optional[str] = None

    # Compliance and governance
    compliance_frameworks: List[ComplianceFramework] = field(default_factory=list)
    data_classification: Optional[str] = None  # public, internal, confidential, restricted
    retention_period_days: Optional[int] = None

    # Multi-tenancy
    tenant_id: Optional[str] = None
    organization_id: Optional[str] = None

    # System context
    hostname: str = field(default_factory=platform.node)
    service_name: str = "eleanor-v8"
    service_version: Optional[str] = None
    environment: str = field(default_factory=lambda: os.getenv("ENVIRONMENT", "production"))

    # Additional structured data
    details: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    # Performance metrics
    duration_ms: Optional[float] = None
    bytes_processed: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary representation."""
        data = asdict(self)
        # Convert enums to strings
        data["level"] = self.level.value
        data["category"] = self.category.value
        data["compliance_frameworks"] = [
            f.value for f in self.compliance_frameworks
        ]
        return data

    def is_critical(self) -> bool:
        """Check if event is critical severity."""
        return self.level == AuditLevel.CRITICAL

    def requires_alert(self) -> bool:
        """Check if event requires real-time alerting."""
        return self.level in {AuditLevel.CRITICAL, AuditLevel.ERROR}

    def is_pii_event(self) -> bool:
        """Check if event involves PII/sensitive data."""
        return self.data_classification in {"confidential", "restricted"}


class AuditLogger:
    """
    Enterprise-grade audit logger with comprehensive features.

    Features:
    - Immutable audit trails via cryptographic ledger
    - Credential sanitization
    - SIEM integration
    - Compliance framework support
    - Real-time alerting
    - Multi-tenancy
    - Retention policies
    """

    def __init__(
        self,
        *,
        service_name: str = "eleanor-v8",
        service_version: Optional[str] = None,
        tenant_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        sanitizer: Optional[CredentialSanitizer] = None,
        enable_siem: bool = True,
        enable_alerts: bool = True,
        default_retention_days: int = 2555,  # ~7 years for compliance
    ):
        self.service_name = service_name
        self.service_version = service_version or os.getenv("SERVICE_VERSION", "unknown")
        self.tenant_id = tenant_id
        self.organization_id = organization_id
        self.sanitizer = sanitizer or CredentialSanitizer()
        self.enable_siem = enable_siem
        self.enable_alerts = enable_alerts
        self.default_retention_days = default_retention_days

        # Get ledger writer for immutable storage
        try:
            self._ledger_writer = get_ledger_writer()
        except Exception as exc:
            logger.warning("audit_ledger_init_failed", extra={"error": str(exc)})
            self._ledger_writer = None

        # SIEM integration (lazy-loaded)
        self._siem_exporters: List[Any] = []
        self._alert_manager: Optional[Any] = None

        # Performance tracking
        self._event_count = 0
        self._failed_writes = 0

    def log_event(self, event: AuditEvent) -> Optional[str]:
        """
        Log an audit event to all configured destinations.

        Args:
            event: The audit event to log

        Returns:
            The event ID if successful, None otherwise
        """
        try:
            # Enrich event with logger context
            if not event.service_name:
                event.service_name = self.service_name
            if not event.service_version:
                event.service_version = self.service_version
            if not event.tenant_id and self.tenant_id:
                event.tenant_id = self.tenant_id
            if not event.organization_id and self.organization_id:
                event.organization_id = self.organization_id

            # Apply retention policy if not set
            if event.retention_period_days is None:
                event.retention_period_days = self._get_retention_period(event)

            # Sanitize sensitive data
            event_dict = event.to_dict()
            sanitized = self._sanitize_event(event_dict)

            # Write to immutable ledger
            ledger_record = self._write_to_ledger(event.event_type, sanitized)

            # Write to standard logging
            self._write_to_log(event, sanitized)

            # Export to SIEM systems
            if self.enable_siem:
                self._export_to_siem(event, sanitized)

            # Send alerts for critical events
            if self.enable_alerts and event.requires_alert():
                self._send_alert(event, sanitized)

            self._event_count += 1
            return event.event_id

        except Exception as exc:
            self._failed_writes += 1
            logger.error(
                "audit_log_failed",
                extra={
                    "error": str(exc),
                    "event_type": event.event_type,
                    "event_id": event.event_id,
                    "traceback": traceback.format_exc(),
                },
            )
            return None

    async def log_event_async(self, event: AuditEvent) -> Optional[str]:
        """Async version of log_event."""
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.log_event, event)

    def _sanitize_event(self, event_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize sensitive data in event."""
        try:
            return self.sanitizer.sanitize_dict(event_dict)
        except Exception as exc:
            logger.warning("audit_sanitization_failed", extra={"error": str(exc)})
            return event_dict

    def _write_to_ledger(self, event_type: str, event_data: Dict[str, Any]) -> Optional[LedgerRecord]:
        """Write event to immutable ledger."""
        if not self._ledger_writer:
            return None
        try:
            return self._ledger_writer.append(event_type, event_data)
        except Exception as exc:
            logger.warning("audit_ledger_write_failed", extra={"error": str(exc)})
            return None

    def _write_to_log(self, event: AuditEvent, sanitized: Dict[str, Any]) -> None:
        """Write event to standard logging system."""
        log_method = getattr(logger, event.level.value, logger.info)
        log_method(event.event_type, extra={"audit_event": sanitized})

    def _export_to_siem(self, event: AuditEvent, sanitized: Dict[str, Any]) -> None:
        """Export event to configured SIEM systems."""
        # Lazy-load SIEM exporters
        if not self._siem_exporters:
            self._init_siem_exporters()

        for exporter in self._siem_exporters:
            try:
                exporter.export(event, sanitized)
            except Exception as exc:
                logger.debug(
                    "siem_export_failed",
                    extra={"siem": exporter.name, "error": str(exc)},
                )

    def _send_alert(self, event: AuditEvent, sanitized: Dict[str, Any]) -> None:
        """Send real-time alert for critical events."""
        if not self._alert_manager:
            self._init_alert_manager()

        if self._alert_manager:
            try:
                self._alert_manager.send_alert(event, sanitized)
            except Exception as exc:
                logger.warning("alert_send_failed", extra={"error": str(exc)})

    def _init_siem_exporters(self) -> None:
        """Initialize SIEM exporters based on configuration."""
        from engine.security.audit.siem import get_siem_exporters

        try:
            self._siem_exporters = get_siem_exporters()
        except Exception as exc:
            logger.warning("siem_init_failed", extra={"error": str(exc)})
            self._siem_exporters = []

    def _init_alert_manager(self) -> None:
        """Initialize alert manager."""
        from engine.security.audit.alerts import get_alert_manager

        try:
            self._alert_manager = get_alert_manager()
        except Exception as exc:
            logger.warning("alert_manager_init_failed", extra={"error": str(exc)})
            self._alert_manager = None

    def _get_retention_period(self, event: AuditEvent) -> int:
        """Determine retention period based on compliance frameworks."""
        # Compliance-driven retention periods
        retention_map = {
            ComplianceFramework.HIPAA: 2555,  # 7 years
            ComplianceFramework.SOC2: 2555,  # 7 years
            ComplianceFramework.ISO27001: 1825,  # 5 years
            ComplianceFramework.PCI_DSS: 365,  # 1 year minimum
            ComplianceFramework.GDPR: 2190,  # 6 years (varies by jurisdiction)
            ComplianceFramework.FedRAMP: 2555,  # 7 years
        }

        # Use maximum retention period from applicable frameworks
        if event.compliance_frameworks:
            periods = [
                retention_map.get(fw, self.default_retention_days)
                for fw in event.compliance_frameworks
            ]
            return max(periods)

        # Critical events get longer retention
        if event.level == AuditLevel.CRITICAL:
            return 3650  # 10 years

        return self.default_retention_days

    def get_metrics(self) -> Dict[str, Any]:
        """Get audit logger metrics."""
        return {
            "events_logged": self._event_count,
            "failed_writes": self._failed_writes,
            "success_rate": (
                (self._event_count - self._failed_writes) / self._event_count
                if self._event_count > 0
                else 1.0
            ),
            "siem_exporters": len(self._siem_exporters),
            "alerts_enabled": self.enable_alerts,
        }


# Global audit logger instance
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger(
    service_name: str = "eleanor-v8",
    **kwargs: Any,
) -> AuditLogger:
    """Get or create the global audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger(service_name=service_name, **kwargs)
    return _audit_logger


__all__ = [
    "AuditEvent",
    "AuditLevel",
    "AuditLogger",
    "ComplianceFramework",
    "EventCategory",
    "get_audit_logger",
]
