"""
Specialized audit event types for different scenarios.

Provides type-safe event classes for common audit scenarios with
appropriate metadata and compliance mappings.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from engine.security.audit.core import (
    AuditEvent,
    AuditLevel,
    ComplianceFramework,
    EventCategory,
)


@dataclass
class AuthenticationEvent(AuditEvent):
    """Authentication-related audit events."""

    event_type: str = "authentication"
    category: EventCategory = field(default=EventCategory.AUTHENTICATION)
    auth_method: Optional[str] = None  # password, api_key, oauth2, saml, certificate
    auth_provider: Optional[str] = None  # local, okta, auth0, azure_ad
    mfa_used: bool = False
    login_success: bool = True
    failure_reason: Optional[str] = None
    source_ip: Optional[str] = None
    device_fingerprint: Optional[str] = None
    geolocation: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        # Authentication failures are always at least WARNING
        if not self.login_success and self.level == AuditLevel.INFO:
            self.level = AuditLevel.WARNING

        # Failed privileged account access is CRITICAL
        if (
            not self.login_success
            and self.details.get("is_admin")
            and self.level != AuditLevel.CRITICAL
        ):
            self.level = AuditLevel.CRITICAL

        # Add compliance frameworks
        if ComplianceFramework.HIPAA not in self.compliance_frameworks:
            self.compliance_frameworks.append(ComplianceFramework.HIPAA)
        if ComplianceFramework.SOC2 not in self.compliance_frameworks:
            self.compliance_frameworks.append(ComplianceFramework.SOC2)


@dataclass
class AuthorizationEvent(AuditEvent):
    """Authorization and access control audit events."""

    event_type: str = "authorization"
    category: EventCategory = field(default=EventCategory.AUTHORIZATION)
    permission_requested: Optional[str] = None
    permission_granted: bool = False
    authorization_policy: Optional[str] = None
    role: Optional[str] = None
    privileges: List[str] = field(default_factory=list)
    denial_reason: Optional[str] = None

    def __post_init__(self) -> None:
        # Authorization denials are WARNING
        if not self.permission_granted and self.level == AuditLevel.INFO:
            self.level = AuditLevel.WARNING

        # Privilege escalation attempts are CRITICAL
        if "admin" in str(self.permission_requested).lower():
            if not self.permission_granted:
                self.level = AuditLevel.CRITICAL
                self.tags.append("privilege_escalation_attempt")

        # Add compliance frameworks
        if ComplianceFramework.SOC2 not in self.compliance_frameworks:
            self.compliance_frameworks.append(ComplianceFramework.SOC2)


@dataclass
class DataAccessEvent(AuditEvent):
    """Data access and manipulation audit events."""

    event_type: str = "data_access"
    category: EventCategory = field(default=EventCategory.DATA_ACCESS)
    operation: Optional[str] = None  # read, write, update, delete, export
    data_type: Optional[str] = None  # pii, phi, financial, credentials
    record_count: Optional[int] = None
    data_volume_bytes: Optional[int] = None
    query_executed: Optional[str] = None
    export_format: Optional[str] = None  # csv, json, pdf
    export_destination: Optional[str] = None

    def __post_init__(self) -> None:
        # PII/PHI access requires elevated classification
        if self.data_type in {"pii", "phi", "credentials"}:
            self.data_classification = "restricted"
            # Add GDPR for PII
            if ComplianceFramework.GDPR not in self.compliance_frameworks:
                self.compliance_frameworks.append(ComplianceFramework.GDPR)
            # Add HIPAA for PHI
            if self.data_type == "phi":
                if ComplianceFramework.HIPAA not in self.compliance_frameworks:
                    self.compliance_frameworks.append(ComplianceFramework.HIPAA)

        # Large data exports are WARNING
        if self.record_count and self.record_count > 1000:
            if self.level == AuditLevel.INFO:
                self.level = AuditLevel.WARNING
            self.tags.append("bulk_data_export")

        # Deletion operations are always WARNING or higher
        if self.operation == "delete" and self.level == AuditLevel.INFO:
            self.level = AuditLevel.WARNING


@dataclass
class ConfigurationChangeEvent(AuditEvent):
    """Configuration and system change audit events."""

    event_type: str = "configuration_change"
    category: EventCategory = field(default=EventCategory.CONFIGURATION)
    config_key: Optional[str] = None
    previous_value: Optional[str] = None
    new_value: Optional[str] = None
    change_reason: Optional[str] = None
    change_approved_by: Optional[str] = None
    rollback_supported: bool = False

    def __post_init__(self) -> None:
        # Configuration changes are at least WARNING
        if self.level == AuditLevel.INFO:
            self.level = AuditLevel.WARNING

        # Security-related config changes are CRITICAL
        security_keywords = {"security", "auth", "credential", "key", "token", "secret"}
        if any(kw in str(self.config_key).lower() for kw in security_keywords):
            self.level = AuditLevel.CRITICAL
            self.tags.append("security_configuration")

        # Add compliance frameworks
        if ComplianceFramework.SOC2 not in self.compliance_frameworks:
            self.compliance_frameworks.append(ComplianceFramework.SOC2)
        if ComplianceFramework.ISO27001 not in self.compliance_frameworks:
            self.compliance_frameworks.append(ComplianceFramework.ISO27001)


@dataclass
class SecurityEvent(AuditEvent):
    """Security-related audit events."""

    event_type: str = "security_event"
    category: EventCategory = field(default=EventCategory.SECURITY)
    level: AuditLevel = field(default=AuditLevel.CRITICAL)
    threat_type: Optional[str] = None  # intrusion, malware, dos, data_breach
    attack_vector: Optional[str] = None
    severity_score: Optional[float] = None  # CVSS score
    mitigated: bool = False
    mitigation_action: Optional[str] = None
    indicators_of_compromise: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Security events are always at least WARNING
        if self.level == AuditLevel.INFO:
            self.level = AuditLevel.WARNING

        # High severity scores are CRITICAL
        if self.severity_score and self.severity_score >= 7.0:
            self.level = AuditLevel.CRITICAL

        # Add all security compliance frameworks
        frameworks = [
            ComplianceFramework.SOC2,
            ComplianceFramework.ISO27001,
            ComplianceFramework.NIST,
        ]
        for fw in frameworks:
            if fw not in self.compliance_frameworks:
                self.compliance_frameworks.append(fw)

        self.tags.append("security_incident")


@dataclass
class GovernanceDecisionEvent(AuditEvent):
    """AI governance decision audit events."""

    event_type: str = "governance_decision"
    category: EventCategory = field(default=EventCategory.GOVERNANCE)
    decision_type: Optional[str] = None  # approval, rejection, escalation
    governance_policy: Optional[str] = None
    risk_score: Optional[float] = None
    ethical_considerations: List[str] = field(default_factory=list)
    human_in_loop_required: bool = False
    decision_rationale: Optional[str] = None
    appeal_available: bool = True

    def __post_init__(self) -> None:
        # High-risk decisions are WARNING
        if self.risk_score and self.risk_score >= 0.7:
            if self.level == AuditLevel.INFO:
                self.level = AuditLevel.WARNING

        # Rejections with no rationale are WARNING
        if self.decision_type == "rejection" and not self.decision_rationale:
            if self.level == AuditLevel.INFO:
                self.level = AuditLevel.WARNING

        # Add AI-specific compliance
        if ComplianceFramework.ISO27001 not in self.compliance_frameworks:
            self.compliance_frameworks.append(ComplianceFramework.ISO27001)

        self.tags.append("ai_governance")


@dataclass
class ModelInferenceEvent(AuditEvent):
    """Model inference and AI operation audit events."""

    event_type: str = "model_inference"
    category: EventCategory = field(default=EventCategory.MODEL_INFERENCE)
    model_id: Optional[str] = None
    model_version: Optional[str] = None
    model_provider: Optional[str] = None  # openai, anthropic, local
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    latency_ms: Optional[float] = None
    cost_usd: Optional[float] = None
    safety_score: Optional[float] = None
    content_filtered: bool = False
    filter_reason: Optional[str] = None

    def __post_init__(self) -> None:
        # Content filtering is WARNING
        if self.content_filtered and self.level == AuditLevel.INFO:
            self.level = AuditLevel.WARNING
            self.tags.append("content_filtered")

        # Low safety scores are CRITICAL
        if self.safety_score and self.safety_score < 0.5:
            self.level = AuditLevel.CRITICAL
            self.tags.append("safety_concern")

        self.tags.append("ai_model")


@dataclass
class SystemEvent(AuditEvent):
    """System-level audit events."""

    event_type: str = "system_event"
    category: EventCategory = field(default=EventCategory.SYSTEM)
    system_component: Optional[str] = None
    event_subtype: Optional[str] = None  # startup, shutdown, error, maintenance
    cpu_usage_percent: Optional[float] = None
    memory_usage_mb: Optional[int] = None
    disk_usage_percent: Optional[float] = None
    network_bytes_sent: Optional[int] = None
    network_bytes_received: Optional[int] = None

    def __post_init__(self) -> None:
        # System errors are ERROR level
        if self.event_subtype == "error" and self.level == AuditLevel.INFO:
            self.level = AuditLevel.ERROR

        # High resource usage is WARNING
        if self.cpu_usage_percent and self.cpu_usage_percent > 90:
            if self.level == AuditLevel.INFO:
                self.level = AuditLevel.WARNING
            self.tags.append("high_cpu_usage")

        if self.memory_usage_mb and self.memory_usage_mb > 8000:  # 8GB
            if self.level == AuditLevel.INFO:
                self.level = AuditLevel.WARNING
            self.tags.append("high_memory_usage")


@dataclass
class AccessEvent(AuditEvent):
    """Generic access audit event (backward compatibility)."""

    event_type: str = "access"
    category: EventCategory = field(default=EventCategory.AUTHORIZATION)
    access_granted: bool = True
    access_type: Optional[str] = None  # read, write, execute, delete

    def __post_init__(self) -> None:
        if not self.access_granted and self.level == AuditLevel.INFO:
            self.level = AuditLevel.WARNING


__all__ = [
    "AccessEvent",
    "AuthenticationEvent",
    "AuthorizationEvent",
    "ConfigurationChangeEvent",
    "DataAccessEvent",
    "GovernanceDecisionEvent",
    "ModelInferenceEvent",
    "SecurityEvent",
    "SystemEvent",
]
