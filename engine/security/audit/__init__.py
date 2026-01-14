"""
Enterprise-Grade Audit System for ELEANOR AI Governance Platform.

Provides comprehensive audit logging with:
- Immutable audit trails with cryptographic verification
- SIEM integration (Splunk, Elastic, Datadog, Azure Sentinel)
- Compliance framework support (GDPR, HIPAA, SOC 2, ISO 27001)
- Real-time alerting for critical security events
- Audit analytics with ML-powered anomaly detection
- Encryption at rest and in transit
- Tamper detection with digital signatures
- Multi-tenancy support
- Retention policies with automated archival
- Query interface for audit investigations
"""

from engine.security.audit.core import (
    AuditEvent,
    AuditLevel,
    AuditLogger,
    ComplianceFramework,
    get_audit_logger,
)
from engine.security.audit.events import (
    AccessEvent,
    AuthenticationEvent,
    AuthorizationEvent,
    ConfigurationChangeEvent,
    DataAccessEvent,
    GovernanceDecisionEvent,
    ModelInferenceEvent,
    SecurityEvent,
    SystemEvent,
)
from engine.security.audit.query import AuditQuery, AuditQueryBuilder

__all__ = [
    # Core
    "AuditEvent",
    "AuditLevel",
    "AuditLogger",
    "ComplianceFramework",
    "get_audit_logger",
    # Events
    "AccessEvent",
    "AuthenticationEvent",
    "AuthorizationEvent",
    "ConfigurationChangeEvent",
    "DataAccessEvent",
    "GovernanceDecisionEvent",
    "ModelInferenceEvent",
    "SecurityEvent",
    "SystemEvent",
    # Query
    "AuditQuery",
    "AuditQueryBuilder",
]
