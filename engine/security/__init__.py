"""
Security module for ELEANOR V8

Provides:
- Secrets management with multiple providers
- Credential sanitization and redaction
- Secure audit logging
- Security configuration validation
"""

from engine.security.secrets import (
    SecretsProvider,
    EnvironmentSecretsProvider,
    AWSSecretsProvider,
    VaultSecretsProvider,
)
from engine.security.sanitizer import SecretsSanitizer
from engine.security.audit import SecureAuditLogger

__all__ = [
    "SecretsProvider",
    "EnvironmentSecretsProvider",
    "AWSSecretsProvider",
    "VaultSecretsProvider",
    "SecretsSanitizer",
    "SecureAuditLogger",
]
