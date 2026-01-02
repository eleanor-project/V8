"""
Security Module for ELEANOR V8

Provides:
- Secrets management (AWS, Vault, Environment)
- Credential sanitization
- Audit trail protection
"""

from engine.security.secrets import (
    SecretsProvider,
    EnvironmentSecretsProvider,
    AWSSecretsProvider,
    VaultSecretsProvider,
    auto_detect_secrets_provider,
)
from engine.security.sanitizer import SecretsSanitizer

__all__ = [
    "SecretsProvider",
    "EnvironmentSecretsProvider",
    "AWSSecretsProvider",
    "VaultSecretsProvider",
    "auto_detect_secrets_provider",
    "SecretsSanitizer",
]
