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
    EnvironmentSecretProvider,
    AWSSecretsProvider,
    VaultSecretsProvider,
    auto_detect_secrets_provider,
    build_secret_provider_from_settings,
    get_llm_api_key,
    get_llm_api_key_sync,
)
from engine.security.sanitizer import SecretsSanitizer
from engine.security.audit import SecureAuditLogger

__all__ = [
    "SecretsProvider",
    "EnvironmentSecretsProvider",
    "EnvironmentSecretProvider",
    "AWSSecretsProvider",
    "VaultSecretsProvider",
    "auto_detect_secrets_provider",
    "build_secret_provider_from_settings",
    "get_llm_api_key",
    "get_llm_api_key_sync",
    "SecretsSanitizer",
    "SecureAuditLogger",
]
