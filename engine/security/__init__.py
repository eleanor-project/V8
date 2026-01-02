"""
ELEANOR V8 — Security Utilities

Secret providers and credential sanitization.
"""

from .secrets import (
    SecretProvider,
    EnvironmentSecretProvider,
    AWSSecretsManagerProvider,
    VaultSecretProvider,
    build_secret_provider,
    build_secret_provider_from_settings,
    get_llm_api_key,
    get_llm_api_key_sync,
)
from .sanitizer import CredentialSanitizer

__all__ = [
    "SecretProvider",
    "EnvironmentSecretProvider",
    "AWSSecretsManagerProvider",
    "VaultSecretProvider",
    "build_secret_provider",
    "build_secret_provider_from_settings",
    "get_llm_api_key",
    "get_llm_api_key_sync",
    "CredentialSanitizer",
]
