"""
ELEANOR V8 — Secret Providers

Provides a pluggable interface for secure secret retrieval with caching.
"""

import asyncio
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_SECRET_KEY_MAP = {
    "openai": "openai-api-key",
    "anthropic": "anthropic-api-key",
    "cohere": "cohere-api-key",
    "xai": "xai-api-key",
    "grok": "xai-api-key",
    "gemini": "gemini-api-key",
}

ENV_SECRET_ALIASES = {
    "openai-api-key": ["OPENAI_API_KEY", "OPENAI_KEY"],
    "anthropic-api-key": ["ANTHROPIC_API_KEY", "ANTHROPIC_KEY"],
    "cohere-api-key": ["COHERE_API_KEY", "COHERE_KEY"],
    "xai-api-key": ["XAI_API_KEY", "XAI_KEY"],
    "gemini-api-key": ["GEMINI_API_KEY", "GEMINI_KEY"],
}


@dataclass
class SecretCacheEntry:
    value: str
    expires_at: float


class SecretProvider(ABC):
    """Abstract interface for secret providers."""

    def __init__(self, *, cache_ttl: int = 300) -> None:
        self.cache_ttl = cache_ttl
        self._cache: dict[str, SecretCacheEntry] = {}

    async def get_secret(self, key: str) -> Optional[str]:
        now = time.time()
        cached = self._cache.get(key)
        if cached and cached.expires_at > now:
            return cached.value

        value = await self._fetch_secret(key)
        if value is not None:
            self._cache[key] = SecretCacheEntry(
                value=value,
                expires_at=now + self.cache_ttl,
            )
        return value

    async def refresh_secrets(self) -> None:
        self._cache.clear()

    @abstractmethod
    async def _fetch_secret(self, key: str) -> Optional[str]:
        raise NotImplementedError


class EnvironmentSecretProvider(SecretProvider):
    """Fallback provider using environment variables (dev only)."""

    async def _fetch_secret(self, key: str) -> Optional[str]:
        value = os.getenv(key)
        if value:
            return value

        key_tail = key.split("/")[-1]
        alt = key_tail.upper().replace("-", "_")
        value = os.getenv(alt)
        if value:
            return value

        aliases = ENV_SECRET_ALIASES.get(key_tail)
        if aliases:
            for alias in aliases:
                value = os.getenv(alias)
                if value:
                    return value
        return None


class AWSSecretsManagerProvider(SecretProvider):
    """AWS Secrets Manager integration."""

    def __init__(
        self,
        *,
        region: str = "us-west-2",
        secret_prefix: Optional[str] = None,
        cache_ttl: int = 300,
    ) -> None:
        super().__init__(cache_ttl=cache_ttl)
        try:
            import boto3  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("boto3 is required for AWS Secrets Manager") from exc
        self.client = boto3.client("secretsmanager", region_name=region)
        self.secret_prefix = secret_prefix or ""

    def _normalize_key(self, key: str) -> str:
        if not self.secret_prefix:
            return key
        if key.startswith(self.secret_prefix):
            return key
        return f"{self.secret_prefix.rstrip('/')}/{key}"

    async def _fetch_secret(self, key: str) -> Optional[str]:
        secret_id = self._normalize_key(key)
        try:
            response = await asyncio.to_thread(self.client.get_secret_value, SecretId=secret_id)
        except Exception as exc:
            logger.error("Failed to retrieve AWS secret", extra={"key": secret_id, "error": str(exc)})
            return None

        secret = response.get("SecretString")
        if secret is None:
            secret_bytes = response.get("SecretBinary")
            if isinstance(secret_bytes, (bytes, bytearray)):
                secret = secret_bytes.decode("utf-8", errors="ignore")
        return secret


class VaultSecretProvider(SecretProvider):
    """HashiCorp Vault integration."""

    def __init__(
        self,
        *,
        address: str,
        token: str,
        mount_path: str = "secret/eleanor",
        cache_ttl: int = 300,
    ) -> None:
        super().__init__(cache_ttl=cache_ttl)
        try:
            import hvac  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("hvac is required for Vault secrets") from exc
        self.client = hvac.Client(url=address, token=token)
        self.mount_path = mount_path.strip("/")

    def _normalize_key(self, key: str) -> str:
        if key.startswith(self.mount_path):
            return key
        return f"{self.mount_path}/{key}"

    async def _fetch_secret(self, key: str) -> Optional[str]:
        secret_path = self._normalize_key(key)
        try:
            response = await asyncio.to_thread(
                self.client.secrets.kv.v2.read_secret_version,
                path=secret_path,
            )
        except Exception as exc:
            logger.error("Failed to retrieve Vault secret", extra={"key": secret_path, "error": str(exc)})
            return None

        data = response.get("data", {}).get("data", {})
        if "value" in data:
            return data.get("value")
        if len(data) == 1:
            return next(iter(data.values()))
        return None


def build_secret_key(provider: str, *, secret_prefix: Optional[str] = None) -> str:
    base = DEFAULT_SECRET_KEY_MAP.get(provider.lower(), provider.lower())
    if not secret_prefix:
        return base
    if base.startswith(secret_prefix):
        return base
    return f"{secret_prefix.rstrip('/')}/{base}"


async def get_llm_api_key(
    provider: str,
    secret_provider: SecretProvider,
    *,
    secret_prefix: Optional[str] = None,
) -> Optional[str]:
    secret_key = build_secret_key(provider, secret_prefix=secret_prefix)
    return await secret_provider.get_secret(secret_key)


def get_llm_api_key_sync(
    provider: str,
    secret_provider: SecretProvider,
    *,
    secret_prefix: Optional[str] = None,
) -> Optional[str]:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(get_llm_api_key(provider, secret_provider, secret_prefix=secret_prefix))

    if loop.is_running():
        logger.warning("Cannot resolve secret synchronously while event loop is running")
        return None
    return loop.run_until_complete(get_llm_api_key(provider, secret_provider, secret_prefix=secret_prefix))


def build_secret_provider(
    *,
    provider: str,
    cache_ttl: int = 300,
    environment: str = "development",
    aws_region: Optional[str] = None,
    aws_secret_prefix: Optional[str] = None,
    vault_address: Optional[str] = None,
    vault_token: Optional[str] = None,
    vault_mount_path: Optional[str] = None,
) -> SecretProvider:
    provider = provider.lower()

    if provider == "aws":
        region = aws_region or "us-west-2"
        return AWSSecretsManagerProvider(
            region=region,
            secret_prefix=aws_secret_prefix,
            cache_ttl=cache_ttl,
        )
    if provider == "vault":
        if not vault_address or not vault_token:
            raise ValueError("Vault address and token are required for vault secret provider")
        return VaultSecretProvider(
            address=vault_address,
            token=vault_token,
            mount_path=vault_mount_path or "secret/eleanor",
            cache_ttl=cache_ttl,
        )
    if provider == "env":
        if environment == "production":
            raise ValueError("Environment secret provider is not allowed in production")
        return EnvironmentSecretProvider(cache_ttl=cache_ttl)

    raise ValueError(f"Unknown secret provider: {provider}")


def build_secret_provider_from_settings(settings: object) -> SecretProvider:
    security = getattr(settings, "security", None)
    if security is None:
        raise ValueError("Settings object missing security configuration")
    return build_secret_provider(
        provider=security.secret_provider,
        cache_ttl=security.secrets_cache_ttl,
        environment=getattr(settings, "environment", "development"),
        aws_region=security.aws.region,
        aws_secret_prefix=security.aws.secret_prefix,
        vault_address=security.vault.address,
        vault_token=security.vault.token,
        vault_mount_path=security.vault.mount_path,
    )


__all__ = [
    "SecretProvider",
    "EnvironmentSecretProvider",
    "AWSSecretsManagerProvider",
    "VaultSecretProvider",
    "build_secret_provider",
    "get_llm_api_key",
    "get_llm_api_key_sync",
    "build_secret_provider_from_settings",
]
