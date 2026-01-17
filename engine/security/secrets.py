"""
Secrets Management for ELEANOR V8

Provides pluggable secrets providers:
- EnvironmentSecretsProvider: Development (env vars)
- AWSSecretsProvider: Production (AWS Secrets Manager)
- VaultSecretsProvider: Production (HashiCorp Vault)
- AzureSecretsProvider: Production (Azure Key Vault)
"""

import asyncio
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Callable, Tuple

from engine.security.audit import audit_log

try:
    import boto3 as _boto3  # type: ignore[import-not-found]
    from botocore.exceptions import ClientError as _ClientError  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    _boto3 = None
    _ClientError = Exception

try:
    import hvac as _hvac  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    _hvac = None

try:
    from azure.identity import DefaultAzureCredential as _DefaultAzureCredential  # type: ignore[import-not-found]
    from azure.keyvault.secrets import SecretClient as _SecretClient  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    _DefaultAzureCredential = None
    _SecretClient = None

boto3 = _boto3
ClientError = _ClientError
hvac = _hvac
DefaultAzureCredential = _DefaultAzureCredential
SecretClient = _SecretClient

logger = logging.getLogger(__name__)


def _format_datetime(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        return value.isoformat()
    except Exception:
        return str(value)


@dataclass
class ValidationResult:
    """Validation result for secret checks."""

    valid: bool
    reason: Optional[str] = None


class SecretValidator:
    """Validate secret formats and basic entropy."""

    @staticmethod
    def _has_sufficient_entropy(value: str) -> bool:
        if len(value) < 16:
            return False
        unique_chars = len(set(value))
        return unique_chars >= max(8, int(len(value) * 0.2))

    def validate_api_key(self, key: str) -> ValidationResult:
        if not key or len(key) < 32:
            return ValidationResult(valid=False, reason="Too short")
        if not self._has_sufficient_entropy(key):
            return ValidationResult(valid=False, reason="Low entropy")
        return ValidationResult(valid=True)


class SecretsProvider(ABC):
    """Abstract base for secrets management providers"""

    def __init__(self):
        """Initialize secrets provider with rotation hooks and audit logging."""
        self._rotation_hooks: List[Callable[[str], None]] = []
        self._audit_log_enabled = True

    @abstractmethod
    def get_secret(self, key: str) -> Optional[str]:
        """Retrieve a secret by key"""
        pass

    @abstractmethod
    def list_secrets(self) -> List[str]:
        """List available secret keys (not values)"""
        pass

    def get_secret_or_raise(self, key: str) -> str:
        """Get secret or raise ValueError if missing"""
        secret = self.get_secret(key)
        if secret is None:
            raise ValueError(f"Required secret not found: {key}")
        return secret

    def get_secret_or_fail(self, key: str) -> str:
        """Get secret or raise ValueError if missing (legacy message)."""
        secret = self.get_secret(key)
        if secret is None:
            raise ValueError(f"Secret '{key}' not found")
        return secret

    async def refresh_secrets(self) -> None:
        """Refresh cached secrets (no-op by default)."""
        return None

    def set_audit_logging(self, enabled: bool) -> None:
        """Enable or disable audit logging for this provider."""
        self._audit_log_enabled = enabled
    
    def add_rotation_hook(self, hook: Callable[[str], None]) -> None:
        """
        Add a hook to be called when a secret is rotated.
        
        Args:
            hook: Callable that takes the secret key as argument
        """
        self._rotation_hooks.append(hook)
    
    def _notify_rotation_hooks(self, key: str) -> None:
        """Notify all rotation hooks that a secret was rotated."""
        for hook in self._rotation_hooks:
            try:
                hook(key)
            except Exception as e:
                logger.error(
                    "rotation_hook_error",
                    extra={"key": key, "error": str(e)},
                    exc_info=True,
                )
    
    def _audit_log_secret_access(self, key: str, action: str, success: bool) -> None:
        """
        Log secret access for audit purposes (without logging values).
        
        Args:
            key: Secret key accessed
            action: Action performed (get, list, rotate, etc.)
            success: Whether the action succeeded
        """
        if not self._audit_log_enabled:
            return
        
        payload = {
            "secret_key": key,
            "action": action,
            "success": success,
            "provider": self.__class__.__name__,
        }
        logger.info("secret_access_audit", extra=payload)
        audit_log("secret_access_audit", extra=payload)
    
    def get_secret_with_version(self, key: str, version: Optional[str] = None) -> Optional[str]:
        """
        Get secret with optional version support.
        
        Args:
            key: Secret key
            version: Optional version identifier (provider-specific)
        
        Returns:
            Secret value or None if not found
        """
        # Default implementation falls back to get_secret
        # Providers can override for version-specific retrieval
        return self.get_secret(key)

    def get_secret_metadata(self, key: str) -> Dict[str, Any]:
        """Return secret metadata if supported by provider."""
        self._audit_log_secret_access(key, "metadata", False)
        return {"supported": False}

    def rotate_secret(self, key: str, new_value: Optional[str] = None) -> bool:
        """Rotate secret value if supported."""
        self._audit_log_secret_access(key, "rotate", False)
        return False


class EnvironmentSecretsProvider(SecretsProvider):
    """
    Development secrets provider using environment variables.

    Warning: Not recommended for production use.
    """

    def __init__(self, prefix: str = "ELEANOR_", cache_ttl: Optional[int] = None):
        """
        Args:
            prefix: Only consider env vars with this prefix
        """
        super().__init__()
        self.prefix = prefix
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Tuple[str, float]] = {}
        logger.info("environment_secrets_provider_initialized", extra={"prefix": prefix})
        logger.warning(
            "Using environment variables for secrets. "
            "Configure AWS Secrets Manager or Vault for production."
        )

    def get_secret(self, key: str) -> Optional[str]:
        """Get secret from environment variable"""
        if self.cache_ttl:
            cached = self._cache.get(key)
            if cached:
                value, expires_at = cached
                if time.time() < expires_at:
                    self._audit_log_secret_access(key, "get_cached", True)
                    return value
                self._cache.pop(key, None)
        # Try with prefix first, then without
        value = os.getenv(f"{self.prefix}{key}")
        if value is None:
            value = os.getenv(key)

        success = value is not None
        self._audit_log_secret_access(key, "get", success)

        if value:
            logger.debug("secret_retrieved_from_environment", extra={"key": key})
        else:
            logger.debug("secret_not_found_in_environment", extra={"key": key})

        if value and self.cache_ttl:
            self._cache[key] = (value, time.time() + self.cache_ttl)
        return value

    def list_secrets(self) -> List[str]:
        """List all environment variables with prefix"""
        return [
            k.replace(self.prefix, "", 1) if k.startswith(self.prefix) else k
            for k in os.environ.keys()
            if k.startswith(self.prefix)
        ]


class EnvironmentSecretProvider(EnvironmentSecretsProvider):
    """Backwards-compatible alias with optional cache_ttl parameter."""

    def __init__(self, prefix: str = "ELEANOR_", cache_ttl: Optional[int] = None):
        super().__init__(prefix=prefix, cache_ttl=cache_ttl)


class AWSSecretsProvider(SecretsProvider):
    """
    Production secrets provider using AWS Secrets Manager.

    Features:
    - Automatic secret rotation support
    - Caching with TTL
    - IAM-based access control
    """

    def __init__(
        self,
        region_name: str = "us-west-2",
        cache_ttl: int = 300,
        prefix: str = "eleanor/",
    ):
        """
        Args:
            region_name: AWS region for Secrets Manager
            cache_ttl: Seconds to cache secrets (default: 5 minutes)
            prefix: Prefix for all secret names
        """
        if boto3 is None:
            raise ImportError(
                "boto3 required for AWSSecretsProvider. " "Install with: pip install boto3"
            )
        super().__init__()
        self.boto3 = boto3
        self.ClientError = ClientError

        self.region_name = region_name
        self.cache_ttl = cache_ttl
        self.prefix = prefix

        self.client = self.boto3.client("secretsmanager", region_name=region_name)

        # Cache: {key: (value, timestamp)}
        self._cache: Dict[str, tuple[str, float]] = {}

        logger.info(
            "aws_secrets_provider_initialized",
            extra={
                "region": region_name,
                "cache_ttl": cache_ttl,
                "prefix": prefix,
            },
        )

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached value is still valid"""
        if key not in self._cache:
            return False

        _, timestamp = self._cache[key]
        age = time.time() - timestamp
        return age < self.cache_ttl

    def get_secret(self, key: str) -> Optional[str]:
        """Get secret from AWS Secrets Manager with caching"""
        # Check cache
        if self._is_cache_valid(key):
            logger.debug("secret_retrieved_from_cache", extra={"key": key})
            value, _ = self._cache[key]
            self._audit_log_secret_access(key, "get_cached", True)
            return value

        # Fetch from AWS
        secret_name = f"{self.prefix}{key}"

        try:
            response = self.client.get_secret_value(SecretId=secret_name)
            secret = response.get("SecretString")
            if secret is None:
                self._audit_log_secret_access(key, "get", False)
                return None
            secret_str = str(secret)

            # Cache it
            self._cache[key] = (secret_str, time.time())

            self._audit_log_secret_access(key, "get", True)
            logger.info("secret_retrieved_from_aws", extra={"key": key, "secret_name": secret_name})

            return secret_str

        except self.ClientError as e:
            error_code = e.response["Error"]["Code"]

            self._audit_log_secret_access(key, "get", False)

            if error_code == "ResourceNotFoundException":
                logger.warning(
                    "secret_not_found_in_aws", extra={"key": key, "secret_name": secret_name}
                )
            else:
                logger.error(
                    "aws_secrets_error",
                    extra={
                        "key": key,
                        "error_code": error_code,
                        "error": str(e),
                    },
                    exc_info=True,
                )

            return None
    
    def get_secret_with_version(self, key: str, version: Optional[str] = None) -> Optional[str]:
        """
        Get secret with version support from AWS Secrets Manager.
        
        Args:
            key: Secret key
            version: AWS Secrets Manager version ID or stage (e.g., "AWSCURRENT", "AWSPREVIOUS")
        
        Returns:
            Secret value or None if not found
        """
        secret_name = f"{self.prefix}{key}"
        
        try:
            params = {"SecretId": secret_name}
            if version:
                params["VersionStage"] = version
            
            response = self.client.get_secret_value(**params)
            secret = response.get("SecretString")
            if secret is None:
                self._audit_log_secret_access(key, f"get_version_{version}", False)
                return None
            
            secret_str = str(secret)
            self._audit_log_secret_access(key, f"get_version_{version}", True)
            return secret_str
            
        except self.ClientError as e:
            self._audit_log_secret_access(key, f"get_version_{version}", False)
            logger.error(
                "aws_secrets_version_error",
                extra={"key": key, "version": version, "error": str(e)},
                exc_info=True,
            )
            return None

    def list_secrets(self) -> List[str]:
        """List all secrets with prefix"""
        try:
            paginator = self.client.get_paginator("list_secrets")
            secrets = []

            for page in paginator.paginate():
                for secret in page["SecretList"]:
                    name = secret["Name"]
                    if name.startswith(self.prefix):
                        # Remove prefix
                        secrets.append(name.replace(self.prefix, "", 1))

            return secrets

        except Exception as e:
            logger.error(
                "failed_to_list_aws_secrets",
                extra={"error": str(e)},
                exc_info=True,
            )
            return []

    def get_secret_metadata(self, key: str) -> Dict[str, Any]:
        """Fetch AWS Secrets Manager metadata for secret."""
        secret_name = f"{self.prefix}{key}"
        try:
            response = self.client.describe_secret(SecretId=secret_name)
            metadata = {
                "name": response.get("Name"),
                "arn": response.get("ARN"),
                "last_changed": _format_datetime(response.get("LastChangedDate")),
                "last_rotated": _format_datetime(response.get("LastRotatedDate")),
                "rotation_enabled": response.get("RotationEnabled"),
                "deleted_date": _format_datetime(response.get("DeletedDate")),
                "created_date": _format_datetime(response.get("CreatedDate")),
            }
            self._audit_log_secret_access(key, "metadata", True)
            return metadata
        except Exception as exc:
            self._audit_log_secret_access(key, "metadata", False)
            logger.error(
                "aws_secrets_metadata_error",
                extra={"key": key, "error": str(exc)},
                exc_info=True,
            )
            return {"supported": True, "error": str(exc)}

    def rotate_secret(self, key: str, new_value: Optional[str] = None) -> bool:
        """Rotate secret with a new value or trigger AWS rotation."""
        secret_name = f"{self.prefix}{key}"
        try:
            if new_value is not None:
                self.client.put_secret_value(SecretId=secret_name, SecretString=new_value)
            else:
                self.client.rotate_secret(SecretId=secret_name)
            self._cache.pop(key, None)
            self._notify_rotation_hooks(key)
            self._audit_log_secret_access(key, "rotate", True)
            return True
        except Exception as exc:
            self._audit_log_secret_access(key, "rotate", False)
            logger.error(
                "aws_secrets_rotation_error",
                extra={"key": key, "error": str(exc)},
                exc_info=True,
            )
            return False

    async def refresh_secrets(self) -> None:
        """Clear cached secrets so they are reloaded on next access."""
        keys_to_notify = list(self._cache.keys())
        self._cache.clear()
        
        # Notify rotation hooks
        for key in keys_to_notify:
            self._notify_rotation_hooks(key)
            self._audit_log_secret_access(key, "refresh", True)


class VaultSecretsProvider(SecretsProvider):
    """
    Production secrets provider using HashiCorp Vault.

    Features:
    - Dynamic secrets generation
    - Automatic secret rotation
    - Audit logging
    - Fine-grained access control
    """

    def __init__(
        self,
        vault_addr: str,
        vault_token: Optional[str] = None,
        mount_point: str = "eleanor",
        cache_ttl: int = 300,
    ):
        """
        Args:
            vault_addr: Vault server address (e.g., https://vault.example.com)
            vault_token: Vault authentication token (or use VAULT_TOKEN env)
            mount_point: KV secrets mount point
        """
        if hvac is None:
            raise ImportError(
                "hvac required for VaultSecretsProvider. " "Install with: pip install hvac"
            )
        super().__init__()
        self.hvac = hvac

        self.vault_addr = vault_addr
        self.mount_point = mount_point
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Tuple[str, float]] = {}

        token = vault_token or os.getenv("VAULT_TOKEN")
        if not token:
            raise ValueError(
                "Vault token required. Set VAULT_TOKEN env var or pass vault_token parameter."
            )

        self.client = self.hvac.Client(url=vault_addr, token=token)

        if not self.client.is_authenticated():
            raise ValueError("Failed to authenticate with Vault")

        logger.info(
            "vault_secrets_provider_initialized",
            extra={
                "vault_addr": vault_addr,
                "mount_point": mount_point,
            },
        )

    def _is_cache_valid(self, key: str) -> bool:
        if key not in self._cache:
            return False
        _, timestamp = self._cache[key]
        age = time.time() - timestamp
        return age < self.cache_ttl

    def get_secret(self, key: str) -> Optional[str]:
        """Get secret from Vault KV v2"""
        try:
            if self._is_cache_valid(key):
                value, _ = self._cache[key]
                self._audit_log_secret_access(key, "get_cached", True)
                return value
            # Read from KV v2 (versioned)
            secret = self.client.secrets.kv.v2.read_secret_version(
                path=key,
                mount_point=self.mount_point,
            )

            value = secret.get("data", {}).get("data", {}).get("value")
            success = value is not None
            self._audit_log_secret_access(key, "get", success)
            
            if value is None:
                return None

            logger.info(
                "secret_retrieved_from_vault", extra={"key": key, "mount_point": self.mount_point}
            )

            secret_str = str(value)
            self._cache[key] = (secret_str, time.time())
            return secret_str

        except Exception as e:
            self._audit_log_secret_access(key, "get", False)
            logger.error(
                "vault_secrets_error",
                extra={
                    "key": key,
                    "error": str(e),
                },
                exc_info=True,
            )
            return None
    
    def get_secret_with_version(self, key: str, version: Optional[int] = None) -> Optional[str]:
        """
        Get secret with version support from Vault KV v2.
        
        Args:
            key: Secret key
            version: Vault secret version number (None for latest)
        
        Returns:
            Secret value or None if not found
        """
        try:
            params = {"path": key, "mount_point": self.mount_point}
            if version is not None:
                params["version"] = version
            
            secret = self.client.secrets.kv.v2.read_secret_version(**params)
            value = secret.get("data", {}).get("data", {}).get("value")
            success = value is not None
            self._audit_log_secret_access(key, f"get_version_{version}", success)
            
            if value is None:
                return None
            
            return str(value)
            
        except Exception as e:
            self._audit_log_secret_access(key, f"get_version_{version}", False)
            logger.error(
                "vault_secrets_version_error",
                extra={"key": key, "version": version, "error": str(e)},
                exc_info=True,
            )
            return None

    def list_secrets(self) -> List[str]:
        """List all secrets in mount point"""
        try:
            response = self.client.secrets.kv.v2.list_secrets(
                path="",
                mount_point=self.mount_point,
            )
            keys = response.get("data", {}).get("keys", [])
            if isinstance(keys, list):
                return [str(key) for key in keys]
            return []
        except Exception as e:
            logger.error(
                "failed_to_list_vault_secrets",
                extra={"error": str(e)},
                exc_info=True,
            )
            return []


class AzureSecretsProvider(SecretsProvider):
    """
    Production secrets provider using Azure Key Vault.
    """

    def __init__(
        self,
        vault_url: str,
        credential: Optional[Any] = None,
        cache_ttl: int = 300,
        prefix: str = "",
    ):
        if SecretClient is None or DefaultAzureCredential is None:
            raise ImportError(
                "azure-identity and azure-keyvault-secrets are required for AzureSecretsProvider."
            )
        super().__init__()
        self.vault_url = vault_url
        self.cache_ttl = cache_ttl
        self.prefix = prefix or ""
        self.credential = credential or DefaultAzureCredential()
        self.client = SecretClient(vault_url=vault_url, credential=self.credential)
        self._cache: Dict[str, Tuple[str, float]] = {}

        logger.info(
            "azure_secrets_provider_initialized",
            extra={"vault_url": vault_url, "prefix": self.prefix},
        )

    def _secret_name(self, key: str) -> str:
        name = f"{self.prefix}{key}" if self.prefix else key
        return name.replace("/", "--")

    def _is_cache_valid(self, key: str) -> bool:
        if key not in self._cache:
            return False
        _, timestamp = self._cache[key]
        return (time.time() - timestamp) < self.cache_ttl

    def get_secret(self, key: str) -> Optional[str]:
        if self._is_cache_valid(key):
            value, _ = self._cache[key]
            self._audit_log_secret_access(key, "get_cached", True)
            return value
        name = self._secret_name(key)
        try:
            secret = self.client.get_secret(name)
            value = secret.value if secret else None
            success = value is not None
            self._audit_log_secret_access(key, "get", success)
            if value is None:
                return None
            secret_str = str(value)
            self._cache[key] = (secret_str, time.time())
            return secret_str
        except Exception as exc:
            self._audit_log_secret_access(key, "get", False)
            logger.error(
                "azure_secrets_error",
                extra={"key": key, "error": str(exc)},
                exc_info=True,
            )
            return None

    def get_secret_with_version(self, key: str, version: Optional[str] = None) -> Optional[str]:
        name = self._secret_name(key)
        try:
            secret = self.client.get_secret(name, version=version)
            value = secret.value if secret else None
            success = value is not None
            self._audit_log_secret_access(key, f"get_version_{version}", success)
            return str(value) if value is not None else None
        except Exception as exc:
            self._audit_log_secret_access(key, f"get_version_{version}", False)
            logger.error(
                "azure_secrets_version_error",
                extra={"key": key, "version": version, "error": str(exc)},
                exc_info=True,
            )
            return None

    def list_secrets(self) -> List[str]:
        try:
            names = []
            for prop in self.client.list_properties_of_secrets():
                name = prop.name
                if self.prefix and not name.startswith(self.prefix):
                    continue
                key = name.replace(self.prefix, "", 1) if self.prefix else name
                names.append(key.replace("--", "/"))
            return names
        except Exception as exc:
            logger.error(
                "failed_to_list_azure_secrets",
                extra={"error": str(exc)},
                exc_info=True,
            )
            return []

    def get_secret_metadata(self, key: str) -> Dict[str, Any]:
        name = self._secret_name(key)
        try:
            secret = self.client.get_secret(name)
            props = secret.properties if secret else None
            metadata = {
                "enabled": getattr(props, "enabled", None),
                "created_on": _format_datetime(getattr(props, "created_on", None)),
                "updated_on": _format_datetime(getattr(props, "updated_on", None)),
                "expires_on": _format_datetime(getattr(props, "expires_on", None)),
                "id": getattr(props, "id", None),
                "tags": getattr(props, "tags", None),
            }
            self._audit_log_secret_access(key, "metadata", True)
            return metadata
        except Exception as exc:
            self._audit_log_secret_access(key, "metadata", False)
            logger.error(
                "azure_secrets_metadata_error",
                extra={"key": key, "error": str(exc)},
                exc_info=True,
            )
            return {"supported": True, "error": str(exc)}

    def rotate_secret(self, key: str, new_value: Optional[str] = None) -> bool:
        if new_value is None:
            self._audit_log_secret_access(key, "rotate", False)
            return False
        name = self._secret_name(key)
        try:
            self.client.set_secret(name, new_value)
            self._cache.pop(key, None)
            self._notify_rotation_hooks(key)
            self._audit_log_secret_access(key, "rotate", True)
            return True
        except Exception as exc:
            self._audit_log_secret_access(key, "rotate", False)
            logger.error(
                "azure_secrets_rotation_error",
                extra={"key": key, "error": str(exc)},
                exc_info=True,
            )
            return False
        except Exception as e:
            logger.error(
                "failed_to_list_vault_secrets",
                extra={"error": str(e)},
                exc_info=True,
            )
            return []

    def get_secret_metadata(self, key: str) -> Dict[str, Any]:
        """Fetch Vault secret metadata (KV v2)."""
        try:
            metadata = self.client.secrets.kv.v2.read_secret_metadata(
                path=key,
                mount_point=self.mount_point,
            )
            data = metadata.get("data", {}) if isinstance(metadata, dict) else {}
            self._audit_log_secret_access(key, "metadata", True)
            return {
                "created_time": data.get("created_time"),
                "updated_time": data.get("updated_time"),
                "current_version": data.get("current_version"),
                "max_versions": data.get("max_versions"),
            }
        except Exception as exc:
            self._audit_log_secret_access(key, "metadata", False)
            logger.error(
                "vault_secrets_metadata_error",
                extra={"key": key, "error": str(exc)},
                exc_info=True,
            )
            return {"supported": True, "error": str(exc)}

    def rotate_secret(self, key: str, new_value: Optional[str] = None) -> bool:
        """Rotate Vault secret by writing a new version."""
        if new_value is None:
            self._audit_log_secret_access(key, "rotate", False)
            return False
        try:
            self.client.secrets.kv.v2.create_or_update_secret(
                path=key,
                mount_point=self.mount_point,
                secret={"value": new_value},
            )
            self._cache.pop(key, None)
            self._notify_rotation_hooks(key)
            self._audit_log_secret_access(key, "rotate", True)
            return True
        except Exception as exc:
            self._audit_log_secret_access(key, "rotate", False)
            logger.error(
                "vault_secrets_rotation_error",
                extra={"key": key, "error": str(exc)},
                exc_info=True,
            )
            return False


_LLM_KEY_MAP = {
    "openai": ("OPENAI_API_KEY", "OPENAI_KEY"),
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_KEY"),
    "xai": ("XAI_API_KEY", "XAI_KEY"),
    "grok": ("XAI_API_KEY", "XAI_KEY"),
}


def _candidate_keys(provider: str) -> Iterable[str]:
    normalized = provider.lower()
    return _LLM_KEY_MAP.get(normalized, (provider.upper(),))


class SecretRotator:
    """Rotate secrets based on metadata."""

    def __init__(
        self,
        provider: SecretsProvider,
        advance_rotation_days: int = 7,
        max_age_days: int = 90,
    ):
        self.provider = provider
        self.advance_rotation_days = advance_rotation_days
        self.max_age_days = max_age_days

    def _parse_timestamp(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        try:
            # Attempt ISO8601 parse via datetime from stdlib
            from datetime import datetime

            return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
        except Exception:
            return None

    def _should_rotate(self, metadata: Dict[str, Any]) -> bool:
        if not metadata:
            return False
        if metadata.get("compromised"):
            return True
        expires_at = metadata.get("expires_at") or metadata.get("expires_on")
        expires_ts = self._parse_timestamp(expires_at)
        if expires_ts:
            days_left = (expires_ts - time.time()) / 86400
            if days_left <= self.advance_rotation_days:
                return True
        last_rotated = (
            metadata.get("last_rotated")
            or metadata.get("last_changed")
            or metadata.get("updated_time")
        )
        last_ts = self._parse_timestamp(last_rotated)
        if last_ts:
            age_days = (time.time() - last_ts) / 86400
            if age_days >= self.max_age_days:
                return True
        return False

    async def rotate_if_needed(self, key: str, new_value: Optional[str] = None) -> bool:
        metadata = await asyncio.to_thread(self.provider.get_secret_metadata, key)
        if not self._should_rotate(metadata):
            return False
        rotated = await asyncio.to_thread(self.provider.rotate_secret, key, new_value)
        return rotated


def get_llm_api_key_sync(provider: str, secret_provider: SecretsProvider) -> Optional[str]:
    """Fetch LLM API key via secret provider with env var fallback."""
    for key in _candidate_keys(provider):
        value = secret_provider.get_secret(key)
        if value:
            return value
        fallback = os.getenv(key)
        if fallback:
            return fallback
    return None


async def get_llm_api_key(provider: str, secret_provider: SecretsProvider) -> Optional[str]:
    """Async wrapper for fetching LLM API keys (supports networked providers)."""
    return await asyncio.to_thread(get_llm_api_key_sync, provider, secret_provider)


def build_secret_provider_from_settings(settings: Any) -> SecretsProvider:
    """Build the configured secret provider from Eleanor settings."""
    provider = getattr(settings.security, "secret_provider", "env")
    provider = str(provider).lower()
    cache_ttl = getattr(settings.security, "secrets_cache_ttl", 300)

    secret_provider: Optional[SecretsProvider] = None

    if provider == "aws":
        aws_cfg = settings.security.aws
        prefix = getattr(aws_cfg, "secret_prefix", "eleanor/")
        if prefix and not prefix.endswith("/"):
            prefix = f"{prefix}/"
        secret_provider = AWSSecretsProvider(
            region_name=getattr(aws_cfg, "region", "us-west-2"),
            cache_ttl=cache_ttl,
            prefix=prefix,
        )
    elif provider == "vault":
        vault_cfg = settings.security.vault
        vault_addr = getattr(vault_cfg, "address", None)
        if not vault_addr:
            raise ValueError("Vault address required for vault secrets provider")
        secret_provider = VaultSecretsProvider(
            vault_addr=vault_addr,
            vault_token=getattr(vault_cfg, "token", None),
            mount_point=getattr(vault_cfg, "mount_path", "secret/eleanor"),
            cache_ttl=cache_ttl,
        )
    elif provider == "azure":
        azure_cfg = getattr(settings.security, "azure", None)
        vault_url = getattr(azure_cfg, "vault_url", None) if azure_cfg else None
        if not vault_url:
            raise ValueError("Azure vault_url required for azure secrets provider")
        prefix = getattr(azure_cfg, "secret_prefix", "") if azure_cfg else ""
        secret_provider = AzureSecretsProvider(
            vault_url=vault_url,
            cache_ttl=cache_ttl,
            prefix=prefix or "",
        )
    elif provider == "env":
        secret_provider = EnvironmentSecretProvider(cache_ttl=cache_ttl)
    else:
        raise ValueError(f"Unknown secret_provider: {provider}")

    enable_audit = getattr(settings.security, "enable_secret_audit", True)
    if hasattr(secret_provider, "set_audit_logging"):
        secret_provider.set_audit_logging(bool(enable_audit))
    return secret_provider


def auto_detect_secrets_provider(cache_ttl: int = 300) -> SecretsProvider:
    """Auto-detect secrets provider based on environment variables."""
    aws_enabled = os.getenv("AWS_SECRETS_MANAGER", "").lower() in ("1", "true", "yes")
    if aws_enabled:
        region = os.getenv("AWS_REGION", "us-west-2")
        prefix = os.getenv("AWS_SECRET_PREFIX") or os.getenv("AWS_SECRETS_PREFIX") or "eleanor/"
        if prefix and not prefix.endswith("/"):
            prefix = f"{prefix}/"
        try:
            return AWSSecretsProvider(region_name=region, cache_ttl=cache_ttl, prefix=prefix)
        except Exception as exc:
            logger.warning("aws_secrets_provider_init_failed", extra={"error": str(exc)})

    azure_vault_url = os.getenv("AZURE_KEY_VAULT_URL") or os.getenv("AZURE_VAULT_URL")
    if azure_vault_url:
        prefix = os.getenv("AZURE_SECRET_PREFIX", "")
        try:
            return AzureSecretsProvider(
                vault_url=azure_vault_url,
                cache_ttl=cache_ttl,
                prefix=prefix or "",
            )
        except Exception as exc:
            logger.warning("azure_secrets_provider_init_failed", extra={"error": str(exc)})

    vault_addr = os.getenv("VAULT_ADDR")
    vault_token = os.getenv("VAULT_TOKEN")
    if vault_addr and vault_token:
        mount_point = os.getenv("VAULT_MOUNT_POINT") or os.getenv("VAULT_MOUNT") or "eleanor"
        try:
            return VaultSecretsProvider(
                vault_addr=vault_addr,
                vault_token=vault_token,
                mount_point=mount_point,
                cache_ttl=cache_ttl,
            )
        except Exception as exc:
            logger.warning("vault_secrets_provider_init_failed", extra={"error": str(exc)})

    return EnvironmentSecretProvider(cache_ttl=cache_ttl)


__all__ = [
    "SecretsProvider",
    "EnvironmentSecretsProvider",
    "EnvironmentSecretProvider",
    "AWSSecretsProvider",
    "VaultSecretsProvider",
    "AzureSecretsProvider",
    "SecretRotator",
    "SecretValidator",
    "ValidationResult",
    "auto_detect_secrets_provider",
    "build_secret_provider_from_settings",
    "get_llm_api_key_sync",
    "get_llm_api_key",
]
