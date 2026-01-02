"""
Secrets Management for ELEANOR V8

Provides secure credential storage and retrieval with multiple backend providers:
- Environment variables (development)
- AWS Secrets Manager (production)
- HashiCorp Vault (production)
"""

import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class SecretsProvider(ABC):
    """Abstract base class for secrets management providers"""

    @abstractmethod
    def get_secret(self, key: str) -> Optional[str]:
        """Retrieve a secret by key"""
        pass

    @abstractmethod
    def list_secrets(self) -> List[str]:
        """List available secret keys (not values)"""
        pass

    def get_secret_or_fail(self, key: str) -> str:
        """Get secret or raise error if not found"""
        secret = self.get_secret(key)
        if secret is None:
            raise ValueError(f"Secret '{key}' not found")
        return secret


class EnvironmentSecretsProvider(SecretsProvider):
    """Development: Use environment variables for secrets"""

    def __init__(self, prefix: str = "ELEANOR_"):
        """
        Args:
            prefix: Environment variable prefix to filter by
        """
        self.prefix = prefix
        logger.info(
            "environment_secrets_provider_initialized",
            extra={"prefix": prefix}
        )

    def get_secret(self, key: str) -> Optional[str]:
        """Retrieve secret from environment variable"""
        value = os.getenv(key)
        if value:
            logger.debug(
                "secret_retrieved",
                extra={"key": key, "source": "environment"}
            )
        return value

    def list_secrets(self) -> List[str]:
        """List environment variables matching prefix"""
        return [
            key for key in os.environ.keys()
            if key.startswith(self.prefix)
        ]


class AWSSecretsProvider(SecretsProvider):
    """Production: Use AWS Secrets Manager"""

    def __init__(
        self,
        region_name: str = "us-west-2",
        cache_ttl: int = 300,
    ):
        """
        Args:
            region_name: AWS region for Secrets Manager
            cache_ttl: Cache TTL in seconds (default: 5 minutes)
        """
        try:
            import boto3
            self.client = boto3.client(
                "secretsmanager",
                region_name=region_name
            )
        except ImportError:
            raise ImportError(
                "boto3 not installed. Install with: pip install boto3"
            )

        self.region_name = region_name
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, tuple[str, float]] = {}  # {key: (value, expiry)}

        logger.info(
            "aws_secrets_provider_initialized",
            extra={
                "region": region_name,
                "cache_ttl": cache_ttl,
            }
        )

    def get_secret(self, key: str) -> Optional[str]:
        """Retrieve secret from AWS Secrets Manager with caching"""
        # Check cache
        if key in self._cache:
            value, expiry = self._cache[key]
            if time.time() < expiry:
                logger.debug(
                    "secret_retrieved_from_cache",
                    extra={"key": key}
                )
                return value
            else:
                # Expired
                del self._cache[key]

        # Fetch from AWS
        try:
            response = self.client.get_secret_value(SecretId=key)
            secret = response["SecretString"]

            # Cache with TTL
            expiry = time.time() + self.cache_ttl
            self._cache[key] = (secret, expiry)

            logger.info(
                "secret_retrieved",
                extra={
                    "key": key,
                    "source": "aws_secrets_manager",
                    "region": self.region_name,
                }
            )
            return secret

        except self.client.exceptions.ResourceNotFoundException:
            logger.warning(
                "secret_not_found",
                extra={"key": key, "source": "aws_secrets_manager"}
            )
            return None
        except Exception as exc:
            logger.error(
                "secret_retrieval_failed",
                extra={
                    "key": key,
                    "error": str(exc),
                    "source": "aws_secrets_manager",
                },
                exc_info=True,
            )
            return None

    def list_secrets(self) -> List[str]:
        """List all secret names in AWS Secrets Manager"""
        try:
            response = self.client.list_secrets()
            return [s["Name"] for s in response["SecretList"]]
        except Exception as exc:
            logger.error(
                "list_secrets_failed",
                extra={"error": str(exc)},
                exc_info=True,
            )
            return []


class VaultSecretsProvider(SecretsProvider):
    """Production: Use HashiCorp Vault"""

    def __init__(
        self,
        vault_addr: str,
        vault_token: str,
        mount_point: str = "secret",
        cache_ttl: int = 300,
    ):
        """
        Args:
            vault_addr: Vault server address (e.g., https://vault.example.com)
            vault_token: Vault authentication token
            mount_point: KV secrets engine mount point
            cache_ttl: Cache TTL in seconds (default: 5 minutes)
        """
        try:
            import hvac
            self.client = hvac.Client(url=vault_addr, token=vault_token)
            
            # Verify connection
            if not self.client.is_authenticated():
                raise ValueError("Failed to authenticate with Vault")
                
        except ImportError:
            raise ImportError(
                "hvac not installed. Install with: pip install hvac"
            )

        self.vault_addr = vault_addr
        self.mount_point = mount_point
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, tuple[str, float]] = {}

        logger.info(
            "vault_secrets_provider_initialized",
            extra={
                "vault_addr": vault_addr,
                "mount_point": mount_point,
                "cache_ttl": cache_ttl,
            }
        )

    def get_secret(self, key: str) -> Optional[str]:
        """Retrieve secret from Vault with caching"""
        # Check cache
        if key in self._cache:
            value, expiry = self._cache[key]
            if time.time() < expiry:
                logger.debug(
                    "secret_retrieved_from_cache",
                    extra={"key": key}
                )
                return value
            else:
                del self._cache[key]

        # Fetch from Vault
        try:
            secret_response = self.client.secrets.kv.v2.read_secret_version(
                path=key,
                mount_point=self.mount_point,
            )
            secret = secret_response["data"]["data"]["value"]

            # Cache with TTL
            expiry = time.time() + self.cache_ttl
            self._cache[key] = (secret, expiry)

            logger.info(
                "secret_retrieved",
                extra={
                    "key": key,
                    "source": "vault",
                    "mount_point": self.mount_point,
                }
            )
            return secret

        except Exception as exc:
            logger.error(
                "secret_retrieval_failed",
                extra={
                    "key": key,
                    "error": str(exc),
                    "source": "vault",
                },
                exc_info=True,
            )
            return None

    def list_secrets(self) -> List[str]:
        """List all secret paths in Vault"""
        try:
            response = self.client.secrets.kv.v2.list_secrets(
                path="",
                mount_point=self.mount_point,
            )
            return response["data"]["keys"]
        except Exception as exc:
            logger.error(
                "list_secrets_failed",
                extra={"error": str(exc)},
                exc_info=True,
            )
            return []


def auto_detect_secrets_provider() -> SecretsProvider:
    """
    Auto-detect and initialize the appropriate secrets provider.

    Detection order:
    1. AWS Secrets Manager (if AWS_SECRETS_MANAGER=true)
    2. HashiCorp Vault (if VAULT_ADDR is set)
    3. Environment variables (fallback)

    Returns:
        Configured SecretsProvider instance
    """
    # Check for AWS Secrets Manager
    if os.getenv("AWS_SECRETS_MANAGER", "").lower() == "true":
        region = os.getenv("AWS_REGION", "us-west-2")
        logger.info(
            "auto_detected_secrets_provider",
            extra={"provider": "aws", "region": region}
        )
        return AWSSecretsProvider(region_name=region)

    # Check for HashiCorp Vault
    vault_addr = os.getenv("VAULT_ADDR")
    vault_token = os.getenv("VAULT_TOKEN")
    if vault_addr and vault_token:
        logger.info(
            "auto_detected_secrets_provider",
            extra={"provider": "vault", "vault_addr": vault_addr}
        )
        return VaultSecretsProvider(
            vault_addr=vault_addr,
            vault_token=vault_token,
        )

    # Fallback to environment variables
    logger.warning(
        "using_environment_secrets_provider",
        extra={
            "message": (
                "Using environment variables for secrets. "
                "Configure AWS Secrets Manager or Vault for production."
            )
        }
    )
    return EnvironmentSecretsProvider()
