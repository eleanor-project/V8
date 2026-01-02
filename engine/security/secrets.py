"""
Secrets Management for ELEANOR V8

Provides pluggable secrets providers:
- EnvironmentSecretsProvider: Development (env vars)
- AWSSecretsProvider: Production (AWS Secrets Manager)
- VaultSecretsProvider: Production (HashiCorp Vault)
"""

import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class SecretsProvider(ABC):
    """Abstract base for secrets management providers"""

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


class EnvironmentSecretsProvider(SecretsProvider):
    """
    Development secrets provider using environment variables.
    
    Warning: Not recommended for production use.
    """

    def __init__(self, prefix: str = "ELEANOR_"):
        """
        Args:
            prefix: Only consider env vars with this prefix
        """
        self.prefix = prefix
        logger.info(
            "environment_secrets_provider_initialized",
            extra={"prefix": prefix}
        )
        logger.warning(
            "Using environment variables for secrets. "
            "Configure AWS Secrets Manager or Vault for production."
        )

    def get_secret(self, key: str) -> Optional[str]:
        """Get secret from environment variable"""
        # Try with prefix first, then without
        value = os.getenv(f"{self.prefix}{key}")
        if value is None:
            value = os.getenv(key)
        
        if value:
            logger.debug(
                "secret_retrieved_from_environment",
                extra={"key": key}
            )
        else:
            logger.debug(
                "secret_not_found_in_environment",
                extra={"key": key}
            )
        
        return value

    def list_secrets(self) -> List[str]:
        """List all environment variables with prefix"""
        return [
            k.replace(self.prefix, "", 1) if k.startswith(self.prefix) else k
            for k in os.environ.keys()
            if k.startswith(self.prefix)
        ]


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
        try:
            import boto3
            from botocore.exceptions import ClientError
            
            self.boto3 = boto3
            self.ClientError = ClientError
        except ImportError:
            raise ImportError(
                "boto3 required for AWSSecretsProvider. "
                "Install with: pip install boto3"
            )

        self.region_name = region_name
        self.cache_ttl = cache_ttl
        self.prefix = prefix
        
        self.client = self.boto3.client(
            "secretsmanager",
            region_name=region_name
        )
        
        # Cache: {key: (value, timestamp)}
        self._cache: Dict[str, tuple[str, float]] = {}
        
        logger.info(
            "aws_secrets_provider_initialized",
            extra={
                "region": region_name,
                "cache_ttl": cache_ttl,
                "prefix": prefix,
            }
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
            logger.debug(
                "secret_retrieved_from_cache",
                extra={"key": key}
            )
            value, _ = self._cache[key]
            return value
        
        # Fetch from AWS
        secret_name = f"{self.prefix}{key}"
        
        try:
            response = self.client.get_secret_value(SecretId=secret_name)
            secret = response["SecretString"]
            
            # Cache it
            self._cache[key] = (secret, time.time())
            
            logger.info(
                "secret_retrieved_from_aws",
                extra={"key": key, "secret_name": secret_name}
            )
            
            return secret
            
        except self.ClientError as e:
            error_code = e.response["Error"]["Code"]
            
            if error_code == "ResourceNotFoundException":
                logger.warning(
                    "secret_not_found_in_aws",
                    extra={"key": key, "secret_name": secret_name}
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
    ):
        """
        Args:
            vault_addr: Vault server address (e.g., https://vault.example.com)
            vault_token: Vault authentication token (or use VAULT_TOKEN env)
            mount_point: KV secrets mount point
        """
        try:
            import hvac
            self.hvac = hvac
        except ImportError:
            raise ImportError(
                "hvac required for VaultSecretsProvider. "
                "Install with: pip install hvac"
            )

        self.vault_addr = vault_addr
        self.mount_point = mount_point
        
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
            }
        )

    def get_secret(self, key: str) -> Optional[str]:
        """Get secret from Vault KV v2"""
        try:
            # Read from KV v2 (versioned)
            secret = self.client.secrets.kv.v2.read_secret_version(
                path=key,
                mount_point=self.mount_point,
            )
            
            value = secret["data"]["data"].get("value")
            
            logger.info(
                "secret_retrieved_from_vault",
                extra={"key": key, "mount_point": self.mount_point}
            )
            
            return value
            
        except Exception as e:
            logger.error(
                "vault_secrets_error",
                extra={
                    "key": key,
                    "error": str(e),
                },
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
            return response["data"]["keys"]
            
        except Exception as e:
            logger.error(
                "failed_to_list_vault_secrets",
                extra={"error": str(e)},
                exc_info=True,
            )
            return []
