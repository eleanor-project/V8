"""
Tests for Secrets Management (Issue #20)
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from engine.security.secrets import (
    EnvironmentSecretsProvider,
    AWSSecretsProvider,
    VaultSecretsProvider,
    auto_detect_secrets_provider,
)
from engine.security.sanitizer import SecretsSanitizer


class TestEnvironmentSecretsProvider:
    """Test environment variable secrets provider"""

    def test_get_secret_exists(self):
        """Test retrieving existing environment variable"""
        with patch.dict(os.environ, {"TEST_SECRET": "test_value"}):
            provider = EnvironmentSecretsProvider()
            assert provider.get_secret("TEST_SECRET") == "test_value"

    def test_get_secret_not_exists(self):
        """Test retrieving non-existent secret"""
        provider = EnvironmentSecretsProvider()
        assert provider.get_secret("NONEXISTENT_SECRET") is None

    def test_list_secrets_with_prefix(self):
        """Test listing secrets with prefix filter"""
        with patch.dict(
            os.environ,
            {
                "ELEANOR_API_KEY": "key1",
                "ELEANOR_SECRET": "key2",
                "OTHER_VAR": "val",
            },
        ):
            provider = EnvironmentSecretsProvider(prefix="ELEANOR_")
            secrets = provider.list_secrets()

            assert "API_KEY" in secrets
            assert "SECRET" in secrets
            assert "OTHER_VAR" not in secrets

    def test_get_secret_or_fail(self):
        """Test get_secret_or_fail raises on missing secret"""
        provider = EnvironmentSecretsProvider()

        with patch.dict(os.environ, {"EXISTS": "value"}):
            assert provider.get_secret_or_fail("EXISTS") == "value"

        with pytest.raises(ValueError, match="Secret 'MISSING' not found"):
            provider.get_secret_or_fail("MISSING")


class TestSecretsSanitizer:
    """Test secrets sanitization"""

    def test_sanitize_openai_key(self):
        """Test OpenAI API key sanitization"""
        text = "API key is sk-abcd1234567890123456789012345678901234567890"
        sanitized = SecretsSanitizer.sanitize_string(text)
        assert "sk-" not in sanitized
        assert "[OPENAI_API_KEY]" in sanitized

    def test_sanitize_anthropic_key(self):
        """Test Anthropic API key sanitization"""
        key = "sk-ant-" + "a" * 95
        sanitized = SecretsSanitizer.sanitize_string(key)
        assert "sk-ant-" not in sanitized
        assert "[ANTHROPIC_API_KEY]" in sanitized

    def test_sanitize_aws_key(self):
        """Test AWS access key sanitization"""
        text = "Access key: AKIAIOSFODNN7EXAMPLE"
        sanitized = SecretsSanitizer.sanitize_string(text)
        assert "AKIA" not in sanitized
        assert "[AWS_ACCESS_KEY]" in sanitized

    def test_sanitize_bearer_token(self):
        """Test Bearer token sanitization"""
        text = "Authorization: Bearer abc123xyz"
        sanitized = SecretsSanitizer.sanitize_string(text)
        assert "abc123xyz" not in sanitized
        assert "[BEARER_TOKEN]" in sanitized

    def test_sanitize_dict_sensitive_keys(self):
        """Test dictionary sanitization with sensitive keys"""
        data = {
            "api_key": "secret123",
            "user": "john",
            "password": "pass123",
            "email": "john@example.com",
        }

        sanitized = SecretsSanitizer.sanitize_dict(data)

        assert sanitized["api_key"] == "[REDACTED]"
        assert sanitized["password"] == "[REDACTED]"
        assert sanitized["user"] == "john"
        assert sanitized["email"] == "john@example.com"

    def test_sanitize_dict_recursive(self):
        """Test recursive dictionary sanitization"""
        data = {
            "config": {
                "api_key": "secret",
                "endpoint": "https://api.example.com",
            },
            "metadata": {"token": "abc123"},
        }

        sanitized = SecretsSanitizer.sanitize_dict(data)

        assert sanitized["config"]["api_key"] == "[REDACTED]"
        assert sanitized["config"]["endpoint"] == "https://api.example.com"
        assert sanitized["metadata"]["token"] == "[REDACTED]"

    def test_sanitize_list(self):
        """Test list sanitization"""
        data = [
            {"api_key": "secret1"},
            {"name": "test"},
            "Bearer token123",
        ]

        sanitized = SecretsSanitizer.sanitize_list(data)

        assert sanitized[0]["api_key"] == "[REDACTED]"
        assert sanitized[1]["name"] == "test"
        assert "token123" not in sanitized[2]

    def test_sanitize_custom_sensitive_keys(self):
        """Test sanitization with custom sensitive keys"""
        data = {"custom_secret": "value", "normal_field": "data"}

        sanitized = SecretsSanitizer.sanitize_dict(data, sensitive_keys={"custom_secret"})

        assert sanitized["custom_secret"] == "[REDACTED]"
        assert sanitized["normal_field"] == "data"

    def test_sanitize_mixed_types(self):
        """Test sanitization of mixed data types"""
        data = {
            "string": "api_key=secret123",
            "number": 42,
            "boolean": True,
            "null": None,
            "list": ["Bearer token"],
        }

        sanitized = SecretsSanitizer.sanitize(data)

        assert "secret123" not in sanitized["string"]
        assert sanitized["number"] == 42
        assert sanitized["boolean"] is True
        assert sanitized["null"] is None


class TestAutoDetectSecretsProvider:
    """Test automatic secrets provider detection"""

    def test_detect_environment_provider(self):
        """Test fallback to environment provider"""
        with patch.dict(os.environ, {}, clear=True):
            provider = auto_detect_secrets_provider()
            assert isinstance(provider, EnvironmentSecretsProvider)

    def test_detect_aws_provider(self):
        """Test detection of AWS Secrets Manager"""
        with patch.dict(
            os.environ,
            {"AWS_SECRETS_MANAGER": "true", "AWS_REGION": "us-east-1"},
        ):
            with patch("engine.security.secrets.boto3"):
                provider = auto_detect_secrets_provider()
                assert isinstance(provider, AWSSecretsProvider)

    def test_detect_vault_provider(self):
        """Test detection of HashiCorp Vault"""
        with patch.dict(
            os.environ,
            {
                "VAULT_ADDR": "https://vault.example.com",
                "VAULT_TOKEN": "test-token",
            },
        ):
            with patch("engine.security.secrets.hvac"):
                mock_client = MagicMock()
                mock_client.is_authenticated.return_value = True

                with patch(
                    "engine.security.secrets.hvac.Client",
                    return_value=mock_client,
                ):
                    provider = auto_detect_secrets_provider()
                    assert isinstance(provider, VaultSecretsProvider)
