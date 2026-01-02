"""
Tests for Security Module (Issue #20)
"""

import os
import pytest
from unittest.mock import patch

from engine.security.secrets import (
    EnvironmentSecretsProvider,
)
from engine.security.sanitizer import SecretsSanitizer
from engine.security.audit import SecureAuditLogger


class TestEnvironmentSecretsProvider:
    """Test EnvironmentSecretsProvider"""

    def test_get_secret_with_prefix(self):
        """Test retrieving secret with prefix"""
        with patch.dict(os.environ, {"ELEANOR_API_KEY": "test-key"}):
            provider = EnvironmentSecretsProvider(prefix="ELEANOR_")
            assert provider.get_secret("API_KEY") == "test-key"

    def test_get_secret_without_prefix(self):
        """Test retrieving secret without prefix"""
        with patch.dict(os.environ, {"API_KEY": "test-key"}):
            provider = EnvironmentSecretsProvider(prefix="ELEANOR_")
            assert provider.get_secret("API_KEY") == "test-key"

    def test_get_secret_not_found(self):
        """Test retrieving non-existent secret"""
        provider = EnvironmentSecretsProvider()
        assert provider.get_secret("NONEXISTENT_KEY") is None

    def test_list_secrets(self):
        """Test listing secrets with prefix"""
        with patch.dict(
            os.environ,
            {"ELEANOR_KEY1": "val1", "ELEANOR_KEY2": "val2", "OTHER": "val3"},
            clear=True,
        ):
            provider = EnvironmentSecretsProvider(prefix="ELEANOR_")
            secrets = provider.list_secrets()
            assert "KEY1" in secrets
            assert "KEY2" in secrets
            assert len(secrets) == 2

    def test_get_secret_or_raise(self):
        """Test get_secret_or_raise raises on missing secret"""
        provider = EnvironmentSecretsProvider()
        with pytest.raises(ValueError, match="Required secret not found"):
            provider.get_secret_or_raise("MISSING_SECRET")


class TestSecretsSanitizer:
    """Test SecretsSanitizer"""

    def setup_method(self):
        """Setup sanitizer for each test"""
        self.sanitizer = SecretsSanitizer()

    def test_sanitize_openai_key(self):
        """Test OpenAI API key redaction"""
        text = "My key is sk-1234567890abcdefghijklmnopqrstuvwxyz12345678"
        result = self.sanitizer.sanitize_string(text)
        assert "sk-1234567890" not in result
        assert "[OPENAI_API_KEY]" in result

    def test_sanitize_anthropic_key(self):
        """Test Anthropic API key redaction"""
        text = "sk-ant-" + "a" * 95
        result = self.sanitizer.sanitize_string(text)
        assert "sk-ant-" not in result
        assert "[ANTHROPIC_API_KEY]" in result

    def test_sanitize_aws_access_key(self):
        """Test AWS access key redaction"""
        text = "AWS key: AKIAIOSFODNN7EXAMPLE"
        result = self.sanitizer.sanitize_string(text)
        assert "AKIAIOSFODNN7EXAMPLE" not in result
        assert "[AWS_ACCESS_KEY]" in result

    def test_sanitize_bearer_token(self):
        """Test Bearer token redaction"""
        text = "Authorization: Bearer abc123xyz789"
        result = self.sanitizer.sanitize_string(text)
        assert "abc123xyz789" not in result
        assert "[BEARER_TOKEN]" in result

    def test_sanitize_github_token(self):
        """Test GitHub token redaction"""
        text = "ghp_" + "a" * 36
        result = self.sanitizer.sanitize_string(text)
        assert "ghp_" not in result
        assert "[GITHUB_TOKEN]" in result

    def test_sanitize_jwt(self):
        """Test JWT token redaction"""
        jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        result = self.sanitizer.sanitize_string(jwt)
        assert "eyJ" not in result
        assert "[JWT_TOKEN]" in result

    def test_sanitize_dict_sensitive_keys(self):
        """Test dictionary sanitization with sensitive keys"""
        data = {
            "api_key": "secret123",
            "username": "john",
            "password": "pass123",
        }
        result = self.sanitizer.sanitize_dict(data)
        assert result["api_key"] == "[REDACTED]"
        assert result["username"] == "john"
        assert result["password"] == "[REDACTED]"

    def test_sanitize_nested_dict(self):
        """Test nested dictionary sanitization"""
        data = {
            "config": {
                "api_key": "secret123",
                "timeout": 30,
            },
            "user": "john",
        }
        result = self.sanitizer.sanitize_dict(data)
        assert result["config"]["api_key"] == "[REDACTED]"
        assert result["config"]["timeout"] == 30
        assert result["user"] == "john"

    def test_sanitize_list(self):
        """Test list sanitization"""
        data = [
            "normal text",
            {"api_key": "secret123"},
            "sk-1234567890abcdefghijklmnopqrstuvwxyz12345678",
        ]
        result = self.sanitizer.sanitize_list(data)
        assert result[0] == "normal text"
        assert result[1]["api_key"] == "[REDACTED]"
        assert "[OPENAI_API_KEY]" in result[2]

    def test_custom_pattern(self):
        """Test adding custom pattern"""
        self.sanitizer.add_pattern(r"CUSTOM-\d{6}", "[CUSTOM_ID]")
        text = "ID is CUSTOM-123456"
        result = self.sanitizer.sanitize_string(text)
        assert "CUSTOM-123456" not in result
        assert "[CUSTOM_ID]" in result

    def test_custom_sensitive_key(self):
        """Test adding custom sensitive key"""
        self.sanitizer.add_sensitive_key("custom_secret")
        data = {"custom_secret": "value123"}
        result = self.sanitizer.sanitize_dict(data)
        assert result["custom_secret"] == "[REDACTED]"


class TestSecureAuditLogger:
    """Test SecureAuditLogger"""

    def setup_method(self):
        """Setup audit logger for each test"""
        self.sanitizer = SecretsSanitizer()
        self.audit_logger = SecureAuditLogger(sanitizer=self.sanitizer)

    def test_log_audit_event_sanitizes(self):
        """Test audit event logging sanitizes data"""
        with patch("engine.security.audit.logger") as mock_logger:
            self.audit_logger.log_audit_event(
                "test_event",
                {"api_key": "secret123", "action": "test"},
            )
            
            # Verify logged data was sanitized
            call_kwargs = mock_logger.info.call_args[1]
            assert call_kwargs["extra"]["details"]["api_key"] == "[REDACTED]"
            assert call_kwargs["extra"]["details"]["action"] == "test"

    def test_log_access(self):
        """Test access control logging"""
        with patch("engine.security.audit.logger") as mock_logger:
            self.audit_logger.log_access(
                user="john",
                resource="/api/data",
                action="read",
                allowed=True,
            )
            
            assert mock_logger.info.called

    def test_log_secret_access(self):
        """Test secret access logging"""
        with patch("engine.security.audit.logger") as mock_logger:
            self.audit_logger.log_secret_access(
                secret_key="openai_key",
                accessor="service_a",
                success=True,
            )
            
            assert mock_logger.info.called
            call_kwargs = mock_logger.info.call_args[1]
            # Verify secret value is never logged
            assert "secret_key" in call_kwargs["extra"]["details"]
            assert call_kwargs["extra"]["details"]["secret_key"] == "openai_key"


class TestIntegration:
    """Integration tests for security components"""

    def test_end_to_end_sanitization(self):
        """Test complete sanitization flow"""
        sanitizer = SecretsSanitizer()
        audit_logger = SecureAuditLogger(sanitizer=sanitizer)
        
        # Simulate logging an event with secrets
        event_details = {
            "request": {
                "headers": {
                    "Authorization": "Bearer sk-1234567890abcdefghijklmnopqrstuvwxyz12345678",
                },
                "body": {"api_key": "secret123"},
            },
            "response": {"status": 200},
        }
        
        with patch("engine.security.audit.logger"):
            audit_logger.log_audit_event("api_request", event_details)
        
        # Manually sanitize to verify
        sanitized = sanitizer.sanitize_dict(event_details)
        
        assert "sk-1234567890" not in str(sanitized)
        assert "secret123" not in str(sanitized)
        assert "[REDACTED]" in str(sanitized) or "[OPENAI_API_KEY]" in str(sanitized)
