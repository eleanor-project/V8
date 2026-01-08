"""
ELEANOR V8 — Comprehensive Configuration Validation
---------------------------------------------------

Enhanced configuration validation with security checks.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from pydantic import ValidationError, field_validator
from .settings import EleanorSettings, SecurityConfig, DatabaseConfig, EngineConfig

logger = logging.getLogger(__name__)


class ConfigValidator:
    """
    Comprehensive configuration validator.
    
    Validates:
    - Security settings
    - Database configuration
    - Engine settings
    - Environment-specific requirements
    - Cross-field dependencies
    """
    
    @staticmethod
    def validate_settings(settings: EleanorSettings) -> List[str]:
        """
        Validate all settings and return list of issues.
        
        Args:
            settings: Settings to validate
        
        Returns:
            List of validation issue messages (empty if valid)
        """
        issues: List[str] = []
        
        # Environment-specific validation
        if settings.is_production:
            issues.extend(ConfigValidator._validate_production(settings))
        elif settings.is_development:
            issues.extend(ConfigValidator._validate_development(settings))
        
        # Cross-field validation
        issues.extend(ConfigValidator._validate_cross_fields(settings))
        
        # Security validation
        issues.extend(ConfigValidator._validate_security(settings.security))
        
        # Database validation
        issues.extend(ConfigValidator._validate_database(settings.database))
        
        # Engine validation
        issues.extend(ConfigValidator._validate_engine(settings.engine))
        
        return issues
    
    @staticmethod
    def _validate_production(settings: EleanorSettings) -> List[str]:
        """Validate production-specific requirements."""
        issues: List[str] = []
        
        # Debug mode must be disabled
        if settings.debug:
            issues.append("Debug mode must be disabled in production")
        
        # Security checks
        if settings.security.jwt_secret.get_secret_value() in ("dev-secret", "changeme", ""):
            issues.append("JWT secret must be set to a secure value in production")
        
        if "*" in settings.security.cors_origins:
            issues.append("CORS origins cannot include wildcard (*) in production")
        
        if not settings.security.cors_origins:
            issues.append("CORS origins must be explicitly configured in production")
        
        # Database checks
        if settings.database.password.get_secret_value() == "":
            issues.append("Database password must be set in production")
        
        # Monitoring checks
        if not settings.monitoring.prometheus_enabled:
            issues.append("Prometheus metrics should be enabled in production")
        
        return issues
    
    @staticmethod
    def _validate_development(settings: EleanorSettings) -> List[str]:
        """Validate development-specific requirements."""
        issues: List[str] = []
        
        # Warn about insecure defaults
        if settings.security.jwt_secret.get_secret_value() == "dev-secret":
            logger.warning("Using default JWT secret in development - change for production")
        
        return issues
    
    @staticmethod
    def _validate_cross_fields(settings: EleanorSettings) -> List[str]:
        """Validate cross-field dependencies."""
        issues: List[str] = []
        
        # API port should not conflict with monitoring port
        if settings.api_port == settings.monitoring.prometheus_port:
            issues.append(
                f"API port ({settings.api_port}) conflicts with Prometheus port "
                f"({settings.monitoring.prometheus_port})"
            )
        
        # Engine timeout should be reasonable
        if settings.engine.timeout_seconds < 1.0:
            issues.append("Engine timeout must be at least 1.0 seconds")
        
        if settings.engine.timeout_seconds > 300.0:
            issues.append("Engine timeout should not exceed 300 seconds (5 minutes)")
        
        # Concurrency limits
        if settings.engine.max_concurrency > 50:
            issues.append("Max concurrency should not exceed 50 to prevent resource exhaustion")
        
        return issues
    
    @staticmethod
    def _validate_security(security: SecurityConfig) -> List[str]:
        """Validate security configuration."""
        issues: List[str] = []
        
        # JWT secret strength
        secret_value = security.jwt_secret.get_secret_value()
        if len(secret_value) < 16:
            issues.append("JWT secret should be at least 16 characters")
        
        # Rate limiting
        if security.rate_limit_enabled:
            if security.rate_limit_requests < 1:
                issues.append("Rate limit requests must be at least 1")
            if security.rate_limit_window < 1:
                issues.append("Rate limit window must be at least 1 second")
        
        return issues
    
    @staticmethod
    def _validate_database(database: DatabaseConfig) -> List[str]:
        """Validate database configuration."""
        issues: List[str] = []
        
        # Connection pool
        if database.pool_size < 1:
            issues.append("Database pool size must be at least 1")
        
        if database.pool_size + database.max_overflow > 100:
            issues.append(
                f"Total database connections ({database.pool_size + database.max_overflow}) "
                "should not exceed 100"
            )
        
        # Timeout
        if database.pool_timeout < 1:
            issues.append("Database pool timeout must be at least 1 second")
        
        # Connection recycle
        if database.pool_recycle < 60:
            issues.append("Database connection recycle time must be at least 60 seconds")
        
        return issues
    
    @staticmethod
    def _validate_engine(engine: EngineConfig) -> List[str]:
        """Validate engine configuration."""
        issues: List[str] = []
        
        # Detail level
        if engine.detail_level < 1 or engine.detail_level > 3:
            issues.append("Engine detail level must be between 1 and 3")
        
        # Concurrency
        if engine.max_concurrency < 1:
            issues.append("Max concurrency must be at least 1")
        
        # Timeout
        if engine.timeout_seconds < 1.0:
            issues.append("Engine timeout must be at least 1.0 seconds")
        
        # Evidence recording
        if engine.evidence_batch_size < 1:
            issues.append("Evidence batch size must be at least 1")
        
        if engine.evidence_flush_interval < 1:
            issues.append("Evidence flush interval must be at least 1 second")
        
        # Circuit breaker
        if engine.circuit_breaker_enabled:
            if engine.circuit_breaker_threshold < 1:
                issues.append("Circuit breaker threshold must be at least 1")
            if engine.circuit_breaker_timeout < 10:
                issues.append("Circuit breaker timeout must be at least 10 seconds")
        
        # Retry
        if engine.retry_enabled:
            if engine.retry_max_attempts < 1:
                issues.append("Retry max attempts must be at least 1")
            if engine.retry_backoff_factor < 1.0:
                issues.append("Retry backoff factor must be at least 1.0")
        
        return issues
    
    @staticmethod
    def validate_and_raise(settings: EleanorSettings) -> None:
        """
        Validate settings and raise ValidationError if issues found.
        
        Args:
            settings: Settings to validate
        
        Raises:
            ValidationError: If validation issues are found
        """
        issues = ConfigValidator.validate_settings(settings)
        if issues:
            error_message = "Configuration validation failed:\n" + "\n".join(f"  - {issue}" for issue in issues)
            raise ValidationError(error_message)


__all__ = ["ConfigValidator"]
