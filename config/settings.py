"""
ELEANOR V8 - Centralized Configuration Management
Secure, type-safe configuration with environment variable support
"""

from typing import Optional, List, Any
from pydantic import Field, field_validator, SecretStr, ValidationInfo
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseConfig(BaseSettings):
    """Database configuration with connection pooling"""

    model_config = SettingsConfigDict(
        env_prefix="DB_", env_file=".env", case_sensitive=False, extra="ignore"
    )

    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, description="Database port")
    name: str = Field(default="eleanor_v8", description="Database name")
    user: str = Field(default="eleanor", description="Database user")
    password: SecretStr = Field(default=SecretStr(""), description="Database password")

    # Connection pooling
    pool_size: int = Field(default=10, ge=1, le=100, description="Connection pool size")
    max_overflow: int = Field(default=20, ge=0, le=100, description="Max overflow connections")
    pool_timeout: int = Field(default=30, ge=1, description="Pool timeout in seconds")
    pool_recycle: int = Field(default=3600, ge=60, description="Connection recycle time")

    @property
    def url(self) -> str:
        """Construct database URL"""
        pwd = self.password.get_secret_value() if self.password else ""
        return f"postgresql+asyncpg://{self.user}:{pwd}@{self.host}:{self.port}/{self.name}"


class WeaviateConfig(BaseSettings):
    """Vector database configuration"""

    model_config = SettingsConfigDict(
        env_prefix="WEAVIATE_", env_file=".env", case_sensitive=False, extra="ignore"
    )

    url: str = Field(default="http://localhost:8080", description="Weaviate URL")
    api_key: Optional[SecretStr] = Field(default=None, description="Weaviate API key")
    timeout: int = Field(default=30, ge=5, description="Request timeout")
    batch_size: int = Field(default=100, ge=1, le=1000, description="Batch import size")


class SecurityConfig(BaseSettings):
    """Security and authentication configuration"""

    model_config = SettingsConfigDict(
        env_prefix="SECURITY_", env_file=".env", case_sensitive=False, extra="ignore"
    )

    jwt_secret: SecretStr = Field(..., description="JWT signing secret")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expiration: int = Field(default=3600, ge=60, le=86400, description="JWT expiration in seconds")

    # API security
    api_key_header: str = Field(default="X-API-Key", description="API key header name")
    cors_origins: List[str] = Field(default=["*"], description="CORS allowed origins")

    # Rate limiting
    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_requests: int = Field(default=100, ge=1, le=10000, description="Requests per window")
    rate_limit_window: int = Field(default=60, ge=1, le=3600, description="Rate limit window in seconds")

    @field_validator("jwt_secret")
    @classmethod
    def validate_jwt_secret(cls, v: SecretStr, info: ValidationInfo) -> SecretStr:
        """Validate JWT secret strength."""
        secret_value = v.get_secret_value()
        # Get environment from environment variable (most reliable)
        # SecurityConfig is instantiated before EleanorSettings, so we can't access parent
        import os
        env = os.getenv("ENVIRONMENT", "development")
        
        # In production, enforce minimum secret length
        if env == "production":
            if len(secret_value) < 32:
                raise ValueError("JWT secret must be at least 32 characters in production")
            if secret_value == "dev-secret" or secret_value == "changeme":
                raise ValueError("JWT secret cannot use default values in production")
        
        return v

    @field_validator("jwt_algorithm")
    @classmethod
    def validate_jwt_algorithm(cls, v: str) -> str:
        """Validate JWT algorithm is secure."""
        allowed_algorithms = {"HS256", "HS384", "HS512", "RS256", "RS384", "RS512", "ES256", "ES384", "ES512"}
        if v.upper() not in allowed_algorithms:
            raise ValueError(f"JWT algorithm must be one of {allowed_algorithms}")
        return v.upper()

    @field_validator("cors_origins")
    @classmethod
    def validate_cors_origins(cls, v: List[str], info: ValidationInfo) -> List[str]:
        """Validate CORS origins configuration."""
        # Get environment from environment variable (most reliable)
        # SecurityConfig is instantiated before EleanorSettings, so we can't access parent
        import os
        env = os.getenv("ENVIRONMENT", "development")
        
        # In production, disallow wildcard
        if env == "production":
            if "*" in v or "" in v:
                raise ValueError("CORS origins cannot include wildcard (*) or empty string in production")
            if not v:
                raise ValueError("CORS origins must be explicitly configured in production")
        
        return v


class EngineConfig(BaseSettings):
    """ELEANOR Engine configuration"""

    model_config = SettingsConfigDict(
        env_prefix="ENGINE_", env_file=".env", case_sensitive=False, extra="ignore"
    )

    # Core settings
    detail_level: int = Field(default=2, ge=1, le=3, description="Output detail level")
    max_concurrency: int = Field(default=6, ge=1, le=50, description="Max concurrent critics")
    timeout_seconds: float = Field(default=10.0, ge=1.0, description="Operation timeout")

    # Feature flags
    enable_reflection: bool = Field(default=True, description="Enable reflection analysis")
    enable_drift_check: bool = Field(default=True, description="Enable drift detection")
    enable_precedent_analysis: bool = Field(default=True, description="Enable precedent analysis")

    # Evidence recording
    jsonl_evidence_path: Optional[str] = Field(
        default="evidence.jsonl", description="Evidence log path"
    )
    evidence_batch_size: int = Field(default=100, ge=1, description="Evidence batch size")
    evidence_flush_interval: int = Field(
        default=30, ge=1, description="Evidence flush interval (seconds)"
    )

    # Circuit breaker settings
    circuit_breaker_enabled: bool = Field(default=True, description="Enable circuit breakers")
    circuit_breaker_threshold: int = Field(default=5, ge=1, description="Failure threshold")
    circuit_breaker_timeout: int = Field(
        default=60, ge=10, description="Circuit breaker timeout (seconds)"
    )

    # Retry settings
    retry_enabled: bool = Field(default=True, description="Enable retries")
    retry_max_attempts: int = Field(default=3, ge=1, le=10, description="Max retry attempts")
    retry_backoff_factor: float = Field(
        default=2.0, ge=1.0, description="Exponential backoff factor"
    )


class MonitoringConfig(BaseSettings):
    """Monitoring and observability configuration"""

    model_config = SettingsConfigDict(
        env_prefix="MONITORING_", env_file=".env", case_sensitive=False, extra="ignore"
    )

    # Prometheus
    prometheus_enabled: bool = Field(default=True, description="Enable Prometheus metrics")
    prometheus_port: int = Field(default=9090, ge=1024, le=65535, description="Prometheus port")

    # OpenTelemetry
    otel_enabled: bool = Field(default=False, description="Enable OpenTelemetry")
    otel_endpoint: Optional[str] = Field(default=None, description="OTEL collector endpoint")

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format (json|text)")
    log_file: Optional[str] = Field(default=None, description="Log file path")


class EleanorSettings(BaseSettings):
    """Main ELEANOR V8 settings"""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # Environment
    environment: str = Field(
        default="development", description="Environment (development|staging|production)"
    )
    debug: bool = Field(default=False, description="Debug mode")

    # API settings
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, ge=1024, le=65535, description="API port")
    api_workers: int = Field(default=4, ge=1, description="API workers")

    # Component configs
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    weaviate: WeaviateConfig = Field(default_factory=WeaviateConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    engine: EngineConfig = Field(default_factory=EngineConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

    @classmethod
    def _get_environment(cls) -> str:
        """Get environment from environment variable or default."""
        import os
        return os.getenv("ENVIRONMENT", "development")

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        allowed = ["development", "staging", "production"]
        if v.lower() not in allowed:
            raise ValueError(f"Environment must be one of {allowed}")
        return v.lower()

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        return self.environment == "development"


# Singleton settings instance
_settings: Optional[EleanorSettings] = None


def get_settings(validate: bool = True) -> EleanorSettings:
    """
    Get application settings (singleton).
    
    Args:
        validate: Whether to validate settings on load
    
    Returns:
        EleanorSettings instance
    
    Raises:
        ValidationError: If validation fails and validate=True
    """
    global _settings
    if _settings is None:
        _settings = EleanorSettings()
        if validate:
            from config.validation import ConfigValidator
            try:
                ConfigValidator.validate_and_raise(_settings)
            except ValidationError as exc:
                import logging
                logger = logging.getLogger(__name__)
                logger.error("settings_validation_failed", extra={"errors": str(exc)})
                raise
    return _settings


def reload_settings(validate: bool = True) -> EleanorSettings:
    """
    Reload settings (useful for testing).
    
    Args:
        validate: Whether to validate settings on reload
    
    Returns:
        EleanorSettings instance
    
    Raises:
        ValidationError: If validation fails and validate=True
    """
    global _settings
    _settings = EleanorSettings()
    if validate:
        from config.validation import ConfigValidator
        try:
            ConfigValidator.validate_and_raise(_settings)
        except ValidationError as exc:
            import logging
            logger = logging.getLogger(__name__)
            logger.error("settings_validation_failed", extra={"errors": str(exc)})
            raise
    return _settings


__all__ = [
    "EleanorSettings",
    "DatabaseConfig",
    "WeaviateConfig",
    "SecurityConfig",
    "EngineConfig",
    "MonitoringConfig",
    "get_settings",
    "reload_settings",
]
