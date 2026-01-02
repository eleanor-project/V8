"""
ELEANOR V8 — Settings Definition

Comprehensive configuration with Pydantic validation.
All settings can be overridden via environment variables with ELEANOR_ prefix.
"""

import os
from typing import Optional, Dict, Any, List
from pathlib import Path
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    ConfigDict,
)
from pydantic_settings import BaseSettings, SettingsConfigDict
import warnings


class LLMConfig(BaseModel):
    """LLM provider configuration."""
    
    provider: str = Field(
        default="ollama",
        description="LLM provider: ollama, openai, anthropic, cohere"
    )
    model_name: str = Field(
        default="llama3.2:3b",
        description="Default model name"
    )
    base_url: Optional[str] = Field(
        default="http://localhost:11434",
        description="Base URL for LLM provider (Ollama)"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for cloud providers"
    )
    timeout: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Request timeout in seconds"
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )


class RouterConfig(BaseModel):
    """Router configuration for model selection."""
    
    health_check_interval: int = Field(
        default=60,
        ge=10,
        description="Health check interval in seconds"
    )
    fallback_model: str = Field(
        default="llama3.2:3b",
        description="Fallback model when primary fails"
    )
    enable_cost_optimization: bool = Field(
        default=True,
        description="Enable cost-aware routing"
    )
    max_latency_ms: float = Field(
        default=5000.0,
        ge=100.0,
        description="Maximum acceptable latency for routing decisions"
    )


class PrecedentConfig(BaseModel):
    """Precedent store configuration."""
    
    backend: str = Field(
        default="none",
        description="Precedent backend: none, chroma, qdrant, pinecone"
    )
    connection_string: Optional[str] = Field(
        default=None,
        description="Connection string for precedent database"
    )
    cache_ttl: int = Field(
        default=3600,
        ge=60,
        description="Cache TTL in seconds"
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum precedent cases to retrieve"
    )
    similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for precedent matching"
    )


class EvidenceConfig(BaseModel):
    """Evidence recording configuration."""
    
    enabled: bool = Field(
        default=True,
        description="Enable evidence recording"
    )
    jsonl_path: str = Field(
        default="evidence.jsonl",
        description="Path to JSONL evidence file"
    )
    buffer_size: int = Field(
        default=1000,
        ge=10,
        le=10000,
        description="Buffer size before flushing to disk"
    )
    flush_interval: float = Field(
        default=5.0,
        ge=1.0,
        description="Auto-flush interval in seconds"
    )
    fail_on_error: bool = Field(
        default=False,
        description="Fail pipeline if evidence recording fails"
    )
    sanitize_secrets: bool = Field(
        default=True,
        description="Sanitize credentials from evidence"
    )


class PerformanceConfig(BaseModel):
    """Performance and concurrency configuration."""
    
    max_concurrency: int = Field(
        default=6,
        ge=1,
        le=50,
        description="Maximum concurrent critic evaluations"
    )
    timeout_seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=300.0,
        description="Timeout for individual critics"
    )
    enable_adaptive_concurrency: bool = Field(
        default=False,
        description="Enable adaptive concurrency control"
    )
    target_latency_ms: float = Field(
        default=500.0,
        ge=100.0,
        description="Target latency for adaptive concurrency"
    )
    
    @field_validator('max_concurrency')
    @classmethod
    def warn_high_concurrency(cls, v: int) -> int:
        if v > 20:
            warnings.warn(
                f"High concurrency ({v}) may cause resource exhaustion. "
                "Consider enabling adaptive concurrency or using circuit breakers.",
                UserWarning
            )
        return v


class CacheConfig(BaseModel):
    """Caching configuration."""
    
    enabled: bool = Field(
        default=True,
        description="Enable caching"
    )
    redis_url: Optional[str] = Field(
        default=None,
        description="Redis URL for distributed cache"
    )
    precedent_ttl: int = Field(
        default=3600,
        ge=60,
        description="Precedent cache TTL in seconds"
    )
    embeddings_ttl: int = Field(
        default=7200,
        ge=60,
        description="Embeddings cache TTL in seconds"
    )
    critics_ttl: int = Field(
        default=1800,
        ge=60,
        description="Critics cache TTL in seconds"
    )
    max_memory_mb: int = Field(
        default=500,
        ge=50,
        le=5000,
        description="Maximum in-memory cache size in MB"
    )


class SecurityConfig(BaseModel):
    """Security configuration."""
    
    max_input_size_bytes: int = Field(
        default=100_000,
        ge=1000,
        le=10_000_000,
        description="Maximum input size"
    )
    max_context_depth: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum context nesting depth"
    )
    enable_prompt_injection_detection: bool = Field(
        default=True,
        description="Enable prompt injection detection"
    )
    secret_provider: str = Field(
        default="env",
        description="Secret provider: env, aws, vault"
    )
    secrets_cache_ttl: int = Field(
        default=300,
        ge=60,
        description="Secrets cache TTL in seconds"
    )


class ObservabilityConfig(BaseModel):
    """Observability and logging configuration."""
    
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR"
    )
    enable_structured_logging: bool = Field(
        default=True,
        description="Enable structured (JSON) logging"
    )
    enable_tracing: bool = Field(
        default=False,
        description="Enable distributed tracing"
    )
    otel_endpoint: Optional[str] = Field(
        default=None,
        description="OpenTelemetry collector endpoint"
    )
    jaeger_endpoint: Optional[str] = Field(
        default=None,
        description="Jaeger endpoint for trace visualization"
    )
    metrics_port: int = Field(
        default=9090,
        ge=1024,
        le=65535,
        description="Prometheus metrics port"
    )
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(
                f"Invalid log_level: {v}. Must be one of {valid_levels}"
            )
        return v_upper


class ResilienceConfig(BaseModel):
    """Resilience and fault tolerance configuration."""
    
    enable_circuit_breakers: bool = Field(
        default=True,
        description="Enable circuit breakers for external dependencies"
    )
    circuit_breaker_threshold: int = Field(
        default=5,
        ge=1,
        description="Failure threshold before opening circuit"
    )
    circuit_breaker_timeout: int = Field(
        default=60,
        ge=10,
        description="Circuit breaker recovery timeout in seconds"
    )
    enable_graceful_degradation: bool = Field(
        default=True,
        description="Enable graceful degradation on component failures"
    )
    max_retry_attempts: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts for transient failures"
    )


class EleanorSettings(BaseSettings):
    """
    Main ELEANOR configuration with hierarchical overrides.
    
    Configuration precedence (highest to lowest):
    1. Command-line arguments (runtime)
    2. Environment variables (ELEANOR_*)
    3. Environment files (.env, .env.{ENV})
    4. YAML configuration (legacy)
    5. Default values
    """
    
    model_config = SettingsConfigDict(
        env_prefix="ELEANOR_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",  # Ignore unknown fields
    )
    
    # Environment
    environment: str = Field(
        default="development",
        description="Environment: development, staging, production"
    )
    
    # Engine Configuration
    detail_level: int = Field(
        default=2,
        ge=1,
        le=3,
        description="Output detail level (1=minimal, 2=standard, 3=forensic)"
    )
    
    # Feature Flags
    enable_reflection: bool = Field(
        default=True,
        description="Enable reflection and uncertainty analysis"
    )
    enable_drift_check: bool = Field(
        default=True,
        description="Enable drift detection"
    )
    enable_precedent_analysis: bool = Field(
        default=True,
        description="Enable precedent-based reasoning"
    )
    
    # Component Configurations
    llm: LLMConfig = Field(default_factory=LLMConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    precedent: PrecedentConfig = Field(default_factory=PrecedentConfig)
    evidence: EvidenceConfig = Field(default_factory=EvidenceConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    resilience: ResilienceConfig = Field(default_factory=ResilienceConfig)
    
    # Critics Configuration (dynamic loading)
    enabled_critics: List[str] = Field(
        default_factory=lambda: [
            "rights",
            "autonomy",
            "fairness",
            "truth",
            "risk",
            "operations",
        ],
        description="List of enabled critics"
    )
    
    @model_validator(mode='after')
    def validate_configuration(self) -> 'EleanorSettings':
        """Cross-field validation."""
        
        # Warn if production without proper backends
        if self.environment == "production":
            if self.precedent.backend == "none":
                warnings.warn(
                    "Production environment without precedent backend configured. "
                    "Set ELEANOR_PRECEDENT__BACKEND for production use.",
                    UserWarning
                )
            
            if self.security.secret_provider == "env":
                warnings.warn(
                    "Production environment using environment variables for secrets. "
                    "Consider AWS Secrets Manager or HashiCorp Vault.",
                    UserWarning
                )
            
            if not self.observability.enable_tracing:
                warnings.warn(
                    "Production environment without distributed tracing. "
                    "Enable tracing for better observability.",
                    UserWarning
                )
        
        # Validate cache configuration
        if self.cache.enabled and self.cache.redis_url is None:
            warnings.warn(
                "Caching enabled without Redis URL. Using in-memory cache only.",
                UserWarning
            )
        
        # Validate tracing endpoints
        if self.observability.enable_tracing:
            if not self.observability.otel_endpoint and not self.observability.jaeger_endpoint:
                warnings.warn(
                    "Tracing enabled but no endpoint configured. "
                    "Set ELEANOR_OBSERVABILITY__OTEL_ENDPOINT or JAEGER_ENDPOINT.",
                    UserWarning
                )
        
        return self
    
    def to_legacy_engine_config(self) -> Dict[str, Any]:
        """Convert to legacy EngineConfig format for backward compatibility."""
        return {
            "detail_level": self.detail_level,
            "max_concurrency": self.performance.max_concurrency,
            "timeout_seconds": self.performance.timeout_seconds,
            "enable_reflection": self.enable_reflection,
            "enable_drift_check": self.enable_drift_check,
            "enable_precedent_analysis": self.enable_precedent_analysis,
            "jsonl_evidence_path": self.evidence.jsonl_path,
        }


# Global settings instance
_settings: Optional[EleanorSettings] = None


def get_settings(
    env_file: Optional[str] = None,
    reload: bool = False
) -> EleanorSettings:
    """
    Get or create global settings instance.
    
    Args:
        env_file: Optional environment file to load
        reload: Force reload settings
    
    Returns:
        EleanorSettings instance
    """
    global _settings
    
    if _settings is None or reload:
        # Determine environment file
        if env_file is None:
            env = os.getenv("ENV", "development")
            env_file = f".env.{env}"
            
            # Fallback to .env if specific file doesn't exist
            if not Path(env_file).exists():
                env_file = ".env"
        
        # Load settings
        if Path(env_file).exists():
            _settings = EleanorSettings(_env_file=env_file)
        else:
            _settings = EleanorSettings()
    
    return _settings


__all__ = [
    "EleanorSettings",
    "get_settings",
    "LLMConfig",
    "RouterConfig",
    "PrecedentConfig",
    "EvidenceConfig",
    "PerformanceConfig",
    "CacheConfig",
    "SecurityConfig",
    "ObservabilityConfig",
    "ResilienceConfig",
]
