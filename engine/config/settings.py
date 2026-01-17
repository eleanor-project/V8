"""
ELEANOR V8 — Settings Definition

Comprehensive configuration with Pydantic validation.
All settings can be overridden via environment variables with ELEANOR_ prefix.
"""

import os
from typing import Optional, Dict, Any, List, Callable, cast
from pathlib import Path
from pydantic import (
    BaseModel,
    Field,
    AliasChoices,
    field_validator,
    model_validator,
    ConfigDict,
)
from pydantic_settings import (  # type: ignore[import-not-found]
    BaseSettings,
    SettingsConfigDict,
    PydanticBaseSettingsSource,
)
import warnings

from engine.config.overlay import load_overlay_payload


def _resolve_environment() -> str:
    return (
        os.getenv("ELEANOR_ENVIRONMENT")
        or os.getenv("ELEANOR_ENV")
        or os.getenv("ENV")
        or "development"
    )


def _load_yaml_config(path: str) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required for YAML config support.") from exc

    yaml_path = Path(path)
    if not yaml_path.exists() or not yaml_path.is_file():
        return {}

    with yaml_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _normalize_yaml_config(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}

    if "environment" in payload:
        normalized["environment"] = payload["environment"]
    if "detail_level" in payload:
        normalized["detail_level"] = payload["detail_level"]
    if "enable_reflection" in payload:
        normalized["enable_reflection"] = payload["enable_reflection"]
    if "enable_drift_check" in payload:
        normalized["enable_drift_check"] = payload["enable_drift_check"]
    if "enable_precedent_analysis" in payload:
        normalized["enable_precedent_analysis"] = payload["enable_precedent_analysis"]

    for key in (
        "llm",
        "router",
        "precedent",
        "evidence",
        "performance",
        "cache",
        "security",
        "observability",
        "resilience",
        "resource_management",
        "gpu",
    ):
        if key in payload:
            normalized[key] = payload[key]

    if isinstance(payload.get("gpu"), dict):
        normalized["gpu"] = payload["gpu"]

    def _map_gpu_section(section_key: str, expected_keys: tuple[str, ...]) -> None:
        section = payload.get(section_key)
        if not isinstance(section, dict):
            return
        if any(key in section for key in expected_keys):
            normalized.setdefault("gpu", {})[section_key] = section

    _map_gpu_section("ollama", ("gpu_layers", "enable_gpu", "num_gpu"))
    _map_gpu_section("embeddings", ("cache_on_gpu", "max_cache_size_gb", "device"))
    _map_gpu_section("critics", ("gpu_batching", "use_gpu"))
    _map_gpu_section("precedent", ("gpu_similarity_search", "cache_embeddings_on_gpu"))

    governance_raw = payload.get("governance")
    governance: Dict[str, Any] = (
        governance_raw if isinstance(governance_raw, dict) else {}
    )
    if "constitutional_config_path" in governance:
        normalized["constitutional_config_path"] = governance["constitutional_config_path"]

    storage_raw = payload.get("storage")
    storage: Dict[str, Any] = storage_raw if isinstance(storage_raw, dict) else {}
    if "evidence_path" in storage:
        normalized.setdefault("evidence", {})["jsonl_path"] = storage["evidence_path"]
    if "replay_log_path" in storage:
        normalized["replay_log_path"] = storage["replay_log_path"]

    precedent_raw = payload.get("precedent")
    precedent: Dict[str, Any] = (
        precedent_raw if isinstance(precedent_raw, dict) else {}
    )
    if "backend" in precedent:
        normalized.setdefault("precedent", {})["backend"] = precedent["backend"]

    logging_raw = payload.get("logging")
    logging_cfg: Dict[str, Any] = (
        logging_raw if isinstance(logging_raw, dict) else {}
    )
    if "level" in logging_cfg:
        normalized.setdefault("observability", {})["log_level"] = logging_cfg["level"]

    return normalized


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    provider: str = Field(
        default="ollama", description="LLM provider: ollama, openai, anthropic, cohere"
    )
    model_name: str = Field(default="llama3.2:3b", description="Default model name")
    base_url: Optional[str] = Field(
        default="http://localhost:11434", description="Base URL for LLM provider (Ollama)"
    )
    api_key: Optional[str] = Field(default=None, description="API key for cloud providers")
    timeout: float = Field(default=30.0, ge=1.0, le=300.0, description="Request timeout in seconds")
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")


class RouterConfig(BaseModel):
    """Router configuration for model selection."""

    health_check_interval: int = Field(
        default=60, ge=10, description="Health check interval in seconds"
    )
    fallback_model: str = Field(
        default="llama3.2:3b", description="Fallback model when primary fails"
    )
    enable_cost_optimization: bool = Field(default=True, description="Enable cost-aware routing")
    max_latency_ms: float = Field(
        default=5000.0, ge=100.0, description="Maximum acceptable latency for routing decisions"
    )


class PrecedentConfig(BaseModel):
    """Precedent store configuration."""

    backend: str = Field(
        default="none", description="Precedent backend: none, chroma, qdrant, pinecone"
    )
    connection_string: Optional[str] = Field(
        default=None, description="Connection string for precedent database"
    )
    cache_ttl: int = Field(default=3600, ge=60, description="Cache TTL in seconds")
    max_results: int = Field(
        default=5, ge=1, le=50, description="Maximum precedent cases to retrieve"
    )
    similarity_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Minimum similarity score for precedent matching"
    )


class EvidenceConfig(BaseModel):
    """Evidence recording configuration."""

    enabled: bool = Field(default=True, description="Enable evidence recording")
    jsonl_path: str = Field(default="evidence.jsonl", description="Path to JSONL evidence file")
    buffer_size: int = Field(
        default=1000, ge=10, le=10000, description="Buffer size before flushing to disk"
    )
    flush_interval: float = Field(default=5.0, ge=1.0, description="Auto-flush interval in seconds")
    fail_on_error: bool = Field(
        default=False, description="Fail pipeline if evidence recording fails"
    )
    sanitize_secrets: bool = Field(default=True, description="Sanitize credentials from evidence")


class PerformanceConfig(BaseModel):
    """Performance and concurrency configuration."""

    max_concurrency: int = Field(
        default=6, ge=1, le=50, description="Maximum concurrent critic evaluations"
    )
    timeout_seconds: float = Field(
        default=10.0, ge=1.0, le=300.0, description="Timeout for individual critics"
    )
    enable_adaptive_concurrency: bool = Field(
        default=False, description="Enable adaptive concurrency control"
    )
    target_latency_ms: float = Field(
        default=500.0, ge=100.0, description="Target latency for adaptive concurrency"
    )

    @field_validator("max_concurrency")
    @classmethod
    def warn_high_concurrency(cls, v: int) -> int:
        if v > 20:
            warnings.warn(
                f"High concurrency ({v}) may cause resource exhaustion. "
                "Consider enabling adaptive concurrency or using circuit breakers.",
                UserWarning,
            )
        return v


class CacheConfig(BaseModel):
    """Caching configuration."""

    enabled: bool = Field(default=True, description="Enable caching")
    redis_url: Optional[str] = Field(default=None, description="Redis URL for distributed cache")
    precedent_ttl: int = Field(default=3600, ge=60, description="Precedent cache TTL in seconds")
    embeddings_ttl: int = Field(default=7200, ge=60, description="Embeddings cache TTL in seconds")
    router_ttl: int = Field(default=1800, ge=60, description="Router cache TTL in seconds")
    critics_ttl: int = Field(default=1800, ge=60, description="Critics cache TTL in seconds")
    detector_ttl: int = Field(default=600, ge=60, description="Detector cache TTL in seconds")
    max_memory_mb: int = Field(
        default=500, ge=50, le=5000, description="Maximum in-memory cache size in MB"
    )


class AWSSecretsConfig(BaseModel):
    """AWS Secrets Manager configuration."""

    region: str = Field(default="us-west-2", description="AWS region for Secrets Manager")
    secret_prefix: Optional[str] = Field(
        default="eleanor", description="Optional prefix for secret names"
    )


class VaultSecretsConfig(BaseModel):
    """HashiCorp Vault configuration."""

    address: Optional[str] = Field(
        default=None, description="Vault address (e.g., https://vault.example.com)"
    )
    token: Optional[str] = Field(
        default=None, description="Vault token (prefer env or sidecar injection)"
    )
    mount_path: str = Field(default="secret/eleanor", description="Vault mount path for secrets")


class AzureSecretsConfig(BaseModel):
    """Azure Key Vault configuration."""

    vault_url: Optional[str] = Field(
        default=None, description="Azure Key Vault URL (e.g., https://vault.vault.azure.net)"
    )
    secret_prefix: str = Field(
        default="", description="Optional prefix for secret names"
    )


class SecretsRotationConfig(BaseModel):
    """Secret rotation configuration."""

    enabled: bool = Field(default=False, description="Enable automatic rotation checks")
    check_interval: int = Field(default=3600, ge=60, description="Rotation check interval in seconds")
    advance_rotation_days: int = Field(default=7, ge=1, description="Rotate secrets expiring within N days")
    max_age_days: int = Field(default=90, ge=1, description="Rotate secrets older than N days")


class SecurityConfig(BaseModel):
    """Security configuration."""

    max_input_size_bytes: int = Field(
        default=100_000, ge=1000, le=10_000_000, description="Maximum input size"
    )
    max_context_depth: int = Field(
        default=5, ge=1, le=20, description="Maximum context nesting depth"
    )
    enable_prompt_injection_detection: bool = Field(
        default=True, description="Enable prompt injection detection"
    )
    secret_provider: str = Field(default="env", description="Secret provider: env, aws, vault, azure")
    secrets_cache_ttl: int = Field(default=300, ge=60, description="Secrets cache TTL in seconds")
    enable_secret_audit: bool = Field(default=True, description="Audit secret access events")
    aws: AWSSecretsConfig = Field(default_factory=AWSSecretsConfig)
    vault: VaultSecretsConfig = Field(default_factory=VaultSecretsConfig)
    azure: AzureSecretsConfig = Field(default_factory=AzureSecretsConfig)
    rotation: SecretsRotationConfig = Field(default_factory=SecretsRotationConfig)


class ObservabilityConfig(BaseModel):
    """Observability and logging configuration."""

    log_level: str = Field(default="INFO", description="Logging level: DEBUG, INFO, WARNING, ERROR")
    enable_structured_logging: bool = Field(
        default=True, description="Enable structured (JSON) logging"
    )
    enable_tracing: bool = Field(default=False, description="Enable distributed tracing")
    otel_endpoint: Optional[str] = Field(
        default=None, description="OpenTelemetry collector endpoint"
    )
    jaeger_endpoint: Optional[str] = Field(
        default=None, description="Jaeger endpoint for trace visualization"
    )
    metrics_port: int = Field(
        default=9090, ge=1024, le=65535, description="Prometheus metrics port"
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Invalid log_level: {v}. Must be one of {valid_levels}")
        return v_upper


class ResilienceConfig(BaseModel):
    """Resilience and fault tolerance configuration."""

    enable_circuit_breakers: bool = Field(
        default=True, description="Enable circuit breakers for external dependencies"
    )
    circuit_breaker_threshold: int = Field(
        default=5, ge=1, description="Failure threshold before opening circuit"
    )
    circuit_breaker_timeout: int = Field(
        default=60, ge=10, description="Circuit breaker recovery timeout in seconds"
    )
    enable_graceful_degradation: bool = Field(
        default=True, description="Enable graceful degradation on component failures"
    )
    max_retry_attempts: int = Field(
        default=3, ge=0, le=10, description="Maximum retry attempts for transient failures"
    )


class ShutdownConfig(BaseModel):
    """Graceful shutdown configuration."""

    graceful_timeout: float = Field(
        default=30.0, ge=1.0, description="Maximum seconds to wait for cleanup"
    )


class ResourceMemoryConfig(BaseModel):
    """Memory pressure monitoring configuration."""

    warning_threshold: float = Field(
        default=0.75, ge=0.1, le=0.99, description="Warn when memory exceeds this fraction"
    )
    critical_threshold: float = Field(
        default=0.90, ge=0.1, le=0.99, description="Critical when memory exceeds this fraction"
    )
    check_interval: float = Field(
        default=30.0, ge=1.0, description="Seconds between memory checks"
    )


class ResourceConnectionPoolConfig(BaseModel):
    """Connection pool settings for external resources."""

    http_pool_size: int = Field(default=100, ge=1, description="Max HTTP connections")
    http_per_host_limit: int = Field(
        default=10, ge=1, description="Max HTTP connections per host"
    )
    http_keepalive_timeout: float = Field(
        default=30.0, ge=1.0, description="HTTP keepalive timeout in seconds"
    )

    database_url: Optional[str] = Field(
        default=None, description="Database DSN for asyncpg pool"
    )
    db_pool_min: int = Field(default=2, ge=0, description="Min DB pool size")
    db_pool_max: int = Field(default=20, ge=1, description="Max DB pool size")
    db_timeout: float = Field(default=10.0, ge=1.0, description="DB command timeout")

    redis_url: Optional[str] = Field(default=None, description="Redis URL for pool")
    redis_pool_size: int = Field(default=50, ge=1, description="Max Redis connections")


class ResourceLimitsConfig(BaseModel):
    """Hard resource limits for enforcement."""

    max_memory_gb: Optional[float] = Field(
        default=None, ge=0.1, description="Max process memory before violation"
    )
    max_open_files: Optional[int] = Field(
        default=None, ge=1, description="Max open file descriptors"
    )
    max_threads: Optional[int] = Field(
        default=None, ge=1, description="Max active threads"
    )
    max_coroutines: Optional[int] = Field(
        default=None, ge=1, description="Max active asyncio tasks"
    )


class EvidenceBufferConfig(BaseModel):
    """Evidence buffer limits."""

    max_buffer_size: int = Field(default=1000, ge=1, description="Max evidence records")
    forensic_output_size: int = Field(
        default=200, ge=1, description="Max evidence records in forensic output"
    )


class ResourceHealthConfig(BaseModel):
    """Health check thresholds for system resources."""

    disk_warning_threshold: float = Field(
        default=0.90, ge=0.1, le=0.99, description="Disk usage warning threshold"
    )
    disk_critical_threshold: float = Field(
        default=0.95, ge=0.1, le=0.99, description="Disk usage critical threshold"
    )


class ResourceManagementConfig(BaseModel):
    """Resource lifecycle configuration."""

    shutdown: ShutdownConfig = Field(default_factory=ShutdownConfig)
    memory: ResourceMemoryConfig = Field(default_factory=ResourceMemoryConfig)
    connections: ResourceConnectionPoolConfig = Field(default_factory=ResourceConnectionPoolConfig)
    limits: ResourceLimitsConfig = Field(default_factory=ResourceLimitsConfig)
    evidence: EvidenceBufferConfig = Field(default_factory=EvidenceBufferConfig)
    health: ResourceHealthConfig = Field(default_factory=ResourceHealthConfig)


class GPUAsyncConfig(BaseModel):
    """Async GPU operation configuration."""

    model_config = ConfigDict(extra="ignore")

    num_streams: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Number of CUDA streams for async GPU execution",
    )


class GPUBatchingConfig(BaseModel):
    """GPU batching configuration."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = Field(default=True, description="Enable GPU batching")
    default_batch_size: int = Field(
        default=8,
        ge=1,
        description="Default batch size for GPU workloads",
    )
    max_batch_size: int = Field(
        default=32,
        ge=1,
        description="Maximum batch size for GPU workloads",
    )
    dynamic_batching: bool = Field(
        default=True,
        description="Dynamically adjust batch size based on observed latency",
    )


class GPUMemoryConfig(BaseModel):
    """GPU memory optimization settings."""

    model_config = ConfigDict(extra="ignore")

    mixed_precision: bool = Field(
        default=True,
        description="Enable mixed precision inference",
    )
    max_memory_per_gpu: Optional[str] = Field(
        default="24GB",
        description="Soft limit for GPU memory usage per device",
    )
    log_memory_stats: bool = Field(
        default=True,
        description="Log GPU memory statistics periodically",
    )
    memory_check_interval: int = Field(
        default=60,
        ge=1,
        description="Memory check interval in seconds",
    )


class GPUMultiConfig(BaseModel):
    """Multi-GPU configuration."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = Field(default=False, description="Enable multi-GPU mode")
    device_ids: List[int] = Field(
        default_factory=list,
        description="GPU device IDs to use (empty = all available)",
    )
    strategy: str = Field(
        default="data_parallel",
        description="Multi-GPU strategy: data_parallel, model_parallel, pipeline_parallel",
    )


class GPUOllamaConfig(BaseModel):
    """Ollama GPU configuration."""

    model_config = ConfigDict(extra="ignore")

    gpu_layers: int = Field(
        default=-1,
        description="Number of Ollama GPU layers (-1 = all, 0 = CPU only)",
    )
    num_gpu: int = Field(
        default=1,
        ge=1,
        description="Number of GPUs per Ollama instance",
    )
    gpu_memory_utilization: float = Field(
        default=0.9,
        ge=0.1,
        le=1.0,
        description="GPU memory utilization ratio for Ollama",
    )
    enable_gpu: bool = Field(
        default=True,
        description="Enable GPU acceleration for Ollama",
    )


class GPUEmbeddingsConfig(BaseModel):
    """GPU embedding configuration."""

    model_config = ConfigDict(extra="ignore")

    device: Optional[str] = Field(
        default=None,
        description="Embedding device override: cuda, mps, cpu",
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        description="Embedding batch size",
    )
    cache_on_gpu: bool = Field(
        default=True,
        description="Cache embeddings on GPU memory",
    )
    max_cache_size_gb: float = Field(
        default=2.0,
        ge=0.1,
        description="Maximum embedding cache size in GB",
    )
    mixed_precision: bool = Field(
        default=True,
        description="Use mixed precision for embedding generation",
    )
    embedding_dim: int = Field(
        default=768,
        ge=64,
        description="Expected embedding vector dimension",
    )


class GPUCriticsConfig(BaseModel):
    """GPU critic execution configuration."""

    model_config = ConfigDict(extra="ignore")

    gpu_batching: bool = Field(
        default=True,
        description="Enable GPU batching for critics",
    )
    batch_size: int = Field(
        default=8,
        ge=1,
        description="Critic batch size",
    )
    use_gpu: bool = Field(
        default=True,
        description="Allow critics to run on GPU",
    )


class GPUPrecedentConfig(BaseModel):
    """GPU precedent retrieval configuration."""

    model_config = ConfigDict(extra="ignore")

    gpu_similarity_search: bool = Field(
        default=True,
        description="Use GPU for precedent similarity search",
    )
    cache_embeddings_on_gpu: bool = Field(
        default=True,
        description="Cache precedent embeddings on GPU",
    )
    max_cached_precedents: int = Field(
        default=10_000,
        ge=1,
        description="Max precedents to cache on GPU",
    )


class GPUSettings(BaseModel):
    """GPU acceleration configuration."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = Field(default=False, description="Enable GPU acceleration")
    device_preference: List[str] = Field(
        default_factory=lambda: ["cuda", "mps", "cpu"],
        description="Device preference order",
    )
    preferred_devices: Optional[List[int]] = Field(
        default=None,
        description="Preferred GPU device IDs",
    )
    multi_gpu: GPUMultiConfig = Field(default_factory=GPUMultiConfig)
    memory: GPUMemoryConfig = Field(default_factory=GPUMemoryConfig)
    batching: GPUBatchingConfig = Field(default_factory=GPUBatchingConfig)
    async_ops: GPUAsyncConfig = Field(
        default_factory=GPUAsyncConfig,
        validation_alias=AliasChoices("async", "async_ops"),
    )
    ollama: GPUOllamaConfig = Field(default_factory=GPUOllamaConfig)
    embeddings: GPUEmbeddingsConfig = Field(default_factory=GPUEmbeddingsConfig)
    critics: GPUCriticsConfig = Field(default_factory=GPUCriticsConfig)
    precedent: GPUPrecedentConfig = Field(default_factory=GPUPrecedentConfig)


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

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        def overlay_settings(_: Optional[BaseSettings] = None) -> Dict[str, Any]:
            return load_overlay_payload()

        def yaml_settings(_: Optional[BaseSettings] = None) -> Dict[str, Any]:
            path = os.getenv("ELEANOR_CONFIG_PATH") or os.getenv("ELEANOR_CONFIG")
            if not path:
                return {}
            if not path.lower().endswith((".yml", ".yaml")):
                return {}
            payload = _load_yaml_config(path)
            return _normalize_yaml_config(payload)

        yaml_source = cast(PydanticBaseSettingsSource, yaml_settings)
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            cast(PydanticBaseSettingsSource, overlay_settings),
            yaml_source,
            file_secret_settings,
        )

    # Environment
    environment: str = Field(
        default="development",
        description="Environment: development, staging, production",
        validation_alias=AliasChoices("ELEANOR_ENVIRONMENT", "ELEANOR_ENV", "ENV"),
    )

    constitutional_config_path: str = Field(
        default="governance/constitutional.yaml",
        description="Path to the constitutional config YAML",
        validation_alias=AliasChoices(
            "ELEANOR_CONSTITUTIONAL_CONFIG_PATH",
            "CONSTITUTIONAL_CONFIG_PATH",
        ),
    )

    replay_log_path: str = Field(
        default="replay_log.jsonl",
        description="Path to replay log JSONL",
        validation_alias=AliasChoices(
            "ELEANOR_REPLAY_LOG_PATH",
            "REPLAY_LOG_PATH",
        ),
    )

    # Engine Configuration
    detail_level: int = Field(
        default=2, ge=1, le=3, description="Output detail level (1=minimal, 2=standard, 3=forensic)"
    )

    # Feature Flags
    enable_reflection: bool = Field(
        default=True, description="Enable reflection and uncertainty analysis"
    )
    enable_drift_check: bool = Field(default=True, description="Enable drift detection")
    enable_precedent_analysis: bool = Field(
        default=True, description="Enable precedent-based reasoning"
    )
    
    # New Feature Flags
    enable_explainable_governance: bool = Field(
        default=False, description="Enable explainable governance with causal reasoning"
    )
    enable_semantic_cache: bool = Field(
        default=False, description="Enable semantic cache for embedding-based similarity caching"
    )
    enable_intelligent_model_selection: bool = Field(
        default=False, description="Enable intelligent model selection for cost optimization"
    )
    enable_anomaly_detection: bool = Field(
        default=False, description="Enable anomaly detection for proactive problem detection"
    )
    enable_streaming_governance: bool = Field(
        default=False, description="Enable streaming governance with WebSocket incremental decisions"
    )
    enable_adaptive_critic_weighting: bool = Field(
        default=False, description="Enable adaptive critic weighting with meta-learning"
    )
    enable_temporal_precedent_evolution: bool = Field(
        default=False, description="Enable temporal precedent evolution tracking"
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
    resource_management: ResourceManagementConfig = Field(default_factory=ResourceManagementConfig)
    gpu: GPUSettings = Field(default_factory=GPUSettings)

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
        description="List of enabled critics",
    )

    @model_validator(mode="after")
    def validate_configuration(self) -> "EleanorSettings":
        """Cross-field validation."""

        # Warn if production without proper backends
        if self.environment == "production":
            if self.precedent.backend == "none":
                warnings.warn(
                    "Production environment without precedent backend configured. "
                    "Set ELEANOR_PRECEDENT__BACKEND for production use.",
                    UserWarning,
                )

            if self.security.secret_provider == "env":
                warnings.warn(
                    "Production environment using environment variables for secrets. "
                    "Consider AWS Secrets Manager or HashiCorp Vault.",
                    UserWarning,
                )

            if not self.observability.enable_tracing:
                warnings.warn(
                    "Production environment without distributed tracing. "
                    "Enable tracing for better observability.",
                    UserWarning,
                )

        # Validate cache configuration
        if self.cache.enabled and self.cache.redis_url is None:
            warnings.warn(
                "Caching enabled without Redis URL. Using in-memory cache only.", UserWarning
            )

        # Validate tracing endpoints
        if self.observability.enable_tracing:
            if not self.observability.otel_endpoint and not self.observability.jaeger_endpoint:
                warnings.warn(
                    "Tracing enabled but no endpoint configured. "
                    "Set ELEANOR_OBSERVABILITY__OTEL_ENDPOINT or JAEGER_ENDPOINT.",
                    UserWarning,
                )

        return self

    def to_legacy_engine_config(self) -> Dict[str, Any]:
        """Convert to legacy EngineConfig format for backward compatibility."""
        return {
            "detail_level": self.detail_level,
            "max_concurrency": self.performance.max_concurrency,
            "timeout_seconds": self.performance.timeout_seconds,
            "enable_adaptive_concurrency": self.performance.enable_adaptive_concurrency,
            "target_latency_ms": self.performance.target_latency_ms,
            "enable_reflection": self.enable_reflection,
            "enable_drift_check": self.enable_drift_check,
            "enable_precedent_analysis": self.enable_precedent_analysis,
            "enable_circuit_breakers": self.resilience.enable_circuit_breakers,
            "circuit_breaker_threshold": self.resilience.circuit_breaker_threshold,
            "circuit_breaker_timeout": self.resilience.circuit_breaker_timeout,
            "enable_graceful_degradation": self.resilience.enable_graceful_degradation,
            "jsonl_evidence_path": self.evidence.jsonl_path,
            "shutdown_timeout_seconds": self.resource_management.shutdown.graceful_timeout,
        }


# Global settings instance
_settings: Optional[EleanorSettings] = None


def get_settings(env_file: Optional[str] = None, reload: bool = False) -> EleanorSettings:
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
            env = _resolve_environment()
            env_file = f".env.{env}"

            # Fallback to .env if specific file doesn't exist
            if not Path(env_file).exists():
                env_file = ".env"

        # Load settings
        if Path(env_file).exists():
            settings_cls = cast(Any, EleanorSettings)
            _settings = settings_cls(_env_file=env_file)
        else:
            _settings = EleanorSettings()

    assert _settings is not None
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
    "AWSSecretsConfig",
    "VaultSecretsConfig",
    "ObservabilityConfig",
    "ResilienceConfig",
    "ResourceManagementConfig",
    "ShutdownConfig",
    "GPUSettings",
    "GPUAsyncConfig",
    "GPUBatchingConfig",
    "GPUMemoryConfig",
    "GPUMultiConfig",
    "GPUOllamaConfig",
    "GPUEmbeddingsConfig",
    "GPUCriticsConfig",
    "GPUPrecedentConfig",
]
