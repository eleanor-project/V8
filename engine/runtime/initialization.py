from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Callable, Dict, Optional, cast

from engine.cache import CacheManager, AdaptiveConcurrencyManager, RouterSelectionCache
from engine.factory import EngineDependencies

# Enhanced features
try:
    from engine.critics.batch_processor import BatchCriticProcessor, BatchCriticConfig
    from engine.cache.warming import CacheWarmer
    from engine.resource.adaptive_limits import AdaptiveResourceLimiter, MemoryMonitor
    ENHANCEMENTS_AVAILABLE = True
except ImportError:
    ENHANCEMENTS_AVAILABLE = False
    BatchCriticProcessor = None
    BatchCriticConfig = None
    CacheWarmer = None
    AdaptiveResourceLimiter = None
    MemoryMonitor = None
from engine.protocols import (
    AggregatorProtocol,
    CriticProtocol,
    DetectorEngineProtocol,
    EvidenceRecorderProtocol,
    PrecedentEngineProtocol,
    PrecedentRetrieverProtocol,
    ReviewTriggerEvaluatorProtocol,
    RouterProtocol,
    UncertaintyEngineProtocol,
)
from engine.runtime.config import EngineConfig
from engine.utils.critic_names import canonicalize_critic_map
from engine.utils.circuit_breaker import CircuitBreakerRegistry
from engine.utils.dependency_tracking import record_dependency_failure

logger = logging.getLogger("engine.engine")


def initialize_engine(
    engine: Any,
    *,
    config: Optional[EngineConfig],
    evidence_recorder: Optional[EvidenceRecorderProtocol],
    detector_engine: Optional[DetectorEngineProtocol],
    precedent_engine: Optional[PrecedentEngineProtocol],
    precedent_retriever: Optional[PrecedentRetrieverProtocol],
    uncertainty_engine: Optional[UncertaintyEngineProtocol],
    aggregator: Optional[AggregatorProtocol],
    router_backend: Optional[Any],
    critics: Optional[Dict[str, CriticProtocol]],
    critic_models: Optional[Dict[str, Any]],
    review_trigger_evaluator: Optional[ReviewTriggerEvaluatorProtocol],
    dependencies: Optional[EngineDependencies],
    error_monitor: Optional[Callable[[Exception, Dict[str, Any]], None]],
    gpu_manager: Optional[Any],
    gpu_executor: Optional[Any],
    gpu_embedding_cache: Optional[Any],
    init_cache_redis: Callable[[Optional[str]], Optional[Any]],
    estimate_embedding_cache_entries: Callable[..., int],
    parse_memory_gb: Callable[[Optional[Any]], Optional[float]],
    resolve_dependencies: Callable[..., EngineDependencies],
) -> None:
    settings = None
    settings_error: Optional[Exception] = None
    try:
        from engine.config import ConfigManager

        settings = ConfigManager().settings
    except Exception as exc:
        settings_error = exc
        record_dependency_failure("engine.config.ConfigManager", exc)

    if config is None:
        if settings is not None:
            config = EngineConfig.from_settings(settings)
            logger.info(
                "Loaded engine configuration from ConfigManager",
                extra={"environment": settings.environment},
            )
        else:
            logger.warning(
                "ConfigManager unavailable; using default EngineConfig",
                extra={"error": str(settings_error) if settings_error else "unknown"},
            )
            config = EngineConfig()
    engine.config = config
    engine.settings = settings

    engine.cache_manager = None
    engine.router_cache = None
    if settings and settings.cache.enabled:
        redis_client = init_cache_redis(settings.cache.redis_url)
        l1_ttls = {
            "precedent": settings.cache.precedent_ttl,
            "embedding": settings.cache.embeddings_ttl,
            "router": settings.cache.router_ttl,
            "critic": settings.cache.critics_ttl,
            "detector": settings.cache.detector_ttl,
        }
        l2_ttls = {key: max(value * 2, value) for key, value in l1_ttls.items()}
        engine.cache_manager = CacheManager(
            redis_client=redis_client,
            l1_ttls=l1_ttls,
            l2_ttls=l2_ttls,
        )
        engine.router_cache = RouterSelectionCache(
            maxsize=200,
            ttl=settings.cache.router_ttl,
        )

    engine.gpu_manager = gpu_manager
    engine.gpu_executor = gpu_executor
    engine.gpu_embedding_cache = gpu_embedding_cache
    engine.gpu_multi_router = None
    engine.gpu_enabled = bool(
        (settings and getattr(settings, "gpu", None) and settings.gpu.enabled)
        or gpu_manager is not None
    )
    engine.gpu_available = False

    if settings and getattr(settings, "gpu", None) and settings.gpu.enabled:
        try:
            from engine.gpu.manager import GPUManager, GPUConfig
            from engine.gpu.async_ops import AsyncGPUExecutor
            from engine.gpu.embeddings import GPUEmbeddingCache
            from engine.gpu.parallelization import MultiGPURouter
        except Exception as exc:
            logger.warning("gpu_modules_unavailable", extra={"error": str(exc)})
            record_dependency_failure("engine.gpu.modules", exc)
        else:
            preferred_devices = settings.gpu.preferred_devices
            if not preferred_devices:
                preferred_devices = settings.gpu.multi_gpu.device_ids or None

            gpu_config = GPUConfig(
                enabled=settings.gpu.enabled,
                device_preference=settings.gpu.device_preference,
                preferred_devices=preferred_devices,
                mixed_precision=settings.gpu.memory.mixed_precision,
                num_streams=settings.gpu.async_ops.num_streams,
                max_memory_per_gpu=parse_memory_gb(settings.gpu.memory.max_memory_per_gpu),
                log_memory_stats=settings.gpu.memory.log_memory_stats,
                memory_check_interval=settings.gpu.memory.memory_check_interval,
                default_batch_size=settings.gpu.batching.default_batch_size,
                max_batch_size=settings.gpu.batching.max_batch_size,
                dynamic_batching=settings.gpu.batching.dynamic_batching,
            )

            if engine.gpu_manager is None:
                engine.gpu_manager = GPUManager(
                    config=gpu_config,
                    preferred_devices=preferred_devices,
                )

            if engine.gpu_manager and engine.gpu_manager.device:
                engine.gpu_available = engine.gpu_manager.is_available()
                if engine.gpu_executor is None:
                    engine.gpu_executor = AsyncGPUExecutor(
                        device=engine.gpu_manager.get_device(),
                        num_streams=gpu_config.num_streams,
                    )

                use_gpu_embedding_cache = (
                    settings.gpu.embeddings.cache_on_gpu
                    or settings.gpu.precedent.cache_embeddings_on_gpu
                )
                if engine.gpu_embedding_cache is None and use_gpu_embedding_cache:
                    bytes_per_value = 2 if settings.gpu.embeddings.mixed_precision else 4
                    cache_entries = estimate_embedding_cache_entries(
                        settings.gpu.embeddings.max_cache_size_gb,
                        settings.gpu.embeddings.embedding_dim,
                        bytes_per_value=bytes_per_value,
                    )
                    if cache_entries > 0:
                        engine.gpu_embedding_cache = GPUEmbeddingCache(
                            device=engine.gpu_manager.get_device(),
                            max_cache_size=cache_entries,
                            embedding_dim=settings.gpu.embeddings.embedding_dim,
                        )

                if settings.gpu.multi_gpu.enabled:
                    device_ids = settings.gpu.multi_gpu.device_ids or None
                    engine.gpu_multi_router = MultiGPURouter(device_ids=device_ids)
    engine.instance_id = str(uuid.uuid4())
    engine._shutdown_event = asyncio.Event()
    engine._cleanup_tasks = []
    
    # Initialize batch critic processor - will be overridden by GPU batcher if enabled
    # GPU batching takes precedence, so only initialize if GPU won't override
    engine.critic_batcher = None
    # Check if GPU batching will be used (check device directly since gpu_available set later)
    gpu_batching_will_override = (
        settings and 
        getattr(settings, "gpu", None) and 
        settings.gpu.critics.gpu_batching and 
        settings.gpu.critics.use_gpu and
        engine.gpu_manager and 
        engine.gpu_manager.device is not None
    )
    
    if not gpu_batching_will_override and ENHANCEMENTS_AVAILABLE and BatchCriticProcessor:
        try:
            batch_config = BatchCriticConfig(enabled=True)
            engine.critic_batcher = BatchCriticProcessor(config=batch_config)
            logger.info("batch_critic_processor_initialized")
        except Exception as exc:
            logger.warning(f"Failed to initialize batch processor: {exc}")
            engine.critic_batcher = None
    
    # Initialize cache warmer if available
    if ENHANCEMENTS_AVAILABLE and CacheWarmer:
        try:
            engine.cache_warmer = CacheWarmer(
                precedent_retriever=engine.precedent_retriever,
                embedding_service=getattr(engine, "embedding_service", None),
                cache_manager=engine.cache_manager,
            )
        except Exception as exc:
            logger.debug(f"Cache warmer not initialized: {exc}")
            engine.cache_warmer = None
    else:
        engine.cache_warmer = None
    
    # Initialize adaptive resource limiter if available
    if ENHANCEMENTS_AVAILABLE and AdaptiveResourceLimiter:
        try:
            base_concurrency = engine.config.max_concurrency if hasattr(engine.config, "max_concurrency") else 6
            engine.adaptive_limiter = AdaptiveResourceLimiter(base_concurrency=base_concurrency)
        except Exception as exc:
            logger.debug(f"Adaptive limiter not initialized: {exc}")
            engine.adaptive_limiter = None
    else:
        engine.adaptive_limiter = None

    engine.circuit_breakers = None
    engine.degradation_enabled = bool(engine.config.enable_graceful_degradation)
    if engine.config.enable_circuit_breakers:
        engine.circuit_breakers = CircuitBreakerRegistry()
    engine._breaker_failure_threshold = engine.config.circuit_breaker_threshold
    engine._breaker_recovery_timeout = engine.config.circuit_breaker_timeout

    deps = resolve_dependencies(
        config=engine.config,
        dependencies=dependencies,
        evidence_recorder=evidence_recorder,
        detector_engine=detector_engine,
        precedent_engine=precedent_engine,
        precedent_retriever=precedent_retriever,
        uncertainty_engine=uncertainty_engine,
        aggregator=aggregator,
        router_backend=router_backend,
        critics=critics,
        critic_models=critic_models,
        review_trigger_evaluator=review_trigger_evaluator,
    )

    engine.router = cast(RouterProtocol, deps.router)
    # Finish wiring cache warmer now that router/router_cache are ready
    if getattr(engine, "cache_warmer", None):
        engine.cache_warmer.router = engine.router
        engine.cache_warmer.router_cache = engine.router_cache
        engine.cache_warmer.cache_manager = engine.cache_manager
        engine.cache_warmer.engine = engine
    engine.critics = canonicalize_critic_map(deps.critics or {})
    engine.critic_models = deps.critic_models or {}
    engine.detector_engine = deps.detector_engine
    engine.recorder = deps.evidence_recorder
    engine.precedent_engine = deps.precedent_engine
    engine.precedent_retriever = deps.precedent_retriever
    engine.uncertainty_engine = deps.uncertainty_engine
    engine.aggregator = deps.aggregator
    engine.review_trigger_evaluator = deps.review_trigger_evaluator
    engine.error_monitor = error_monitor
    precedent_cache_enabled = True
    if settings and getattr(settings, "gpu", None):
        precedent_cache_enabled = settings.gpu.precedent.cache_embeddings_on_gpu
    if precedent_cache_enabled and engine.precedent_retriever and engine.gpu_embedding_cache:
        if hasattr(engine.precedent_retriever, "embedding_cache"):
            engine.precedent_retriever.embedding_cache = engine.gpu_embedding_cache

    engine.adaptive_concurrency = None
    if engine.config.enable_adaptive_concurrency:
        initial_limit = min(engine.config.max_concurrency, 6)
        min_limit = min(2, initial_limit)
        max_limit = max(engine.config.max_concurrency, initial_limit)
        engine.adaptive_concurrency = AdaptiveConcurrencyManager(
            initial_limit=initial_limit,
            min_limit=min_limit,
            max_limit=max_limit,
            target_latency_ms=engine.config.target_latency_ms,
        )
        engine.semaphore = engine.adaptive_concurrency.semaphore
    else:
        engine.semaphore = asyncio.Semaphore(engine.config.max_concurrency)

    # GPU-based critic batching takes precedence over enhancement-based batching
    if settings and getattr(settings, "gpu", None):
        if settings.gpu.critics.gpu_batching and settings.gpu.critics.use_gpu:
            if engine.gpu_manager and engine.gpu_available:
                try:
                    from engine.gpu.batch_processor import BatchProcessor
                except Exception as exc:
                    logger.warning("gpu_batch_processor_unavailable", extra={"error": str(exc)})
                    record_dependency_failure("engine.gpu.batch_processor", exc)
                else:
                    # Override any previously set critic_batcher with GPU version
                    if engine.critic_batcher is not None:
                        logger.info("gpu_critic_batcher_overriding_enhancement_batcher")
                    engine.critic_batcher = BatchProcessor(
                        process_fn=engine._process_critic_batch,
                        device=engine.gpu_manager.get_device(),
                        initial_batch_size=settings.gpu.critics.batch_size,
                        max_batch_size=settings.gpu.batching.max_batch_size,
                        min_batch_size=1,
                        dynamic_sizing=settings.gpu.batching.dynamic_batching,
                    )
                    logger.info("gpu_critic_batcher_initialized", extra={
                        "device": str(engine.gpu_manager.device),
                        "batch_size": settings.gpu.critics.batch_size,
                    })

    # Traffic Light governance hook (external governor).
    # This is an *observer* hook by default: it does not override critic outcomes.
    engine.traffic_light_governance = None
    try:
        from engine.integrations.traffic_light_governance import TrafficLightGovernanceHook
        engine.traffic_light_governance = TrafficLightGovernanceHook.from_env(
            enabled=bool(getattr(engine.config, 'enable_traffic_light_governance', True)),
            router_config_path=str(getattr(engine.config, 'traffic_light_router_config_path', 'governance/router_config.yaml')),
            events_jsonl_path=getattr(engine.config, 'governance_events_jsonl_path', 'governance_events.jsonl'),
            mode='observe',
        )
    except Exception as exc:
        logger.debug(f"Traffic Light governance hook not initialized: {exc}")

    logger.info(
        "engine_initialized",
        extra={"instance_id": getattr(engine, "instance_id", None)},
    )
