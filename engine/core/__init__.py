"""
ELEANOR V8 — Engine Builder (async runtime)
-------------------------------------------

Bridges the API layer to the async engine/engine.py implementation.
Configurable via kwargs or environment variables for adapters, precedent,
embeddings, governance, and evidence sinks.
"""

import os
import json
import inspect
import logging
from typing import Any, Dict, Optional, Callable, List, cast

from engine.engine import create_engine, EngineConfig
from engine.router.router import RouterV8
from engine.router.adapters import bootstrap_default_registry
from engine.aggregator.aggregator import AggregatorV8
from engine.uncertainty.uncertainty import UncertaintyEngineV8
from engine.precedent.alignment import PrecedentAlignmentEngineV8
from engine.precedent.retrieval import PrecedentRetrievalV8
from engine.precedent.embeddings import bootstrap_embedding_registry
from engine.precedent.store import (
    WeaviatePrecedentStore,
    PgVectorStore,
)
from engine.recorder.evidence_recorder import EvidenceRecorder
from engine.detectors.engine import DetectorEngineV8
from engine.governance.opa_client import OPAClientV8
from engine.security.secrets import (
    EnvironmentSecretProvider,
    build_secret_provider_from_settings,
    get_llm_api_key_sync,
)

logger = logging.getLogger(__name__)


def _wrap_llm_adapter(fn):
    async def _adapter(text: str, context: Optional[Dict[str, Any]] = None):
        if not callable(fn):
            return text
        try:
            sig = inspect.signature(fn)
            has_context = "context" in sig.parameters
        except (TypeError, ValueError):
            has_context = False

        result = fn(text, context=context) if has_context else fn(text)
        if hasattr(result, "__await__"):
            result = await result
        return result

    return _adapter


def _parse_memory_gb(value: Optional[Any]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        raw = value.strip().lower()
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            pass
        suffixes = {"gb": 1.0, "g": 1.0, "mb": 1.0 / 1024, "m": 1.0 / 1024}
        for suffix, factor in suffixes.items():
            if raw.endswith(suffix):
                number = raw[: -len(suffix)].strip()
                try:
                    return float(number) * factor
                except ValueError:
                    return None
    return None


def _build_router(
    adapters: Optional[Dict[str, Any]],
    router_policy: Optional[Dict[str, Any]],
    *,
    openai_key: Optional[str],
    anthropic_key: Optional[str],
    xai_key: Optional[str],
    hf_device: Optional[str] = None,
):
    def _parse_mapping(env_var: str) -> Dict[str, float]:
        raw = os.getenv(env_var)
        if not raw:
            return {}
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return {str(k): float(v) for k, v in data.items()}
        except Exception:
            pass
        # fall back to comma-separated adapter:value
        mapping = {}
        for chunk in raw.split(","):
            if ":" in chunk:
                k, v = chunk.split(":", 1)
                try:
                    mapping[k.strip()] = float(v)
                except ValueError:
                    continue
        return mapping

    if adapters is None:
        registry = bootstrap_default_registry(
            openai_key=openai_key,
            anthropic_key=anthropic_key,
            xai_key=xai_key,
            hf_device=hf_device,
        )
        adapters = {}
        for name in registry.list():
            adapters[name] = registry.get(name)

    adapters = adapters or {"default": lambda text, context=None: text}

    if not router_policy:
        primary = next(iter(adapters.keys()))
        router_policy = {
            "primary": primary,
            "fallback_order": [k for k in adapters.keys() if k != primary],
            "max_retries": 2,
            "circuit_breaker": {"enabled": False},
            "adapter_costs": _parse_mapping("ROUTER_ADAPTER_COSTS"),
            "max_cost": _parse_float_env("ROUTER_MAX_COST"),
            "adapter_latency": _parse_mapping("ROUTER_ADAPTER_LATENCIES"),
            "latency_budget_ms": _parse_float_env("ROUTER_LATENCY_BUDGET_MS"),
        }

    return lambda: RouterV8(adapters=adapters, routing_policy=router_policy)


def _build_precedent_layer(
    precedent_store: Optional[Any],
    embed_backend: str,
    openai_key: Optional[str],
    anthropic_key: Optional[str],
    xai_key: Optional[str],
    embedding_device: Optional[str] = None,
):
    retriever = None
    store = precedent_store

    # Embeddings
    embed_registry = bootstrap_embedding_registry(
        openai_key=openai_key,
        anthropic_key=anthropic_key,
        xai_key=xai_key,
        device=embedding_device,
    )
    embed_adapter = None
    if embed_backend in embed_registry.list():
        embed_adapter = embed_registry.get(embed_backend)
    elif embed_registry.list():
        embed_adapter = embed_registry.get(embed_registry.list()[0])
    embed_fn = embed_adapter.embed if embed_adapter else None

    # Precedent store from env if not provided
    if store is None:
        backend = os.getenv("PRECEDENT_BACKEND", "memory").lower()
        if backend == "weaviate":
            try:
                import weaviate

                client = weaviate.Client(url=os.getenv("WEAVIATE_URL", "http://localhost:8080"))  # type: ignore[call-arg]
                store = WeaviatePrecedentStore(
                    client=client,
                    class_name=os.getenv("WEAVIATE_CLASS_NAME", "Precedent"),
                    embed_fn=embed_fn,
                )
            except Exception:
                store = None
        elif backend == "pgvector":
            conn = os.getenv("PG_CONN_STRING")
            table = os.getenv("PG_TABLE", "precedent")
            if conn:
                try:
                    store = PgVectorStore(
                        connection_string=conn,
                        table_name=table,
                    )
                except Exception:
                    store = None
        else:

            class MemoryStore:
                def __init__(self, embed_fn=None):
                    self.items: List[Dict[str, Any]] = []
                    self.embed_fn: Callable[[str], List[float]] = embed_fn or (lambda x: [])

                def add(self, text, metadata=None):
                    self.items.append(
                        {"text": text, "metadata": metadata or {}, "embedding": self.embed_fn(text)}
                    )

                def search(self, q, top_k=5):
                    # naive search; return top_k in insertion order
                    return self.items[:top_k]

            store = MemoryStore(embed_fn=embed_fn)

    if store and PrecedentRetrievalV8 is not None:
        retriever = PrecedentRetrievalV8(store_client=store, embedding_fn=embed_fn)

    return retriever


def build_eleanor_engine_v8(
    *,
    llm_fn=None,
    constitutional_config: Optional[Dict[str, Any]] = None,
    router_adapters: Optional[Dict[str, Any]] = None,
    router_policy: Optional[Dict[str, Any]] = None,
    precedent_store: Optional[Any] = None,
    opa_callback=None,
    evidence_path: Optional[str] = None,
    critic_models: Optional[Dict[str, Any]] = None,
    settings_override: Optional["EleanorSettings"] = None,
) -> Any:
    """
    Build a fully operational async engine instance.

    Args mirror the API/websocket bootstrap while remaining optional so that
    local development can run with minimal configuration.
    """

    settings = settings_override
    environment = os.getenv("ELEANOR_ENVIRONMENT") or os.getenv("ELEANOR_ENV") or "development"
    cache_ttl = 300
    secret_provider = None

    if settings is None:
        try:
            from engine.config import ConfigManager

            settings = ConfigManager().settings
            environment = settings.environment
            cache_ttl = settings.security.secrets_cache_ttl
            secret_provider = build_secret_provider_from_settings(settings)
        except Exception as exc:
            if environment == "production":
                raise
            logger.warning(
                "Secret provider setup failed; falling back to environment provider",
                extra={"error": str(exc)},
            )
    else:
        environment = settings.environment
        cache_ttl = settings.security.secrets_cache_ttl
        try:
            secret_provider = build_secret_provider_from_settings(settings)
        except Exception as exc:
            if environment == "production":
                raise
            logger.warning(
                "Secret provider setup failed; falling back to environment provider",
                extra={"error": str(exc)},
            )

    if secret_provider is None:
        if environment == "production":
            raise RuntimeError("Secret provider required in production")
        secret_provider = EnvironmentSecretProvider(cache_ttl=cache_ttl)

    openai_key = get_llm_api_key_sync("openai", secret_provider)
    anthropic_key = get_llm_api_key_sync("anthropic", secret_provider)
    xai_key = get_llm_api_key_sync("xai", secret_provider)
    embedding_backend = os.getenv("EMBEDDING_BACKEND", "gpt")
    gpu_manager = None
    gpu_device_name = None

    if settings and getattr(settings, "gpu", None) and settings.gpu.enabled:
        try:
            from engine.gpu.manager import GPUManager, GPUConfig as ManagerConfig

            preferred_devices = settings.gpu.preferred_devices
            if not preferred_devices:
                preferred_devices = settings.gpu.multi_gpu.device_ids or None

            gpu_config = ManagerConfig(
                enabled=settings.gpu.enabled,
                device_preference=settings.gpu.device_preference,
                preferred_devices=preferred_devices,
                mixed_precision=settings.gpu.memory.mixed_precision,
                num_streams=settings.gpu.async_ops.num_streams,
                max_memory_per_gpu=_parse_memory_gb(settings.gpu.memory.max_memory_per_gpu),
                log_memory_stats=settings.gpu.memory.log_memory_stats,
                memory_check_interval=settings.gpu.memory.memory_check_interval,
                default_batch_size=settings.gpu.batching.default_batch_size,
                max_batch_size=settings.gpu.batching.max_batch_size,
                dynamic_batching=settings.gpu.batching.dynamic_batching,
            )
            gpu_manager = GPUManager(
                config=gpu_config,
                preferred_devices=preferred_devices,
            )
            if gpu_manager.device:
                gpu_device_name = str(gpu_manager.device)
        except Exception as exc:
            logger.warning(
                "GPU manager initialization failed",
                extra={"error": str(exc)},
            )
            gpu_manager = None
            gpu_device_name = None

    def _resolve_gpu_device_name(preferred: Optional[str]) -> Optional[str]:
        if preferred:
            if gpu_manager and gpu_manager.device:
                if preferred.startswith("cuda") and gpu_manager.device.type != "cuda":
                    return str(gpu_manager.device)
                if preferred == "mps" and gpu_manager.device.type != "mps":
                    return str(gpu_manager.device)
            return preferred
        if gpu_device_name:
            return gpu_device_name
        return None

    adapters = router_adapters or None
    if llm_fn and not adapters:
        adapters = {"primary": _wrap_llm_adapter(llm_fn)}
        router_policy = router_policy or {"primary": "primary", "fallback_order": []}

    hf_device = os.getenv("HF_DEVICE")
    if settings and getattr(settings, "gpu", None) and hf_device is None:
        if settings.gpu.enabled:
            hf_device = _resolve_gpu_device_name(None)
        else:
            hf_device = "cpu"

    router_backend = _build_router(
        adapters=adapters,
        router_policy=router_policy,
        openai_key=openai_key,
        anthropic_key=anthropic_key,
        xai_key=xai_key,
        hf_device=hf_device,
    )

    embedding_device = None
    if settings and getattr(settings, "gpu", None):
        embedding_device = _resolve_gpu_device_name(settings.gpu.embeddings.device)
        if (
            embedding_device is None
            and settings.gpu.enabled
            and settings.gpu.precedent.gpu_similarity_search
        ):
            embedding_device = _resolve_gpu_device_name(None)

    precedent_retriever = _build_precedent_layer(
        precedent_store=precedent_store,
        embed_backend=embedding_backend,
        openai_key=openai_key,
        anthropic_key=anthropic_key,
        xai_key=xai_key,
        embedding_device=embedding_device,
    )

    evidence_buffer = settings.evidence.buffer_size if settings else None
    evidence_flush_interval = settings.evidence.flush_interval if settings else None
    recorder = EvidenceRecorder(
        jsonl_path=evidence_path or "evidence.jsonl",
        buffer_size=evidence_buffer,
        flush_interval=evidence_flush_interval,
    )

    # Detectors
    # Production posture: use the real detector engine (auto-loads built-in detectors).
    # No heuristic stubs in the core runtime.
    detector_timeout = float(os.getenv("DETECTOR_TIMEOUT_SECONDS", "2.0"))
    detector_engine = DetectorEngineV8(timeout_seconds=detector_timeout)

    uncertainty = UncertaintyEngineV8() if UncertaintyEngineV8 is not None else None
    aggregator = AggregatorV8()
    precedent_engine = (
        PrecedentAlignmentEngineV8() if PrecedentAlignmentEngineV8 is not None else None
    )
    engine_config = EngineConfig.from_settings(settings) if settings else EngineConfig()

    engine_instance = create_engine(
        config=engine_config,
        evidence_recorder=recorder,
        detector_engine=detector_engine,
        precedent_engine=precedent_engine,
        precedent_retriever=precedent_retriever,
        uncertainty_engine=uncertainty,
        aggregator=aggregator,
        router_backend=router_backend,
        critic_models=critic_models,
        gpu_manager=gpu_manager,
    )

    # Governance (OPA)
    if opa_callback:
        setattr(engine_instance, "opa_callback", opa_callback)
    else:
        opa_client = OPAClientV8(
            base_url=os.getenv("OPA_URL", "http://localhost:8181"),
            policy_path=os.getenv("OPA_POLICY_PATH", "v1/data/eleanor/decision"),
        )
        setattr(engine_instance, "opa_callback", opa_client.evaluate)
    
    # Integrate optional features based on feature flags
    try:
        from engine.core.feature_integration import integrate_optional_features
        integrate_optional_features(engine_instance, settings)
    except Exception as exc:
        logger.warning(
            "Failed to integrate optional features",
            extra={"error": str(exc)},
            exc_info=True
        )

    return engine_instance


__all__ = ["build_eleanor_engine_v8"]


def _parse_float_env(var_name: str) -> Optional[float]:
    val = os.getenv(var_name)
    if not val:
        return None
    try:
        return float(val)
    except ValueError:
        return None
