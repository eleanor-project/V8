import inspect
from typing import Any, Callable, Dict, Optional, cast

from engine.factory import EngineDependencies
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


def resolve_router_backend(
    router_backend: Optional[Any],
    *,
    dependency_factory: Any,
) -> RouterProtocol:
    if router_backend is None:
        return dependency_factory.create_router()
    if isinstance(router_backend, str):
        return dependency_factory.create_router(backend=router_backend)
    if inspect.isclass(router_backend):
        return cast(RouterProtocol, router_backend())
    if callable(router_backend):
        return cast(RouterProtocol, router_backend())
    return cast(RouterProtocol, router_backend)


def init_cache_redis(
    redis_url: Optional[str],
    *,
    logger: Any,
    record_dependency_failure: Callable[[str, Exception], None],
) -> Optional[Any]:
    if not redis_url:
        return None
    try:
        import redis  # type: ignore[import-not-found]
    except ImportError:
        logger.warning("Redis cache requested but redis package is not installed")
        record_dependency_failure("redis", ImportError("redis not installed"))
        return None
    try:
        return redis.Redis.from_url(redis_url, decode_responses=True)
    except Exception as exc:
        logger.warning("Redis cache initialization failed", extra={"error": str(exc)})
        record_dependency_failure("redis", exc)
        return None


def estimate_embedding_cache_entries(
    max_cache_size_gb: float,
    embedding_dim: int,
    *,
    bytes_per_value: int = 4,
) -> int:
    if max_cache_size_gb <= 0 or embedding_dim <= 0 or bytes_per_value <= 0:
        return 0
    bytes_per_embedding = embedding_dim * bytes_per_value
    estimated = int((max_cache_size_gb * 1024**3) / bytes_per_embedding)
    return max(1, estimated)


def parse_memory_gb(value: Optional[Any]) -> Optional[float]:
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


def resolve_dependencies(
    *,
    config: EngineConfig,
    dependencies: Optional[EngineDependencies],
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
    dependency_factory: Any,
    resolve_router_backend: Callable[[Optional[Any]], RouterProtocol],
) -> EngineDependencies:
    if dependencies is not None:
        return dependencies

    router_backend_name = router_backend if isinstance(router_backend, str) else None

    deps = dependency_factory.create_all_dependencies(
        router_backend=router_backend_name,
        jsonl_evidence_path=config.jsonl_evidence_path,
        critics=critics,
        critic_models=critic_models,
        enable_precedent=config.enable_precedent_analysis,
        enable_uncertainty=config.enable_reflection,
    )

    if router_backend is not None and not isinstance(router_backend, str):
        deps.router = resolve_router_backend(router_backend)
    if detector_engine is not None:
        deps.detector_engine = detector_engine
    if evidence_recorder is not None:
        deps.evidence_recorder = evidence_recorder
    if precedent_engine is not None:
        deps.precedent_engine = precedent_engine
    if precedent_retriever is not None:
        deps.precedent_retriever = precedent_retriever
    if uncertainty_engine is not None:
        deps.uncertainty_engine = uncertainty_engine
    if aggregator is not None:
        deps.aggregator = aggregator
    if review_trigger_evaluator is not None:
        deps.review_trigger_evaluator = review_trigger_evaluator

    return deps
