"""
ELEANOR V8 - Enterprise Constitutional Engine Runtime
Dual API (run + run_stream)
Dependency Injection Ready
Full Evidence Recorder Integration
Precedent Alignment + Uncertainty Engine Hooks
Forensic Detail-Level Output Mode
"""

import inspect
import json
import logging
from typing import Any, Callable, Dict, Optional

from engine.factory import DependencyFactory, EngineDependencies
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
from engine.runtime.governance import run_governance_review_gate
from engine.runtime.helpers import (
    estimate_embedding_cache_entries as _estimate_embedding_cache_entries_impl,
    init_cache_redis as _init_cache_redis_impl,
    parse_memory_gb as _parse_memory_gb_impl,
    resolve_dependencies as _resolve_dependencies_impl,
    resolve_router_backend as _resolve_router_backend_impl,
)
from engine.runtime.initialization import initialize_engine
from engine.runtime.mixins import EngineRuntimeMixin
from engine.runtime.models import (  # noqa: F401
    EngineCriticFinding,
    EngineForensicData,
    EngineModelInfo,
    EngineResult,
)
from engine.validation import validate_input  # compatibility for tests/monkeypatching
from engine.utils.dependency_tracking import record_dependency_failure
from governance.review_packets import build_review_packet
from governance.review_triggers import Case
from engine.replay_store import store_review_packet

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Dependency Resolution
# ---------------------------------------------------------


def _resolve_router_backend(router_backend: Optional[Any]) -> RouterProtocol:
    return _resolve_router_backend_impl(
        router_backend,
        dependency_factory=DependencyFactory,
    )


def _init_cache_redis(redis_url: Optional[str]) -> Optional[Any]:
    return _init_cache_redis_impl(
        redis_url,
        logger=logger,
        record_dependency_failure=record_dependency_failure,
    )


def _estimate_embedding_cache_entries(
    max_cache_size_gb: float,
    embedding_dim: int,
    *,
    bytes_per_value: int = 4,
) -> int:
    return _estimate_embedding_cache_entries_impl(
        max_cache_size_gb,
        embedding_dim,
        bytes_per_value=bytes_per_value,
    )


def _parse_memory_gb(value: Optional[Any]) -> Optional[float]:
    return _parse_memory_gb_impl(value)


def _resolve_dependencies(
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
) -> EngineDependencies:
    return _resolve_dependencies_impl(
        config=config,
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
        dependency_factory=DependencyFactory,
        resolve_router_backend=_resolve_router_backend,
    )


# ---------------------------------------------------------
# ELEANOR ENGINE V8
# ---------------------------------------------------------


class EleanorEngineV8(EngineRuntimeMixin):
    def __init__(
        self,
        *,
        config: Optional[EngineConfig] = None,
        evidence_recorder: Optional[EvidenceRecorderProtocol] = None,
        detector_engine: Optional[DetectorEngineProtocol] = None,
        precedent_engine: Optional[PrecedentEngineProtocol] = None,
        precedent_retriever: Optional[PrecedentRetrieverProtocol] = None,
        uncertainty_engine: Optional[UncertaintyEngineProtocol] = None,
        aggregator: Optional[AggregatorProtocol] = None,
        router_backend: Optional[Any] = None,
        critics: Optional[Dict[str, CriticProtocol]] = None,
        critic_models: Optional[Dict[str, Any]] = None,
        review_trigger_evaluator: Optional[ReviewTriggerEvaluatorProtocol] = None,
        dependencies: Optional[EngineDependencies] = None,
        error_monitor: Optional[Callable[[Exception, Dict[str, Any]], None]] = None,
        gpu_manager: Optional[Any] = None,
        gpu_executor: Optional[Any] = None,
        gpu_embedding_cache: Optional[Any] = None,
    ):
        self._json_module = json
        self._inspect_module = inspect
        initialize_engine(
            self,
            config=config,
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
            dependencies=dependencies,
            error_monitor=error_monitor,
            gpu_manager=gpu_manager,
            gpu_executor=gpu_executor,
            gpu_embedding_cache=gpu_embedding_cache,
            init_cache_redis=_init_cache_redis,
            estimate_embedding_cache_entries=_estimate_embedding_cache_entries,
            parse_memory_gb=_parse_memory_gb,
            resolve_dependencies=_resolve_dependencies,
        )

    def _run_governance_review_gate(self, case: Case):
        return run_governance_review_gate(
            self,
            case,
            build_review_packet=build_review_packet,
            store_review_packet=store_review_packet,
        )


# ---------------------------------------------------------
# ENGINE FACTORY
# ---------------------------------------------------------


def create_engine(
    config: Optional[EngineConfig] = None,
    *,
    evidence_recorder: Optional[EvidenceRecorderProtocol] = None,
    detector_engine: Optional[DetectorEngineProtocol] = None,
    precedent_engine: Optional[PrecedentEngineProtocol] = None,
    precedent_retriever: Optional[PrecedentRetrieverProtocol] = None,
    uncertainty_engine: Optional[UncertaintyEngineProtocol] = None,
    aggregator: Optional[AggregatorProtocol] = None,
    router_backend: Optional[Any] = None,
    critics: Optional[Dict[str, CriticProtocol]] = None,
    critic_models: Optional[Dict[str, Any]] = None,
    review_trigger_evaluator: Optional[ReviewTriggerEvaluatorProtocol] = None,
    dependencies: Optional[EngineDependencies] = None,
    error_monitor: Optional[Callable[[Exception, Dict[str, Any]], None]] = None,
    gpu_manager: Optional[Any] = None,
    gpu_executor: Optional[Any] = None,
    gpu_embedding_cache: Optional[Any] = None,
) -> EleanorEngineV8:
    engine = EleanorEngineV8(
        config=config,
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
        dependencies=dependencies,
        error_monitor=error_monitor,
        gpu_manager=gpu_manager,
        gpu_executor=gpu_executor,
        gpu_embedding_cache=gpu_embedding_cache,
    )

    logger.info(
        "engine_created",
        extra={"instance_id": getattr(engine, "instance_id", None)},
    )
    return engine


__all__ = [
    "EleanorEngineV8",
    "EngineConfig",
    "EngineResult",
    "EngineCriticFinding",
    "EngineModelInfo",
    "EngineForensicData",
    "create_engine",
]
