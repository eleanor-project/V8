"""
ELEANOR V8 — Engine Type Definitions
-------------------------------------

Type definitions to replace Any types in engine code.
"""

from typing import Protocol, Optional, Dict, List, Any, Callable, Awaitable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engine.protocols import (
        CriticProtocol,
        RouterProtocol,
        AggregatorProtocol,
        EvidenceRecorderProtocol,
        PrecedentRetrieverProtocol,
        DetectorEngineProtocol,
        UncertaintyEngineProtocol,
    )
    from engine.cache import CacheManager
    from engine.runtime.config import EngineConfig
    from engine.schemas.pipeline_types import CriticResult


class EngineProtocol(Protocol):
    """Protocol defining engine interface for type safety."""
    
    config: "EngineConfig"
    instance_id: str
    cache_manager: Optional["CacheManager"]
    critics: Dict[str, "CriticProtocol"]
    critic_models: Dict[str, Any]
    router: "RouterProtocol"
    aggregator: Optional["AggregatorProtocol"]
    recorder: Optional["EvidenceRecorderProtocol"]
    precedent_retriever: Optional["PrecedentRetrieverProtocol"]
    detector_engine: Optional["DetectorEngineProtocol"]
    uncertainty_engine: Optional["UncertaintyEngineProtocol"]
    gpu_manager: Optional[Any]
    gpu_enabled: bool
    _shutdown_event: Any
    _cleanup_tasks: List[Any]
    
    def _emit_error(
        self,
        exc: Exception,
        *,
        stage: str,
        trace_id: Optional[str] = None,
        critic: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None: ...
    
    def _emit_validation_error(
        self,
        exc: Exception,
        *,
        text: Any,
        context: Any,
        trace_id: Optional[str],
    ) -> None: ...
    
    def _validate_inputs(
        self,
        text: str,
        context: Optional[dict],
        trace_id: Optional[str],
        detail_level: Optional[int],
    ) -> tuple[str, Dict[str, Any], str, int]: ...
    
    def _build_critic_error_result(
        self,
        critic_name: str,
        error: Exception,
        duration_ms: Optional[float] = None,
        *,
        degraded: bool = False,
        degradation_reason: Optional[str] = None,
    ) -> "CriticResult": ...


class ModelAdapterProtocol(Protocol):
    """Protocol for model adapters."""
    
    async def generate(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str: ...


class CriticRefType:
    """Type alias for critic references."""
    pass


# Type aliases for common patterns
EngineType = EngineProtocol
CriticRef = CriticRefType | "CriticProtocol" | type["CriticProtocol"]
ModelAdapter = ModelAdapterProtocol | Callable[[str], Awaitable[str]]


__all__ = [
    "EngineProtocol",
    "ModelAdapterProtocol",
    "CriticRefType",
    "EngineType",
    "CriticRef",
    "ModelAdapter",
]
