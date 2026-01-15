"""
ELEANOR V8 — Critic Infrastructure Integration (Production Orchestrator)
------------------------------------------------------------------------

This module provides the infrastructure layer around critic execution,
integrating with caching, circuit breakers, evidence recording, events,
and degradation strategies.

NOW USES: ProductionOrchestrator with advanced features:
- Policy-based gating (cost optimization)
- Staged execution with dependencies
- Retry strategies
- Resource management
- Result validation
- Priority scheduling
"""

import asyncio
import inspect
import logging
import os
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from engine.types.engine_types import EngineType, CriticRef

from engine.cache import CacheKey
from engine.exceptions import CriticEvaluationError, EvidenceRecordingError
from engine.resilience.degradation import DegradationStrategy
from engine.schemas.pipeline_types import CriticResult, CriticResultsMap
from engine.utils.circuit_breaker import CircuitBreakerOpen

# Import Production Orchestrator
from engine.orchestrator.orchestrator_production import (
    ProductionOrchestrator,
    CriticConfig,
    OrchestratorConfig,
    OrchestratorHooks,
    CriticInput,
    ExecutionStage,
    ExecutionPolicy,
)

# Enhanced observability
try:
    from engine.observability.tracing import run_critic_with_trace
    from engine.observability.correlation import get_correlation_id
    from engine.observability.business_metrics import record_critic_agreement
    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False
    run_critic_with_trace = None
    get_correlation_id = None
    record_critic_agreement = None

# Event bus
try:
    from engine.events.event_bus import get_event_bus, CriticEvaluatedEvent, EventType
    EVENT_BUS_AVAILABLE = True
except ImportError:
    EVENT_BUS_AVAILABLE = False
    get_event_bus = None
    CriticEvaluatedEvent = None
    EventType = None

logger = logging.getLogger("engine.engine")


# ============================================================================
# Configuration
# ============================================================================


def _get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean from environment variable."""
    value = os.getenv(key, "").strip().lower()
    if not value:
        return default
    return value in ("1", "true", "yes", "on")


def _get_env_int(key: str, default: int) -> int:
    """Get integer from environment variable."""
    value = os.getenv(key, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning(f"Invalid {key}={value}, using default {default}")
        return default


def _get_env_float(key: str, default: float) -> float:
    """Get float from environment variable."""
    value = os.getenv(key, "").strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning(f"Invalid {key}={value}, using default {default}")
        return default


def get_orchestrator_config() -> OrchestratorConfig:
    """
    Create orchestrator config from environment variables.
    
    Environment Variables:
    - ELEANOR_ORCHESTRATOR_MAX_CONCURRENT: Max concurrent critics (default: 10)
    - ELEANOR_ORCHESTRATOR_ENABLE_GATING: Enable policy gating (default: true)
    - ELEANOR_ORCHESTRATOR_ENABLE_RETRIES: Enable retry logic (default: true)
    - ELEANOR_ORCHESTRATOR_STRICT_VALIDATION: Strict validation (default: true)
    - ELEANOR_ORCHESTRATOR_GLOBAL_TIMEOUT: Global timeout in seconds (default: none)
    """
    return OrchestratorConfig(
        max_concurrent_critics=_get_env_int("ELEANOR_ORCHESTRATOR_MAX_CONCURRENT", 10),
        enable_policy_gating=_get_env_bool("ELEANOR_ORCHESTRATOR_ENABLE_GATING", True),
        enable_retries=_get_env_bool("ELEANOR_ORCHESTRATOR_ENABLE_RETRIES", True),
        strict_validation=_get_env_bool("ELEANOR_ORCHESTRATOR_STRICT_VALIDATION", True),
        fail_on_validation_error=_get_env_bool("ELEANOR_ORCHESTRATOR_FAIL_ON_VALIDATION", False),
        global_timeout_seconds=_get_env_float("ELEANOR_ORCHESTRATOR_GLOBAL_TIMEOUT", 0.0) or None,
    )


# ============================================================================
# Critic Classification and Configuration
# ============================================================================


def classify_critic_stage(critic_name: str, critic_ref: Any) -> ExecutionStage:
    """
    Automatically classify critic into execution stage based on name and attributes.
    
    This provides sensible defaults. Can be overridden with critic metadata.
    """
    name_lower = critic_name.lower()
    
    # Fast critics (< 100ms expected)
    if any(keyword in name_lower for keyword in ["quick", "fast", "simple", "basic"]):
        return ExecutionStage.FAST_CRITICS
    
    # Pre-validation
    if any(keyword in name_lower for keyword in ["validation", "sanity", "check"]):
        return ExecutionStage.PRE_VALIDATION
    
    # Deep/expensive analysis
    if any(keyword in name_lower for keyword in ["deep", "comprehensive", "detailed", "llm", "external"]):
        return ExecutionStage.DEEP_ANALYSIS
    
    # Post-processing
    if any(keyword in name_lower for keyword in ["aggregate", "summary", "synthesis"]):
        return ExecutionStage.POST_PROCESSING
    
    # Default: Core analysis
    return ExecutionStage.CORE_ANALYSIS


def get_critic_priority(critic_name: str, critic_ref: Any) -> int:
    """
    Automatically determine critic priority (1=highest, 10=lowest).
    
    Based on name and safety criticality.
    """
    name_lower = critic_name.lower()
    
    # Critical safety checks - highest priority
    if any(keyword in name_lower for keyword in ["safety", "security", "critical"]):
        return 1
    
    # Rights and bias - high priority
    if any(keyword in name_lower for keyword in ["rights", "bias", "discrimination"]):
        return 2
    
    # Fairness and ethics - medium-high priority
    if any(keyword in name_lower for keyword in ["fairness", "ethics", "harm"]):
        return 3
    
    # Privacy and compliance - medium priority
    if any(keyword in name_lower for keyword in ["privacy", "pii", "compliance"]):
        return 4
    
    # Standard analysis - medium priority
    if any(keyword in name_lower for keyword in ["analyze", "assess", "evaluate"]):
        return 5
    
    # Style and quality - lower priority
    if any(keyword in name_lower for keyword in ["style", "quality", "format"]):
        return 7
    
    # Documentation and metadata - lowest priority
    if any(keyword in name_lower for keyword in ["document", "metadata", "log"]):
        return 9
    
    # Default: medium priority
    return 5


def get_critic_policy(critic_name: str, context: Dict[str, Any]) -> ExecutionPolicy:
    """
    Automatically determine execution policy based on critic name and context.
    """
    name_lower = critic_name.lower()
    
    # Always run safety-critical checks
    if any(keyword in name_lower for keyword in ["safety", "security", "critical"]):
        return ExecutionPolicy.ALWAYS
    
    # Run deep analysis only if violations found
    if any(keyword in name_lower for keyword in ["deep", "comprehensive", "detailed"]):
        return ExecutionPolicy.ON_VIOLATION
    
    # Run PII detection only for high-risk
    if any(keyword in name_lower for keyword in ["pii", "personal", "sensitive"]):
        return ExecutionPolicy.ON_HIGH_RISK
    
    # Default: always run
    return ExecutionPolicy.ALWAYS


# ============================================================================
# Infrastructure Adapter
# ============================================================================


class CriticInfrastructureAdapter:
    """
    Adapter that bridges Production Orchestrator and engine infrastructure.
    
    Provides hooks for:
    - Caching
    - Circuit breakers
    - Evidence recording
    - Event emission
    - Degradation strategies
    - Adaptive concurrency
    """
    
    def __init__(
        self,
        engine: "EngineType",
        evidence_records: Optional[List[Any]] = None,
        degraded_components: Optional[List[str]] = None,
    ):
        self.engine = engine
        self.evidence_records = evidence_records or []
        self.degraded_components = degraded_components or []
    
    async def check_cache(
        self,
        critic_name: str,
        input_snapshot: CriticInput,
    ) -> Optional[Dict[str, Any]]:
        """Check cache for critic result."""
        if not self.engine.cache_manager:
            return None
        
        try:
            cache_key = CacheKey.from_data(
                "critic",
                critic=critic_name,
                input_text=input_snapshot.input_text,
                model_response=input_snapshot.model_response,
                context=input_snapshot.context,
            )
            cached = await self.engine.cache_manager.get(cache_key)
            if cached is not None:
                cached_result = dict(cached)
                cached_result.setdefault("critic", critic_name)
                cached_result["duration_ms"] = 0.0
                cached_result["from_cache"] = True
                logger.debug(f"Cache hit for critic '{critic_name}'")
                return cached_result
        except Exception as e:
            logger.warning(f"Cache check failed for critic '{critic_name}': {e}")
        
        return None
    
    async def record_evidence(
        self,
        critic_name: str,
        result: Dict[str, Any],
        trace_id: str,
        model_response: str,
        context: Dict[str, Any],
    ) -> None:
        """Record evidence for this critic execution."""
        if not self.engine.recorder:
            return
        
        try:
            record = await self.engine.recorder.record(
                critic=critic_name,
                rule_id=str(critic_name),
                severity=result.get("severity") or "INFO",
                violation_description=str(result.get("justification", "")),
                confidence=float(result.get("confidence", 0.0)),
                mitigation=result.get("mitigation"),
                redundancy_group=None,
                detector_metadata=result.get("details", {}),
                context=context,
                raw_text=model_response,
                trace_id=trace_id,
            )
            self.evidence_records.append(record)
        except Exception as record_exc:
            error = EvidenceRecordingError(
                "Evidence recording failed",
                details={"error": str(record_exc), "critic": critic_name},
            )
            self.engine._emit_error(
                error, 
                stage="evidence", 
                trace_id=trace_id, 
                critic=critic_name
            )
    
    async def emit_event(
        self,
        critic_name: str,
        result: Dict[str, Any],
        trace_id: str,
        duration_ms: float,
    ) -> None:
        """Emit critic evaluated event."""
        if not EVENT_BUS_AVAILABLE or not get_event_bus or not EventType:
            return
        
        try:
            event_bus = get_event_bus()
            severity = float(result.get("severity", 0.0))
            violations = list(result.get("violations", []))
            
            event = CriticEvaluatedEvent(
                event_type=EventType.CRITIC_EVALUATED,
                trace_id=trace_id,
                data={"duration_ms": duration_ms},
                critic_name=critic_name,
                severity=severity,
                violations=violations,
            )
            await event_bus.publish(event)
        except Exception as exc:
            logger.debug(f"Failed to publish critic event: {exc}")
    
    async def handle_degradation(
        self,
        critic_name: str,
        error: Exception,
        duration_ms: float,
    ) -> Optional[Dict[str, Any]]:
        """Handle degradation for failed critic."""
        if not self.engine.degradation_enabled:
            return None
        
        try:
            self.degraded_components.append(f"critic:{critic_name}")
            
            fallback = await DegradationStrategy.critic_fallback(
                critic_name=critic_name,
                error=error,
                context={"duration_ms": duration_ms},
            )
            
            return self.engine._build_critic_error_result(
                critic_name=critic_name,
                error=error,
                duration_ms=duration_ms,
                degraded=True,
                degradation_reason=fallback.get("degradation_reason"),
            )
        except Exception as deg_error:
            logger.error(
                f"Degradation strategy failed for critic '{critic_name}': {deg_error}",
                exc_info=True
            )
            return None
    
    def update_adaptive_concurrency(self, duration_ms: float) -> None:
        """Update adaptive concurrency metrics."""
        if self.engine.adaptive_concurrency:
            self.engine.adaptive_concurrency.record_latency(duration_ms)


# ============================================================================
# Critic Callable Creation
# ============================================================================


def create_critic_callable(
    engine: "EngineType",
    critic_name: str,
    critic_ref: "CriticRef",
    model_response: str,
) -> Any:
    """
    Create an async callable for a critic that the orchestrator can execute.
    """
    
    async def wrapped_critic(input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Wrapped critic that conforms to orchestrator interface."""
        # Extract inputs
        input_text = input_dict["input_text"]
        context = input_dict["context"]
        
        # Get or create critic instance
        critic = critic_ref if not inspect.isclass(critic_ref) else critic_ref()
        
        # Resolve model adapter
        bound_adapter = None if context.get("force_model_output") else engine.critic_models.get(critic_name)
        
        if bound_adapter is None:
            # Static response adapter
            class _StaticModelResponse:
                def __init__(self, response: str):
                    self.response = response
                
                async def generate(self, prompt: str, context: Optional[Dict[str, Any]] = None):
                    return self.response
            
            model_adapter = _StaticModelResponse(model_response)
        else:
            # Bound adapter logic
            if hasattr(bound_adapter, "generate"):
                model_adapter = bound_adapter
            elif callable(bound_adapter):
                class _BoundCallable:
                    def __init__(self, fn):
                        self.fn = fn
                    
                    async def generate(self, prompt: str, context: Optional[Dict[str, Any]] = None):
                        res = (
                            self.fn(prompt, context=context)
                            if "context" in inspect.signature(self.fn).parameters
                            else self.fn(prompt)
                        )
                        return await res if inspect.isawaitable(res) else res
                
                model_adapter = _BoundCallable(bound_adapter)
            else:
                # Fallback to static
                class _StaticModelFallback:
                    def __init__(self, response: str):
                        self.response = response
                    
                    async def generate(self, prompt: str, context: Optional[Dict[str, Any]] = None):
                        return self.response
                
                model_adapter = _StaticModelFallback(model_response)
        
        # Execute critic
        evaluate_fn = critic.evaluate
        if inspect.iscoroutinefunction(evaluate_fn):
            result = await evaluate_fn(model_adapter, input_text=input_text, context=context)
        else:
            result = await asyncio.to_thread(
                evaluate_fn, model_adapter, input_text=input_text, context=context
            )
            if inspect.isawaitable(result):
                result = await result
        
        return result
    
    return wrapped_critic


# ============================================================================
# Main Entry Point
# ============================================================================


async def run_critics_with_orchestrator(
    engine: "EngineType",
    model_response: str,
    input_text: str,
    context: dict,
    trace_id: str,
    degraded_components: Optional[List[str]] = None,
    evidence_records: Optional[List[Any]] = None,
) -> CriticResultsMap:
    """
    Run all critics using Production Orchestrator with full infrastructure integration.
    
    This is the main entry point that uses the production-grade orchestrator
    with policy gating, staged execution, retries, and resource management.
    """
    # Create infrastructure adapter
    infrastructure = CriticInfrastructureAdapter(
        engine=engine,
        evidence_records=evidence_records,
        degraded_components=degraded_components,
    )
    
    # Create input snapshot
    input_snapshot = CriticInput(
        model_response=model_response,
        input_text=input_text,
        context=context,
        trace_id=trace_id,
    )
    
    # Define infrastructure hooks
    async def pre_execution_hook(
        critic_name: str,
        critic_input: CriticInput,
    ) -> Optional[Dict[str, Any]]:
        """Check cache before execution."""
        return await infrastructure.check_cache(critic_name, critic_input)
    
    async def post_execution_hook(
        critic_name: str,
        result: Dict[str, Any],
        duration_ms: float,
    ) -> None:
        """Record evidence and emit events after successful execution."""
        # Update adaptive concurrency
        infrastructure.update_adaptive_concurrency(duration_ms)
        
        # Record evidence
        await infrastructure.record_evidence(
            critic_name=critic_name,
            result=result,
            trace_id=trace_id,
            model_response=model_response,
            context=context,
        )
        
        # Emit event
        await infrastructure.emit_event(
            critic_name=critic_name,
            result=result,
            trace_id=trace_id,
            duration_ms=duration_ms,
        )
    
    async def on_failure_hook(
        critic_name: str,
        error: Exception,
        duration_ms: float,
    ) -> Optional[Dict[str, Any]]:
        """Handle failures with degradation strategy."""
        # Update adaptive concurrency
        infrastructure.update_adaptive_concurrency(duration_ms)
        
        # Try degradation
        fallback = await infrastructure.handle_degradation(
            critic_name=critic_name,
            error=error,
            duration_ms=duration_ms,
        )
        
        if fallback:
            return fallback
        
        # Log error
        critic_error = CriticEvaluationError(
            critic_name=critic_name,
            message=str(error),
            trace_id=trace_id,
            details={
                "duration_ms": duration_ms,
                "error_type": type(error).__name__,
            },
        )
        engine._emit_error(
            critic_error,
            stage="critic",
            trace_id=trace_id,
            critic=critic_name,
            context=context,
        )
        
        return None
    
    # Create hooks
    hooks = OrchestratorHooks(
        pre_execution=pre_execution_hook,
        post_execution=post_execution_hook,
        on_failure=on_failure_hook,
    )
    
    # Build critic configurations with production features
    critic_configs = {}
    for critic_name, critic_ref in engine.critics.items():
        # Get or detect configuration
        stage = classify_critic_stage(critic_name, critic_ref)
        priority = get_critic_priority(critic_name, critic_ref)
        policy = get_critic_policy(critic_name, context)
        
        # Get timeout from engine config or use smart defaults
        timeout = engine.config.timeout_seconds
        if stage == ExecutionStage.FAST_CRITICS:
            timeout = min(timeout, 1.0)  # Fast critics max 1s
        elif stage == ExecutionStage.DEEP_ANALYSIS:
            timeout = max(timeout, 5.0)  # Deep analysis min 5s
        
        # Create callable
        critic_callable = create_critic_callable(
            engine=engine,
            critic_name=critic_name,
            critic_ref=critic_ref,
            model_response=model_response,
        )
        
        # Create config with production features
        critic_configs[critic_name] = CriticConfig(
            name=critic_name,
            callable=critic_callable,
            stage=stage,
            priority=priority,
            timeout_seconds=timeout,
            execution_policy=policy,
            max_retries=2 if policy != ExecutionPolicy.ALWAYS else 0,  # Retry non-critical
            retry_on_timeout=True,
            required_output_fields={"value", "score"},  # Basic validation
        )
    
    # Get orchestrator config
    config = get_orchestrator_config()
    
    # Create and run orchestrator
    orchestrator = ProductionOrchestrator(
        critics=critic_configs,
        config=config,
        hooks=hooks,
    )
    
    logger.info(
        f"Running {len(critic_configs)} critics with Production Orchestrator "
        f"(gating={'on' if config.enable_policy_gating else 'off'}, "
        f"retries={'on' if config.enable_retries else 'off'})"
    )
    
    # Execute all critics
    results = await orchestrator.run_all(input_snapshot)
    
    # Log metrics
    metrics = orchestrator.metrics
    logger.info(
        f"Orchestrator execution complete: "
        f"total={metrics.total_executions}, "
        f"success={metrics.successful}, "
        f"failed={metrics.failed}, "
        f"gated={metrics.gated}, "
        f"retried={metrics.retried}"
    )
    
    return results


__all__ = [
    "CriticInfrastructureAdapter",
    "run_critics_with_orchestrator",
    "create_critic_callable",
    "get_orchestrator_config",
]
