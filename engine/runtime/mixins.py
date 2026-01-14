from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, List, Optional, Type, cast
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engine.runtime.models import EngineResult
    from engine.schemas.pipeline_types import (
        AggregationOutput,
        CriticResult,
        CriticResultsMap,
        PrecedentAlignmentResult,
        UncertaintyResult,
    )

from engine.runtime.config import EngineConfig, load_config_from_yaml
from engine.runtime.critics import (
    process_critic_batch,
    run_single_critic,
    run_single_critic_with_breaker,
)
from engine.runtime.critic_infrastructure import run_critics_with_orchestrator
from engine.runtime.errors import (
    build_aggregation_fallback,
    build_critic_error_result,
    emit_error,
    emit_validation_error,
    validate_inputs,
)
from engine.runtime.governance import (
    build_case_for_review,
    calculate_critic_disagreement,
    collect_citations,
    collect_uncertainty_flags,
)
from engine.runtime.lifecycle import setup_resources, shutdown_engine
from engine.runtime.models import EngineResult
from engine.runtime.pipeline import (
    aggregate_results,
    execute_with_degradation,
    run_precedent_alignment,
    run_uncertainty_engine,
)
from engine.runtime.routing import run_detectors, select_model
from engine.runtime.run import run_engine
from engine.runtime.streaming import run_stream_engine
from engine.schemas.pipeline_types import (
    AggregationOutput,
    CriticResult,
    CriticResultsMap,
    PrecedentAlignmentResult,
    UncertaintyResult,
)
from governance.review_triggers import Case


class EngineRuntimeMixin:
    """
    Engine runtime mixin with proper async context manager implementation.
    
    Fixed issues:
    - Proper exception handling in __aexit__
    - Resource cleanup on exceptions
    - Type safety improvements
    """
    
    async def __aenter__(self) -> "EngineRuntimeMixin":
        """
        Async context manager entry.
        
        Sets up resources and returns self.
        Raises exception if setup fails.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            await self._setup_resources()
            return self
        except Exception as exc:
            logger.error(
                "engine_setup_failed",
                extra={"error": str(exc), "error_type": type(exc).__name__},
                exc_info=True,
            )
            # Clean up any partially initialized resources
            try:
                await self.shutdown(timeout=5.0)
            except Exception as cleanup_exc:
                logger.warning(
                    "cleanup_after_setup_failure_failed",
                    extra={"error": str(cleanup_exc)},
                )
            raise

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> bool:
        """
        Async context manager exit.
        
        Properly handles exceptions and ensures cleanup.
        
        Args:
            exc_type: Exception type if exception occurred
            exc_val: Exception value if exception occurred
            exc_tb: Exception traceback if exception occurred
        
        Returns:
            False to propagate exceptions, True to suppress them
        """
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            await self.shutdown()
        except Exception as shutdown_exc:
            logger.error(
                "engine_shutdown_failed",
                extra={
                    "error": str(shutdown_exc),
                    "error_type": type(shutdown_exc).__name__,
                    "original_exception": str(exc_val) if exc_val else None,
                },
                exc_info=True,
            )
            # Don't suppress original exception if shutdown fails
            if exc_val:
                raise exc_val from shutdown_exc
        
        # Return False to propagate any exceptions that occurred
        return False

    async def _setup_resources(self) -> None:
        await setup_resources(self)

    async def shutdown(self, *, timeout: Optional[float] = None) -> None:
        await shutdown_engine(self, timeout=timeout)

    def _emit_error(
        self,
        exc: Exception,
        *,
        stage: str,
        trace_id: Optional[str] = None,
        critic: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        emit_error(
            self,
            exc,
            stage=stage,
            trace_id=trace_id,
            critic=critic,
            context=context,
            extra=extra,
        )

    def _emit_validation_error(
        self,
        exc: "InputValidationError",
        *,
        text: Any,
        context: Any,
        trace_id: Optional[str],
    ) -> None:
        emit_validation_error(
            self,
            exc,
            text=text,
            context=context,
            trace_id=trace_id,
            json_module=self._json_module,
        )

    def _validate_inputs(
        self,
        text: str,
        context: Optional[dict],
        trace_id: Optional[str],
        detail_level: Optional[int],
    ) -> tuple[str, Dict[str, Any], str, int]:
        return validate_inputs(self, text, context, trace_id, detail_level)

    def _build_critic_error_result(
        self,
        critic_name: str,
        error: Exception,
        duration_ms: Optional[float] = None,
        *,
        degraded: bool = False,
        degradation_reason: Optional[str] = None,
    ) -> CriticResult:
        return cast(
            CriticResult,
            build_critic_error_result(
                critic_name,
                error,
                duration_ms,
                degraded=degraded,
                degradation_reason=degradation_reason,
            ),
        )

    def _build_aggregation_fallback(
        self,
        model_response: str,
        precedent_data: Optional[PrecedentAlignmentResult],
        uncertainty_data: Optional[UncertaintyResult],
        error: Exception,
    ) -> AggregationOutput:
        return build_aggregation_fallback(
            model_response,
            precedent_data,
            uncertainty_data,
            error,
        )

    def _get_circuit_breaker(self, name: str):
        registry = getattr(self, "_circuit_breakers", None) or self.circuit_breakers
        if not registry:
            return None
        return registry.get_or_create(
            name,
            failure_threshold=self._breaker_failure_threshold,
            recovery_timeout=self._breaker_recovery_timeout,
            success_threshold=2,
        )

    async def _run_detectors(
        self,
        text: str,
        context: Dict[str, Any],
        timings: Dict[str, float],
    ) -> Optional[Dict[str, Any]]:
        return await run_detectors(self, text, context, timings)

    async def _select_model(
        self,
        text: str,
        context: Optional[dict] = None,
        timings: Optional[Dict[str, float]] = None,
        router_diagnostics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return await select_model(self, text, context or {}, timings, router_diagnostics)

    async def _select_router(
        self,
        text: str,
        context: Optional[dict] = None,
        timings: Optional[Dict[str, float]] = None,
        router_diagnostics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return await self._select_model(text, context or {}, timings, router_diagnostics)

    async def _run_single_critic(
        self,
        name: str,
        critic_ref: Any,
        model_response: str,
        input_text: str,
        context: dict,
        trace_id: str,
        evidence_records: Optional[List[Any]] = None,
    ) -> CriticResult:
        return await run_single_critic(
            self,
            name,
            critic_ref,
            model_response,
            input_text,
            context,
            trace_id,
            evidence_records,
        )

    async def _run_single_critic_with_breaker(
        self,
        name: str,
        critic_ref: Any,
        model_response: str,
        input_text: str,
        context: dict,
        trace_id: str,
        degraded_components: Optional[List[str]] = None,
        evidence_records: Optional[List[Any]] = None,
    ) -> CriticResult:
        return await run_single_critic_with_breaker(
            self,
            name,
            critic_ref,
            model_response,
            input_text,
            context,
            trace_id,
            degraded_components,
            evidence_records,
        )

    async def _process_critic_batch(
        self,
        items: List[tuple[Any, ...]],
    ) -> List[Any]:
        return await process_critic_batch(self, items)

    async def _run_critics_parallel(
        self,
        model_response: str,
        context: dict,
        trace_id: str,
        input_text: Optional[str] = None,
        degraded_components: Optional[List[str]] = None,
        evidence_records: Optional[List[Any]] = None,
    ) -> CriticResultsMap:
        # Ensure we always route through OrchestratorV2 (single critic execution path).
        effective_input_text = context.get("input_text_override") or input_text or ""
        if not isinstance(effective_input_text, str):
            effective_input_text = str(effective_input_text)

        return await run_critics_with_orchestrator(
            self,
            model_response=model_response,
            input_text=effective_input_text,
            context=context,
            trace_id=trace_id,
            degraded_components=degraded_components,
            evidence_records=evidence_records,
        )

    async def _execute_with_degradation(
        self,
        *,
        stage: str,
        run_fn: Callable[[], Awaitable[Any]],
        fallback_fn: Callable[[Exception], Awaitable[Any]],
        error_type: Type[Exception],
        degraded_components: List[str],
        degrade_component: str,
        context: Dict[str, Any],
        trace_id: Optional[str],
        fallback_on_error: Optional[Callable[[Exception], Any]] = None,
    ) -> tuple[Any, bool, Optional[Exception]]:
        return await execute_with_degradation(
            self,
            stage=stage,
            run_fn=run_fn,
            fallback_fn=fallback_fn,
            error_type=error_type,
            degraded_components=degraded_components,
            degrade_component=degrade_component,
            context=context,
            trace_id=trace_id,
            fallback_on_error=fallback_on_error,
        )

    async def _run_precedent_alignment(
        self,
        critic_results: CriticResultsMap,
        trace_id: str,
        text: str = "",
        timings: Optional[Dict[str, float]] = None,
    ) -> Optional[PrecedentAlignmentResult]:
        return await run_precedent_alignment(
            self,
            critic_results,
            trace_id,
            text=text,
            timings=timings,
            inspect_module=self._inspect_module,
        )

    async def _run_uncertainty_engine(
        self,
        precedent_alignment: Optional[PrecedentAlignmentResult],
        critic_results: CriticResultsMap,
        model_name: str = "unknown-model",
        timings: Optional[Dict[str, float]] = None,
    ) -> Optional[UncertaintyResult]:
        return await run_uncertainty_engine(
            self,
            precedent_alignment,
            critic_results,
            model_name=model_name,
            timings=timings,
        )

    async def _aggregate_results(
        self,
        critic_results: CriticResultsMap,
        model_response: str,
        precedent_data: Optional[PrecedentAlignmentResult] = None,
        uncertainty_data: Optional[UncertaintyResult] = None,
        timings: Optional[Dict[str, float]] = None,
    ) -> AggregationOutput:
        return await aggregate_results(
            self,
            critic_results,
            model_response,
            precedent_data,
            uncertainty_data,
            timings,
        )

    def _calculate_critic_disagreement(self, critic_outputs: CriticResultsMap) -> float:
        return calculate_critic_disagreement(critic_outputs)

    def _collect_citations(self, critic_outputs: CriticResultsMap) -> Dict[str, Any]:
        return collect_citations(critic_outputs)

    def _collect_uncertainty_flags(
        self, uncertainty_data: Optional[UncertaintyResult]
    ) -> List[str]:
        return collect_uncertainty_flags(uncertainty_data)

    def _build_case_for_review(
        self,
        trace_id: str,
        context: Dict[str, Any],
        aggregated: AggregationOutput,
        critic_results: CriticResultsMap,
        precedent_data: Optional[PrecedentAlignmentResult],
        uncertainty_data: Optional[UncertaintyResult],
    ) -> Case:
        return build_case_for_review(
            trace_id,
            context,
            aggregated,
            critic_results,
            precedent_data,
            uncertainty_data,
        )

    async def run(
        self,
        text: str,
        context: Optional[dict] = None,
        *,
        detail_level: Optional[int] = None,
        trace_id: Optional[str] = None,
    ) -> EngineResult:
        return await run_engine(
            self,
            text,
            context,
            detail_level=detail_level,
            trace_id=trace_id,
        )

    async def run_stream(
        self,
        text: str,
        context: Optional[dict] = None,
        *,
        detail_level: Optional[int] = None,
        trace_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        async for event in run_stream_engine(
            self,
            text,
            context,
            detail_level=detail_level,
            trace_id=trace_id,
        ):
            yield event

    @staticmethod
    def load_config_from_yaml(path: str) -> EngineConfig:
        return load_config_from_yaml(path)
