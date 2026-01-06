import asyncio
import inspect
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional, Type, cast

from engine.cache import CacheKey
from engine.exceptions import (
    AggregationError,
    PrecedentRetrievalError,
    UncertaintyComputationError,
)
from engine.schemas.pipeline_types import (
    AggregationOutput,
    CriticResultsMap,
    PrecedentAlignmentResult,
    PrecedentRetrievalResult,
    UncertaintyResult,
)
from engine.utils.circuit_breaker import CircuitBreakerOpen

logger = logging.getLogger("engine.engine")


async def execute_with_degradation(
    engine: Any,
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
    try:
        breaker = engine._get_circuit_breaker(stage)
        if breaker is not None:
            result = await breaker.call(run_fn)
        else:
            result = await run_fn()
        return result, False, None
    except CircuitBreakerOpen as exc:
        if engine.degradation_enabled:
            degraded_components.append(degrade_component)
            fallback = await fallback_fn(exc)
            return fallback, True, exc
        raise
    except error_type as exc:
        engine._emit_error(exc, stage=stage, trace_id=trace_id, context=context)
        if engine.degradation_enabled:
            degraded_components.append(degrade_component)
            fallback = await fallback_fn(exc)
            return fallback, True, exc
        fallback_result = fallback_on_error(exc) if fallback_on_error else None
        return fallback_result, False, exc


async def run_precedent_alignment(
    engine: Any,
    critic_results: CriticResultsMap,
    trace_id: str,
    text: str = "",
    timings: Optional[Dict[str, float]] = None,
    inspect_module: Any = inspect,
) -> Optional[PrecedentAlignmentResult]:
    if not engine.precedent_engine:
        return None
    start = asyncio.get_event_loop().time()
    cases: List[Dict[str, Any]] = []
    query_embedding: List[float] = []
    retrieval_meta: Optional[PrecedentRetrievalResult] = None

    if engine.precedent_retriever:
        try:
            retrieve_fn = engine.precedent_retriever.retrieve

            async def _call_retriever() -> Optional[PrecedentRetrievalResult]:
                result = retrieve_fn(
                    text,
                    list(critic_results.values()),
                )
                if inspect_module.isawaitable(result):
                    return cast(Optional[PrecedentRetrievalResult], await result)
                return cast(Optional[PrecedentRetrievalResult], result)

            if engine.cache_manager:
                cache_key = CacheKey.from_data(
                    "precedent",
                    query_text=text,
                    critics=critic_results,
                )
                cached = await engine.cache_manager.get(cache_key)
                if cached is not None:
                    retrieval_meta = cast(PrecedentRetrievalResult, cached)
                else:
                    if inspect_module.iscoroutinefunction(retrieve_fn):
                        retrieval_meta = await _call_retriever()
                    else:
                        retrieval_meta = await asyncio.to_thread(
                            retrieve_fn, text, list(critic_results.values())
                        )
                        if inspect_module.isawaitable(retrieval_meta):
                            retrieval_meta = await retrieval_meta
                    if retrieval_meta is not None:
                        await engine.cache_manager.set(cache_key, retrieval_meta)
            else:
                if inspect_module.iscoroutinefunction(retrieve_fn):
                    retrieval_meta = await _call_retriever()
                else:
                    retrieval_meta = await asyncio.to_thread(
                        retrieve_fn, text, list(critic_results.values())
                    )
                    if inspect_module.isawaitable(retrieval_meta):
                        retrieval_meta = await retrieval_meta
            retrieval_meta_dict = cast(PrecedentRetrievalResult, retrieval_meta or {})
            cases = cast(
                List[Dict[str, Any]],
                retrieval_meta_dict.get("precedent_cases")
                or retrieval_meta_dict.get("cases")
                or [],
            )
            query_embedding = cast(
                List[float],
                retrieval_meta_dict.get("query_embedding") or [],
            )
        except Exception as exc:
            raise PrecedentRetrievalError(
                "Precedent retrieval failed",
                details={"error": str(exc), "trace_id": trace_id},
            ) from exc

    analyze_fn = getattr(engine.precedent_engine, "analyze", None)
    try:
        if analyze_fn:
            out = analyze_fn(
                critics=critic_results,
                precedent_cases=cases,
                query_embedding=query_embedding,
            )
        else:
            out = None
    except Exception as exc:
        raise PrecedentRetrievalError(
            "Precedent alignment failed",
            details={"error": str(exc), "trace_id": trace_id},
        ) from exc

    out_result: Optional[PrecedentAlignmentResult] = cast(
        Optional[PrecedentAlignmentResult],
        out,
    )
    if retrieval_meta:
        out_result = cast(
            PrecedentAlignmentResult,
            {**(out_result or {}), "retrieval": retrieval_meta},
        )

    end = asyncio.get_event_loop().time()
    if timings is not None:
        timings["precedent_alignment_ms"] = (end - start) * 1000
    return out_result


async def run_uncertainty_engine(
    engine: Any,
    precedent_alignment: Optional[PrecedentAlignmentResult],
    critic_results: CriticResultsMap,
    model_name: str = "unknown-model",
    timings: Optional[Dict[str, float]] = None,
) -> Optional[UncertaintyResult]:
    if not engine.uncertainty_engine:
        return None
    start = asyncio.get_event_loop().time()
    compute_fn = getattr(engine.uncertainty_engine, "compute", None) or getattr(
        engine.uncertainty_engine, "evaluate", None
    )
    if not compute_fn:
        return None

    try:
        out = compute_fn(
            critics=critic_results,
            model_used=model_name,
            precedent_alignment=precedent_alignment or {},
        )
        if inspect.isawaitable(out):
            out = await out
    except Exception as exc:
        raise UncertaintyComputationError(
            "Uncertainty computation failed",
            details={"error": str(exc)},
        ) from exc
    end = asyncio.get_event_loop().time()
    if timings is not None:
        timings["uncertainty_engine_ms"] = (end - start) * 1000
    return cast(UncertaintyResult, out)


async def aggregate_results(
    engine: Any,
    critic_results: CriticResultsMap,
    model_response: str,
    precedent_data: Optional[PrecedentAlignmentResult] = None,
    uncertainty_data: Optional[UncertaintyResult] = None,
    timings: Optional[Dict[str, float]] = None,
) -> AggregationOutput:
    if not engine.aggregator:
        raise AggregationError("AggregatorV8 not available")
    start = asyncio.get_event_loop().time()
    try:
        agg_result = engine.aggregator.aggregate(
            critics=critic_results,
            precedent=precedent_data or {},
            uncertainty=uncertainty_data or {},
            model_output=model_response,
        )
        out = await agg_result if inspect.isawaitable(agg_result) else agg_result
    except Exception as exc:
        raise AggregationError(
            "Aggregation failed",
            details={"error": str(exc)},
        ) from exc

    end = asyncio.get_event_loop().time()
    if timings is not None:
        timings["aggregation_ms"] = (end - start) * 1000
    return cast(AggregationOutput, out)
