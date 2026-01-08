import asyncio
import inspect
import logging
import time
from typing import Any, Dict, Optional, cast

from engine.cache import CacheKey
from engine.exceptions import DetectorExecutionError, RouterSelectionError

# Enhanced observability
try:
    from engine.observability.tracing import run_router_with_trace
    from engine.observability.cost_tracking import record_llm_call, extract_token_usage
    from engine.events.event_bus import get_event_bus, RouterSelectedEvent, EventType
    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False
    run_router_with_trace = None
    record_llm_call = None
    extract_token_usage = None
    get_event_bus = None
    RouterSelectedEvent = None
    EventType = None

logger = logging.getLogger("engine.engine")


async def run_detectors(
    engine: Any,
    text: str,
    context: Dict[str, Any],
    timings: Dict[str, float],
) -> Optional[Dict[str, Any]]:
    if not engine.detector_engine:
        return None
    cache_key: Optional[CacheKey] = None
    if engine.cache_manager:
        cache_key = CacheKey.from_data("detector", text=text, context=context)
        cached = await engine.cache_manager.get(cache_key)
        if cached is not None:
            return cast(Optional[Dict[str, Any]], cached)

    start = asyncio.get_event_loop().time()
    try:
        signals = await engine.detector_engine.detect_all(text, context)
        summary = engine.detector_engine.aggregate_signals(signals)
    except Exception as exc:
        raise DetectorExecutionError(
            "Detector execution failed",
            details={"error": str(exc)},
        ) from exc
    end = asyncio.get_event_loop().time()
    timings["detectors_ms"] = (end - start) * 1000

    converted_signals = {
        name: (sig.model_dump() if hasattr(sig, "model_dump") else sig)
        for name, sig in signals.items()
    }
    result = {
        **summary,
        "signals": converted_signals,
    }
    if engine.cache_manager and cache_key:
        await engine.cache_manager.set(cache_key, result)
    return result


async def select_model(
    engine: Any,
    text: str,
    context: dict,
    timings: Optional[Dict[str, float]] = None,
    router_diagnostics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if timings is None:
        timings = {}
    if router_diagnostics is None:
        router_diagnostics = {}
    start = asyncio.get_event_loop().time()
    cache_key: Optional[CacheKey] = None

    if engine.cache_manager:
        cache_key = CacheKey.from_data("router", text=text, context=context)
        cached = await engine.cache_manager.get(cache_key)
        if cached is not None:
            router_diagnostics.update({"cache": "exact"})
            return cast(Dict[str, Any], cached)

    if engine.router_cache:
        similar = engine.router_cache.get_similar(text, context)
        if similar is not None:
            router_diagnostics.update({"cache": "similar"})
            if engine.cache_manager and cache_key:
                await engine.cache_manager.set(cache_key, similar)
            return similar

    try:
        # Use tracing if available
        if OBSERVABILITY_AVAILABLE and run_router_with_trace:
            router_result = await run_router_with_trace(
                engine.router.route,
                text,
                context,
            )
            if inspect.isawaitable(router_result):
                router_result = await router_result
        else:
            call = engine.router.route(text=text, context=context)
            router_result = await call if inspect.isawaitable(call) else call
    except RouterSelectionError:
        end = asyncio.get_event_loop().time()
        timings["router_selection_ms"] = (end - start) * 1000
        raise
    except Exception as exc:
        end = asyncio.get_event_loop().time()
        timings["router_selection_ms"] = (end - start) * 1000
        raise RouterSelectionError(
            "Router failed to select a model",
            details={"error": str(exc)},
        ) from exc

    if not router_result or router_result.get("response_text") is None:
        end = asyncio.get_event_loop().time()
        timings["router_selection_ms"] = (end - start) * 1000
        raise RouterSelectionError(
            "Router returned no response",
            details={"router_result": router_result},
        )

    end = asyncio.get_event_loop().time()
    timings["router_selection_ms"] = (end - start) * 1000
    router_diagnostics.update(router_result.get("diagnostics", {}) or {})

    model_info = {
        "model_name": router_result.get("model_name"),
        "model_version": router_result.get("model_version"),
        "router_selection_reason": router_result.get("reason"),
        "health_score": router_result.get("health_score"),
        "cost_estimate": router_result.get("cost"),
    }

    selection = {
        "model_info": model_info,
        "response_text": router_result.get("response_text") or "",
    }
    
    # Record cost tracking if available
    if OBSERVABILITY_AVAILABLE and record_llm_call:
        try:
            # Extract token usage from router result
            input_tokens, output_tokens = extract_token_usage(router_result)
            latency = timings.get("router_selection_ms", 0) / 1000.0
            model_name = model_info.get("model_name", "unknown")
            provider = "unknown"  # Could be extracted from router_result
            record_llm_call(
                model=model_name,
                provider=provider,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_seconds=latency,
            )
        except Exception as exc:
            logger.debug(f"Failed to record LLM cost: {exc}")
    
    # Publish router event if event bus available
    if OBSERVABILITY_AVAILABLE and get_event_bus and EventType is not None:
        try:
            event_bus = get_event_bus()
            trace_id = context.get("trace_id", "unknown")
            event = RouterSelectedEvent(
                event_type=EventType.ROUTER_SELECTED,
                timestamp=None,  # Will be set in __post_init__
                trace_id=trace_id,
                data={"diagnostics": router_diagnostics},
                model_name=model_info.get("model_name", "unknown"),
                selection_reason=model_info.get("router_selection_reason", "unknown"),
                cost_estimate=model_info.get("cost_estimate"),
            )
            await event_bus.publish(event)
        except Exception as exc:
            logger.debug(f"Failed to publish router event: {exc}")
    
    if engine.cache_manager and cache_key:
        await engine.cache_manager.set(cache_key, selection)
    if engine.router_cache:
        engine.router_cache.set(text, context, selection)
    return selection
