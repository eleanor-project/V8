import asyncio
import inspect
import logging
from typing import Any, Dict, Optional, cast

from engine.cache import CacheKey
from engine.exceptions import DetectorExecutionError, RouterSelectionError

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
    if engine.cache_manager and cache_key:
        await engine.cache_manager.set(cache_key, selection)
    if engine.router_cache:
        engine.router_cache.set(text, context, selection)
    return selection
