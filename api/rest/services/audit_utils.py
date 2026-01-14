from __future__ import annotations

from typing import Any, Dict, List, Optional

from engine.logging_config import get_logger

logger = get_logger(__name__)


def _call_first(obj: Any, candidates: List[str], *args, **kwargs):
    """
    Call the first available method name in `candidates` on obj.
    Returns (called, result).
    """
    for name in candidates:
        fn = getattr(obj, name, None)
        if callable(fn):
            return True, fn(*args, **kwargs)
    return False, None


async def _call_first_async(obj: Any, candidates: List[str], *args, **kwargs):
    for name in candidates:
        fn = getattr(obj, name, None)
        if callable(fn):
            result = fn(*args, **kwargs)
            if hasattr(result, "__await__"):
                return True, await result
            return True, result
    return False, None


async def fetch_trace(replay_store: Any, trace_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve the stored trace packet for a trace_id.
    Expected result shape:
      {"trace_id", "input", "context", "response", "timestamp", ...}
    """
    called, result = await _call_first_async(
        replay_store,
        ["get", "get_trace", "load", "load_trace", "fetch", "fetch_trace", "read"],
        trace_id,
    )
    if called and result:
        return result

    # Some implementations require positional args
    called, result = await _call_first_async(
        replay_store,
        ["get", "get_trace", "load", "load_trace", "fetch", "fetch_trace", "read"],
        trace_id,
    )
    if called and result:
        return result

    return None


async def search_traces(
    replay_store: Any,
    *,
    query: Optional[str] = None,
    user_id: Optional[str] = None,
    decision: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """
    Search traces. We support multiple store APIs:
      - replay_store.search(...)
      - replay_store.query(...)
      - replay_store.list(...) with filters
      - replay_store.recent(limit=...)
    """
    # Primary (explicit search)
    called, result = await _call_first_async(
        replay_store,
        candidates=["search", "query", "find"],
        query=query,
        user_id=user_id,
        decision=decision,
        limit=limit,
    )
    if called and isinstance(result, list):
        return result

    # Secondary (list/recent)
    called, result = await _call_first_async(
        replay_store,
        candidates=["list", "recent", "latest"],
        limit=limit,
    )
    if called and isinstance(result, list):
        # apply basic filtering if store does not
        filtered = []
        for item in result:
            if not isinstance(item, dict):
                continue
            if decision and str(item.get("response", {}).get("final_decision", "")).lower() != decision.lower():
                continue
            if user_id and str(item.get("user_id", "")) != str(user_id):
                continue
            if query:
                blob = str(item.get("input", "")) + " " + str(item.get("trace_id", ""))
                if query.lower() not in blob.lower():
                    continue
            filtered.append(item)
        return filtered[:limit]

    return []


async def replay_trace(engine: Any, stored: Dict[str, Any]) -> Dict[str, Any]:
    """
    Re-run the engine on the stored input/context. Returns engine response payload.
    """
    input_text = stored.get("input") or ""
    context = stored.get("context") or {}
    trace_id = stored.get("trace_id")

    run_fn = getattr(engine, "run", None)
    if run_fn is None:
        raise RuntimeError("Engine does not expose run() for replay")

    result = await run_fn(input_text, context=context, trace_id=trace_id, detail_level=3)
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if hasattr(result, "dict"):
        return result.dict()
    return result
