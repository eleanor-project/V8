from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict

logger = logging.getLogger(__name__)

_dependency_failures: Dict[str, Dict[str, Any]] = {}


def record_dependency_failure(name: str, exc: Exception) -> None:
    """Record and log when a dependency cannot be loaded."""
    now = datetime.now(timezone.utc).isoformat()
    entry = _dependency_failures.setdefault(
        name,
        {"count": 0, "last_failure": None, "last_error": None},
    )
    entry["count"] += 1
    entry["last_failure"] = now
    entry["last_error"] = str(exc)
    logger.warning(
        "dependency_failed_to_load",
        extra={"dependency": name, "error": str(exc)},
        exc_info=exc,
    )


def get_dependency_metrics() -> Dict[str, Any]:
    """Return a snapshot of dependency load failures."""
    failures = {name: info.copy() for name, info in _dependency_failures.items()}
    total_failures = sum(info.get("count", 0) for info in failures.values())
    return {
        "failures": failures,
        "total_failures": total_failures,
        "tracked_dependencies": len(failures),
        "last_checked": datetime.now(timezone.utc).isoformat(),
    }


__all__ = ["record_dependency_failure", "get_dependency_metrics"]
