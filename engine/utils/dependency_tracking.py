from __future__ import annotations

import logging
from collections import Counter
from typing import Dict

logger = logging.getLogger(__name__)

_dependency_failures: Counter[str] = Counter()


def record_dependency_failure(name: str, exc: Exception) -> None:
    """Record and log when a dependency cannot be loaded."""
    _dependency_failures[name] += 1
    logger.warning(
        "dependency_failed_to_load",
        extra={"dependency": name, "error": str(exc)},
        exc_info=exc,
    )


def get_dependency_metrics() -> Dict[str, int]:
    """Return a snapshot of dependency load failures."""
    return dict(_dependency_failures)


__all__ = ["record_dependency_failure", "get_dependency_metrics"]
