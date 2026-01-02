"""
ELEANOR V8 — Observability Framework

Structured logging and distributed tracing for production observability.
"""

from .logging import configure_logging, get_logger
from .tracing import configure_tracing, get_tracer, TraceContext

__all__ = [
    "configure_logging",
    "get_logger",
    "configure_tracing",
    "get_tracer",
    "TraceContext",
]
