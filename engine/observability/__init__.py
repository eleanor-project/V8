"""
ELEANOR V8 — Observability Framework

Structured logging and distributed tracing for production observability.
"""

from .logging import configure_logging, get_logger
try:
    from .tracing import (
        configure_tracing,
        get_tracer,
        TraceContext,
        trace_operation,
        trace_span,
        run_critic_with_trace,
        run_router_with_trace,
    )
    TRACING_AVAILABLE = True
except ImportError:
    TRACING_AVAILABLE = False
    TraceContext = None
    trace_operation = None
    trace_span = None
    run_critic_with_trace = None
    run_router_with_trace = None

from .business_metrics import (
    record_decision,
    record_escalation,
    record_critic_agreement,
    record_uncertainty,
    record_severity,
    record_critic_disagreement,
    set_active_traces,
    set_degraded_components,
    record_engine_result,
)

from .correlation import (
    CorrelationContext,
    with_correlation_id,
    get_correlation_id,
)

from .cost_tracking import (
    calculate_cost,
    record_llm_call,
    extract_token_usage,
)

__all__ = [
    "configure_logging",
    "get_logger",
    "TraceContext",
    "trace_operation",
    "trace_span",
    "run_critic_with_trace",
    "run_router_with_trace",
    "record_decision",
    "record_escalation",
    "record_critic_agreement",
    "record_uncertainty",
    "record_severity",
    "record_critic_disagreement",
    "set_active_traces",
    "set_degraded_components",
    "record_engine_result",
    "CorrelationContext",
    "with_correlation_id",
    "get_correlation_id",
    "calculate_cost",
    "record_llm_call",
    "extract_token_usage",
]
