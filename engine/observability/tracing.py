"""
ELEANOR V8 — Enhanced Distributed Tracing
-----------------------------------------

Custom spans and trace context management for better observability.
"""

import logging
from typing import Optional, Dict, Any, Callable
from functools import wraps
from contextlib import contextmanager

logger = logging.getLogger(__name__)

try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None
    Status = None
    StatusCode = None
    TraceContextTextMapPropagator = None


class TraceContext:
    """Manage trace context across async operations."""
    
    _propagator = TraceContextTextMapPropagator() if OTEL_AVAILABLE else None
    
    @staticmethod
    def get_current_trace_id() -> Optional[str]:
        """Get current trace ID from context."""
        if not OTEL_AVAILABLE:
            return None
        
        span = trace.get_current_span()
        if span and span.get_span_context().is_valid:
            return format(span.get_span_context().trace_id, "032x")
        return None
    
    @staticmethod
    def get_current_span_id() -> Optional[str]:
        """Get current span ID from context."""
        if not OTEL_AVAILABLE:
            return None
        
        span = trace.get_current_span()
        if span and span.get_span_context().is_valid:
            return format(span.get_span_context().span_id, "016x")
        return None
    
    @staticmethod
    def inject_context(carrier: Dict[str, str]) -> None:
        """Inject trace context into carrier dict."""
        if not OTEL_AVAILABLE or not TraceContext._propagator:
            return
        
        span = trace.get_current_span()
        if span:
            TraceContext._propagator.inject(carrier, span.get_span_context())
    
    @staticmethod
    def extract_context(carrier: Dict[str, str]) -> Optional[Any]:
        """Extract trace context from carrier dict."""
        if not OTEL_AVAILABLE or not TraceContext._propagator:
            return None
        
        return TraceContext._propagator.extract(carrier)


def get_tracer(name: str):
    """Get tracer instance."""
    if not OTEL_AVAILABLE:
        return None
    return trace.get_tracer(name)


def trace_operation(
    operation_name: str,
    attributes: Optional[Dict[str, Any]] = None,
    record_exception: bool = True,
):
    """
    Decorator to trace an operation.
    
    Usage:
        @trace_operation("critic.evaluate", attributes={"critic.name": "rights"})
        async def evaluate(...):
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not OTEL_AVAILABLE:
                return await func(*args, **kwargs)
            
            tracer = get_tracer(func.__module__)
            if not tracer:
                return await func(*args, **kwargs)
            
            with tracer.start_as_current_span(operation_name) as span:
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, str(value))
                
                try:
                    result = await func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as exc:
                    if record_exception:
                        span.record_exception(exc)
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not OTEL_AVAILABLE:
                return func(*args, **kwargs)
            
            tracer = get_tracer(func.__module__)
            if not tracer:
                return func(*args, **kwargs)
            
            with tracer.start_as_current_span(operation_name) as span:
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, str(value))
                
                try:
                    result = func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as exc:
                    if record_exception:
                        span.record_exception(exc)
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    raise
        
        # Return appropriate wrapper based on function type
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


@contextmanager
def trace_span(operation_name: str, attributes: Optional[Dict[str, Any]] = None):
    """
    Context manager for tracing operations.
    
    Usage:
        with trace_span("database.query", attributes={"query": "SELECT ..."}):
            result = await db.execute(...)
    """
    if not OTEL_AVAILABLE:
        yield
        return
    
    tracer = get_tracer(__name__)
    if not tracer:
        yield
        return
    
    with tracer.start_as_current_span(operation_name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, str(value))
        yield span


async def run_critic_with_trace(
    critic_name: str,
    critic_func: Callable,
    *args,
    **kwargs,
) -> Any:
    """
    Run critic evaluation with tracing.
    
    Args:
        critic_name: Name of the critic
        critic_func: Critic evaluation function
        *args: Positional arguments
        **kwargs: Keyword arguments (should include input_text, model_response, etc.)
    
    Returns:
        Critic result
    """
    if not OTEL_AVAILABLE:
        return await critic_func(*args, **kwargs)
    
    tracer = get_tracer(__name__)
    if not tracer:
        return await critic_func(*args, **kwargs)
    
    input_text = kwargs.get("input_text", "")
    model_response = kwargs.get("model_response", "")
    
    with tracer.start_as_current_span(
        f"critic.{critic_name}",
        attributes={
            "critic.name": critic_name,
            "input.length": len(input_text),
            "model_response.length": len(model_response),
        },
    ) as span:
        try:
            result = await critic_func(*args, **kwargs)
            
            # Add result attributes
            if hasattr(result, "severity"):
                span.set_attribute("critic.severity", float(result.severity))
            if hasattr(result, "confidence"):
                span.set_attribute("critic.confidence", float(result.confidence))
            if hasattr(result, "violations"):
                span.set_attribute("critic.violations.count", len(result.violations))
            
            span.set_status(Status(StatusCode.OK))
            return result
        
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise


async def run_router_with_trace(
    router_func: Callable,
    text: str,
    context: Dict[str, Any],
    *args,
    **kwargs,
) -> Any:
    """Run router selection with tracing."""
    if not OTEL_AVAILABLE:
        return await router_func(text, context, *args, **kwargs)
    
    tracer = get_tracer(__name__)
    if not tracer:
        return await router_func(text, context, *args, **kwargs)
    
    with tracer.start_as_current_span(
        "router.select_model",
        attributes={
            "input.length": len(text),
            "context.keys": list(context.keys()),
        },
    ) as span:
        try:
            result = await router_func(text, context, *args, **kwargs)
            
            # Add result attributes
            if isinstance(result, dict):
                model_name = result.get("model_name") or result.get("model_info", {}).get("model_name")
                if model_name:
                    span.set_attribute("router.model_selected", str(model_name))
            
            span.set_status(Status(StatusCode.OK))
            return result
        
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise


__all__ = [
    "TraceContext",
    "get_tracer",
    "trace_operation",
    "trace_span",
    "run_critic_with_trace",
    "run_router_with_trace",
]
