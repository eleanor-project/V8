"""
ELEANOR V8 — Distributed Tracing

OpenTelemetry distributed tracing integration.
"""

import contextvars
from typing import Optional, Any, Dict
import logging

logger = logging.getLogger(__name__)

# Context variable for trace context
_trace_context = contextvars.ContextVar('trace_context', default={})


class TraceContext:
    """
    Manages trace context across async calls.
    
    Uses contextvars for thread-safe context propagation.
    """
    
    @staticmethod
    def set_trace_id(trace_id: str) -> None:
        """Set trace ID in context."""
        ctx = _trace_context.get().copy()
        ctx['trace_id'] = trace_id
        _trace_context.set(ctx)
    
    @staticmethod
    def get_trace_id() -> Optional[str]:
        """Get trace ID from context."""
        return _trace_context.get().get('trace_id')
    
    @staticmethod
    def set_span_id(span_id: str) -> None:
        """Set span ID in context."""
        ctx = _trace_context.get().copy()
        ctx['span_id'] = span_id
        _trace_context.set(ctx)
    
    @staticmethod
    def get_span_id() -> Optional[str]:
        """Get span ID from context."""
        return _trace_context.get().get('span_id')
    
    @staticmethod
    def get_context() -> Dict[str, Any]:
        """Get full trace context."""
        return _trace_context.get().copy()
    
    @staticmethod
    def clear() -> None:
        """Clear trace context."""
        _trace_context.set({})


def configure_tracing(
    service_name: str = "eleanor-v8",
    jaeger_endpoint: Optional[str] = None,
    otel_endpoint: Optional[str] = None,
    enabled: bool = True,
) -> Optional[Any]:
    """
    Configure OpenTelemetry distributed tracing.
    
    Args:
        service_name: Service name for tracing
        jaeger_endpoint: Jaeger collector endpoint
        otel_endpoint: OpenTelemetry collector endpoint
        enabled: Enable tracing
    
    Returns:
        Tracer instance or None if disabled
    """
    if not enabled:
        logger.info("Distributed tracing disabled")
        return None
    
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.resources import Resource
        
        # Create resource
        resource = Resource.create({
            "service.name": service_name,
            "service.version": "8.0.0",
        })
        
        # Create tracer provider
        provider = TracerProvider(resource=resource)
        
        # Add exporters
        if jaeger_endpoint:
            try:
                from opentelemetry.exporter.jaeger.thrift import JaegerExporter
                jaeger_exporter = JaegerExporter(
                    agent_host_name=jaeger_endpoint.split(':')[0],
                    agent_port=int(jaeger_endpoint.split(':')[1]) if ':' in jaeger_endpoint else 6831,
                )
                provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))
                logger.info(f"Jaeger tracing configured: {jaeger_endpoint}")
            except ImportError:
                logger.warning("Jaeger exporter not available")
        
        if otel_endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
                otlp_exporter = OTLPSpanExporter(endpoint=otel_endpoint)
                provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
                logger.info(f"OTLP tracing configured: {otel_endpoint}")
            except ImportError:
                logger.warning("OTLP exporter not available")
        
        # Set global tracer provider
        trace.set_tracer_provider(provider)
        
        logger.info(f"Distributed tracing enabled for {service_name}")
        return provider
        
    except ImportError:
        logger.warning(
            "OpenTelemetry not installed. Install with: "
            "pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-jaeger"
        )
        return None
    except Exception as e:
        logger.error(f"Failed to configure tracing: {e}")
        return None


def get_tracer(name: Optional[str] = None) -> Any:
    """
    Get tracer instance.
    
    Args:
        name: Tracer name
    
    Returns:
        Tracer instance or no-op tracer
    """
    try:
        from opentelemetry import trace
        return trace.get_tracer(name or __name__)
    except ImportError:
        # Return no-op tracer
        class NoOpTracer:
            def start_as_current_span(self, *args, **kwargs):
                from contextlib import nullcontext
                return nullcontext()
        return NoOpTracer()


__all__ = ["configure_tracing", "get_tracer", "TraceContext"]
