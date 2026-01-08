"""
ELEANOR V8 — Correlation ID Management
---------------------------------------

Manage correlation IDs across async operations for better traceability.
"""

import contextvars
import uuid
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Context variable for correlation ID
_correlation_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "correlation_id", default=None
)


class CorrelationContext:
    """Manage correlation IDs across async operations."""
    
    @staticmethod
    def get() -> Optional[str]:
        """Get current correlation ID from context."""
        return _correlation_id.get()
    
    @staticmethod
    def set(correlation_id: str) -> None:
        """Set correlation ID in context."""
        _correlation_id.set(correlation_id)
    
    @staticmethod
    def generate() -> str:
        """Generate new correlation ID."""
        return str(uuid.uuid4())
    
    @staticmethod
    def get_or_generate() -> str:
        """Get existing correlation ID or generate new one."""
        existing = CorrelationContext.get()
        if existing:
            return existing
        new_id = CorrelationContext.generate()
        CorrelationContext.set(new_id)
        return new_id
    
    @staticmethod
    def clear() -> None:
        """Clear correlation ID from context."""
        _correlation_id.set(None)


def with_correlation_id(correlation_id: Optional[str] = None):
    """
    Decorator to set correlation ID for function execution.
    
    Usage:
        @with_correlation_id("my-correlation-id")
        async def my_function():
            # correlation_id available via CorrelationContext.get()
            pass
    """
    def decorator(func):
        import inspect
        
        if inspect.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                original_id = CorrelationContext.get()
                try:
                    if correlation_id:
                        CorrelationContext.set(correlation_id)
                    elif not CorrelationContext.get():
                        CorrelationContext.set(CorrelationContext.generate())
                    return await func(*args, **kwargs)
                finally:
                    CorrelationContext.set(original_id)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                original_id = CorrelationContext.get()
                try:
                    if correlation_id:
                        CorrelationContext.set(correlation_id)
                    elif not CorrelationContext.get():
                        CorrelationContext.set(CorrelationContext.generate())
                    return func(*args, **kwargs)
                finally:
                    CorrelationContext.set(original_id)
            return sync_wrapper
    
    return decorator


def get_correlation_id() -> str:
    """Get or generate correlation ID."""
    return CorrelationContext.get_or_generate()


__all__ = [
    "CorrelationContext",
    "with_correlation_id",
    "get_correlation_id",
]
