"""
ELEANOR V8 — Resilience Framework

Circuit breakers and graceful degradation for fault tolerance.
"""

from .circuit_breaker import CircuitBreaker, CircuitState, CircuitBreakerOpenError
from .degradation import DegradationStrategy
from .health import ComponentHealthChecker

__all__ = [
    "CircuitBreaker",
    "CircuitState",
    "CircuitBreakerOpenError",
    "DegradationStrategy",
    "ComponentHealthChecker",
]
