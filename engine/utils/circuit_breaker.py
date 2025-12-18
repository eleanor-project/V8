"""
ELEANOR V8 — Circuit Breaker Pattern
------------------------------------

Implements the circuit breaker pattern for resilient LLM and external service calls.

States:
- CLOSED: Normal operation, requests pass through
- OPEN: Failing, requests are rejected immediately
- HALF_OPEN: Testing recovery, limited requests allowed

Usage:
    breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30)

    try:
        result = await breaker.call(llm_adapter.generate, prompt)
    except CircuitBreakerOpen:
        # Handle circuit open - use fallback
        result = await fallback_adapter.generate(prompt)
"""

import asyncio
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Callable, Any, Optional, Dict
import threading


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreakerOpen(Exception):
    """Exception raised when circuit is open."""

    def __init__(self, name: str, recovery_time: float):
        self.name = name
        self.recovery_time = recovery_time
        super().__init__(
            f"Circuit breaker '{name}' is open. "
            f"Recovery in {recovery_time:.1f} seconds."
        )


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""

    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes in half-open before closing
    recovery_timeout: float = 30.0  # Seconds before attempting recovery
    half_open_max_calls: int = 3  # Max concurrent calls in half-open state


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker monitoring."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    state_changes: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None


class CircuitBreaker:
    """
    Circuit breaker for external service calls.

    Thread-safe implementation supporting both sync and async calls.
    """

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 5,
        success_threshold: int = 2,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3,
        on_state_change: Optional[Callable[[str, CircuitState, CircuitState], None]] = None
    ):
        self.name = name
        self.config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            recovery_timeout=recovery_timeout,
            half_open_max_calls=half_open_max_calls
        )

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._lock = threading.Lock()

        self._on_state_change: Optional[Callable[[str, CircuitState, CircuitState], None]] = on_state_change
        self.metrics = CircuitBreakerMetrics()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            self._check_state_transition()
            return self._state

    def _check_state_transition(self) -> None:
        """Check if state should transition based on time."""
        if self._state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to(CircuitState.HALF_OPEN)

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self._last_failure_time is None:
            return True
        return time.time() - self._last_failure_time >= self.config.recovery_timeout

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._state
        if old_state == new_state:
            return

        self._state = new_state
        self.metrics.state_changes += 1

        # Reset counters on state change
        if new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._success_count = 0
            self._half_open_calls = 0

        # Callback for monitoring
        if self._on_state_change is not None:
            try:
                self._on_state_change(self.name, old_state, new_state)
            except Exception:
                pass  # Don't let callback errors affect circuit

    def _on_success(self) -> None:
        """Handle successful call."""
        with self._lock:
            self.metrics.successful_calls += 1
            self.metrics.last_success_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)

    def _on_failure(self, error: Exception) -> None:
        """Handle failed call."""
        with self._lock:
            self.metrics.failed_calls += 1
            self.metrics.last_failure_time = time.time()
            self._last_failure_time = time.time()
            self._failure_count += 1

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open immediately opens circuit
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)

    def _can_execute(self) -> bool:
        """Check if a call can be executed."""
        with self._lock:
            self._check_state_transition()

            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                return False

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

            return False

    def _get_recovery_time(self) -> float:
        """Get time until recovery attempt."""
        if self._last_failure_time is None:
            return 0.0
        elapsed = time.time() - self._last_failure_time
        return max(0.0, self.config.recovery_timeout - elapsed)

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function through the circuit breaker.

        Supports both sync and async functions.

        Raises:
            CircuitBreakerOpen: If circuit is open
        """
        self.metrics.total_calls += 1

        if not self._can_execute():
            self.metrics.rejected_calls += 1
            raise CircuitBreakerOpen(self.name, self._get_recovery_time())

        try:
            # Handle both sync and async functions
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            self._on_success()
            return result

        except Exception as e:
            self._on_failure(e)
            raise

    def call_sync(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a synchronous function through the circuit breaker.

        Raises:
            CircuitBreakerOpen: If circuit is open
        """
        self.metrics.total_calls += 1

        if not self._can_execute():
            self.metrics.rejected_calls += 1
            raise CircuitBreakerOpen(self.name, self._get_recovery_time())

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure(e)
            raise

    def reset(self) -> None:
        """Reset the circuit breaker to closed state."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None

    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        with self._lock:
            self._check_state_transition()
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "recovery_time_remaining": self._get_recovery_time(),
                "metrics": {
                    "total_calls": self.metrics.total_calls,
                    "successful_calls": self.metrics.successful_calls,
                    "failed_calls": self.metrics.failed_calls,
                    "rejected_calls": self.metrics.rejected_calls,
                    "state_changes": self.metrics.state_changes
                }
            }


class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers.

    Usage:
        registry = CircuitBreakerRegistry()
        gpt_breaker = registry.get_or_create("gpt-4", failure_threshold=3)
        claude_breaker = registry.get_or_create("claude-3", failure_threshold=5)
    """

    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()

    def get_or_create(
        self,
        name: str,
        **kwargs
    ) -> CircuitBreaker:
        """Get an existing circuit breaker or create a new one."""
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name=name, **kwargs)
            return self._breakers[name]

    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get an existing circuit breaker."""
        return self._breakers.get(name)

    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers."""
        return {name: cb.get_status() for name, cb in self._breakers.items()}

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for breaker in self._breakers.values():
            breaker.reset()


# Global registry
circuit_breaker_registry = CircuitBreakerRegistry()
