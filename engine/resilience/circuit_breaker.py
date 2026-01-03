"""
ELEANOR V8 — Circuit Breaker Implementation

Circuit breaker pattern for external dependencies.
"""

import inspect
import logging
from enum import Enum
from datetime import datetime, timedelta
from typing import Callable, Any, Optional, Type

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""

    pass


class CircuitBreaker:
    """
    Circuit breaker for external dependencies.

    Implements the circuit breaker pattern:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, reject requests immediately
    - HALF_OPEN: Testing if service has recovered
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 2,
        expected_exception: Type[Exception] = Exception,
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting reset
            success_threshold: Successful calls needed in HALF_OPEN to close
            expected_exception: Exception type to count as failure
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = CircuitState.CLOSED

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerOpenError: If circuit is open
        """
        # Check circuit state
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker open. Last failure: {self.last_failure_time}"
                )

        # Execute function
        try:
            if inspect.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            self._on_success()
            return result

        except self.expected_exception as e:
            self._on_failure(e)
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self.last_failure_time is None:
            return False

        elapsed = datetime.now() - self.last_failure_time
        return elapsed > timedelta(seconds=self.recovery_timeout)

    def _on_success(self) -> None:
        """Handle successful execution."""
        self.failure_count = 0

        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            logger.debug(
                f"Circuit breaker success in HALF_OPEN: "
                f"{self.success_count}/{self.success_threshold}"
            )

            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED
                self.success_count = 0
                logger.info("Circuit breaker closed (recovered)")

    def _on_failure(self, exception: Exception) -> None:
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        logger.warning(
            f"Circuit breaker failure: {self.failure_count}/{self.failure_threshold} "
            f"({type(exception).__name__}: {exception})"
        )

        if self.state == CircuitState.HALF_OPEN:
            # Immediate reopen on failure in HALF_OPEN
            self.state = CircuitState.OPEN
            logger.warning("Circuit breaker reopened from HALF_OPEN")
        elif self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.error(f"Circuit breaker opened after {self.failure_count} failures")

    def get_state(self) -> dict:
        """Get current circuit breaker state."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "healthy": self.state == CircuitState.CLOSED,
        }

    def reset(self) -> None:
        """Manually reset circuit breaker."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        logger.info("Circuit breaker manually reset")


__all__ = ["CircuitBreaker", "CircuitState", "CircuitBreakerOpenError"]
