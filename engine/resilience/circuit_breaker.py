"""
ELEANOR V8 - Circuit Breaker Pattern
Prevents cascading failures by temporarily blocking failing operations
"""

import asyncio
import time
from enum import Enum
from typing import Callable, Optional, Any, TypeVar, Generic
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5
    timeout_seconds: int = 60
    half_open_max_calls: int = 3
    success_threshold: int = 2  # Successes needed in half-open to close


@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    state_changes: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    
    @property
    def success_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open"""
    pass


class CircuitBreaker(Generic[T]):
    """
    Circuit Breaker implementation for protecting against cascading failures.
    
    Usage:
        breaker = CircuitBreaker(name="my-service")
        
        async with breaker:
            result = await risky_operation()
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._last_failure_time: Optional[float] = None
        self._last_state_change: float = time.time()
        
        self.stats = CircuitBreakerStats()
        self._lock = asyncio.Lock()
    
    @property
    def state(self) -> CircuitBreakerState:
        return self._state
    
    @property
    def is_closed(self) -> bool:
        return self._state == CircuitBreakerState.CLOSED
    
    @property
    def is_open(self) -> bool:
        return self._state == CircuitBreakerState.OPEN
    
    @property
    def is_half_open(self) -> bool:
        return self._state == CircuitBreakerState.HALF_OPEN
    
    async def __aenter__(self):
        await self._check_state()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            await self._record_success()
        else:
            await self._record_failure(exc_val)
        return False
    
    async def _check_state(self):
        """Check and update circuit breaker state"""
        async with self._lock:
            if self._state == CircuitBreakerState.OPEN:
                # Check if timeout has elapsed
                if self._last_failure_time:
                    elapsed = time.time() - self._last_failure_time
                    if elapsed >= self.config.timeout_seconds:
                        logger.info(f"[CircuitBreaker:{self.name}] Transitioning to HALF_OPEN")
                        await self._transition_to_half_open()
                    else:
                        raise CircuitBreakerError(
                            f"Circuit breaker '{self.name}' is OPEN. "
                            f"Retry in {self.config.timeout_seconds - elapsed:.1f}s"
                        )
                else:
                    raise CircuitBreakerError(f"Circuit breaker '{self.name}' is OPEN")
            
            elif self._state == CircuitBreakerState.HALF_OPEN:
                if self._half_open_calls >= self.config.half_open_max_calls:
                    raise CircuitBreakerError(
                        f"Circuit breaker '{self.name}' in HALF_OPEN: max calls reached"
                    )
                self._half_open_calls += 1
    
    async def _record_success(self):
        """Record successful operation"""
        async with self._lock:
            self.stats.total_calls += 1
            self.stats.successful_calls += 1
            self.stats.last_success_time = time.time()
            
            if self._state == CircuitBreakerState.HALF_OPEN:
                self._success_count += 1
                logger.debug(
                    f"[CircuitBreaker:{self.name}] Success in HALF_OPEN "
                    f"({self._success_count}/{self.config.success_threshold})"
                )
                
                if self._success_count >= self.config.success_threshold:
                    logger.info(f"[CircuitBreaker:{self.name}] Transitioning to CLOSED")
                    await self._transition_to_closed()
            
            elif self._state == CircuitBreakerState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0
    
    async def _record_failure(self, exception: Exception):
        """Record failed operation"""
        async with self._lock:
            self.stats.total_calls += 1
            self.stats.failed_calls += 1
            self.stats.last_failure_time = time.time()
            self._last_failure_time = time.time()
            
            if self._state == CircuitBreakerState.HALF_OPEN:
                logger.warning(
                    f"[CircuitBreaker:{self.name}] Failure in HALF_OPEN, reopening circuit"
                )
                await self._transition_to_open()
            
            elif self._state == CircuitBreakerState.CLOSED:
                self._failure_count += 1
                logger.debug(
                    f"[CircuitBreaker:{self.name}] Failure recorded "
                    f"({self._failure_count}/{self.config.failure_threshold}): {exception}"
                )
                
                if self._failure_count >= self.config.failure_threshold:
                    logger.warning(
                        f"[CircuitBreaker:{self.name}] Failure threshold reached, opening circuit"
                    )
                    await self._transition_to_open()
    
    async def _transition_to_open(self):
        """Transition to OPEN state"""
        self._state = CircuitBreakerState.OPEN
        self._last_state_change = time.time()
        self.stats.state_changes += 1
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
    
    async def _transition_to_half_open(self):
        """Transition to HALF_OPEN state"""
        self._state = CircuitBreakerState.HALF_OPEN
        self._last_state_change = time.time()
        self.stats.state_changes += 1
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
    
    async def _transition_to_closed(self):
        """Transition to CLOSED state"""
        self._state = CircuitBreakerState.CLOSED
        self._last_state_change = time.time()
        self.stats.state_changes += 1
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
    
    def get_stats(self) -> dict:
        """Get circuit breaker statistics"""
        return {
            "name": self.name,
            "state": self._state.value,
            "total_calls": self.stats.total_calls,
            "successful_calls": self.stats.successful_calls,
            "failed_calls": self.stats.failed_calls,
            "success_rate": self.stats.success_rate,
            "state_changes": self.stats.state_changes,
            "failure_count": self._failure_count,
            "last_failure_time": self._last_failure_time,
            "last_state_change": self._last_state_change,
        }
    
    async def reset(self):
        """Reset circuit breaker to CLOSED state"""
        async with self._lock:
            logger.info(f"[CircuitBreaker:{self.name}] Manual reset to CLOSED")
            await self._transition_to_closed()


class CircuitBreakerRegistry:
    """Global registry for circuit breakers"""
    
    _instance: Optional['CircuitBreakerRegistry'] = None
    
    def __init__(self):
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()
    
    @classmethod
    def get_instance(cls) -> 'CircuitBreakerRegistry':
        if cls._instance is None:
            cls._instance = CircuitBreakerRegistry()
        return cls._instance
    
    async def get_or_create(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ) -> CircuitBreaker:
        """Get existing or create new circuit breaker"""
        async with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name=name, config=config)
            return self._breakers[name]
    
    def get_all_stats(self) -> dict[str, dict]:
        """Get statistics for all circuit breakers"""
        return {name: breaker.get_stats() for name, breaker in self._breakers.items()}
    
    async def reset_all(self):
        """Reset all circuit breakers"""
        async with self._lock:
            for breaker in self._breakers.values():
                await breaker.reset()
