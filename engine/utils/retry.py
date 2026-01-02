"""
ELEANOR V8 — Retry with Exponential Backoff
-------------------------------------------

Provides retry functionality with exponential backoff for resilient
external service calls.

Usage:
    @retry_with_backoff(max_retries=3, base_delay=1.0)
    async def call_llm(prompt: str) -> str:
        return await llm.generate(prompt)

    # Or with custom config
    config = RetryConfig(max_retries=5, base_delay=0.5, max_delay=60.0)

    @retry_with_backoff(config=config)
    async def call_api():
        ...
"""

import asyncio
import inspect
import random
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Optional, Tuple, Type
import logging

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 3
    base_delay: float = 1.0  # Initial delay in seconds
    max_delay: float = 30.0  # Maximum delay in seconds
    exponential_base: float = 2.0  # Base for exponential backoff
    jitter: bool = True  # Add random jitter to delays
    jitter_factor: float = 0.1  # Jitter as fraction of delay

    # Exceptions to retry on (empty means retry on all exceptions)
    retryable_exceptions: Tuple[Type[BaseException], ...] = field(
        default_factory=lambda: (Exception,)
    )

    # Exceptions to never retry on
    non_retryable_exceptions: Tuple[Type[BaseException], ...] = field(
        default_factory=lambda: (KeyboardInterrupt, SystemExit)
    )


@dataclass
class RetryMetrics:
    """Metrics for retry operations."""

    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0
    total_retries: int = 0
    total_delay_seconds: float = 0.0


class RetryExhausted(Exception):
    """Raised when all retry attempts are exhausted."""

    def __init__(
        self,
        message: str,
        attempts: int,
        last_exception: Exception,
        total_delay: float
    ):
        self.attempts = attempts
        self.last_exception = last_exception
        self.total_delay = total_delay
        super().__init__(
            f"{message}. Attempts: {attempts}, "
            f"Total delay: {total_delay:.2f}s, "
            f"Last error: {last_exception}"
        )


def calculate_delay(
    attempt: int,
    config: RetryConfig
) -> float:
    """
    Calculate the delay before the next retry attempt.

    Uses exponential backoff with optional jitter.
    """
    # Exponential backoff: base_delay * (exponential_base ^ attempt)
    delay = config.base_delay * (config.exponential_base ** attempt)

    # Cap at max_delay
    delay = min(delay, config.max_delay)

    # Add jitter if enabled
    if config.jitter:
        jitter_range = delay * config.jitter_factor
        delay += random.uniform(-jitter_range, jitter_range)

    return max(0, delay)


def should_retry(
    exception: Exception,
    config: RetryConfig
) -> bool:
    """
    Determine if an exception should trigger a retry.
    """
    # Never retry on non-retryable exceptions
    if isinstance(exception, config.non_retryable_exceptions):
        return False

    # Retry on retryable exceptions
    if isinstance(exception, config.retryable_exceptions):
        return True

    return False


def retry_with_backoff(
    max_retries: Optional[int] = None,
    base_delay: Optional[float] = None,
    max_delay: Optional[float] = None,
    exponential_base: Optional[float] = None,
    jitter: Optional[bool] = None,
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator for retry with exponential backoff.

    Supports both sync and async functions.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential calculation
        jitter: Whether to add random jitter
        config: Full RetryConfig object (overrides individual params)
        on_retry: Callback called before each retry (attempt, exception, delay)

    Usage:
        @retry_with_backoff(max_retries=3, base_delay=1.0)
        async def call_api():
            ...

        @retry_with_backoff(config=RetryConfig(max_retries=5))
        def sync_call():
            ...
    """
    # Build config from parameters or use provided config
    if config is None:
        config = RetryConfig()

    if max_retries is not None:
        config.max_retries = max_retries
    if base_delay is not None:
        config.base_delay = base_delay
    if max_delay is not None:
        config.max_delay = max_delay
    if exponential_base is not None:
        config.exponential_base = exponential_base
    if jitter is not None:
        config.jitter = jitter

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            last_exception: Optional[Exception] = None
            total_delay = 0.0

            for attempt in range(config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)

                except Exception as e:
                    last_exception = e

                    if not should_retry(e, config):
                        raise

                    if attempt >= config.max_retries:
                        raise RetryExhausted(
                            f"Retry exhausted for {func.__name__}",
                            attempts=attempt + 1,
                            last_exception=e,
                            total_delay=total_delay
                        )

                    delay = calculate_delay(attempt, config)
                    total_delay += delay

                    logger.warning(
                        f"Retry attempt {attempt + 1}/{config.max_retries} "
                        f"for {func.__name__} after {delay:.2f}s. "
                        f"Error: {e}"
                    )

                    if on_retry:
                        try:
                            on_retry(attempt + 1, e, delay)
                        except Exception:
                            pass  # Don't let callback errors affect retry

                    await asyncio.sleep(delay)

            # Should never reach here, but just in case
            last_exc = last_exception or Exception("Unknown error")
            raise RetryExhausted(
                f"Retry exhausted for {func.__name__}",
                attempts=config.max_retries + 1,
                last_exception=last_exc,
                total_delay=total_delay
            )

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            last_exception: Optional[Exception] = None
            total_delay = 0.0

            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)

                except Exception as e:
                    last_exception = e

                    if not should_retry(e, config):
                        raise

                    if attempt >= config.max_retries:
                        raise RetryExhausted(
                            f"Retry exhausted for {func.__name__}",
                            attempts=attempt + 1,
                            last_exception=e,
                            total_delay=total_delay
                        )

                    delay = calculate_delay(attempt, config)
                    total_delay += delay

                    logger.warning(
                        f"Retry attempt {attempt + 1}/{config.max_retries} "
                        f"for {func.__name__} after {delay:.2f}s. "
                        f"Error: {e}"
                    )

                    if on_retry:
                        try:
                            on_retry(attempt + 1, e, delay)
                        except Exception:
                            pass

                    time.sleep(delay)

            last_exc = last_exception or Exception("Unknown error")
            raise RetryExhausted(
                f"Retry exhausted for {func.__name__}",
                attempts=config.max_retries + 1,
                last_exception=last_exc,
                total_delay=total_delay
            )

        # Return appropriate wrapper based on function type
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


async def retry_async(
    func: Callable[..., Any],
    *args,
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
    **kwargs
) -> Any:
    """
    Retry an async function with exponential backoff.

    Alternative to decorator for dynamic retry configuration.

    Usage:
        result = await retry_async(
            api_call,
            param1, param2,
            config=RetryConfig(max_retries=5),
            kwarg1="value"
        )
    """
    config = config or RetryConfig()
    last_exception: Optional[Exception] = None
    total_delay = 0.0

    for attempt in range(config.max_retries + 1):
        try:
            if inspect.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            return func(*args, **kwargs)

        except Exception as e:
            last_exception = e

            if not should_retry(e, config):
                raise

            if attempt >= config.max_retries:
                raise RetryExhausted(
                    "Retry exhausted",
                    attempts=attempt + 1,
                    last_exception=e,
                    total_delay=total_delay
                )

            delay = calculate_delay(attempt, config)
            total_delay += delay

            if on_retry:
                try:
                    on_retry(attempt + 1, e, delay)
                except Exception:
                    pass

            await asyncio.sleep(delay)

    last_exc = last_exception or Exception("Unknown error")
    raise RetryExhausted(
        "Retry exhausted",
        attempts=config.max_retries + 1,
        last_exception=last_exc,
        total_delay=total_delay
    )
