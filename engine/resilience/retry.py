"""
ELEANOR V8 - Retry Logic with Exponential Backoff
"""

import asyncio
import random
import logging
from typing import Callable, TypeVar, Optional, Type, Tuple
from dataclasses import dataclass
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class RetryPolicy:
    """Retry policy configuration"""
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    jitter: bool = True
    retry_exceptions: Tuple[Type[Exception], ...] = (Exception,)


def retry_with_backoff(
    policy: Optional[RetryPolicy] = None,
    **policy_kwargs
):
    """
    Decorator for retrying async functions with exponential backoff.
    
    Usage:
        @retry_with_backoff(max_attempts=3, backoff_factor=2.0)
        async def risky_operation():
            ...
    """
    if policy is None:
        policy = RetryPolicy(**policy_kwargs)
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception: Optional[Exception] = None
            
            for attempt in range(1, policy.max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                
                except policy.retry_exceptions as e:
                    last_exception = e
                    
                    if attempt == policy.max_attempts:
                        logger.error(
                            f"[Retry] {func.__name__} failed after {attempt} attempts: {e}"
                        )
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = min(
                        policy.initial_delay * (policy.backoff_factor ** (attempt - 1)),
                        policy.max_delay
                    )
                    
                    # Add jitter to prevent thundering herd
                    if policy.jitter:
                        delay *= (0.5 + random.random())
                    
                    logger.warning(
                        f"[Retry] {func.__name__} attempt {attempt}/{policy.max_attempts} "
                        f"failed: {e}. Retrying in {delay:.2f}s"
                    )
                    
                    await asyncio.sleep(delay)
            
            # Should never reach here, but for type safety
            raise last_exception or Exception("Retry failed")
        
        return wrapper
    
    return decorator


async def retry_async(
    func: Callable[..., T],
    *args,
    policy: Optional[RetryPolicy] = None,
    **kwargs
) -> T:
    """
    Programmatic retry with exponential backoff.
    
    Usage:
        result = await retry_async(
            risky_function,
            arg1, arg2,
            policy=RetryPolicy(max_attempts=5)
        )
    """
    if policy is None:
        policy = RetryPolicy()
    
    last_exception: Optional[Exception] = None
    
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return await func(*args, **kwargs)
        
        except policy.retry_exceptions as e:
            last_exception = e
            
            if attempt == policy.max_attempts:
                logger.error(
                    f"[Retry] {func.__name__} failed after {attempt} attempts: {e}"
                )
                raise
            
            delay = min(
                policy.initial_delay * (policy.backoff_factor ** (attempt - 1)),
                policy.max_delay
            )
            
            if policy.jitter:
                delay *= (0.5 + random.random())
            
            logger.warning(
                f"[Retry] {func.__name__} attempt {attempt}/{policy.max_attempts} "
                f"failed: {e}. Retrying in {delay:.2f}s"
            )
            
            await asyncio.sleep(delay)
    
    raise last_exception or Exception("Retry failed")
