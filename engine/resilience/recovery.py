"""
ELEANOR V8 — Error Recovery Strategies
---------------------------------------

Configurable error recovery strategies for different error types.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, Callable
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Error categories for classification."""
    TRANSIENT = "transient"  # Temporary, can retry
    PERMANENT = "permanent"  # Won't succeed on retry
    UNKNOWN = "unknown"  # Unknown category


class ErrorClassifier:
    """Classify errors for appropriate handling."""
    
    TRANSIENT_ERRORS = (
        TimeoutError,
        ConnectionError,
        asyncio.TimeoutError,
        OSError,  # Network errors
    )
    
    PERMANENT_ERRORS = (
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        AssertionError,
    )
    
    @classmethod
    def classify(cls, error: Exception) -> ErrorCategory:
        """
        Classify error as transient, permanent, or unknown.
        
        Args:
            error: Exception to classify
        
        Returns:
            ErrorCategory
        """
        error_type = type(error)
        
        # Check transient errors
        if isinstance(error, cls.TRANSIENT_ERRORS):
            return ErrorCategory.TRANSIENT
        
        # Check permanent errors
        if isinstance(error, cls.PERMANENT_ERRORS):
            return ErrorCategory.PERMANENT
        
        # Check error message for transient indicators
        error_msg = str(error).lower()
        transient_indicators = [
            "timeout",
            "connection",
            "network",
            "temporary",
            "retry",
            "unavailable",
        ]
        if any(indicator in error_msg for indicator in transient_indicators):
            return ErrorCategory.TRANSIENT
        
        return ErrorCategory.UNKNOWN


class ErrorRecoveryStrategy(ABC):
    """Base class for error recovery strategies."""
    
    @abstractmethod
    async def recover(self, error: Exception, context: Dict[str, Any]) -> Any:
        """
        Attempt to recover from error.
        
        Args:
            error: The exception that occurred
            context: Context information
        
        Returns:
            Recovered result or raises exception
        
        Raises:
            Exception: If recovery fails
        """
        pass


class RetryStrategy(ErrorRecoveryStrategy):
    """Retry with exponential backoff."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
    ):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
    
    async def recover(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Retry operation with exponential backoff."""
        operation = context.get("operation")
        if not operation:
            raise error
        
        delay = self.initial_delay
        
        for attempt in range(1, self.max_attempts + 1):
            try:
                logger.info(
                    "retry_attempt",
                    extra={
                        "attempt": attempt,
                        "max_attempts": self.max_attempts,
                        "error": str(error),
                    },
                )
                
                if asyncio.iscoroutinefunction(operation):
                    return await operation()
                else:
                    return operation()
            
            except Exception as retry_error:
                if attempt == self.max_attempts:
                    logger.error(
                        "retry_exhausted",
                        extra={
                            "attempts": attempt,
                            "final_error": str(retry_error),
                        },
                    )
                    raise retry_error
                
                # Wait before retry
                await asyncio.sleep(min(delay, self.max_delay))
                delay *= self.backoff_factor
        
        raise error


class FallbackStrategy(ErrorRecoveryStrategy):
    """Fallback to alternative operation."""
    
    def __init__(self, fallback_operation: Callable):
        self.fallback_operation = fallback_operation
    
    async def recover(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Execute fallback operation."""
        logger.warning(
            "fallback_triggered",
            extra={
                "error": str(error),
                "fallback": self.fallback_operation.__name__,
            },
        )
        
        if asyncio.iscoroutinefunction(self.fallback_operation):
            return await self.fallback_operation(**context)
        else:
            return self.fallback_operation(**context)


class DegradeStrategy(ErrorRecoveryStrategy):
    """Degrade functionality but continue."""
    
    def __init__(self, degraded_operation: Callable, component_name: str):
        self.degraded_operation = degraded_operation
        self.component_name = component_name
    
    async def recover(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Execute degraded operation."""
        logger.warning(
            "degradation_triggered",
            extra={
                "component": self.component_name,
                "error": str(error),
            },
        )
        
        context["degraded"] = True
        context["degraded_component"] = self.component_name
        
        if asyncio.iscoroutinefunction(self.degraded_operation):
            return await self.degraded_operation(**context)
        else:
            return self.degraded_operation(**context)


class RecoveryManager:
    """Manage error recovery strategies."""
    
    def __init__(self):
        self._strategies: Dict[Type[Exception], ErrorRecoveryStrategy] = {}
        self._default_strategy: Optional[ErrorRecoveryStrategy] = None
    
    def register_strategy(
        self,
        error_type: Type[Exception],
        strategy: ErrorRecoveryStrategy,
    ) -> None:
        """Register recovery strategy for error type."""
        self._strategies[error_type] = strategy
    
    def set_default_strategy(self, strategy: ErrorRecoveryStrategy) -> None:
        """Set default recovery strategy."""
        self._default_strategy = strategy
    
    async def recover(
        self,
        error: Exception,
        context: Dict[str, Any],
        operation: Optional[Callable] = None,
    ) -> Any:
        """
        Attempt to recover from error.
        
        Args:
            error: The exception
            context: Context information
            operation: Optional operation to retry
        
        Returns:
            Recovered result
        
        Raises:
            Exception: If recovery fails
        """
        if operation:
            context["operation"] = operation
        
        # Find strategy for this error type
        error_type = type(error)
        strategy = self._strategies.get(error_type)
        
        # Try base classes
        if not strategy:
            for base_type in error_type.__mro__:
                if base_type in self._strategies:
                    strategy = self._strategies[base_type]
                    break
        
        # Use default if no specific strategy
        if not strategy:
            strategy = self._default_strategy
        
        if not strategy:
            # No recovery strategy available
            raise error
        
        return await strategy.recover(error, context)


__all__ = [
    "ErrorCategory",
    "ErrorClassifier",
    "ErrorRecoveryStrategy",
    "RetryStrategy",
    "FallbackStrategy",
    "DegradeStrategy",
    "RecoveryManager",
]
