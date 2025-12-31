"""Custom exception types for ELEANOR V8 Engine.

Provides structured error handling across the engine pipeline.
"""

from typing import Any, Dict, Optional


class EleanorV8Exception(Exception):
    """Base exception for all ELEANOR V8 errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class CriticEvaluationError(EleanorV8Exception):
    """Raised when a critic fails to evaluate properly."""

    def __init__(self, critic_name: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.critic_name = critic_name


class RouterSelectionError(EleanorV8Exception):
    """Raised when the router cannot select an appropriate model."""

    pass


class PrecedentRetrievalError(EleanorV8Exception):
    """Raised when precedent retrieval fails."""

    pass


class UncertaintyComputationError(EleanorV8Exception):
    """Raised when uncertainty engine computation fails."""

    pass


class AggregationError(EleanorV8Exception):
    """Raised when aggregation of critic results fails."""

    pass


class ConfigurationError(EleanorV8Exception):
    """Raised when engine configuration is invalid."""

    pass


class EvidenceRecordingError(EleanorV8Exception):
    """Raised when evidence recording fails critically.
    
    This is distinct from best-effort logging failures.
    Used when evidence recording is required for compliance.
    """

    pass


class DetectorExecutionError(EleanorV8Exception):
    """Raised when detector execution fails."""

    pass


class GovernanceEvaluationError(EleanorV8Exception):
    """Raised when governance review evaluation fails."""

    pass


class InputValidationError(EleanorV8Exception):
    """Raised when input validation fails."""

    pass


class TimeoutError(EleanorV8Exception):
    """Raised when an operation exceeds configured timeout."""

    pass
