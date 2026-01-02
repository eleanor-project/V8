"""Custom exception types for ELEANOR V8 Engine.

Provides structured error handling across the engine pipeline.
"""

from typing import Any, Dict, Optional


class EleanorV8Exception(Exception):
    """Base exception for all ELEANOR V8 errors."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        super().__init__(message)
        self.message = message
        merged: Dict[str, Any] = dict(details or {})
        merged.update(kwargs)
        self.details = merged


class EleanorEngineError(EleanorV8Exception):
    """Compatibility alias for engine-wide errors."""

    pass


class CriticEvaluationError(EleanorV8Exception):
    """Raised when a critic fails to evaluate properly."""

    def __init__(
        self,
        critic_name: str,
        message: str,
        trace_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        super().__init__(
            message,
            details,
            critic_name=critic_name,
            trace_id=trace_id,
            **kwargs,
        )
        self.critic_name = critic_name
        self.trace_id = trace_id


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


class ConstitutionalSignal(EleanorV8Exception):
    """Base class for constitutional signals (not failures)."""

    pass


class EscalationRequired(ConstitutionalSignal):
    """Signal that mandatory human escalation is required."""

    def __init__(
        self,
        message: str,
        *,
        critic: str,
        clause: str,
        tier: int,
        severity: float,
        rationale: str,
        trace_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            message,
            details,
            critic=critic,
            clause=clause,
            tier=tier,
            severity=severity,
            rationale=rationale,
            trace_id=trace_id,
            **kwargs,
        )
        self.critic = critic
        self.clause = clause
        self.tier = tier
        self.severity = severity
        self.rationale = rationale
        self.trace_id = trace_id


class UncertaintyBoundaryExceeded(ConstitutionalSignal):
    """Signal that uncertainty exceeded a competence boundary."""

    def __init__(
        self,
        message: str,
        *,
        uncertainty_score: float,
        sources: list[str],
        recommendation: str,
        trace_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            message,
            details,
            uncertainty_score=uncertainty_score,
            sources=sources,
            recommendation=recommendation,
            trace_id=trace_id,
            **kwargs,
        )
        self.uncertainty_score = uncertainty_score
        self.sources = sources
        self.recommendation = recommendation
        self.trace_id = trace_id


class DissentPreservationRequired(ConstitutionalSignal):
    """Signal that dissent preservation is required."""

    def __init__(
        self,
        message: str,
        *,
        critic: Optional[str] = None,
        severity: Optional[float] = None,
        trace_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            message,
            details,
            critic=critic,
            severity=severity,
            trace_id=trace_id,
            **kwargs,
        )
        self.critic = critic
        self.severity = severity
        self.trace_id = trace_id


def is_constitutional_signal(exc: BaseException) -> bool:
    """Return True when the exception represents a constitutional signal."""

    return isinstance(exc, ConstitutionalSignal)
