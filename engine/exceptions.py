"""
ELEANOR V8 — Constitutional Exception Hierarchy

This module defines exceptions that preserve the distinction between:
1. Constitutional governance concerns (signals, not errors)
2. Technical failures (true errors requiring recovery)
3. Uncertainty (epistemic limitation, not malfunction)

Principle: Escalation and uncertainty are governance signals, not error conditions.
"""

from typing import Any, Dict, List, Optional


# ============================================================================
# BASE EXCEPTIONS
# ============================================================================

class EleanorException(Exception):
    """Base exception for all ELEANOR V8 errors."""
    
    def __init__(self, message: str, *, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}


class ConstitutionalSignal(EleanorException):
    """
    Base class for constitutional governance signals.
    
    These are NOT errors - they are legitimate governance outputs that require
    special handling (human review, escalation, uncertainty preservation).
    
    Subclasses should never be caught and silenced; they represent the system
    working as designed to surface constitutional concerns.
    """
    pass


# ============================================================================
# GOVERNANCE SIGNALS (Not Errors)
# ============================================================================

class EscalationRequired(ConstitutionalSignal):
    """
    A critic has triggered mandatory human review.
    
    This is a governance signal, not an error. The system is correctly
    identifying a situation where automation lacks legitimacy to proceed.
    
    Attributes:
        critic: Name of the critic triggering escalation
        clause: Specific clause violated (e.g., 'A2', 'U3')
        tier: Required escalation tier (2 or 3)
        severity: Constitutional severity score
        rationale: Critic's justification
        trace_id: Audit trail identifier
    """
    
    def __init__(
        self,
        message: str,
        *,
        critic: str,
        clause: str,
        tier: int,
        severity: float,
        rationale: str,
        trace_id: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, context=context)
        self.critic = critic
        self.clause = clause
        self.tier = tier
        self.severity = severity
        self.rationale = rationale
        self.trace_id = trace_id


class UncertaintyBoundaryExceeded(ConstitutionalSignal):
    """
    The system's epistemic competence boundary has been exceeded.
    
    This is NOT an error - it's the system correctly recognizing it lacks
    sufficient grounding to proceed with confidence. Uncertainty is a signal.
    
    Attributes:
        uncertainty_score: Quantified uncertainty (0.0-1.0)
        sources: Components contributing uncertainty
        recommendation: Suggested handling (escalate, defer, acknowledge)
        trace_id: Audit trail identifier
    """
    
    def __init__(
        self,
        message: str,
        *,
        uncertainty_score: float,
        sources: List[str],
        recommendation: str,
        trace_id: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, context=context)
        self.uncertainty_score = uncertainty_score
        self.sources = sources
        self.recommendation = recommendation
        self.trace_id = trace_id


class DissentPreservationRequired(ConstitutionalSignal):
    """
    Critical dissent detected that must not be averaged away.
    
    Raised when minority critic opinions represent legitimate constitutional
    concerns that would be silenced by pure aggregation.
    
    Attributes:
        dissenting_critics: Critics with minority position
        consensus_critics: Critics in majority
        dissent_severity: Maximum severity among dissenters
        trace_id: Audit trail identifier
    """
    
    def __init__(
        self,
        message: str,
        *,
        dissenting_critics: List[str],
        consensus_critics: List[str],
        dissent_severity: float,
        trace_id: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, context=context)
        self.dissenting_critics = dissenting_critics
        self.consensus_critics = consensus_critics
        self.dissent_severity = dissent_severity
        self.trace_id = trace_id


# ============================================================================
# TECHNICAL ERRORS (Actual Failures)
# ============================================================================

class CriticEvaluationError(EleanorException):
    """
    A critic failed to complete evaluation due to technical error.
    
    This IS an error - the critic could not perform its function.
    Recovery: retry, fallback, or fail-safe to most conservative position.
    
    Attributes:
        critic_name: Name of failed critic
        failure_type: Category of failure (timeout, exception, invalid_output)
        trace_id: Audit trail identifier
    """
    
    def __init__(
        self,
        message: str,
        *,
        critic_name: str,
        failure_type: str,
        trace_id: str,
        original_error: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, context=context)
        self.critic_name = critic_name
        self.failure_type = failure_type
        self.trace_id = trace_id
        self.original_error = original_error


class RouterSelectionError(EleanorException):
    """
    Router failed to select a model or produce a response.
    
    Recovery: fallback model, cached response, or fail-safe rejection.
    
    Attributes:
        router_diagnostics: Debug information from router
        attempted_models: Models that were considered
        trace_id: Audit trail identifier
    """
    
    def __init__(
        self,
        message: str,
        *,
        router_diagnostics: Dict[str, Any],
        attempted_models: List[str],
        trace_id: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, context=context)
        self.router_diagnostics = router_diagnostics
        self.attempted_models = attempted_models
        self.trace_id = trace_id


class PrecedentRetrievalError(EleanorException):
    """
    Precedent engine failed to retrieve or align cases.
    
    Recovery: proceed without precedent (with uncertainty flag), or defer.
    
    Attributes:
        retrieval_type: Type of precedent operation (search, alignment, embedding)
        trace_id: Audit trail identifier
    """
    
    def __init__(
        self,
        message: str,
        *,
        retrieval_type: str,
        trace_id: str,
        original_error: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, context=context)
        self.retrieval_type = retrieval_type
        self.trace_id = trace_id
        self.original_error = original_error


class AggregationError(EleanorException):
    """
    Aggregator failed to synthesize critic results.
    
    Recovery: surface individual critic results without synthesis, or fail-safe.
    
    Attributes:
        aggregation_stage: Stage of failure (weighting, synthesis, dissent_check)
        partial_results: Any partial aggregation completed
        trace_id: Audit trail identifier
    """
    
    def __init__(
        self,
        message: str,
        *,
        aggregation_stage: str,
        partial_results: Optional[Dict[str, Any]] = None,
        trace_id: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, context=context)
        self.aggregation_stage = aggregation_stage
        self.partial_results = partial_results or {}
        self.trace_id = trace_id


class EvidenceRecordingError(EleanorException):
    """
    Evidence recorder failed to persist audit trail.
    
    This is CRITICAL for governance compliance. Depending on configuration,
    this should either block execution or trigger immediate escalation.
    
    Attributes:
        record_type: Type of evidence (critic_evaluation, escalation, human_review)
        storage_backend: Failed storage system
        trace_id: Audit trail identifier
    """
    
    def __init__(
        self,
        message: str,
        *,
        record_type: str,
        storage_backend: str,
        trace_id: str,
        original_error: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, context=context)
        self.record_type = record_type
        self.storage_backend = storage_backend
        self.trace_id = trace_id
        self.original_error = original_error


# ============================================================================
# CONFIGURATION & VALIDATION ERRORS
# ============================================================================

class ConfigurationError(EleanorException):
    """
    Invalid configuration detected at startup or runtime.
    
    Recovery: use defaults (if safe), or fail-fast to prevent undefined behavior.
    
    Attributes:
        config_field: Specific configuration field
        invalid_value: The problematic value
        expected_type: Expected value type/range
    """
    
    def __init__(
        self,
        message: str,
        *,
        config_field: str,
        invalid_value: Any,
        expected_type: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, context=context)
        self.config_field = config_field
        self.invalid_value = invalid_value
        self.expected_type = expected_type


class InputValidationError(EleanorException):
    """
    Invalid input detected (potential security concern or malformed request).
    
    Recovery: reject input with clear error to user.
    
    Attributes:
        validation_type: Type of validation failed (size, format, injection)
        field: Input field that failed validation
    """
    
    def __init__(
        self,
        message: str,
        *,
        validation_type: str,
        field: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, context=context)
        self.validation_type = validation_type
        self.field = field


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def is_constitutional_signal(exc: Exception) -> bool:
    """
    Check if an exception is a constitutional signal (not an error).
    
    Constitutional signals should be propagated and handled specially,
    never silently caught or logged as errors.
    """
    return isinstance(exc, ConstitutionalSignal)


def is_recoverable_error(exc: Exception) -> bool:
    """
    Check if an error is recoverable with fallback logic.
    
    Recoverable errors allow degraded operation; unrecoverable errors
    should fail-fast or escalate to human review.
    """
    recoverable_types = (
        RouterSelectionError,
        PrecedentRetrievalError,
    )
    return isinstance(exc, recoverable_types)


def requires_immediate_escalation(exc: Exception) -> bool:
    """
    Check if an exception requires immediate human escalation.
    
    These are either governance signals or critical failures that
    compromise constitutional guarantees.
    """
    escalation_types = (
        EscalationRequired,
        UncertaintyBoundaryExceeded,
        DissentPreservationRequired,
        EvidenceRecordingError,
    )
    return isinstance(exc, escalation_types)


__all__ = [
    # Base
    "EleanorException",
    "ConstitutionalSignal",
    # Governance Signals
    "EscalationRequired",
    "UncertaintyBoundaryExceeded",
    "DissentPreservationRequired",
    # Technical Errors
    "CriticEvaluationError",
    "RouterSelectionError",
    "PrecedentRetrievalError",
    "AggregationError",
    "EvidenceRecordingError",
    # Configuration
    "ConfigurationError",
    "InputValidationError",
    # Utilities
    "is_constitutional_signal",
    "is_recoverable_error",
    "requires_immediate_escalation",
]
