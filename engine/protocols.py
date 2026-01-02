"""
ELEANOR V8 - Protocol Definitions for Dependency Injection

Defines protocol interfaces for all major engine components to enable
dependency injection, testing with mocks, and loose coupling.
"""

from typing import Awaitable, Protocol, Any, Dict, List, Optional, runtime_checkable
from abc import abstractmethod

from engine.schemas.pipeline_types import (
    AggregationOutput,
    CriticResult,
    PrecedentAlignmentResult,
    PrecedentRetrievalResult,
    UncertaintyResult,
)

@runtime_checkable
class RouterProtocol(Protocol):
    """Protocol for model router implementations."""

    @abstractmethod
    def route(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any] | Awaitable[Dict[str, Any]]:
        """Route request to appropriate model and return response.
        
        Args:
            text: Input text to process
            context: Additional context for routing decision
            
        Returns:
            Dict with keys:
                - model_name: str
                - model_version: Optional[str]
                - response_text: str
                - reason: Optional[str]
                - health_score: Optional[float]
                - cost: Optional[Dict[str, Any]]
                - diagnostics: Optional[Dict[str, Any]]
        """
        ...


@runtime_checkable
class CriticProtocol(Protocol):
    """Protocol for critic implementations."""

    @abstractmethod
    def evaluate(
        self,
        model_adapter: Any,
        input_text: str,
        context: Dict[str, Any],
    ) -> CriticResult:
        """Evaluate model output against critic's principles.
        
        Args:
            model_adapter: Adapter for generating model responses
            input_text: Original input text
            context: Evaluation context
            
        Returns:
            Dict with keys:
                - severity: float (0.0-1.0)
                - violations: List[Dict[str, Any]]
                - justification: str
                - score: Optional[float]
                - principle: Optional[str]
                - evidence: Optional[Dict[str, Any]]
        """
        ...

    def severity(self, score: float) -> str:
        """Convert numeric score to severity label."""
        ...


@runtime_checkable
class DetectorEngineProtocol(Protocol):
    """Protocol for detector engine implementations."""

    @abstractmethod
    async def detect_all(
        self,
        text: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run all detectors on input text.
        
        Args:
            text: Input text to analyze
            context: Detection context
            
        Returns:
            Dict mapping detector names to signal results
        """
        ...

    @abstractmethod
    def aggregate_signals(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate detector signals into summary.
        
        Args:
            signals: Detector signals from detect_all()
            
        Returns:
            Aggregated summary dict
        """
        ...


@runtime_checkable
class EvidenceRecorderProtocol(Protocol):
    """Protocol for evidence recording implementations."""

    @abstractmethod
    async def record(
        self,
        *,
        critic: str,
        rule_id: str,
        severity: str,
        violation_description: str,
        confidence: float,
        mitigation: Optional[str],
        redundancy_group: Optional[str],
        detector_metadata: Dict[str, Any],
        context: Dict[str, Any],
        raw_text: str,
        trace_id: str,
    ) -> Any:
        """Record evaluation evidence.
        
        Args:
            critic: Name of critic that generated evidence
            rule_id: Rule identifier
            severity: Severity label (INFO, WARNING, ERROR, CRITICAL)
            violation_description: Description of violation
            confidence: Confidence score (0.0-1.0)
            mitigation: Optional mitigation suggestion
            redundancy_group: Optional grouping for deduplication
            detector_metadata: Metadata from detectors
            context: Evaluation context
            raw_text: Raw model output
            trace_id: Trace identifier for correlation
        """
        ...


@runtime_checkable
class PrecedentEngineProtocol(Protocol):
    """Protocol for precedent alignment engine implementations."""

    @abstractmethod
    def analyze(
        self,
        critics: Dict[str, Any],
        precedent_cases: List[Dict[str, Any]],
        query_embedding: List[float],
    ) -> PrecedentAlignmentResult:
        """Analyze precedent alignment.
        
        Args:
            critics: Critic evaluation results
            precedent_cases: Retrieved precedent cases
            query_embedding: Query embedding vector
            
        Returns:
            Dict with keys:
                - alignment_score: float
                - novel: bool
                - similar_cases: List[Dict[str, Any]]
                - reasoning: str
        """
        ...


@runtime_checkable
class PrecedentRetrieverProtocol(Protocol):
    """Protocol for precedent retrieval implementations."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        critic_results: List[CriticResult],
        top_k: int = 5,
    ) -> PrecedentRetrievalResult:
        """Retrieve relevant precedent cases.
        
        Args:
            query: Query text
            critic_results: Critic evaluation results for context
            top_k: Number of cases to retrieve
            
        Returns:
            Dict with keys:
                - precedent_cases: List[Dict[str, Any]]
                - query_embedding: List[float]
                - retrieval_metadata: Dict[str, Any]
        """
        ...


@runtime_checkable
class UncertaintyEngineProtocol(Protocol):
    """Protocol for uncertainty quantification engine implementations."""

    @abstractmethod
    def compute(
        self,
        critics: Dict[str, CriticResult],
        model_used: str,
        precedent_alignment: PrecedentAlignmentResult,
    ) -> UncertaintyResult:
        """Compute uncertainty metrics.
        
        Args:
            critics: Critic evaluation results
            model_used: Name of model that generated output
            precedent_alignment: Precedent alignment analysis
            
        Returns:
            Dict with keys:
                - overall_uncertainty: float (0.0-1.0)
                - needs_escalation: bool
                - confidence_intervals: Dict[str, Any]
                - risk_factors: List[str]
        """
        ...


@runtime_checkable
class AggregatorProtocol(Protocol):
    """Protocol for result aggregation implementations."""

    @abstractmethod
    def aggregate(
        self,
        critics: Dict[str, CriticResult],
        precedent: PrecedentAlignmentResult,
        uncertainty: UncertaintyResult,
        model_output: str,
    ) -> AggregationOutput:
        """Aggregate all analysis results.
        
        Args:
            critics: Critic evaluation results
            precedent: Precedent alignment analysis
            uncertainty: Uncertainty quantification results
            model_output: Original model output text
            
        Returns:
            Dict with keys:
                - final_output: str
                - score: Dict[str, Any]
                - rights_impacted: List[str]
                - dissent: Optional[Dict[str, Any]]
        """
        ...


@runtime_checkable
class ReviewTriggerEvaluatorProtocol(Protocol):
    """Protocol for governance review trigger evaluation."""

    @abstractmethod
    def evaluate(self, case: Any) -> Dict[str, Any]:
        """Evaluate if case requires human review.
        
        Args:
            case: Case object with evaluation results
            
        Returns:
            Dict with keys:
                - review_required: bool
                - triggers: List[str]
                - priority: Optional[str]
        """
        ...


__all__ = [
    "RouterProtocol",
    "CriticProtocol",
    "DetectorEngineProtocol",
    "EvidenceRecorderProtocol",
    "PrecedentEngineProtocol",
    "PrecedentRetrieverProtocol",
    "UncertaintyEngineProtocol",
    "AggregatorProtocol",
    "ReviewTriggerEvaluatorProtocol",
]
