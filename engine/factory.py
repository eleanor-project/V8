"""
ELEANOR V8 - Dependency Injection Factory

Provides factory functions for creating engine instances with proper
dependency injection, making it easy to swap implementations and create
test instances with mocks.
"""

import logging
from typing import Any, Dict, Optional
from dataclasses import dataclass

from engine.protocols import (
    RouterProtocol,
    DetectorEngineProtocol,
    EvidenceRecorderProtocol,
    PrecedentEngineProtocol,
    PrecedentRetrieverProtocol,
    UncertaintyEngineProtocol,
    AggregatorProtocol,
    ReviewTriggerEvaluatorProtocol,
)

logger = logging.getLogger(__name__)


@dataclass
class EngineDependencies:
    """Container for all engine dependencies."""

    router: RouterProtocol
    detector_engine: DetectorEngineProtocol
    evidence_recorder: EvidenceRecorderProtocol
    critics: Dict[str, Any]
    review_trigger_evaluator: ReviewTriggerEvaluatorProtocol
    precedent_engine: Optional[PrecedentEngineProtocol] = None
    precedent_retriever: Optional[PrecedentRetrieverProtocol] = None
    uncertainty_engine: Optional[UncertaintyEngineProtocol] = None
    aggregator: Optional[AggregatorProtocol] = None
    critic_models: Optional[Dict[str, Any]] = None


class DependencyFactory:
    """Factory for creating engine dependencies."""

    @staticmethod
    def create_router(
        backend: Optional[str] = None,
        **kwargs: Any,
    ) -> RouterProtocol:
        """Create router instance.

        Args:
            backend: Router backend to use ("v8", "ollama", etc.)
            **kwargs: Additional arguments for router initialization

        Returns:
            Router instance implementing RouterProtocol
        """
        if backend == "mock":
            from engine.mocks import MockRouter

            return MockRouter(**kwargs)

        # Default: Use RouterV8
        from engine.router.router import RouterV8

        return RouterV8(**kwargs)

    @staticmethod
    def create_detector_engine(
        detectors: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> DetectorEngineProtocol:
        """Create detector engine instance.

        Args:
            detectors: Dict of detector implementations
            **kwargs: Additional arguments for detector engine

        Returns:
            DetectorEngine instance
        """
        if detectors and "mock" in detectors:
            from engine.mocks import MockDetectorEngine

            return MockDetectorEngine()

        from engine.detectors.engine import DetectorEngineV8

        return DetectorEngineV8(detectors=detectors or {}, **kwargs)

    @staticmethod
    def create_evidence_recorder(
        jsonl_path: Optional[str] = "evidence.jsonl",
        **kwargs: Any,
    ) -> EvidenceRecorderProtocol:
        """Create evidence recorder instance.

        Args:
            jsonl_path: Path to JSONL evidence file
            **kwargs: Additional arguments for recorder

        Returns:
            EvidenceRecorder instance
        """
        if jsonl_path == "mock":
            from engine.mocks import MockEvidenceRecorder

            return MockEvidenceRecorder()

        from engine.recorder.evidence_recorder import EvidenceRecorder

        return EvidenceRecorder(jsonl_path=jsonl_path, **kwargs)

    @staticmethod
    def create_critics(
        custom_critics: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Create default or custom critics.

        Args:
            custom_critics: Optional custom critic implementations
            **kwargs: Additional arguments for critics

        Returns:
            Dict mapping critic names to implementations
        """
        if custom_critics:
            if "mock" in custom_critics:
                from engine.mocks import MockCritic

                return {"mock": MockCritic}
            return custom_critics

        # Default critics
        from engine.critics.rights import RightsCriticV8
        from engine.critics.risk import RiskCriticV8
        from engine.critics.fairness import FairnessCriticV8
        from engine.critics.pragmatics import PragmaticsCriticV8
        from engine.critics.truth import TruthCriticV8
        from engine.critics.autonomy import AutonomyCriticV8

        return {
            "rights": RightsCriticV8,
            "autonomy": AutonomyCriticV8,
            "fairness": FairnessCriticV8,
            "truth": TruthCriticV8,
            "risk": RiskCriticV8,
            "operations": PragmaticsCriticV8,
        }

    @staticmethod
    def create_precedent_engine(
        enabled: bool = True,
        **kwargs: Any,
    ) -> Optional[PrecedentEngineProtocol]:
        """Create precedent alignment engine instance.

        Args:
            enabled: Whether to enable precedent engine
            **kwargs: Additional arguments for precedent engine

        Returns:
            PrecedentEngine instance or None if disabled
        """
        if not enabled:
            return None

        if kwargs.get("mock"):
            from engine.mocks import MockPrecedentEngine

            return MockPrecedentEngine()

        from engine.precedent.alignment import PrecedentAlignmentEngineV8

        return PrecedentAlignmentEngineV8(**kwargs)

    @staticmethod
    def create_precedent_retriever(
        enabled: bool = True,
        store_client: Optional[Any] = None,
        **kwargs: Any,
    ) -> Optional[PrecedentRetrieverProtocol]:
        """Create precedent retriever instance.

        Args:
            enabled: Whether to enable precedent retriever
            store_client: Storage client for precedents
            **kwargs: Additional arguments for retriever

        Returns:
            PrecedentRetriever instance or None if disabled
        """
        if not enabled:
            return None

        if kwargs.get("mock"):
            from engine.mocks import MockPrecedentRetriever

            return MockPrecedentRetriever()

        from engine.precedent.retrieval import PrecedentRetrievalV8

        if store_client is None:
            # Create null store for development
            class _NullStore:
                def search(self, q: str, top_k: int = 5):
                    return []

            store_client = _NullStore()
            logger.warning(
                "Using null precedent store. Configure PRECEDENT_BACKEND for production."
            )

        return PrecedentRetrievalV8(store_client=store_client, **kwargs)

    @staticmethod
    def create_uncertainty_engine(
        enabled: bool = True,
        **kwargs: Any,
    ) -> Optional[UncertaintyEngineProtocol]:
        """Create uncertainty quantification engine instance.

        Args:
            enabled: Whether to enable uncertainty engine
            **kwargs: Additional arguments for uncertainty engine

        Returns:
            UncertaintyEngine instance or None if disabled
        """
        if not enabled:
            return None

        if kwargs.get("mock"):
            from engine.mocks import MockUncertaintyEngine

            return MockUncertaintyEngine()

        from engine.uncertainty.uncertainty import UncertaintyEngineV8

        return UncertaintyEngineV8(**kwargs)

    @staticmethod
    def create_aggregator(
        enabled: bool = True,
        **kwargs: Any,
    ) -> Optional[AggregatorProtocol]:
        """Create result aggregator instance.

        Args:
            enabled: Whether to enable aggregator
            **kwargs: Additional arguments for aggregator

        Returns:
            Aggregator instance or None if disabled
        """
        if not enabled:
            return None

        if kwargs.get("mock"):
            from engine.mocks import MockAggregator

            return MockAggregator()

        from engine.aggregator.aggregator import AggregatorV8

        return AggregatorV8(**kwargs)

    @staticmethod
    def create_review_trigger_evaluator(
        **kwargs: Any,
    ) -> ReviewTriggerEvaluatorProtocol:
        """Create governance review trigger evaluator.

        Args:
            **kwargs: Additional arguments for evaluator

        Returns:
            ReviewTriggerEvaluator instance
        """
        if kwargs.get("mock"):
            from engine.mocks import MockReviewTriggerEvaluator

            return MockReviewTriggerEvaluator()

        from governance.review_triggers import ReviewTriggerEvaluator

        return ReviewTriggerEvaluator(**kwargs)

    @classmethod
    def create_all_dependencies(
        cls,
        *,
        router_backend: Optional[str] = None,
        detectors: Optional[Dict[str, Any]] = None,
        jsonl_evidence_path: Optional[str] = "evidence.jsonl",
        critics: Optional[Dict[str, Any]] = None,
        critic_models: Optional[Dict[str, Any]] = None,
        enable_precedent: bool = True,
        enable_uncertainty: bool = True,
        precedent_store: Optional[Any] = None,
        mock_all: bool = False,
        **kwargs: Any,
    ) -> EngineDependencies:
        """Create all engine dependencies with sensible defaults.

        Args:
            router_backend: Router backend to use
            detectors: Detector implementations
            jsonl_evidence_path: Path to evidence file
            critics: Custom critic implementations
            critic_models: Model adapters for critics
            enable_precedent: Enable precedent analysis
            enable_uncertainty: Enable uncertainty quantification
            precedent_store: Storage client for precedents
            mock_all: Use mock implementations for testing
            **kwargs: Additional configuration

        Returns:
            EngineDependencies with all components initialized
        """
        mock_kwargs = {"mock": True} if mock_all else {}

        return EngineDependencies(
            router=cls.create_router(backend=router_backend, **mock_kwargs),
            detector_engine=cls.create_detector_engine(detectors=detectors, **mock_kwargs),
            evidence_recorder=cls.create_evidence_recorder(
                jsonl_path=jsonl_evidence_path if not mock_all else "mock", **mock_kwargs
            ),
            critics=cls.create_critics(custom_critics=critics, **mock_kwargs),
            precedent_engine=cls.create_precedent_engine(enabled=enable_precedent, **mock_kwargs),
            precedent_retriever=cls.create_precedent_retriever(
                enabled=enable_precedent, store_client=precedent_store, **mock_kwargs
            ),
            uncertainty_engine=cls.create_uncertainty_engine(
                enabled=enable_uncertainty, **mock_kwargs
            ),
            aggregator=cls.create_aggregator(**mock_kwargs),
            review_trigger_evaluator=cls.create_review_trigger_evaluator(**mock_kwargs),
            critic_models=critic_models,
        )


__all__ = [
    "EngineDependencies",
    "DependencyFactory",
]
