"""
ELEANOR V8 - Mock Implementations for Dependency Injection

Lightweight mocks for routing, critics, detectors, evidence recording,
precedent, uncertainty, aggregation, and governance review evaluation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class MockRouter:
    def __init__(
        self,
        *,
        response_text: str = "mock response",
        model_name: str = "mock-model",
        model_version: str = "0.0",
        reason: str = "mock",
    ):
        self.response_text = response_text
        self.model_name = model_name
        self.model_version = model_version
        self.reason = reason

    def route(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {
            "response_text": self.response_text,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "reason": self.reason,
            "health_score": 1.0,
            "cost": None,
            "diagnostics": {"mock": True},
        }


class MockCritic:
    def __init__(self, name: str = "mock", score: float = 0.05):
        self.name = name
        self._score = score

    async def evaluate(self, model_adapter: Any, input_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "severity": self._score,
            "violations": [],
            "justification": f"{self.name} mock evaluation",
            "score": self._score,
        }

    def severity(self, score: float) -> str:
        if score < 0.2:
            return "INFO"
        if score < 0.45:
            return "WARNING"
        if score < 0.75:
            return "VIOLATION"
        return "CRITICAL"


class MockDetectorEngine:
    async def detect_all(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    def aggregate_signals(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        return {"summary": "mock", "score": 0.0}


class MockEvidenceRecorder:
    def __init__(self):
        self.buffer: List[Dict[str, Any]] = []

    async def record(self, **kwargs: Any) -> Dict[str, Any]:
        self.buffer.append(dict(kwargs))
        return dict(kwargs)

    def latest(self, n: int = 100) -> List[Dict[str, Any]]:
        return self.buffer[-n:]


class MockPrecedentEngine:
    def analyze(
        self,
        critics: Dict[str, Any],
        precedent_cases: List[Dict[str, Any]],
        query_embedding: List[float],
    ) -> Dict[str, Any]:
        return {
            "alignment_score": 0.0,
            "support_strength": 0.0,
            "conflict_level": 0.0,
            "drift_score": 0.0,
            "clusters": [],
            "is_novel": True,
            "analysis": "mock precedent analysis",
        }


class MockPrecedentRetriever:
    def retrieve(
        self,
        query: str,
        critic_results: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        return {
            "precedent_cases": [],
            "query_embedding": [],
        }


class MockUncertaintyEngine:
    def compute(
        self,
        critics: Dict[str, Any],
        model_used: str,
        precedent_alignment: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "overall_uncertainty": 0.0,
            "needs_escalation": False,
            "explanation": "mock uncertainty",
        }


class MockAggregator:
    def aggregate(
        self,
        critics: Dict[str, Any],
        precedent: Dict[str, Any],
        uncertainty: Dict[str, Any],
        model_output: str = "",
    ) -> Dict[str, Any]:
        return {
            "decision": "allow",
            "final_output": model_output or "",
            "score": {"average_severity": 0.0},
            "rights_impacted": [],
            "dissent": None,
            "precedent": precedent or {},
            "uncertainty": uncertainty or {},
        }


class MockReviewTriggerEvaluator:
    def evaluate(self, case: Any) -> Dict[str, Any]:
        return {"review_required": False, "triggers": []}


__all__ = [
    "MockRouter",
    "MockCritic",
    "MockDetectorEngine",
    "MockEvidenceRecorder",
    "MockPrecedentEngine",
    "MockPrecedentRetriever",
    "MockUncertaintyEngine",
    "MockAggregator",
    "MockReviewTriggerEvaluator",
]
