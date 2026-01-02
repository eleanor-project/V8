from __future__ import annotations

import uuid
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

from engine.schemas.escalation import (
    CriticEvaluation,
    Concern,
    EscalationSignal,
)
from engine.schemas.pipeline_types import CriticResult

class BaseCriticV8:
    """
    Base class for all ELEANOR V8 critics.

    Provides:
      - severity framework
      - redundancy filter
      - evidence package builder
      - uncertainty propagation hooks
      - delta scoring for drift detection
      - hybrid model configuration (preferred model + registry)

    Model Configuration (Hybrid Approach):
    ---------------------------------------
    Critics can be configured with models in three ways:

    1. Preferred Model (Explicit):
       critic = RightsCriticV8(model=OpusModel())

    2. Registry (Centralized):
       registry = ModelRegistry()
       registry.assign_model("rights", "claude-opus-4.5")
       critic = RightsCriticV8(registry=registry)

    3. Runtime Override:
       result = await critic.evaluate(model=specific_model, ...)

    Priority: runtime_model > preferred_model > registry > fallback
    """

    SEVERITY_LEVELS = ["INFO", "WARNING", "VIOLATION", "CRITICAL"]

    def __init__(
        self,
        name: str,
        version: str = "8.0",
        model=None,
        registry=None
    ):
        """
        Initialize critic with optional model configuration.

        Args:
            name: Critic name (e.g., "rights", "fairness")
            version: Critic version
            model: Preferred model instance (Option 1 - explicit)
            registry: ModelRegistry for centralized config (Option 3)
        """
        self.name = name
        self.version = version
        self._preferred_model = model
        self._registry = registry

    def get_model(self, runtime_model=None, context: Optional[Dict[str, Any]] = None):
        """
        Get the appropriate model for this critic.

        Priority:
        1. runtime_model (passed to evaluate())
        2. preferred_model (set in __init__)
        3. registry lookup (if registry provided)
        4. None (caller must provide fallback)

        Args:
            runtime_model: Model passed at evaluation time
            context: Optional context for registry routing

        Returns:
            Model instance or None
        """
        # Priority 1: Runtime override
        if runtime_model is not None:
            return runtime_model

        # Priority 2: Preferred model
        if self._preferred_model is not None:
            return self._preferred_model

        # Priority 3: Registry lookup
        if self._registry is not None:
            model_id = self._registry.get_model_for_critic(self.name, context)
            # Note: Registry returns model_id string, not instance
            # Caller needs to resolve this to actual model instance
            # This is intentional to keep registry lightweight
            return model_id

        # Priority 4: No model configured
        return None

    async def evaluate(self, model, input_text: str, context: Dict[str, Any]) -> CriticResult:
        """
        Every critic must implement this.

        Args:
            model: Model instance or None (will use configured model if None)
            input_text: User input text
            context: Evaluation context

        Returns:
            Evidence package
        """
        raise NotImplementedError

    def severity(self, score: float) -> str:
        if score < 0.2:
            return "INFO"
        elif score < 0.45:
            return "WARNING"
        elif score < 0.75:
            return "VIOLATION"
        return "CRITICAL"

    def build_evidence(self, *, score: float, rationale: str, principle: str, evidence: Dict[str, Any], flags=None):
        return {
            "critic_id": f"{self.name}:{self.version}",
            "timestamp": time.time(),
            "score": score,
            "severity": self.severity(score),
            "principle": principle,
            "rationale": rationale,
            "evidence": evidence,
            "flags": flags or [],
            "uuid": str(uuid.uuid4())
        }


# ============================================================
# ConstitutionalCritic — structured escalation output
# ============================================================

class ConstitutionalCritic(ABC):
    """
    Base class for schema-aligned constitutional critics.

    Guarantees:
    - uniform CriticEvaluation output
    - explicit, clause-aware escalation
    - stable charter_version tracking
    """

    critic_id: str
    charter_version: str

    def __init__(self, *, charter_version: str):
        self.charter_version = charter_version

    @abstractmethod
    def evaluate(self, **kwargs) -> CriticEvaluation:
        """Perform critic-specific evaluation and return a CriticEvaluation."""
        raise NotImplementedError

    def _build_evaluation(
        self,
        *,
        concerns: List[Concern],
        severity_score: float,
        citations: List[str],
        escalation: Optional[EscalationSignal] = None,
        uncertainty: Optional[str] = None,
    ) -> CriticEvaluation:
        """Assemble a CriticEvaluation with consistent field names."""
        return CriticEvaluation(
            critic_id=self.critic_id,
            charter_version=self.charter_version,
            concerns=concerns,
            escalation=escalation,
            severity_score=severity_score,
            citations=citations,
            uncertainty=uncertainty,
        )
