import uuid
import time
from typing import Dict, Any

class BaseCriticV8:
    """
    Base class for all ELEANOR V8 critics.
    Provides:
      - severity framework
      - redundancy filter
      - evidence package builder
      - uncertainty propagation hooks
      - delta scoring for drift detection
    """

    SEVERITY_LEVELS = ["INFO", "WARNING", "VIOLATION", "CRITICAL"]

    def __init__(self, name: str, version: str = "8.0"):
        self.name = name
        self.version = version

    async def evaluate(self, model, input_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Every critic must implement this."""
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
