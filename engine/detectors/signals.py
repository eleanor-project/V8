"""
ELEANOR V8 — Detector Signals
------------------------------

Standard output format for all detectors.
"""

from pydantic import BaseModel, Field
from typing import Any, Dict, List


class DetectorSignal(BaseModel):
    """
    Standard output format for all detectors.

    Attributes:
        detector_name: Name of the detector that generated this signal
        severity: Float 0.0-1.0 where 0.0=no concern, 0.3=minor, 0.6=moderate, 0.9+=severe
        violations: List of violation categories detected
        evidence: Supporting evidence dictionary
        flags: Flags for escalation/routing to critics
    """
    detector_name: str
    severity: float = Field(ge=0.0, le=1.0)  # 0.0 - 1.0
    violations: List[str] = Field(default_factory=list)  # List of violation categories
    evidence: Dict[str, Any] = Field(default_factory=dict)  # Supporting evidence
    flags: List[str] = Field(default_factory=list)  # Flags for escalation/routing

    @property
    def violation(self) -> bool:
        """Compatibility alias: True if any violation, False otherwise."""
        return bool(self.violations)

    @property
    def metadata(self) -> Dict[str, Any]:
        """Alias for evidence to satisfy legacy consumers."""
        return self.evidence

    @property
    def mitigation(self) -> Any:
        """Optional mitigation guidance if provided."""
        return self.evidence.get("mitigation")

    @property
    def description(self) -> Any:
        """Optional description if provided."""
        return self.evidence.get("description")
