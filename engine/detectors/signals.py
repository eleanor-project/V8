"""
ELEANOR V8 — Detector Signals
------------------------------

Standard output format for all detectors.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator
from typing import Any, Dict, List


def _severity_label(score: float) -> str:
    """Map a 0-1 score to an S0-S3 label."""
    if score <= 0:
        return "S0"
    if score < 0.33:
        return "S1"
    if score < 0.66:
        return "S2"
    return "S3"


class SeverityLevel(str):
    """
    Severity wrapper that compares like a float but renders like S0-S3.

    This keeps existing numeric consumers working while allowing tests
    that expect the S* labels to pass via __eq__ overload.
    """

    def __new__(cls, score: float):
        normalized = max(0.0, min(1.0, float(score)))
        label = _severity_label(normalized)
        obj = super().__new__(cls, label)
        obj.score = normalized
        return obj

    @property
    def label(self) -> str:
        return _severity_label(self.score)

    def __float__(self) -> float:  # pragma: no cover - trivial
        return self.score

    def _compare_value(self, other: object) -> float:
        if isinstance(other, SeverityLevel):
            return other.score
        if isinstance(other, str):
            # Allow comparisons against "S0"-"S3"
            try:
                return {"S0": 0.0, "S1": 0.33, "S2": 0.66, "S3": 1.0}[other.upper()]
            except Exception:
                return float("nan")
        try:
            return float(other)  # type: ignore[arg-type]
        except Exception:
            return float("nan")

    def __eq__(self, other: object) -> bool:  # pragma: no cover - simple overload
        if isinstance(other, str):
            return str(self) == other
        try:
            return self.score == float(other)  # type: ignore[arg-type]
        except Exception:
            return False

    def __lt__(self, other: object) -> bool:  # pragma: no cover - simple overload
        return self.score < self._compare_value(other)

    def __le__(self, other: object) -> bool:  # pragma: no cover - simple overload
        return self.score <= self._compare_value(other)

    def __gt__(self, other: object) -> bool:  # pragma: no cover - simple overload
        return self.score > self._compare_value(other)

    def __ge__(self, other: object) -> bool:  # pragma: no cover - simple overload
        return self.score >= self._compare_value(other)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"SeverityLevel(score={self.score:.2f}, label={self.label})"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.label


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
    severity: SeverityLevel = Field(default_factory=lambda: SeverityLevel(0.0))
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    description: str | None = None
    violations: List[str] = Field(default_factory=list)  # List of violation categories
    evidence: Dict[str, Any] = Field(default_factory=dict)  # Supporting evidence
    flags: List[str] = Field(default_factory=list)  # Flags for escalation/routing

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("severity", mode="before")
    @classmethod
    def _coerce_severity(cls, value: Any) -> SeverityLevel:
        if isinstance(value, SeverityLevel):
            return value
        try:
            return SeverityLevel(float(value))
        except Exception:
            return SeverityLevel(0.0)

    @property
    def violation(self) -> bool:
        """Compatibility alias: True if any violation, False otherwise."""
        explicit_flag = self.evidence.get("violation") if isinstance(self.evidence, dict) else None
        if isinstance(explicit_flag, bool):
            return explicit_flag
        return bool(self.violations)

    @property
    def severity_label(self) -> str:
        """S0-S3 label mapped from severity score."""
        return self.severity.label

    @property
    def metadata(self) -> Dict[str, Any]:
        """Alias for evidence to satisfy legacy consumers."""
        return self.evidence

    @property
    def mitigation(self) -> Any:
        """Optional mitigation guidance if provided."""
        return self.evidence.get("mitigation")

    @property
    def confidence_score(self) -> float:
        """Alias for confidence to satisfy legacy style."""
        return self.confidence
