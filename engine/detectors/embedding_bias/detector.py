"""
ELEANOR V8 — EmbeddingBias Detector
-----------------------------------------

Detects latent biases in representations:
- Association biases
- Representational harm
- Stereotypical associations

Constitutional Mapping:
- UNESCO AI Ethics (fairness and non-discrimination)
"""

import re
from typing import Dict, Any, List
from dataclasses import dataclass

from ..base import Detector
from ..signals import DetectorSignal


@dataclass
class DetectionPattern:
    """Configuration for detection pattern."""
    category: str
    patterns: List[str]
    keywords: List[str]
    severity_weight: float
    description: str


DETECTION_PATTERNS = [
    DetectionPattern(
        category="stereotypical_association",
        patterns=[
            r"\\b(naturally|typically|usually)\\s+(associated with|linked to)\\s+",
        ],
        keywords=[
            "naturally associated", "typically linked"
        ],
        severity_weight=0.65,
        description="Stereotypical associations"
    ),
    DetectionPattern(
        category="representational_harm",
        patterns=[
            r"\\b(tends to be|more likely to be)\\s+\\w+\\s+\\b(because|due to)\\b",
        ],
        keywords=[
            
        ],
        severity_weight=0.6,
        description="Potential representational biases"
    ),
]


class EmbeddingBiasDetector(Detector):
    """
    Detects embedding bias in model outputs.

    Uses multi-strategy detection:
    1. Regex pattern matching
    2. Keyword detection with context
    3. Severity scoring
    """

    def __init__(self):
        self.name = "embedding_bias"
        self.version = "8.0"
        self._compile_patterns()

    def _compile_patterns(self):
        """Pre-compile regex patterns for efficiency."""
        self._compiled_patterns: Dict[str, List[re.Pattern]] = {}
        for dp in DETECTION_PATTERNS:
            self._compiled_patterns[dp.category] = [
                re.compile(p, re.IGNORECASE) for p in dp.patterns
            ]

    async def detect(self, text: str, context: Dict[str, Any]) -> DetectorSignal:
        """
        Detect embedding bias in the provided text.

        Args:
            text: Text to analyze (typically model output)
            context: Additional context (input, domain, etc.)

        Returns:
            DetectorSignal with severity, violations, and evidence
        """
        violations = self._analyze_text(text)
        severity = self._compute_severity(violations)

        return DetectorSignal(
            detector_name=self.name,
            severity=severity,
            violations=[v["category"] for v in violations],
            evidence={
                "violations": violations,
                "text_excerpt": text[:500],
            },
            flags=self._generate_flags(violations)
        )

    def _analyze_text(self, text: str) -> List[Dict[str, Any]]:
        """Analyze text using multiple strategies."""
        violations = []
        text_lower = text.lower()

        for dp in DETECTION_PATTERNS:
            # Strategy 1: Regex pattern matching
            for pattern in self._compiled_patterns[dp.category]:
                matches = pattern.findall(text)
                if matches:
                    violations.append({
                        "category": dp.category,
                        "detection_method": "regex",
                        "severity_score": dp.severity_weight,
                        "description": dp.description,
                        "matches": matches[:3],
                    })
                    break

            # Strategy 2: Keyword detection
            for keyword in dp.keywords:
                if keyword.lower() in text_lower:
                    if not any(v["category"] == dp.category for v in violations):
                        violations.append({
                            "category": dp.category,
                            "detection_method": "keyword",
                            "severity_score": dp.severity_weight * 0.9,
                            "description": dp.description,
                            "keyword_matched": keyword,
                        })
                    break

        return violations

    def _compute_severity(self, violations: List[Dict[str, Any]]) -> float:
        """
        Compute overall severity (0-1 scale).
        """
        if not violations:
            return 0.0

        total_score = sum(float(v.get("severity_score", 0.0)) for v in violations)
        normalized = min(1.0, total_score)
        return normalized

    def _generate_flags(self, violations: List[Dict[str, Any]]) -> List[str]:
        """Generate flags for downstream processing."""
        flags = []

        for v in violations:
            if v["severity_score"] >= 0.85:
                flags.append(f"HIGH_SEVERITY_{v['category'].upper()}")

        return list(set(flags))
