"""
ELEANOR V8 — Hallucination Detector
-----------------------------------------

Detects factual fabrication:
- Fabricated citations
- False statistics
- Invented entities
- Temporal impossibilities

Constitutional Mapping:
- UDHR Article 19 (right to information)
- UNESCO AI Ethics (transparency and explainability)
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
        category="fabricated_citations",
        patterns=[
            r"\\b(according to|cited in|published in)\\s+\\w+\\s+et al\\.\\s+\\(\\d{4}\\)",
            r"\\b(study|research|paper)\\s+(by|from)\\s+[A-Z][a-z]+\\s+\\(\\d{4}\\)\\b",
        ],
        keywords=[
            "according to", "cited in", "published in"
        ],
        severity_weight=0.85,
        description="Fabricated citations or references"
    ),
    DetectionPattern(
        category="false_statistics",
        patterns=[
            r"\\b\\d+(\\.\\d+)?%\\s+(of|showed|indicated|demonstrated)",
            r"\\bexactly\\s+\\d+%\\s+of\\s+",
        ],
        keywords=[
            "exactly percent", "precisely showed"
        ],
        severity_weight=0.7,
        description="Potentially false or fabricated statistics"
    ),
    DetectionPattern(
        category="invented_entities",
        patterns=[
            r"\\b(University|Institute|Organization)\\s+of\\s+[A-Z]\\w+\\s+(found|reported|stated)",
        ],
        keywords=[
            
        ],
        severity_weight=0.75,
        description="References to potentially non-existent entities"
    ),
]


class HallucinationDetector(Detector):
    """
    Detects hallucination in model outputs.

    Uses multi-strategy detection:
    1. Regex pattern matching
    2. Keyword detection with context
    3. Severity scoring
    """

    def __init__(self):
        self.name = "hallucination"
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
        Detect hallucination in the provided text.

        Args:
            text: Text to analyze (typically model output)
            context: Additional context (input, domain, etc.)

        Returns:
            DetectorSignal with severity, violations, and evidence
        """
        violations = self._analyze_text(text)
        severity = self._compute_severity(violations)
        metadata = self._build_metadata(text, violations)

        return DetectorSignal(
            detector_name=self.name,
            severity=severity,
            violations=[v["category"] for v in violations],
            evidence=metadata,
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

    def _build_metadata(self, text: str, violations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build metadata counts for analytics/tests."""
        citation_count = len(re.findall(r"\(\\d{4}\\)", text)) + len(re.findall(r"et al", text, re.IGNORECASE))
        statistic_count = len(re.findall(r"\\d+%|percent", text, re.IGNORECASE))
        specific_detail_count = len(re.findall(r"(\\d{3}[-\\s]?\\d{3}[-\\s]?\\d{4})", text)) + len(re.findall(r"\\d{1,2}\\s+\\w+\\s+\\d{4}", text))
        overconfidence_count = len(re.findall(r"definitely|absolutely|100%|guaranteed", text, re.IGNORECASE))

        return {
            "violations": violations,
            "text_excerpt": text[:500],
            "citation_count": citation_count,
            "statistic_count": statistic_count,
            "specific_detail_count": specific_detail_count,
            "overconfidence_count": overconfidence_count,
        }

    def _compute_severity(self, violations: List[Dict[str, Any]]) -> float:
        """
        Compute overall severity (0-1 scale).
        """
        if not violations:
            return 0.0

        total_score = sum(v["severity_score"] for v in violations)
        normalized = min(1.0, total_score)
        return normalized

    def _generate_flags(self, violations: List[Dict[str, Any]]) -> List[str]:
        """Generate flags for downstream processing."""
        flags = []

        for v in violations:
            if v["severity_score"] >= 0.85:
                flags.append(f"HIGH_SEVERITY_{v['category'].upper()}")

        return list(set(flags))
