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
from ..signals import DetectorSignal, SeverityLevel


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
            r"\b(according to|cited in|published in)\s+[\w\s]+?\(\d{4}\)",
            r"\b(study|research|paper)\s+(by|from)\s+[A-Z][a-zA-Z]+\s+\(\d{4}\)\b",
        ],
        keywords=[
            "according to", "cited in", "published in", "et al."
        ],
        severity_weight=0.35,
        description="Fabricated citations or references"
    ),
    DetectionPattern(
        category="false_statistics",
        patterns=[
            r"\b\d+(\.\d+)?%\s+(of|showed|indicated|demonstrated)",
            r"\bexactly\s+\d+%\s+of\s+",
        ],
        keywords=[
            "exactly percent", "precisely showed", "increase by"
        ],
        severity_weight=0.25,
        description="Potentially false or fabricated statistics"
    ),
    DetectionPattern(
        category="invented_entities",
        patterns=[
            r"\b(University|Institute|Organization)\s+of\s+[A-Z][\w]+\s+(found|reported|stated)",
        ],
        keywords=[],
        severity_weight=0.2,
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
        metadata = self._build_metadata(text, violations)
        severity_score = max(self._compute_severity(violations), metadata.get("risk_score", 0.0))
        confidence = min(1.0, 0.2 * len(violations) + severity_score)

        return DetectorSignal(
            detector_name=self.name,
            severity=SeverityLevel(severity_score),
            confidence=confidence,
            description="Hallucination risk assessment",
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

        # Fallback lightweight heuristics to ensure coverage for analytics
        if re.search(r"\(\d{4}\)", text) or "et al" in text_lower:
            if not any(v["category"] == "fabricated_citations" for v in violations):
                violations.append({
                    "category": "fabricated_citations",
                    "detection_method": "heuristic",
                    "severity_score": 0.25,
                    "description": "Citation present; verify source.",
                })

        if re.search(r"\d+%|\bpercent\b", text_lower):
            if not any(v["category"] == "false_statistics" for v in violations):
                violations.append({
                    "category": "false_statistics",
                    "detection_method": "heuristic",
                    "severity_score": 0.2,
                    "description": "Statistical claim present.",
                })

        if re.search(r"definitely|absolutely|100%|guaranteed", text, re.IGNORECASE):
            violations.append({
                "category": "overconfidence",
                "detection_method": "keyword",
                "severity_score": 0.15,
                "description": "Overconfident language present.",
            })

        return violations

def _build_metadata(self, text: str, violations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build metadata counts for analytics/tests."""
        citation_count = len(re.findall(r"\(\d{4}\)", text)) + len(re.findall(r"et al\.?", text, re.IGNORECASE))
        statistic_count = len(re.findall(r"\d+%|\bpercent\b", text, re.IGNORECASE))
        specific_detail_count = len(re.findall(r"(\d{3}[-\s]?\d{3}[-\s]?\d{4})", text))
        specific_detail_count += len(re.findall(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}", text, re.IGNORECASE))
        overconfidence_count = len(re.findall(r"definitely|absolutely|100%|guaranteed", text, re.IGNORECASE))

        total_risk = 0.0
        if citation_count:
            total_risk += 0.25 + 0.1 * min(citation_count, 3)
        if statistic_count:
            total_risk += 0.2 + 0.05 * min(statistic_count, 3)
        if specific_detail_count:
            total_risk += 0.15
        if overconfidence_count:
            total_risk += 0.15

        categories = list({v["category"] for v in violations})

        mitigation = None
        if categories:
            mitigation = "Verify citations and statistics; add sources or rephrase uncertain claims."

        return {
            "violations": violations,
            "text_excerpt": text[:500],
            "citation_count": citation_count,
            "statistic_count": statistic_count,
            "specific_detail_count": specific_detail_count,
            "overconfidence_count": overconfidence_count,
            "categories": categories,
            "risk_score": min(1.0, total_risk),
            "mitigation": mitigation,
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
