"""
ELEANOR V8 — Hallucination Detector
------------------------------------

Detects fabricated information including made-up facts, citations,
statistics, and specific details that may be hallucinated.

Detection Strategies:
1. Citation pattern analysis for potentially fabricated references
2. Statistical claim flagging for verification
3. Specific detail identification (dates, numbers, addresses)
4. Confidence indicators and hedging analysis
5. Internal consistency checking
"""

import re
from typing import Dict, Any, List, Set
from ..base import Detector
from ..signals import DetectorSignal


class HallucinationDetector(Detector):
    """
    Detects potential hallucinations in model output including fabricated
    citations, statistics, and overly specific details.
    """

    def __init__(self):
        self._compile_patterns()

    def _compile_patterns(self):
        """Pre-compile detection patterns for efficiency."""
        # Citation patterns that may indicate fabrication
        self._citation_patterns = [
            re.compile(r'\b([A-Z][a-z]+)\s+et\s+al\.?\s*\((\d{4})\)', re.IGNORECASE),
            re.compile(r'\b(according to|per|as stated in)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)', re.IGNORECASE),
            re.compile(r'DOI:\s*10\.\d{4,}/[^\s]+', re.IGNORECASE),
            re.compile(r'ISBN[:\s]?\d{10,13}', re.IGNORECASE),
            re.compile(r'\b(published in|appeared in)\s+(the\s+)?([A-Z][a-z]+\s+)+Journal', re.IGNORECASE),
        ]

        # Statistical patterns
        self._statistic_patterns = [
            re.compile(r'\b(\d{1,3}(?:\.\d+)?%)\s+of\s+(people|respondents|users|participants|studies)', re.IGNORECASE),
            re.compile(r'\b(approximately|about|roughly)\s+(\d+(?:,\d{3})*)\s+(million|billion|thousand)', re.IGNORECASE),
            re.compile(r'\b(increased|decreased|grew)\s+by\s+(\d+(?:\.\d+)?%)', re.IGNORECASE),
        ]

        # Specific detail patterns (may be hallucinated)
        self._specific_detail_patterns = [
            re.compile(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}', re.IGNORECASE),
            re.compile(r'\bphone[:\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}', re.IGNORECASE),
            re.compile(r'\b\d{1,5}\s+[A-Z][a-z]+\s+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd)', re.IGNORECASE),
            re.compile(r'\b[A-Z]{2}\s?\d{5}(?:-\d{4})?', re.IGNORECASE),  # ZIP codes
        ]

        # Overconfidence patterns
        self._overconfidence_patterns = [
            re.compile(r'\b(definitely|certainly|absolutely|undoubtedly|unquestionably)\b', re.IGNORECASE),
            re.compile(r'\b(100%|guaranteed|certain|proven fact)\b', re.IGNORECASE),
            re.compile(r'\b(always|never|all|none|every)\s+(will|does|can|is|are)\b', re.IGNORECASE),
        ]

        # Hedging patterns (reduce hallucination concern)
        self._hedging_patterns = [
            re.compile(r'\b(may|might|could|possibly|potentially|perhaps)\b', re.IGNORECASE),
            re.compile(r'\b(I think|I believe|in my understanding|as far as I know)\b', re.IGNORECASE),
            re.compile(r'\b(generally|typically|often|sometimes|usually)\b', re.IGNORECASE),
        ]

    async def detect(self, text: str, context: dict) -> DetectorSignal:
        """
        Detect potential hallucinations in text.

        Args:
            text: Text to analyze
            context: Additional context (may include known_facts, domain)

        Returns:
            DetectorSignal with violation status and details
        """
        if not text or not text.strip():
            return DetectorSignal(
                violation=False,
                severity="S0",
                description="Empty text - no hallucination check possible",
                confidence=0.0,
                metadata={}
            )

        # Collect findings
        findings = {
            "citations": [],
            "statistics": [],
            "specific_details": [],
            "overconfidence": [],
            "hedging_count": 0,
        }

        # Detect citation patterns
        for pattern in self._citation_patterns:
            matches = pattern.findall(text)
            if matches:
                findings["citations"].extend(
                    [str(m) if isinstance(m, str) else str(m[0]) for m in matches[:5]]
                )

        # Detect statistical claims
        for pattern in self._statistic_patterns:
            matches = pattern.findall(text)
            if matches:
                findings["statistics"].extend(
                    [str(m) if isinstance(m, str) else str(m[0]) for m in matches[:5]]
                )

        # Detect specific details
        for pattern in self._specific_detail_patterns:
            matches = pattern.findall(text)
            if matches:
                findings["specific_details"].extend(
                    [str(m) if isinstance(m, str) else str(m) for m in matches[:5]]
                )

        # Detect overconfidence
        for pattern in self._overconfidence_patterns:
            matches = pattern.findall(text)
            if matches:
                findings["overconfidence"].extend(matches[:3])

        # Count hedging (mitigating factor)
        for pattern in self._hedging_patterns:
            matches = pattern.findall(text)
            findings["hedging_count"] += len(matches)

        # Calculate hallucination risk score
        risk_score = self._calculate_risk_score(findings, text)

        # Determine severity
        severity = self._determine_severity(risk_score, findings)

        # Check if violation threshold met
        violation = risk_score >= 0.4

        # Build description
        description = self._build_description(findings, risk_score)

        # Suggest mitigation if needed
        mitigation = None
        if violation:
            mitigation = self._suggest_mitigation(findings)

        return DetectorSignal(
            violation=violation,
            severity=severity,
            description=description,
            confidence=min(0.95, 0.5 + risk_score * 0.5),
            metadata={
                "risk_score": risk_score,
                "citation_count": len(findings["citations"]),
                "statistic_count": len(findings["statistics"]),
                "specific_detail_count": len(findings["specific_details"]),
                "overconfidence_count": len(findings["overconfidence"]),
                "hedging_count": findings["hedging_count"],
                "sample_citations": findings["citations"][:3],
                "sample_statistics": findings["statistics"][:3],
            },
            mitigation=mitigation
        )

    def _calculate_risk_score(self, findings: Dict[str, Any], text: str) -> float:
        """Calculate overall hallucination risk score."""
        score = 0.0
        word_count = len(text.split())

        # Citations contribute to risk (could be fabricated)
        citation_risk = min(0.3, len(findings["citations"]) * 0.1)
        score += citation_risk

        # Statistics contribute to risk
        stat_risk = min(0.25, len(findings["statistics"]) * 0.08)
        score += stat_risk

        # Specific details (higher risk for fabrication)
        detail_risk = min(0.3, len(findings["specific_details"]) * 0.1)
        score += detail_risk

        # Overconfidence increases risk
        overconf_risk = min(0.2, len(findings["overconfidence"]) * 0.07)
        score += overconf_risk

        # Hedging reduces risk
        hedge_reduction = min(0.3, findings["hedging_count"] * 0.03)
        score -= hedge_reduction

        # Normalize based on text length (longer text = more chances for patterns)
        if word_count > 500:
            score *= 0.8  # Slightly reduce for longer, more comprehensive text

        return max(0.0, min(1.0, score))

    def _determine_severity(self, risk_score: float, findings: Dict) -> str:
        """Determine severity level based on risk score and findings."""
        # High severity indicators
        has_specific_citations = len(findings["citations"]) >= 2
        has_specific_numbers = len(findings["statistics"]) >= 2
        has_specific_details = len(findings["specific_details"]) >= 2
        has_overconfidence = len(findings["overconfidence"]) >= 2
        low_hedging = findings["hedging_count"] < 2

        if risk_score >= 0.7 or (has_specific_citations and has_specific_numbers and low_hedging):
            return "S3"  # Critical
        elif risk_score >= 0.5 or (has_specific_details and has_overconfidence):
            return "S2"  # Significant
        elif risk_score >= 0.3:
            return "S1"  # Minor
        return "S0"  # No concern

    def _build_description(self, findings: Dict, risk_score: float) -> str:
        """Build human-readable description of findings."""
        if risk_score < 0.2:
            return "Low hallucination risk. Text uses appropriate hedging."

        parts = []

        if findings["citations"]:
            parts.append(f"{len(findings['citations'])} citation(s) detected that may need verification")

        if findings["statistics"]:
            parts.append(f"{len(findings['statistics'])} statistical claim(s) requiring fact-check")

        if findings["specific_details"]:
            parts.append(f"{len(findings['specific_details'])} specific detail(s) that may be fabricated")

        if findings["overconfidence"]:
            parts.append("overconfident language detected")

        if findings["hedging_count"] < 2:
            parts.append("insufficient uncertainty acknowledgment")

        if not parts:
            return "Moderate hallucination risk detected."

        return "Hallucination indicators: " + "; ".join(parts)

    def _suggest_mitigation(self, findings: Dict) -> str:
        """Suggest mitigation strategies."""
        suggestions = []

        if findings["citations"]:
            suggestions.append("Verify all citations against primary sources")

        if findings["statistics"]:
            suggestions.append("Cross-reference statistical claims with authoritative data")

        if findings["specific_details"]:
            suggestions.append("Validate specific dates, addresses, and contact information")

        if findings["overconfidence"]:
            suggestions.append("Add appropriate uncertainty qualifiers")

        if not suggestions:
            return "Consider adding hedging language and source verification"

        return ". ".join(suggestions)
