"""
ELEANOR V8 — Discrimination Detector
-------------------------------------

Detects discriminatory content based on protected characteristics
including explicit bias, stereotyping, and differential treatment.

Detection Strategies:
1. Protected characteristic mention with negative context
2. Stereotyping pattern recognition
3. Differential treatment indicators
4. Exclusionary language detection
5. Comparative bias analysis
"""

import re
from typing import Dict, Any, List, Set, Tuple
from ..base import Detector
from ..signals import DetectorSignal


class DiscriminationDetector(Detector):
    """
    Detects discriminatory content targeting protected characteristics
    including race, gender, religion, age, disability, and more.
    """

    # Protected characteristics and associated terms
    PROTECTED_CHARACTERISTICS = {
        "race_ethnicity": [
            "black", "white", "asian", "hispanic", "latino", "latina",
            "african", "african american", "caucasian", "arab", "middle eastern",
            "native american", "indigenous", "pacific islander"
        ],
        "gender": [
            "male", "female", "man", "woman", "men", "women", "boy", "girl",
            "masculine", "feminine"
        ],
        "gender_identity": [
            "transgender", "trans", "cisgender", "non-binary", "nonbinary",
            "genderqueer", "genderfluid"
        ],
        "sexual_orientation": [
            "gay", "lesbian", "bisexual", "homosexual", "heterosexual",
            "straight", "queer", "lgbtq", "lgbt"
        ],
        "religion": [
            "muslim", "christian", "jewish", "hindu", "buddhist", "sikh",
            "atheist", "agnostic", "religious", "catholic", "protestant"
        ],
        "age": [
            "old", "young", "elderly", "senior", "aged", "youth", "teen",
            "teenager", "millennial", "boomer", "gen z", "geriatric"
        ],
        "disability": [
            "disabled", "handicapped", "blind", "deaf", "autistic",
            "mentally ill", "wheelchair", "impaired", "special needs"
        ],
        "nationality": [
            "immigrant", "foreigner", "refugee", "migrant", "alien",
            "foreign", "native", "citizen"
        ],
        "socioeconomic": [
            "poor", "rich", "wealthy", "homeless", "unemployed",
            "welfare", "low-income", "working class"
        ],
    }

    # Discriminatory language patterns
    DISCRIMINATION_PATTERNS = [
        # Explicit discrimination
        (r"\b(all|every|typical)\s+({group})\s+(are|is|always)\s+(lazy|stupid|criminal|dangerous|violent|inferior)", "explicit", 0.95),
        (r"\b({group})\s+(can't|cannot|shouldn't|are not capable)\s+(of|to)", "capability_denial", 0.85),
        (r"\b(don't trust|never trust|can't trust)\s+(any|a|the)?\s*({group})", "distrust", 0.9),

        # Stereotyping
        (r"\b({group})\s+(are|is)\s+(naturally|inherently|genetically)\s+\w+", "stereotyping", 0.8),
        (r"\b(typical|stereotypical)\s+({group})", "stereotyping", 0.7),
        (r"\blike\s+(all|most|typical)\s+({group})", "stereotyping", 0.75),

        # Differential treatment
        (r"\b(only|just)\s+({group})\s+(can|should|are allowed)", "differential", 0.8),
        (r"\b(different|separate|special)\s+(rules?|treatment|standards?)\s+for\s+({group})", "differential", 0.85),
        (r"\b(not for|not meant for|not designed for)\s+({group})", "exclusionary", 0.8),

        # Derogatory
        (r"\b({group})\s+(are|is)\s+(animals?|vermin|parasites?|trash|scum)", "dehumanization", 0.95),
        (r"\b(dirty|filthy|disgusting)\s+({group})", "derogatory", 0.9),
    ]

    # Negative context indicators
    NEGATIVE_CONTEXTS = [
        "stupid", "lazy", "criminal", "dangerous", "violent", "inferior",
        "worthless", "useless", "disgusting", "dirty", "evil", "bad",
        "untrustworthy", "dishonest", "corrupt", "incompetent", "weak",
        "backward", "primitive", "savage", "uncivilized", "terrorist"
    ]

    def __init__(self):
        self._compile_patterns()

    def _compile_patterns(self):
        """Pre-compile discrimination detection patterns."""
        self._compiled_patterns = []

        # Build group term regex
        all_terms = []
        for terms in self.PROTECTED_CHARACTERISTICS.values():
            all_terms.extend(terms)

        group_pattern = "|".join(re.escape(term) for term in all_terms)

        for pattern_template, category, severity in self.DISCRIMINATION_PATTERNS:
            pattern = pattern_template.replace("{group}", f"({group_pattern})")
            try:
                compiled = re.compile(pattern, re.IGNORECASE)
                self._compiled_patterns.append((compiled, category, severity))
            except re.error:
                continue

        # Compile negative context pattern
        negative_terms = "|".join(re.escape(t) for t in self.NEGATIVE_CONTEXTS)
        self._negative_pattern = re.compile(
            rf"\b({group_pattern})\s+\w*\s*(are|is|always|never|tend to be)\s+\w*\s*({negative_terms})",
            re.IGNORECASE
        )

    async def detect(self, text: str, context: dict) -> DetectorSignal:
        """
        Detect discriminatory content in text.

        Args:
            text: Text to analyze
            context: Additional context

        Returns:
            DetectorSignal with violation status and details
        """
        if not text or not text.strip():
            return DetectorSignal(
                violation=False,
                severity="S0",
                description="Empty text - no discrimination check possible",
                confidence=0.0,
                metadata={}
            )

        findings = {
            "violations": [],
            "protected_groups_mentioned": set(),
            "categories": set(),
            "max_severity": 0.0,
        }

        text_lower = text.lower()

        # Check which protected groups are mentioned
        for category, terms in self.PROTECTED_CHARACTERISTICS.items():
            for term in terms:
                if term.lower() in text_lower:
                    findings["protected_groups_mentioned"].add(category)
                    break

        # Check discrimination patterns
        for pattern, category, severity in self._compiled_patterns:
            matches = pattern.findall(text)
            if matches:
                findings["violations"].append({
                    "category": category,
                    "severity": severity,
                    "matches": [str(m)[:100] for m in matches[:3]],
                })
                findings["categories"].add(category)
                findings["max_severity"] = max(findings["max_severity"], severity)

        # Check negative context pattern
        negative_matches = self._negative_pattern.findall(text)
        if negative_matches:
            findings["violations"].append({
                "category": "negative_association",
                "severity": 0.85,
                "matches": [str(m)[:100] for m in negative_matches[:3]],
            })
            findings["categories"].add("negative_association")
            findings["max_severity"] = max(findings["max_severity"], 0.85)

        # Calculate overall risk
        risk_score = self._calculate_risk_score(findings)

        # Determine severity level
        severity = self._determine_severity(risk_score, findings)

        # Check if violation threshold met
        violation = risk_score >= 0.4 or findings["max_severity"] >= 0.8

        # Build description
        description = self._build_description(findings, risk_score)

        # Suggest mitigation
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
                "violation_count": len(findings["violations"]),
                "protected_groups": list(findings["protected_groups_mentioned"]),
                "categories": list(findings["categories"]),
                "max_pattern_severity": findings["max_severity"],
            },
            mitigation=mitigation
        )

    def _calculate_risk_score(self, findings: Dict[str, Any]) -> float:
        """Calculate overall discrimination risk score."""
        if not findings["violations"]:
            return 0.0

        # Base score from max severity
        score = findings["max_severity"]

        # Increase for multiple violation types
        category_count = len(findings["categories"])
        if category_count > 1:
            score = min(1.0, score + 0.1 * (category_count - 1))

        # Increase for multiple protected groups targeted
        group_count = len(findings["protected_groups_mentioned"])
        if group_count > 1:
            score = min(1.0, score + 0.05 * (group_count - 1))

        # Critical categories get boost
        critical_categories = {"explicit", "dehumanization", "derogatory"}
        if findings["categories"] & critical_categories:
            score = min(1.0, score + 0.1)

        return score

    def _determine_severity(self, risk_score: float, findings: Dict) -> str:
        """Determine severity level."""
        critical_categories = {"explicit", "dehumanization", "derogatory"}

        if risk_score >= 0.85 or (findings["categories"] & critical_categories):
            return "S3"  # Critical
        elif risk_score >= 0.6:
            return "S2"  # Significant
        elif risk_score >= 0.4:
            return "S1"  # Minor
        return "S0"

    def _build_description(self, findings: Dict, risk_score: float) -> str:
        """Build human-readable description."""
        if not findings["violations"]:
            return "No discriminatory content detected."

        parts = []

        # Summarize violation types
        if findings["categories"]:
            cats = ", ".join(findings["categories"])
            parts.append(f"Discrimination types: {cats}")

        # Note protected groups
        if findings["protected_groups_mentioned"]:
            groups = ", ".join(findings["protected_groups_mentioned"])
            parts.append(f"Protected groups referenced: {groups}")

        # Severity note
        if risk_score >= 0.8:
            parts.append("CRITICAL: Severe discriminatory content detected")
        elif risk_score >= 0.6:
            parts.append("WARNING: Significant discriminatory language")
        elif risk_score >= 0.4:
            parts.append("CAUTION: Potential discriminatory content")

        return ". ".join(parts)

    def _suggest_mitigation(self, findings: Dict) -> str:
        """Suggest mitigation strategies."""
        suggestions = []

        if "explicit" in findings["categories"] or "dehumanization" in findings["categories"]:
            suggestions.append("Remove explicit discriminatory statements")

        if "stereotyping" in findings["categories"]:
            suggestions.append("Avoid generalizations about protected groups")

        if "differential" in findings["categories"]:
            suggestions.append("Ensure equal treatment language")

        if "exclusionary" in findings["categories"]:
            suggestions.append("Use inclusive language")

        if not suggestions:
            return "Review content for bias and ensure respectful treatment of all groups"

        return ". ".join(suggestions)
