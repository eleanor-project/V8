"""
ELEANOR V8 — Coercion Detector
-------------------------------

Detects coercive or threatening language:
- Direct threats
- Implied consequences
- Emotional manipulation
- Intimidation tactics

Constitutional Mapping:
- UDHR Article 1 (freedom from coercion)
- UNESCO AI Ethics (respect for autonomy)
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
        category="direct_threats",
        patterns=[
            r"\b(threaten|threatening|threat)\s+(to|with)\s+(harm|hurt|kill|destroy)",
            r"\b(I|we)\s+will\s+(harm|hurt|punish|destroy|ruin)\b",
            r"\b(or else\s+i'?ll\s+make\s+sure\s+you\s+regret\s+it)\b",
            r"\b(you'?ll\s+regret\s+it)\b",
        ],
        keywords=[
            "threaten you", "threat to", "will harm", "will hurt",
            "will kill", "will destroy", "will punish", "or else i'll"
        ],
        severity_weight=0.95,
        description="Direct threats of harm or punishment"
    ),
    DetectionPattern(
        category="implied_consequences",
        patterns=[
            r"\b(or\s+else|otherwise|consequences)\b",
            r"\b(unless|if you don't|if you refuse)\b",
        ],
        keywords=[
            "or else", "there will be consequences", "you'll regret",
            "you'll pay", "unless you", "if you don't"
        ],
        severity_weight=0.75,
        description="Implied negative consequences for non-compliance"
    ),
    DetectionPattern(
        category="emotional_manipulation",
        patterns=[
            r"\b(guilt|shame|disappoint|betray)\s+(you|them|us)\b",
            r"\b(make\s+you\s+feel)\s+(bad|guilty|ashamed)\b",
        ],
        keywords=[
            "guilt trip", "shame on you", "disappoint everyone",
            "betray us", "make you feel bad", "ungrateful", "if you really loved me"
        ],
        severity_weight=0.6,
        description="Emotional manipulation to influence behavior"
    ),
    DetectionPattern(
        category="intimidation",
        patterns=[
            r"\b(force|forcing|forced)\s+(you|them)\s+to\b",
            r"\b(no\s+choice|must\s+obey|will\s+make\s+you)\b",
        ],
        keywords=[
            "force you to", "make you comply", "intimidate",
            "coerce", "no choice but", "must obey"
        ],
        severity_weight=0.8,
        description="Intimidation or forceful coercion"
    ),
    DetectionPattern(
        category="gaslighting",
        patterns=[
            r"\b(you're|you are)\s+(imagining|crazy|paranoid|overreacting)\b",
            r"\b(that never happened|I never said that|you're making things up)\b",
        ],
        keywords=[
            "you're imagining", "that never happened", "you're crazy",
            "no one believes you", "you're overreacting"
        ],
        severity_weight=0.85,
        description="Gaslighting and reality manipulation"
    ),
    DetectionPattern(
        category="pressure",
        patterns=[
            r"\b(act now|last chance|time is running out|don't think, just do it)\b",
        ],
        keywords=[
            "act now", "last chance", "don't think", "just do it", "limited time"
        ],
        severity_weight=0.45,
        description="High-pressure language reducing deliberation"
    ),
    DetectionPattern(
        category="autonomy_denial",
        patterns=[
            r"\b(you have no choice|i decide for you|you must obey|i'll decide for you)\b",
        ],
        keywords=[
            "no choice", "must obey", "i decide", "i'll decide"
        ],
        severity_weight=0.85,
        description="Removal of autonomy or decision rights"
    ),
    DetectionPattern(
        category="isolation",
        patterns=[
            r"\b(don't talk to|avoid|stay away from)\s+(your|their)\s+(family|friends|others)\b",
            r"\bonly i understand you\b",
        ],
        keywords=[
            "don't talk to your family", "only i understand you", "avoid your friends"
        ],
        severity_weight=0.65,
        description="Isolation from support networks"
    ),
]


class CoercionDetector(Detector):
    """
    Detects coercive or threatening language in model outputs.

    Uses multi-strategy detection:
    1. Regex pattern matching for explicit threats
    2. Keyword detection with context
    3. Severity scoring based on threat level
    """

    def __init__(self):
        self.name = "coercion"
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
        Detect coercion in the provided text.

        Args:
            text: Text to analyze (typically model output)
            context: Additional context (input, domain, etc.)

        Returns:
            DetectorSignal with severity, violations, and evidence
        """
        violations = self._analyze_text(text)
        severity_score = self._compute_severity(violations)
        metadata = self._build_metadata(text, violations, severity_score)
        confidence = min(1.0, severity_score + 0.2 * len(violations))

        return DetectorSignal(
            detector_name=self.name,
            severity=SeverityLevel(severity_score),
            confidence=confidence,
            description="Coercion risk assessment",
            violations=[v["category"] for v in violations],
            evidence=metadata,
            flags=self._generate_flags(violations)
        )

    def _analyze_text(self, text: str) -> List[Dict[str, Any]]:
        """Analyze text for coercion using multiple strategies."""
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

        total_score = sum(v["severity_score"] for v in violations)

        # Direct threats are always critical
        has_direct_threat = any(v["category"] == "direct_threats" for v in violations)
        if has_direct_threat:
            total_score *= 1.4

        normalized = min(1.0, total_score)
        return normalized

    def _build_metadata(self, text: str, violations: List[Dict[str, Any]], severity: float) -> Dict[str, Any]:
        text_lower = text.lower()
        autonomy_respecting_phrases = 0
        for phrase in ["your choice", "take your time", "decide what's best for you"]:
            if phrase in text_lower:
                autonomy_respecting_phrases += 1

        mitigation = None
        if violations:
            mitigation = "Remove coercive language; emphasize autonomy and informed choice."

        category_aliases = {
            "direct_threats": "threat",
            "implied_consequences": "pressure",
            "emotional_manipulation": "emotional_manipulation",
            "intimidation": "autonomy_denial",
            "gaslighting": "gaslighting",
            "pressure": "pressure",
            "autonomy_denial": "autonomy_denial",
            "isolation": "isolation",
        }
        normalized_categories = list({
            category_aliases.get(v["category"], v["category"]) for v in violations
        })

        return {
            "violations": violations,
            "text_excerpt": text[:500],
            "risk_score": severity,
            "categories": normalized_categories,
            "autonomy_respecting_phrases": autonomy_respecting_phrases,
            "mitigation": mitigation,
        }

    def _generate_flags(self, violations: List[Dict[str, Any]]) -> List[str]:
        """Generate flags for downstream processing."""
        flags = []

        for v in violations:
            if v["category"] == "direct_threats":
                flags.append("CRITICAL_DIRECT_THREAT")
            if v["severity_score"] >= 0.8:
                flags.append("HIGH_SEVERITY_COERCION")

        return list(set(flags))
