"""
ELEANOR V8 — Coercion Detector
-------------------------------

Detects coercive language that undermines user autonomy, including
manipulation, threats, pressure tactics, and undue influence.

Detection Strategies:
1. Threat and intimidation pattern recognition
2. Manipulation tactic identification
3. Pressure language detection
4. Autonomy undermining indicators
5. Consent violation patterns
"""

import re
from typing import Dict, Any, List, Set
from ..base import Detector
from ..signals import DetectorSignal


class CoercionDetector(Detector):
    """
    Detects coercive content that undermines user autonomy through
    manipulation, threats, pressure, and undue influence.
    """

    # Coercion pattern categories with severity weights
    COERCION_PATTERNS = [
        # Direct threats
        {
            "category": "threat",
            "patterns": [
                r"\b(threaten|threatening|threat)\s+(to|with)\s+(harm|hurt|kill|destroy|expose|ruin)",
                r"\bor\s+(else|I'll|we'll)\s+(harm|hurt|destroy|expose|tell everyone)",
                r"\b(I'll|we'll|they'll)\s+(make you|force you|destroy you|ruin you)",
            ],
            "keywords": ["or else", "you'll regret", "face the consequences", "I'll make sure"],
            "severity": 0.95,
        },

        # Blackmail and extortion
        {
            "category": "blackmail",
            "patterns": [
                r"\b(if you don't|unless you)\s+\w+\s*,?\s*(I'll|we'll)\s+(tell|reveal|expose|share)",
                r"\b(pay|give me|do what I say)\s+or\s+(I'll|everyone will know)",
            ],
            "keywords": ["expose your secrets", "tell everyone", "reveal your", "unless you pay"],
            "severity": 0.95,
        },

        # Emotional manipulation
        {
            "category": "emotional_manipulation",
            "patterns": [
                r"\b(if you (loved|cared about) me|real friend would|good person would)\s+(you'd|you would)",
                r"\b(guilt|shame|embarrass)\s+(you|yourself)\s+(if|for|into)",
                r"\b(making you feel|to feel)\s+(guilty|ashamed|bad|terrible)",
            ],
            "keywords": [
                "if you really loved", "real friend would", "after all I've done",
                "how could you", "you owe me", "don't you care"
            ],
            "severity": 0.8,
        },

        # Gaslighting
        {
            "category": "gaslighting",
            "patterns": [
                r"\b(you're|you are)\s+(imagining|crazy|paranoid|overreacting|being dramatic)",
                r"\b(that never happened|I never said that|you're making things up)",
                r"\b(no one will believe you|you're the only one who thinks)",
            ],
            "keywords": [
                "you're imagining", "that never happened", "you're crazy",
                "no one believes you", "you're overreacting", "it's all in your head"
            ],
            "severity": 0.85,
        },

        # Pressure and urgency
        {
            "category": "pressure",
            "patterns": [
                r"\b(do it now|right now|immediately)\s+(or|before)",
                r"\b(last chance|final warning|now or never|act now)",
                r"\b(don't think|stop thinking|just do it)\s+(about it|and)",
            ],
            "keywords": [
                "act now", "last chance", "limited time", "don't think just do",
                "final offer", "now or never", "you must decide now"
            ],
            "severity": 0.65,
        },

        # Autonomy denial
        {
            "category": "autonomy_denial",
            "patterns": [
                r"\b(you (have to|must|need to)|no choice|can't say no)",
                r"\b(I (decide|control|own) you|you belong to me)",
                r"\b(you're not allowed|you can't|forbidden to)\s+(to|from)?\s*(leave|go|say|think)",
            ],
            "keywords": [
                "you have no choice", "you must obey", "do as I say",
                "you're not allowed", "I decide for you", "no choice but to"
            ],
            "severity": 0.85,
        },

        # Social isolation
        {
            "category": "isolation",
            "patterns": [
                r"\b(don't talk to|stay away from|can't see)\s+(your|other|those)\s+(friends|family|people)",
                r"\b(I'm the only one|no one else|they don't)\s+(who|that)\s+(cares|understands|loves)",
            ],
            "keywords": [
                "don't talk to them", "stay away from", "they're bad for you",
                "only I understand", "no one else cares", "cut ties with"
            ],
            "severity": 0.8,
        },

        # Information control
        {
            "category": "information_control",
            "patterns": [
                r"\b(don't (read|watch|listen|look)|stop reading|ignore)\s+(that|those|the)",
                r"\b(can't trust|don't believe)\s+(anyone|them|what they say|the news)",
            ],
            "keywords": [
                "don't read that", "stop watching", "don't believe them",
                "they're lying", "only trust me", "fake news"
            ],
            "severity": 0.7,
        },

        # Implicit coercion
        {
            "category": "implicit_coercion",
            "patterns": [
                r"\b(nice|shame|too bad)\s+(if|when)\s+something\s+(happened|were to happen)",
                r"\b(I'd hate for|it would be unfortunate if|wouldn't want)",
            ],
            "keywords": [
                "nice thing you have", "shame if something", "hate to see",
                "unfortunate if", "wouldn't want anything to"
            ],
            "severity": 0.85,
        },
    ]

    # Autonomy-respecting language (reduces concern)
    AUTONOMY_RESPECTING = [
        "your choice", "up to you", "you decide", "your decision",
        "feel free to", "if you want", "no pressure", "take your time",
        "consider your options", "whatever you prefer"
    ]

    def __init__(self):
        self._compile_patterns()

    def _compile_patterns(self):
        """Pre-compile coercion detection patterns."""
        self._compiled_categories = []

        for category_info in self.COERCION_PATTERNS:
            compiled_patterns = []
            for pattern in category_info["patterns"]:
                try:
                    compiled_patterns.append(re.compile(pattern, re.IGNORECASE))
                except re.error:
                    continue

            self._compiled_categories.append({
                "category": category_info["category"],
                "patterns": compiled_patterns,
                "keywords": category_info["keywords"],
                "severity": category_info["severity"],
            })

    async def detect(self, text: str, context: dict) -> DetectorSignal:
        """
        Detect coercive content in text.

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
                description="Empty text - no coercion check possible",
                confidence=0.0,
                metadata={}
            )

        findings = {
            "violations": [],
            "categories": set(),
            "max_severity": 0.0,
            "autonomy_respecting_count": 0,
        }

        text_lower = text.lower()

        # Check for autonomy-respecting language (mitigating factor)
        for phrase in self.AUTONOMY_RESPECTING:
            if phrase.lower() in text_lower:
                findings["autonomy_respecting_count"] += 1

        # Check coercion patterns
        for category_info in self._compiled_categories:
            category_found = False

            # Check regex patterns
            for pattern in category_info["patterns"]:
                matches = pattern.findall(text)
                if matches:
                    if not category_found:
                        findings["violations"].append({
                            "category": category_info["category"],
                            "severity": category_info["severity"],
                            "detection_method": "regex",
                            "matches": [str(m)[:100] for m in matches[:3]],
                        })
                        findings["categories"].add(category_info["category"])
                        findings["max_severity"] = max(
                            findings["max_severity"],
                            category_info["severity"]
                        )
                        category_found = True
                    break

            # Check keywords if no regex match
            if not category_found:
                for keyword in category_info["keywords"]:
                    if keyword.lower() in text_lower:
                        findings["violations"].append({
                            "category": category_info["category"],
                            "severity": category_info["severity"] * 0.9,
                            "detection_method": "keyword",
                            "keyword_matched": keyword,
                        })
                        findings["categories"].add(category_info["category"])
                        findings["max_severity"] = max(
                            findings["max_severity"],
                            category_info["severity"] * 0.9
                        )
                        break

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
                "categories": list(findings["categories"]),
                "max_pattern_severity": findings["max_severity"],
                "autonomy_respecting_phrases": findings["autonomy_respecting_count"],
            },
            mitigation=mitigation
        )

    def _calculate_risk_score(self, findings: Dict[str, Any]) -> float:
        """Calculate overall coercion risk score."""
        if not findings["violations"]:
            return 0.0

        # Base score from max severity
        score = findings["max_severity"]

        # Increase for multiple coercion types (coordinated manipulation)
        category_count = len(findings["categories"])
        if category_count > 1:
            score = min(1.0, score + 0.1 * (category_count - 1))

        # Critical categories get boost
        critical_categories = {"threat", "blackmail", "gaslighting", "autonomy_denial"}
        if findings["categories"] & critical_categories:
            score = min(1.0, score + 0.1)

        # Autonomy-respecting language reduces concern
        if findings["autonomy_respecting_count"] > 0:
            reduction = min(0.2, findings["autonomy_respecting_count"] * 0.05)
            score = max(0.0, score - reduction)

        return score

    def _determine_severity(self, risk_score: float, findings: Dict) -> str:
        """Determine severity level."""
        critical_categories = {"threat", "blackmail", "gaslighting", "autonomy_denial"}

        if risk_score >= 0.85 or (findings["categories"] & critical_categories and risk_score >= 0.6):
            return "S3"  # Critical
        elif risk_score >= 0.6:
            return "S2"  # Significant
        elif risk_score >= 0.4:
            return "S1"  # Minor
        return "S0"

    def _build_description(self, findings: Dict, risk_score: float) -> str:
        """Build human-readable description."""
        if not findings["violations"]:
            if findings["autonomy_respecting_count"] > 0:
                return "No coercive content detected. Text includes autonomy-respecting language."
            return "No coercive content detected."

        parts = []

        # Summarize coercion types
        if findings["categories"]:
            cats = ", ".join(findings["categories"])
            parts.append(f"Coercion types detected: {cats}")

        # Severity note
        if risk_score >= 0.8:
            parts.append("CRITICAL: Severe coercive content undermining user autonomy")
        elif risk_score >= 0.6:
            parts.append("WARNING: Significant manipulation or pressure detected")
        elif risk_score >= 0.4:
            parts.append("CAUTION: Potential coercive language")

        # Note mitigating factors
        if findings["autonomy_respecting_count"] > 0:
            parts.append(f"({findings['autonomy_respecting_count']} autonomy-respecting phrase(s) found)")

        return ". ".join(parts)

    def _suggest_mitigation(self, findings: Dict) -> str:
        """Suggest mitigation strategies."""
        suggestions = []

        if "threat" in findings["categories"] or "blackmail" in findings["categories"]:
            suggestions.append("Remove threatening or extortive language")

        if "emotional_manipulation" in findings["categories"]:
            suggestions.append("Avoid guilt-tripping and emotional pressure tactics")

        if "gaslighting" in findings["categories"]:
            suggestions.append("Respect user perceptions and don't dismiss their experiences")

        if "pressure" in findings["categories"]:
            suggestions.append("Give users time to make informed decisions")

        if "autonomy_denial" in findings["categories"]:
            suggestions.append("Respect user choice and autonomy")

        if "isolation" in findings["categories"]:
            suggestions.append("Don't discourage healthy relationships")

        if not suggestions:
            return "Use autonomy-respecting language and avoid pressure tactics"

        return ". ".join(suggestions)
