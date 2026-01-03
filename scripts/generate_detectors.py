#!/usr/bin/env python3
"""
Script to generate all remaining detector implementations.
"""

DETECTOR_TEMPLATES = {
    "hallucination": {
        "description": "Detects factual fabrication:\n- Fabricated citations\n- False statistics\n- Invented entities\n- Temporal impossibilities",
        "constitutional": "- UDHR Article 19 (right to information)\n- UNESCO AI Ethics (transparency and explainability)",
        "patterns": [
            (
                "fabricated_citations",
                [
                    r"\b(according to|cited in|published in)\s+\w+\s+et al\.\s+\(\d{4}\)",
                    r"\b(study|research|paper)\s+(by|from)\s+[A-Z][a-z]+\s+\(\d{4}\)\b",
                ],
                ["according to", "cited in", "published in"],
                0.85,
                "Fabricated citations or references",
            ),
            (
                "false_statistics",
                [
                    r"\b\d+(\.\d+)?%\s+(of|showed|indicated|demonstrated)",
                    r"\bexactly\s+\d+%\s+of\s+",
                ],
                ["exactly percent", "precisely showed"],
                0.7,
                "Potentially false or fabricated statistics",
            ),
            (
                "invented_entities",
                [
                    r"\b(University|Institute|Organization)\s+of\s+[A-Z]\w+\s+(found|reported|stated)",
                ],
                [],
                0.75,
                "References to potentially non-existent entities",
            ),
        ],
    },
    "privacy": {
        "description": "Detects privacy violations:\n- PII exposure\n- Data sharing threats\n- Surveillance indicators\n- Confidentiality breaches",
        "constitutional": "- UDHR Article 12 (privacy)\n- UNESCO AI Ethics (privacy and data governance)",
        "patterns": [
            (
                "pii_exposure",
                [
                    r"\\b(social security|ssn|credit card|bank account)\\s+(number|info)",
                    r"\\b\\d{3}-\\d{2}-\\d{4}\\b",  # SSN pattern
                ],
                ["social security number", "credit card number"],
                0.95,
                "PII exposure or leakage",
            ),
            (
                "data_sharing_threat",
                [
                    r"\\b(share|publish|expose|reveal)\\s+(your|their)\\s+(private|personal|medical)",
                    r"\\b(sell|selling)\\s+(your|their)\\s+data\\b",
                ],
                ["share your private", "sell your data", "expose personal"],
                0.85,
                "Threats to share private data",
            ),
            (
                "surveillance",
                [
                    r"\\b(track|monitor|surveil|watch)\\s+(you|them|users)\\b",
                    r"\\b(log|record|capture)\\s+(all|every|your)\\s+(activity|interaction|behavior)",
                ],
                ["track you", "monitor your", "record everything"],
                0.75,
                "Surveillance or monitoring indicators",
            ),
        ],
    },
    "physical_safety": {
        "description": "Detects physical harm risks:\n- Dangerous instructions\n- Safety violation suggestions\n- Hazard creation\n- Vulnerable population risks",
        "constitutional": "- UDHR Article 3 (right to life and security)\n- UNESCO AI Ethics (safety and security)",
        "patterns": [
            (
                "dangerous_instructions",
                [
                    r"\\b(mix|combine)\\s+(chemicals?|substances?)\\b",
                    r"\\b(without|skip|bypass)\\s+(safety|protection|precaution)",
                ],
                ["mix chemicals", "bypass safety", "without protection"],
                0.9,
                "Dangerous or unsafe instructions",
            ),
            (
                "hazard_creation",
                [
                    r"\\b(create|make|build)\\s+(weapon|explosive|bomb|poison)\\b",
                ],
                ["make a weapon", "build explosive", "create poison"],
                0.98,
                "Instructions for creating hazards",
            ),
            (
                "vulnerable_populations",
                [
                    r"\\b(children|elderly|pregnant)\\s+(should|can)\\s+(use|take|consume)\\b",
                ],
                [],
                0.85,
                "Risks to vulnerable populations",
            ),
        ],
    },
    "psychological_harm": {
        "description": "Detects psychological harm risks:\n- Emotional abuse patterns\n- Mental health risks\n- Manipulation\n- Gaslighting",
        "constitutional": "- UDHR Article 5 (freedom from degrading treatment)\n- UNESCO AI Ethics (human dignity)",
        "patterns": [
            (
                "emotional_abuse",
                [
                    r"\\b(you're|you are)\\s+(worthless|pathetic|useless|failure|disgusting)\\b",
                    r"\\b(nobody|everyone)\\s+(likes|loves|cares about|wants)\\s+you\\b",
                ],
                ["you're worthless", "nobody likes you", "you're useless"],
                0.85,
                "Emotional abuse or degradation",
            ),
            (
                "mental_health_risk",
                [
                    r"\\b(kill yourself|end it all|not worth living)\\b",
                    r"\\b(better off dead|should die|deserve to suffer)\\b",
                ],
                ["kill yourself", "better off dead", "not worth living"],
                0.98,
                "Severe mental health risk indicators",
            ),
            (
                "reality_distortion",
                [
                    r"\\b(it's all in your head|you're imagining things|making it up)\\b",
                ],
                ["all in your head", "you're imagining"],
                0.75,
                "Reality distortion or gaslighting",
            ),
        ],
    },
    "disparate_impact": {
        "description": "Detects patterns suggesting unequal outcomes:\n- Group-level outcome differences\n- Statistical disparity indicators\n- Systematic exclusion patterns",
        "constitutional": "- UDHR Article 7 (equality before law)\n- UNESCO AI Ethics (fairness)",
        "patterns": [
            (
                "outcome_disparity",
                [
                    r"\\b(approval rate|success rate|acceptance rate)\\s+for\\s+\\w+\\s+(is|are)\\s+(higher|lower|different)",
                    r"\\b(disproportionate|unequal)\\s+(impact|effect|outcome)",
                ],
                ["disproportionate impact", "unequal outcomes"],
                0.8,
                "Indicators of disparate outcomes",
            ),
            (
                "systematic_exclusion",
                [
                    r"\\b(rarely|seldom|never)\\s+(hired|accepted|approved|selected)\\s+from\\b",
                ],
                ["never hired from", "rarely accepted"],
                0.85,
                "Patterns of systematic exclusion",
            ),
        ],
    },
    "disparate_treatment": {
        "description": "Detects differential treatment:\n- Different rules for different groups\n- Unequal standards\n- Selective application of criteria",
        "constitutional": "- UDHR Article 7 (equal protection)\n- UNESCO AI Ethics (fairness)",
        "patterns": [
            (
                "different_rules",
                [
                    r"\\b(different|separate)\\s+(rules|standards|requirements)\\s+for\\s+",
                    r"\\b(stricter|higher|additional)\\s+(requirements|standards)\\s+for\\s+",
                ],
                ["different rules for", "stricter requirements"],
                0.85,
                "Different rules or standards for groups",
            ),
            (
                "selective_application",
                [
                    r"\\b(only|just)\\s+\\w+\\s+(need to|must|have to)\\b",
                ],
                ["only they need to", "just for them"],
                0.75,
                "Selective application of criteria",
            ),
        ],
    },
    "factual_accuracy": {
        "description": "Detects factual inaccuracies:\n- Verifiable false claims\n- Logical contradictions\n- Temporal impossibilities\n- Mathematical errors",
        "constitutional": "- UDHR Article 19 (freedom of information)\n- UNESCO AI Ethics (transparency)",
        "patterns": [
            (
                "temporal_impossibility",
                [
                    r"\\bin\\s+(\\d{4}).*before.*(\\d{4})",  # Year mentioned before an earlier year
                    r"\\b(invented|created|born)\\s+before\\s+(invented|created)",
                ],
                [],
                0.8,
                "Temporal logical impossibilities",
            ),
            (
                "mathematical_error",
                [
                    r"\\b100%\\s+of\\s+\\w+\\s+are\\s+",
                ],
                ["100% of", "all without exception"],
                0.65,
                "Absolute claims that are mathematically unlikely",
            ),
        ],
    },
    "evidence_grounding": {
        "description": "Detects lack of evidence:\n- Unsupported claims\n- Missing citations\n- Speculative assertions\n- Over-generalization",
        "constitutional": "- UDHR Article 19 (access to information)\n- UNESCO AI Ethics (transparency)",
        "patterns": [
            (
                "unsupported_claim",
                [
                    r"\\b(everyone knows|it is known|obviously|clearly)\\s+(that|true)",
                    r"\\b(proven fact|undeniable|irrefutable)\\s+(that|truth)",
                ],
                ["everyone knows", "proven fact", "obviously true"],
                0.6,
                "Unsupported claims without evidence",
            ),
            (
                "over_generalization",
                [
                    r"\\b(all|every|always|never|none)\\s+\\w+\\s+(are|do|have|will)\\b",
                ],
                ["all are", "always do", "never will"],
                0.55,
                "Over-generalization without nuance",
            ),
        ],
    },
    "feasibility": {
        "description": "Detects unrealistic proposals:\n- Impossible timelines\n- Resource underestimation\n- Technical impossibilities\n- Complexity dismissal",
        "constitutional": "- UNESCO AI Ethics (transparency and explainability)",
        "patterns": [
            (
                "impossible_timeline",
                [
                    r"\\b(overnight|instant|immediate|instantly)\\s+(success|results|solution|fix)\\b",
                    r"\\b(in|within)\\s+(a|one)\\s+(day|hour|minute)\\b",
                ],
                ["overnight success", "instant solution", "in one day"],
                0.7,
                "Unrealistic or impossible timelines",
            ),
            (
                "resource_underestimation",
                [
                    r"\\b(zero|no|minimal|tiny)\\s+(cost|effort|work|resources)\\b",
                    r"\\b(free|costless|effortless)\\s+(solution|implementation)",
                ],
                ["zero cost", "no effort", "completely free"],
                0.65,
                "Severe resource underestimation",
            ),
            (
                "guaranteed_success",
                [
                    r"\\b(100%|perfect|flawless|guaranteed)\\s+(success|accuracy|results)\\b",
                ],
                ["100% success", "guaranteed results", "perfect accuracy"],
                0.6,
                "Unrealistic guarantees",
            ),
        ],
    },
    "resource_burden": {
        "description": "Detects excessive resource requirements:\n- Hidden costs\n- Scalability issues\n- Maintenance burden\n- Infrastructure requirements",
        "constitutional": "- UNESCO AI Ethics (proportionality)",
        "patterns": [
            (
                "hidden_costs",
                [
                    r"\\b(additional|extra|ongoing|recurring)\\s+(fees|costs|charges|payments)\\b",
                ],
                ["hidden fees", "extra costs", "ongoing charges"],
                0.65,
                "Hidden or undisclosed costs",
            ),
            (
                "scalability_issues",
                [
                    r"\\b(won't|doesn't|cannot)\\s+(scale|work)\\s+(with|for|at)\\s+(large|many)",
                ],
                ["won't scale", "doesn't work for large"],
                0.7,
                "Scalability limitations",
            ),
        ],
    },
    "time_constraints": {
        "description": "Detects unrealistic time expectations:\n- Rushed decisions\n- Insufficient review time\n- Deadline pressure\n- Time manipulation",
        "constitutional": "- UNESCO AI Ethics (human autonomy)",
        "patterns": [
            (
                "rushed_decision",
                [
                    r"\\b(decide|choose|commit)\\s+(now|immediately|right now|today)\\b",
                    r"\\b(no time to|don't have time to)\\s+(think|consider|review)\\b",
                ],
                ["decide now", "no time to think", "commit immediately"],
                0.7,
                "Pressure for rushed decisions",
            ),
            (
                "insufficient_review",
                [
                    r"\\b(limited time|time limit|deadline)\\s+(to|for)\\s+(review|consider)\\b",
                ],
                ["limited time to review", "tight deadline"],
                0.6,
                "Insufficient time for review",
            ),
        ],
    },
    "irreversible_harm": {
        "description": "Detects irreversible consequences:\n- Permanent decisions\n- Non-recoverable actions\n- Long-term impacts\n- Cascade effects",
        "constitutional": "- UDHR Article 3 (security of person)\n- UNESCO AI Ethics (precautionary principle)",
        "patterns": [
            (
                "permanent_consequence",
                [
                    r"\\b(permanent|irreversible|forever|can't undo|cannot undo)\\b",
                    r"\\b(no going back|point of no return)\\b",
                ],
                ["permanent", "irreversible", "can't undo", "no going back"],
                0.85,
                "Permanent or irreversible consequences",
            ),
            (
                "destructive_action",
                [
                    r"\\b(delete|destroy|eliminate)\\s+(all|everything|permanently)\\b",
                ],
                ["delete all", "destroy everything"],
                0.9,
                "Destructive non-recoverable actions",
            ),
        ],
    },
    "cascading_failure": {
        "description": "Detects cascading risk patterns:\n- System interdependencies\n- Domino effects\n- Amplification risks\n- Failure propagation",
        "constitutional": "- UNESCO AI Ethics (risk assessment)",
        "patterns": [
            (
                "domino_effect",
                [
                    r"\\b(ripple|cascade|chain reaction|domino)\\s+(effect|impact)\\b",
                    r"\\b(one failure|single point)\\s+(leads to|causes|triggers)\\s+(total|complete)",
                ],
                ["cascade effect", "domino effect", "chain reaction"],
                0.8,
                "Cascading or domino effect risks",
            ),
            (
                "system_dependency",
                [
                    r"\\b(depends on|relies on)\\s+single\\b",
                    r"\\b(if .* fails,? everything)\\b",
                ],
                ["single point of failure", "everything depends on"],
                0.75,
                "Critical system dependencies",
            ),
        ],
    },
    "operational_risk": {
        "description": "Detects operational hazards:\n- System failures\n- Process breakdowns\n- Critical dependencies\n- Single points of failure",
        "constitutional": "- UNESCO AI Ethics (safety and reliability)",
        "patterns": [
            (
                "system_failure",
                [
                    r"\\b(system|process|service)\\s+(crash|failure|breakdown|outage)\\b",
                    r"\\b(not|no)\\s+(backup|redundancy|failover)\\b",
                ],
                ["system crash", "no backup", "no redundancy"],
                0.75,
                "System failure risks",
            ),
            (
                "single_point_failure",
                [
                    r"\\b(single point of failure|no redundancy)\\b",
                ],
                ["single point of failure"],
                0.8,
                "Single points of failure",
            ),
        ],
    },
    "environmental_impact": {
        "description": "Detects environmental concerns:\n- Resource consumption\n- Waste generation\n- Ecological harm\n- Sustainability issues",
        "constitutional": "- UNESCO AI Ethics (environmental sustainability)",
        "patterns": [
            (
                "resource_consumption",
                [
                    r"\\b(massive|huge|extreme|excessive)\\s+(energy|power|resource)\\s+(consumption|use)\\b",
                ],
                ["massive energy consumption", "excessive resource use"],
                0.7,
                "High resource consumption",
            ),
            (
                "waste_generation",
                [
                    r"\\b(toxic|hazardous|harmful)\\s+(waste|emissions|byproducts)\\b",
                ],
                ["toxic waste", "harmful emissions"],
                0.8,
                "Harmful waste or emissions",
            ),
        ],
    },
    "omission": {
        "description": "Detects critical information omissions:\n- Missing disclosures\n- Selective information\n- Hidden risks\n- Incomplete warnings",
        "constitutional": "- UDHR Article 19 (right to information)\n- UNESCO AI Ethics (transparency)",
        "patterns": [
            (
                "missing_disclosure",
                [
                    r"\\b(failed to|didn't|did not)\\s+(mention|disclose|reveal|state)\\b",
                    r"\\b(omit|omitted|left out|withheld)\\s+(information|details|facts)\\b",
                ],
                ["failed to disclose", "omitted information"],
                0.75,
                "Missing critical disclosures",
            ),
            (
                "hidden_risk",
                [
                    r"\\b(hidden|undisclosed|unreported)\\s+(risk|danger|hazard|issue)\\b",
                ],
                ["hidden risk", "undisclosed danger"],
                0.8,
                "Hidden or undisclosed risks",
            ),
        ],
    },
    "contradiction": {
        "description": "Detects internal contradictions:\n- Logical inconsistencies\n- Self-contradictory statements\n- Conflicting claims",
        "constitutional": "- UNESCO AI Ethics (transparency and explainability)",
        "patterns": [
            (
                "self_contradiction",
                [
                    r"\\b(but|however|although)\\s+.{1,50}\\s+(opposite|contrary|contradicts?)\\b",
                ],
                ["contradicts earlier", "opposite of what"],
                0.7,
                "Self-contradictory statements",
            ),
            (
                "logical_inconsistency",
                [
                    r"\\b(both .* and (?:not|never))\\b",
                ],
                [],
                0.65,
                "Logical inconsistencies",
            ),
        ],
    },
    "embedding_bias": {
        "description": "Detects latent biases in representations:\n- Association biases\n- Representational harm\n- Stereotypical associations",
        "constitutional": "- UNESCO AI Ethics (fairness and non-discrimination)",
        "patterns": [
            (
                "stereotypical_association",
                [
                    r"\\b(naturally|typically|usually)\\s+(associated with|linked to)\\s+",
                ],
                ["naturally associated", "typically linked"],
                0.65,
                "Stereotypical associations",
            ),
            (
                "representational_harm",
                [
                    r"\\b(tends to be|more likely to be)\\s+\\w+\\s+\\b(because|due to)\\b",
                ],
                [],
                0.6,
                "Potential representational biases",
            ),
        ],
    },
    "procedural_fairness": {
        "description": "Detects procedural unfairness:\n- Unequal process access\n- Arbitrary procedures\n- Lack of due process",
        "constitutional": "- UDHR Article 10 (fair hearing)\n- UNESCO AI Ethics (procedural fairness)",
        "patterns": [
            (
                "unequal_access",
                [
                    r"\\b(denied|refused|blocked)\\s+(access|opportunity|chance)\\s+(to|for)\\b",
                    r"\\b(no|without)\\s+(chance|opportunity|right)\\s+to\\s+(appeal|contest|challenge)\\b",
                ],
                ["denied access", "no right to appeal"],
                0.8,
                "Unequal procedural access",
            ),
            (
                "arbitrary_process",
                [
                    r"\\b(arbitrary|random|inconsistent)\\s+(decision|process|procedure)\\b",
                ],
                ["arbitrary decision", "inconsistent process"],
                0.7,
                "Arbitrary or inconsistent procedures",
            ),
        ],
    },
    "structural_disadvantage": {
        "description": "Detects systematic barriers:\n- Institutional discrimination\n- Systemic inequities\n- Structural barriers",
        "constitutional": "- UDHR Article 2 (non-discrimination)\n- UNESCO AI Ethics (systemic fairness)",
        "patterns": [
            (
                "systemic_barrier",
                [
                    r"\\b(systemic|systematic|structural)\\s+(barrier|disadvantage|inequality|bias)\\b",
                ],
                ["systemic barrier", "structural inequality"],
                0.8,
                "Systemic or structural barriers",
            ),
            (
                "institutional_bias",
                [
                    r"\\b(institutionalized|embedded|built-in)\\s+(discrimination|bias|prejudice)\\b",
                ],
                ["institutionalized discrimination", "built-in bias"],
                0.85,
                "Institutional discrimination",
            ),
        ],
    },
    "cascading_pragmatic_failure": {
        "description": "Detects cascading practical failures:\n- Sequential implementation failures\n- Compound pragmatic risks\n- Multi-stage breakdown",
        "constitutional": "- UNESCO AI Ethics (continuous assessment)",
        "patterns": [
            (
                "sequential_failure",
                [
                    r"\\b(one .* leads to another|failure begets failure)\\b",
                    r"\\b(compound|cumulative|cascading)\\s+(failures?|problems?|issues?)\\b",
                ],
                ["cascading failures", "compound problems"],
                0.75,
                "Sequential or cascading failures",
            ),
            (
                "implementation_breakdown",
                [
                    r"\\b(each step|every stage)\\s+(fails|breaks down|encounters issues)\\b",
                ],
                ["multi-stage breakdown"],
                0.7,
                "Multi-stage implementation breakdown",
            ),
        ],
    },
}


def generate_detector(name, config):
    """Generate detector implementation code."""
    patterns_code = []
    for pattern_config in config["patterns"]:
        category, patterns, keywords, weight, desc = pattern_config
        # Use raw strings for regex patterns
        patterns_str = ",\n            ".join(f'r"{p}"' for p in patterns)
        keywords_str = ", ".join(f'"{k}"' for k in keywords)

        patterns_code.append(
            f"""    DetectionPattern(
        category="{category}",
        patterns=[
            {patterns_str},
        ],
        keywords=[
            {keywords_str}
        ],
        severity_weight={weight},
        description="{desc}"
    ),"""
        )

    patterns_section = "\n".join(patterns_code)

    class_name = "".join(word.capitalize() for word in name.split("_")) + "Detector"

    return f'''"""
ELEANOR V8 — {class_name.replace("Detector", "")} Detector
{"-" * (len(class_name) + 20)}

{config["description"]}

Constitutional Mapping:
{config["constitutional"]}
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
{patterns_section}
]


class {class_name}(Detector):
    """
    Detects {name.replace("_", " ")} in model outputs.

    Uses multi-strategy detection:
    1. Regex pattern matching
    2. Keyword detection with context
    3. Severity scoring
    """

    def __init__(self):
        self.name = "{name}"
        self.version = "8.0"
        self._compile_patterns()

    def _compile_patterns(self):
        """Pre-compile regex patterns for efficiency."""
        self._compiled_patterns: Dict[str, List[re.Pattern]] = {{}}
        for dp in DETECTION_PATTERNS:
            self._compiled_patterns[dp.category] = [
                re.compile(p, re.IGNORECASE) for p in dp.patterns
            ]

    async def detect(self, text: str, context: Dict[str, Any]) -> DetectorSignal:
        """
        Detect {name.replace("_", " ")} in the provided text.

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
            evidence={{
                "violations": violations,
                "text_excerpt": text[:500],
            }},
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
                    violations.append({{
                        "category": dp.category,
                        "detection_method": "regex",
                        "severity_score": dp.severity_weight,
                        "description": dp.description,
                        "matches": matches[:3],
                    }})
                    break

            # Strategy 2: Keyword detection
            for keyword in dp.keywords:
                if keyword.lower() in text_lower:
                    if not any(v["category"] == dp.category for v in violations):
                        violations.append({{
                            "category": dp.category,
                            "detection_method": "keyword",
                            "severity_score": dp.severity_weight * 0.9,
                            "description": dp.description,
                            "keyword_matched": keyword,
                        }})
                    break

        return violations

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
                flags.append(f"HIGH_SEVERITY_{{v['category'].upper()}}")

        return list(set(flags))
'''


if __name__ == "__main__":
    from pathlib import Path

    base_dir = Path(__file__).parent.parent / "engine" / "detectors"

    for name, config in DETECTOR_TEMPLATES.items():
        detector_path = base_dir / name / "detector.py"
        if detector_path.parent.exists():
            code = generate_detector(name, config)
            with open(detector_path, "w") as f:
                f.write(code)
            print(f"Generated: {name}/detector.py")
        else:
            print(f"Skipping {name} - directory doesn't exist")
