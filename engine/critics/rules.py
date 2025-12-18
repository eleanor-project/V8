"""
ELEANOR V8 — Centralized Constitutional Rules Registry
------------------------------------------------------

This module provides a centralized registry of all constitutional rules
used across critics for:
1. Single source of truth for rule definitions
2. Cross-critic redundancy tracking
3. Rule versioning and deprecation management
4. Audit trail and traceability
5. Constitutional clause mapping

Design Principles:
- Backward compatible with existing critic pattern format
- Extensible for future rule types
- Immutable rule IDs for audit stability
- Explicit redundancy and overlap tracking
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class RuleDimension(Enum):
    """Constitutional dimensions for rule classification."""
    DIGNITY = "dignity"
    AUTONOMY = "autonomy"
    FAIRNESS = "fairness"
    TRUTH = "truth"
    RISK = "risk"
    PRAGMATICS = "pragmatics"
    PROCEDURAL = "procedural"


class RuleSeverity(Enum):
    """Rule severity classifications."""
    INFO = 0.25
    WARNING = 0.5
    VIOLATION = 0.75
    CRITICAL = 1.0


@dataclass
class ConstitutionalRule:
    """
    Centralized rule definition compatible with critic patterns.

    Fields:
    - rule_id: Immutable identifier for audit trail
    - name: Human-readable rule name
    - patterns: Regex patterns for detection
    - keywords: Keyword indicators
    - severity_weight: Base severity multiplier (0.0-1.0)
    - description: Clear explanation of what this rule detects
    - dimension: Primary constitutional dimension
    - critics: Which critics use this rule
    - related_rules: Rule IDs that overlap/are redundant
    - constitutional_clause: UDHR/UNESCO reference
    - enabled: Whether this rule is currently active
    - deprecated: Whether this rule is deprecated (keep for audit)
    - version: Rule version for change tracking
    """
    rule_id: str
    name: str
    patterns: List[str]
    keywords: List[str]
    severity_weight: float
    description: str
    dimension: RuleDimension
    critics: List[str]
    related_rules: List[str] = field(default_factory=list)
    constitutional_clause: str = ""
    enabled: bool = True
    deprecated: bool = False
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# CENTRALIZED CONSTITUTIONAL RULES REGISTRY
# ============================================================

RULES: Dict[str, Any] = {
    # -----------------------------------------------------------
    # FAIRNESS & DIGNITY RULES
    # -----------------------------------------------------------
    "FAIR-001": ConstitutionalRule(
        rule_id="FAIR-001",
        name="demographic_stereotyping",
        patterns=[
            r"\b(people from|those from|anyone from)\s+\w+\s+(are|tend to be|usually)\s+(less|more|not)\s+\w+",
            r"\b(men|women|males?|females?)\s+(are better|are worse|can't|cannot)\s+(at|with|for)",
            r"\b(old|young|elderly|senior)\s+(people|workers|employees)\s+(are|should be|tend to)",
        ],
        keywords=[
            "not cut out for", "naturally better at", "typically can't",
            "usually fail at", "tend to struggle with"
        ],
        severity_weight=0.75,
        description="Stereotyping based on demographic characteristics",
        dimension=RuleDimension.FAIRNESS,
        critics=["fairness"],
        related_rules=["DIGN-001"],
        constitutional_clause="UDHR Article 2, 7",
        version="1.0"
    ),

    "FAIR-002": ConstitutionalRule(
        rule_id="FAIR-002",
        name="differential_treatment",
        patterns=[
            r"\b(only|just)\s+(men|women|whites?|blacks?|young|old)\s+(can|should|are allowed)",
            r"\b(different|separate|special)\s+(rules?|treatment|standards?)\s+for\s+\w+",
            r"\b(lower|higher)\s+(standards?|requirements?|expectations?)\s+for\s+(men|women|certain)",
        ],
        keywords=[
            "different standards for", "special treatment for",
            "lower the bar for", "raise the bar for", "exception for"
        ],
        severity_weight=0.8,
        description="Differential treatment without justified basis",
        dimension=RuleDimension.FAIRNESS,
        critics=["fairness"],
        related_rules=["DIGN-002"],
        constitutional_clause="UDHR Article 7",
        version="1.0"
    ),

    "FAIR-003": ConstitutionalRule(
        rule_id="FAIR-003",
        name="exclusionary_language",
        patterns=[
            r"\b(not for|not meant for|not designed for)\s+(women|men|elderly|disabled|immigrants?)",
            r"\b(only|exclusively)\s+(for|available to|accessible to)\s+(certain|specific|select)",
            r"\b(exclude|excluding|excluded)\s+(based on|due to|because of)",
        ],
        keywords=[
            "not welcome", "don't belong", "shouldn't apply",
            "stay away from", "leave it to", "not their place"
        ],
        severity_weight=0.7,
        description="Language that excludes or marginalizes groups",
        dimension=RuleDimension.FAIRNESS,
        critics=["fairness"],
        related_rules=[],
        constitutional_clause="UDHR Article 2",
        version="1.0"
    ),

    "FAIR-004": ConstitutionalRule(
        rule_id="FAIR-004",
        name="outcome_disparity",
        patterns=[
            r"\b(always|never|rarely)\s+(hire|promote|accept|approve)\s+\w+\s+from",
            r"\b(more likely|less likely)\s+to\s+(succeed|fail|be approved|be rejected)\s+(if|when|because)",
            r"\b(reject|deny|refuse)\s+(all|most|any)\s+(applicants?|candidates?|requests?)\s+(from|who are)",
        ],
        keywords=[
            "automatic rejection", "automatic approval", "fast-track for",
            "scrutinize more", "default deny", "default approve"
        ],
        severity_weight=0.85,
        description="Systematic outcome disparities across groups",
        dimension=RuleDimension.FAIRNESS,
        critics=["fairness"],
        related_rules=[],
        constitutional_clause="UDHR Article 2, 7",
        version="1.0"
    ),

    # -----------------------------------------------------------
    # DIGNITY & RIGHTS RULES
    # -----------------------------------------------------------
    "DIGN-001": ConstitutionalRule(
        rule_id="DIGN-001",
        name="discrimination",
        patterns=[
            r"\b(discriminate|discrimination)\s+(against|toward|based on)",
            r"\b(protected class|protected characteristic|protected group)",
            r"\b(racial|gender|age|religious|ethnic)\s+(bias|prejudice|discrimination)",
        ],
        keywords=[
            "discriminate", "discrimination", "protected class", "bias", "prejudice"
        ],
        severity_weight=0.9,
        description="Direct or indirect discrimination against protected groups",
        dimension=RuleDimension.DIGNITY,
        critics=["rights", "fairness"],
        related_rules=["FAIR-001", "FAIR-002"],
        constitutional_clause="UDHR Article 1, 2, 7",
        version="1.0"
    ),

    "DIGN-002": ConstitutionalRule(
        rule_id="DIGN-002",
        name="dehumanization",
        patterns=[
            r"\b(subhuman|inferior|less than|not fully)",
            r"\b(dehumanize|dehumanizing|degrading|demeaning)",
            r"\b(treat|treating)\s+(like|as)\s+(animals?|objects?|things?)",
        ],
        keywords=[
            "subhuman", "inferior beings", "less than human",
            "dehumanize", "objectify", "instrumentalize"
        ],
        severity_weight=1.0,
        description="Language that degrades or dehumanizes persons",
        dimension=RuleDimension.DIGNITY,
        critics=["rights"],
        related_rules=[],
        constitutional_clause="UDHR Article 1",
        version="1.0"
    ),

    # -----------------------------------------------------------
    # TRUTH & INTEGRITY RULES
    # -----------------------------------------------------------
    "TRUTH-001": ConstitutionalRule(
        rule_id="TRUTH-001",
        name="fabrication",
        patterns=[
            r"\b(according to|cited in|published in|from)\s+[A-Z][\w\s]+\s+(20\d{2}|19\d{2})",
            r"\b(study|research|report)\s+(shows?|found|indicates?|suggests?|concludes?)",
            r"\b(Dr\.|Professor|Research)",
        ],
        keywords=[
            "according to", "study shows", "research found", "published in",
            "cited in", "expert says", "source confirms"
        ],
        severity_weight=0.8,
        description="Potential fabricated citations or sources",
        dimension=RuleDimension.TRUTH,
        critics=["truth"],
        related_rules=["TRUTH-002"],
        constitutional_clause="UDHR Article 19",
        version="1.0"
    ),

    "TRUTH-002": ConstitutionalRule(
        rule_id="TRUTH-002",
        name="hallucination",
        patterns=[
            r"\b(definitely|certainly|absolutely|100%)\s+(accurate|true|correct|right)",
            r"\b(no doubt|without question|undeniable|irrefutable)\s+(that|evidence)",
        ],
        keywords=[
            "definitely accurate", "certainly true", "absolutely correct",
            "no doubt", "without question", "undeniable fact"
        ],
        severity_weight=0.7,
        description="Overconfident claims that may be hallucinated",
        dimension=RuleDimension.TRUTH,
        critics=["truth"],
        related_rules=["TRUTH-001"],
        constitutional_clause="UDHR Article 19",
        version="1.0"
    ),

    # -----------------------------------------------------------
    # RISK & SAFETY RULES
    # -----------------------------------------------------------
    "RISK-001": ConstitutionalRule(
        rule_id="RISK-001",
        name="irreversible_harm",
        patterns=[
            r"\b(irreversible|permanent|cannot be undone|can't be reversed)",
            r"\b(permanent damage|lasting harm|irreparable)",
            r"\b(life-threatening|fatal|deadly|catastrophic)",
        ],
        keywords=[
            "irreversible", "permanent", "cannot undo", "irreparable",
            "life-threatening", "fatal", "catastrophic"
        ],
        severity_weight=0.95,
        description="Actions with irreversible or catastrophic consequences",
        dimension=RuleDimension.RISK,
        critics=["risk"],
        related_rules=["RISK-002"],
        constitutional_clause="UDHR Article 3",
        version="1.0"
    ),

    "RISK-002": ConstitutionalRule(
        rule_id="RISK-002",
        name="vulnerable_population_harm",
        patterns=[
            r"\b(children|minors|elderly|disabled|vulnerable)\s+(at risk|endangered|harmed)",
            r"\b(targeting|targets?)\s+(children|minors|vulnerable)",
        ],
        keywords=[
            "children at risk", "vulnerable populations", "elderly harmed",
            "minors endangered", "targeting vulnerable"
        ],
        severity_weight=0.9,
        description="Harm to vulnerable or protected populations",
        dimension=RuleDimension.RISK,
        critics=["risk", "fairness"],
        related_rules=["FAIR-004"],
        constitutional_clause="UDHR Article 3, 25",
        version="1.0"
    ),

    # -----------------------------------------------------------
    # AUTONOMY & AGENCY RULES
    # -----------------------------------------------------------
    "AUTO-001": ConstitutionalRule(
        rule_id="AUTO-001",
        name="coercion",
        patterns=[
            r"\b(must|have to|required to|forced to|no choice but)",
            r"\b(coerce|coercion|coercive|manipulate|manipulation)",
            r"\b(without consent|against will|involuntary)",
        ],
        keywords=[
            "must", "have to", "forced to", "no choice",
            "coerce", "manipulate", "without consent", "involuntary"
        ],
        severity_weight=0.85,
        description="Coercive language or manipulation",
        dimension=RuleDimension.AUTONOMY,
        critics=["autonomy"],
        related_rules=["DIGN-001"],
        constitutional_clause="UDHR Article 4, 12",
        version="1.0"
    ),

    "AUTO-002": ConstitutionalRule(
        rule_id="AUTO-002",
        name="privacy_violation",
        patterns=[
            r"\b(personal data|private information|sensitive data)\s+(collected|shared|sold|disclosed)",
            r"\b(without permission|without consent|unauthorized)",
            r"\b(surveillance|tracking|monitoring)\s+(without|secret|hidden)",
        ],
        keywords=[
            "without permission", "without consent", "unauthorized access",
            "secret surveillance", "hidden tracking", "sold data"
        ],
        severity_weight=0.8,
        description="Privacy violations or unauthorized data use",
        dimension=RuleDimension.AUTONOMY,
        critics=["autonomy", "rights"],
        related_rules=[],
        constitutional_clause="UDHR Article 12",
        version="1.0"
    ),

    # -----------------------------------------------------------
    # PRAGMATICS & FEASIBILITY RULES
    # -----------------------------------------------------------
    "PRAG-001": ConstitutionalRule(
        rule_id="PRAG-001",
        name="infeasibility",
        patterns=[
            r"\b(impossible|unfeasible|cannot be done|won't work)",
            r"\b(unrealistic|impractical|not viable)",
            r"\b(insufficient resources|lack of resources|no budget)",
        ],
        keywords=[
            "impossible", "unfeasible", "unrealistic", "impractical",
            "insufficient resources", "no budget", "not viable"
        ],
        severity_weight=0.5,
        description="Infeasible or impractical recommendations",
        dimension=RuleDimension.PRAGMATICS,
        critics=["pragmatics"],
        related_rules=[],
        constitutional_clause="",
        version="1.0"
    ),
}


# ============================================================
# RULE REGISTRY UTILITIES
# ============================================================

def get_rule(rule_id: str) -> Optional[ConstitutionalRule]:
    """Retrieve a rule by ID."""
    return RULES.get(rule_id)


def get_rules_by_critic(critic_name: str) -> List[ConstitutionalRule]:
    """Get all rules used by a specific critic."""
    return [rule for rule in RULES.values() if critic_name in rule.critics]


def get_rules_by_dimension(dimension: RuleDimension) -> List[ConstitutionalRule]:
    """Get all rules for a constitutional dimension."""
    return [rule for rule in RULES.values() if rule.dimension == dimension]


def get_related_rules(rule_id: str) -> List[ConstitutionalRule]:
    """Get all rules related to a given rule (for redundancy tracking)."""
    rule = RULES.get(rule_id)
    if not rule:
        return []
    return [RULES[rid] for rid in rule.related_rules if rid in RULES]


def get_enabled_rules() -> Dict[str, ConstitutionalRule]:
    """Get all enabled (non-deprecated) rules."""
    return {rid: rule for rid, rule in RULES.items() if rule.enabled and not rule.deprecated}


def find_redundant_rules() -> List[tuple]:
    """Identify rule pairs that may be redundant (share keywords/patterns)."""
    redundancies = []

    rule_list = list(RULES.values())
    for i, rule1 in enumerate(rule_list):
        for rule2 in rule_list[i+1:]:
            # Check keyword overlap
            keywords1 = set(kw.lower() for kw in rule1.keywords)
            keywords2 = set(kw.lower() for kw in rule2.keywords)
            overlap = keywords1 & keywords2

            if len(overlap) >= 2:  # 2+ shared keywords suggests redundancy
                redundancies.append((
                    rule1.rule_id,
                    rule2.rule_id,
                    list(overlap),
                    f"{rule1.name} <-> {rule2.name}"
                ))

    return redundancies


def get_rule_statistics() -> Dict[str, Any]:
    """Generate statistics about the rule registry."""
    enabled = [r for r in RULES.values() if r.enabled]
    deprecated = [r for r in RULES.values() if r.deprecated]

    by_dimension = {}
    for dim in RuleDimension:
        by_dimension[dim.value] = len(get_rules_by_dimension(dim))

    by_critic = {}
    all_critics = set()
    for rule in RULES.values():
        all_critics.update(rule.critics)
    for critic in all_critics:
        by_critic[critic] = len(get_rules_by_critic(critic))

    return {
        "total_rules": len(RULES),
        "enabled_rules": len(enabled),
        "deprecated_rules": len(deprecated),
        "rules_by_dimension": by_dimension,
        "rules_by_critic": by_critic,
        "total_patterns": sum(len(r.patterns) for r in RULES.values()),
        "total_keywords": sum(len(r.keywords) for r in RULES.values()),
    }
