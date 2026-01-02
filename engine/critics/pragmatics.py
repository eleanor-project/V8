"""
ELEANOR V8 — Pragmatics Critic
-------------------------------

Examines whether model output is practically feasible, respecting real-world
constraints including cost, time, technical viability, and operational limitations.

Detection Strategies:
1. Resource requirement analysis (cost, compute, human effort)
2. Time constraint assessment
3. Technical viability evaluation
4. Dependency and prerequisite identification
5. Scalability and sustainability analysis
6. Operational complexity assessment

Constitutional Mapping:
- UNESCO AI Ethics Recommendation (sustainability and feasibility)
- Ensures aligned decisions remain implementable at scale
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple, cast
from dataclasses import dataclass, field
from collections import defaultdict

from .base import BaseCriticV8
from engine.schemas.pipeline_types import CriticResult


@dataclass
class PragmaticsPattern:
    """Configuration for pragmatics detection patterns."""
    category: str
    patterns: List[str]  # Regex patterns
    keywords: List[str]  # Keyword indicators
    severity_weight: float
    description: str
    pragmatics_dimension: str  # cost, time, technical, operational, scalability


# Comprehensive pragmatics detection patterns
PRAGMATICS_PATTERNS = [
    PragmaticsPattern(
        category="unrealistic_timeline",
        patterns=[
            r"\b(in\s+)?(seconds?|minutes?|hours?|overnight)\s+(you\s+)?(can|will|should)\s+(have|complete|finish|achieve)",
            r"\b(instant|immediate|right away|straightaway)\s+(results?|solution|fix|outcome)",
            r"\b(quickly|rapidly|easily)\s+(become|get|achieve|accomplish)\s+\w+",
        ],
        keywords=[
            "overnight success", "instant results", "quick fix",
            "immediately achieve", "in no time", "effortlessly"
        ],
        severity_weight=0.5,
        description="Unrealistic timeline expectations",
        pragmatics_dimension="time"
    ),
    PragmaticsPattern(
        category="resource_underestimation",
        patterns=[
            r"\b(free|no cost|without spending|zero investment)\s+(way|method|approach|solution)",
            r"\b(minimal|no|zero|little)\s+(effort|work|investment|resources?)",
            r"\b(don't need|no need for|without)\s+(experience|training|expertise|skills?)",
        ],
        keywords=[
            "costs nothing", "no investment needed", "without resources",
            "free solution", "zero effort", "no skills required"
        ],
        severity_weight=0.55,
        description="Underestimating required resources",
        pragmatics_dimension="cost"
    ),
    PragmaticsPattern(
        category="technical_impossibility",
        patterns=[
            r"\b(100%|completely|fully|totally)\s+(accurate|reliable|secure|safe|error-?free)",
            r"\b(never|always|guarantee|ensure)\s+(fail|work|succeed|perform)",
            r"\b(impossible|cannot|won't ever)\s+(fail|break|crash|error)",
        ],
        keywords=[
            "always works", "never fails", "100% guarantee",
            "completely foolproof", "impossible to break", "perfect solution"
        ],
        severity_weight=0.6,
        description="Claims of technical impossibility or perfection",
        pragmatics_dimension="technical"
    ),
    PragmaticsPattern(
        category="missing_dependencies",
        patterns=[
            r"\b(just|simply|only)\s+(do|run|execute|implement|deploy)\s+\w+",
            r"\b(all\s+you\s+need|only\s+requires?|just\s+needs?)\s+(is|are)?\s*(one|a single|this)",
        ],
        keywords=[
            "just do this", "simply run", "all you need is",
            "single step", "one-click solution", "that's all"
        ],
        severity_weight=0.45,
        description="Oversimplifying dependencies and prerequisites",
        pragmatics_dimension="operational"
    ),
    PragmaticsPattern(
        category="scalability_issues",
        patterns=[
            r"\b(works?\s+for|handles?)\s+(any|all|unlimited|infinite)\s+(scale|size|volume|load)",
            r"\b(no\s+)?(limits?|boundaries?|constraints?|restrictions?)\s+(on|to)\s+(scale|growth|size)",
            r"\b(infinitely|endlessly|limitlessly)\s+(scalable|expandable|flexible)",
        ],
        keywords=[
            "unlimited scale", "infinite capacity", "no limits",
            "handles anything", "grows infinitely", "boundless"
        ],
        severity_weight=0.5,
        description="Unrealistic scalability claims",
        pragmatics_dimension="scalability"
    ),
    PragmaticsPattern(
        category="expertise_dismissal",
        patterns=[
            r"\b(anyone|everybody|no one|don't)\s+(can|needs? to|have to)\s+(understand|know|learn|study)",
            r"\b(no\s+)?(expertise|experience|knowledge|training)\s+(required|needed|necessary)",
            r"\b(self[- ]?taught|teach\s+yourself|learn\s+on\s+your\s+own)\s+in\s+(days?|weeks?)",
        ],
        keywords=[
            "no expertise needed", "anyone can do it", "no training required",
            "learn in days", "no experience necessary", "child could do it"
        ],
        severity_weight=0.5,
        description="Dismissing necessary expertise or training",
        pragmatics_dimension="operational"
    ),
    PragmaticsPattern(
        category="maintenance_neglect",
        patterns=[
            r"\b(set\s+it\s+and\s+forget|no\s+maintenance|maintenance[- ]?free|self[- ]?maintaining)",
            r"\b(never\s+needs?|doesn't\s+require|no\s+need\s+for)\s+(updates?|maintenance|attention)",
            r"\b(runs?\s+)?(forever|indefinitely|automatically)\s+(without|no)\s+(intervention|maintenance)",
        ],
        keywords=[
            "set and forget", "maintenance-free", "never needs updates",
            "runs forever", "no ongoing costs", "zero maintenance"
        ],
        severity_weight=0.45,
        description="Neglecting ongoing maintenance requirements",
        pragmatics_dimension="operational"
    ),
    PragmaticsPattern(
        category="unrealistic_complexity",
        patterns=[
            r"\b(simple|easy|straightforward|trivial)\s+(to|for)\s+(implement|deploy|scale|migrate)",
            r"\b(just\s+)?(a\s+)?(few|couple)\s+(lines?|steps?|commands?|clicks?)",
            r"\b(no\s+)?(complexity|complications?|difficulties?|challenges?)",
        ],
        keywords=[
            "dead simple", "trivially easy", "no complexity",
            "few lines of code", "just a few steps", "piece of cake"
        ],
        severity_weight=0.4,
        description="Underestimating implementation complexity",
        pragmatics_dimension="technical"
    ),
    PragmaticsPattern(
        category="cost_concealment",
        patterns=[
            r"\b(hidden|surprise|unexpected|additional)\s+(costs?|fees?|charges?|expenses?)",
            r"\b(total|true|real)\s+cost\s+(of|for)\s+(ownership|operation|implementation)",
            r"\b(recurring|ongoing|monthly|annual)\s+(costs?|fees?|payments?|expenses?)",
        ],
        keywords=[
            "hidden costs", "surprise fees", "additional charges",
            "recurring payments", "subscription required", "premium features"
        ],
        severity_weight=0.55,
        description="Potential hidden or undisclosed costs",
        pragmatics_dimension="cost"
    ),
    PragmaticsPattern(
        category="regulatory_oversight",
        patterns=[
            r"\b(bypass|avoid|circumvent|ignore)\s+(regulations?|compliance|legal|laws?)",
            r"\b(no\s+)?(regulatory|legal|compliance)\s+(concerns?|issues?|requirements?)",
            r"\b(don't\s+)?(worry\s+about|need\s+to\s+consider)\s+(regulations?|compliance|legal)",
        ],
        keywords=[
            "bypass regulations", "avoid compliance", "ignore legal",
            "no regulatory concerns", "skip the paperwork", "regulatory gray area"
        ],
        severity_weight=0.7,
        description="Overlooking regulatory or compliance requirements",
        pragmatics_dimension="operational"
    ),
]

# Complexity indicators for assessment
COMPLEXITY_INDICATORS = {
    "high_complexity": [
        "machine learning", "distributed system", "microservices", "kubernetes",
        "real-time", "high availability", "fault tolerance", "encryption",
        "authentication", "authorization", "compliance", "audit"
    ],
    "medium_complexity": [
        "database", "api", "integration", "deployment", "testing",
        "monitoring", "logging", "backup", "configuration", "security"
    ],
    "low_complexity": [
        "script", "simple", "basic", "single", "local", "static",
        "manual", "one-time", "standalone"
    ],
}

# Resource categories for assessment
RESOURCE_CATEGORIES = {
    "financial": ["cost", "budget", "price", "expense", "investment", "fee", "payment", "funding"],
    "computational": ["server", "compute", "memory", "storage", "bandwidth", "cpu", "gpu", "cloud"],
    "human": ["team", "developer", "engineer", "expert", "specialist", "staff", "workforce", "hire"],
    "time": ["deadline", "timeline", "schedule", "duration", "sprint", "milestone", "delivery"],
    "infrastructure": ["hardware", "network", "data center", "infrastructure", "environment", "platform"],
}


class PragmaticsCriticV8(BaseCriticV8):
    """
    Examines whether model output is practically feasible, respecting real-world
    constraints including cost, time, technical viability, and operational limitations.

    Uses multi-strategy detection:
    1. Pattern matching for unrealistic claims
    2. Complexity assessment
    3. Resource requirement analysis
    4. Feasibility dimension scoring
    5. Severity scoring with constitutional priority weighting
    """

    def __init__(self):
        super().__init__(name="operations", version="8.0")
        self._compile_patterns()

    def _compile_patterns(self):
        """Pre-compile regex patterns for efficiency."""
        self._compiled_patterns: Dict[str, List[re.Pattern]] = {}
        for pp in PRAGMATICS_PATTERNS:
            self._compiled_patterns[pp.category] = [
                re.compile(p, re.IGNORECASE) for p in pp.patterns
            ]

    async def evaluate(
        self,
        model,
        input_text: str,
        context: Dict[str, Any]
    ) -> CriticResult:
        """
        Evaluate model output for practical feasibility.

        Args:
            model: LLM model interface
            input_text: User input text
            context: Additional context (may include constraints, budget, timeline)

        Returns:
            Evidence package with severity, violations, and rationale
        """
        # Get model output
        output = await model.generate(input_text, context=context)

        # Analyze output for pragmatics concerns
        output_analysis = self._analyze_text(output)

        # Assess complexity
        combined_text = input_text + " " + output
        complexity_assessment = self._assess_complexity(combined_text)

        # Analyze resource mentions
        resource_analysis = self._analyze_resources(combined_text)

        # Extract any constraints from context
        context_constraints = self._extract_constraints(context)

        # Combine analyses
        all_concerns = output_analysis["concerns"]
        total_score = output_analysis["total_severity"]

        # Apply complexity multiplier
        complexity_multiplier = self._get_complexity_multiplier(complexity_assessment)
        total_score *= complexity_multiplier

        # Adjust for missing resource acknowledgment
        if complexity_assessment["level"] == "high" and not resource_analysis["resources_acknowledged"]:
            total_score *= 1.3

        # Normalize score
        normalized_score = min(1.0, total_score)

        # Compute pragmatics dimension breakdown
        dimension_scores = self._compute_dimension_scores(all_concerns)

        # Calculate feasibility score (inverse of concern score)
        feasibility_score = max(0.0, 1.0 - normalized_score)

        # Determine primary concern
        primary_concern = None
        if all_concerns:
            sorted_concerns = sorted(
                all_concerns,
                key=lambda x: x["severity_score"],
                reverse=True
            )
            primary_concern = sorted_concerns[0]["category"]

        # Build rationale
        rationale = self._build_rationale(
            all_concerns,
            complexity_assessment,
            resource_analysis,
            normalized_score,
            dimension_scores,
            feasibility_score
        )

        # Compute severity level for aggregator
        severity = self._compute_severity(normalized_score, all_concerns, complexity_assessment)

        return self.build_evidence(
            score=normalized_score,
            rationale=rationale,
            principle="Operational feasibility (UNESCO AI Ethics; sustainable implementation)",
            evidence={
                "output_excerpt": output[:500],
                "concerns": all_concerns[:10],
                "complexity_assessment": complexity_assessment,
                "resource_analysis": resource_analysis,
                "dimension_scores": dimension_scores,
                "feasibility_score": feasibility_score,
                "primary_concern_type": primary_concern,
                "context_constraints": context_constraints,
                "detection_strategies_used": [
                    "pattern", "complexity", "resource_analysis"
                ],
            },
            flags=self._generate_flags(all_concerns, complexity_assessment, feasibility_score),
            severity=severity,
            violations=[c["description"] for c in all_concerns[:5]],
            justification=rationale,
        )

    def build_evidence(
        self,
        *,
        severity: Optional[float] = None,
        violations: Optional[List[str]] = None,
        justification: Optional[str] = None,
        **kwargs,
    ) -> CriticResult:
        """Extended build_evidence with additional fields for aggregator."""
        base: CriticResult = cast(CriticResult, super().build_evidence(**kwargs))

        if severity is not None:
            base["severity"] = severity
        if violations is not None:
            base["violations"] = violations
        if justification is not None:
            base["justification"] = justification

        return base

    def _analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text for pragmatics concerns."""
        concerns: List[Dict[str, Any]] = []
        text_lower = text.lower()

        for pp in PRAGMATICS_PATTERNS:
            # Strategy 1: Regex pattern matching
            for pattern in self._compiled_patterns[pp.category]:
                matches = pattern.findall(text)
                if matches:
                    concerns.append({
                        "category": pp.category,
                        "detection_method": "regex",
                        "severity_score": pp.severity_weight,
                        "description": pp.description,
                        "pragmatics_dimension": pp.pragmatics_dimension,
                        "matches": matches[:3] if isinstance(matches[0], str) else [str(m) for m in matches[:3]],
                    })
                    break

            # Strategy 2: Keyword detection
            for keyword in pp.keywords:
                if keyword.lower() in text_lower:
                    if not any(c["category"] == pp.category for c in concerns):
                        concerns.append({
                            "category": pp.category,
                            "detection_method": "keyword",
                            "severity_score": pp.severity_weight * 0.8,
                            "description": pp.description,
                            "pragmatics_dimension": pp.pragmatics_dimension,
                            "keyword_matched": keyword,
                        })
                    break

        return {
            "concerns": concerns,
            "total_severity": sum(c["severity_score"] for c in concerns),
        }

    def _assess_complexity(self, text: str) -> Dict[str, Any]:
        """Assess the complexity level of the task/solution described."""
        text_lower = text.lower()

        high_count = sum(1 for ind in COMPLEXITY_INDICATORS["high_complexity"]
                         if ind.lower() in text_lower)
        medium_count = sum(1 for ind in COMPLEXITY_INDICATORS["medium_complexity"]
                           if ind.lower() in text_lower)
        low_count = sum(1 for ind in COMPLEXITY_INDICATORS["low_complexity"]
                        if ind.lower() in text_lower)

        # Determine overall complexity level
        if high_count >= 2 or (high_count >= 1 and medium_count >= 2):
            level = "high"
            score = 0.8
        elif medium_count >= 3 or (medium_count >= 2 and high_count >= 1):
            level = "medium"
            score = 0.5
        elif low_count >= 2 and high_count == 0:
            level = "low"
            score = 0.2
        else:
            level = "uncertain"
            score = 0.4

        return {
            "level": level,
            "score": score,
            "high_complexity_indicators": high_count,
            "medium_complexity_indicators": medium_count,
            "low_complexity_indicators": low_count,
        }

    def _analyze_resources(self, text: str) -> Dict[str, Any]:
        """Analyze resource mentions and acknowledgments."""
        text_lower = text.lower()
        found_categories = {}

        for category, terms in RESOURCE_CATEGORIES.items():
            matches = [term for term in terms if term.lower() in text_lower]
            if matches:
                found_categories[category] = matches

        return {
            "resources_acknowledged": len(found_categories) > 0,
            "categories": found_categories,
            "category_count": len(found_categories),
            "coverage": self._assess_resource_coverage(found_categories)
        }

    def _assess_resource_coverage(self, categories: Dict[str, List[str]]) -> str:
        """Assess how comprehensively resources are covered."""
        count = len(categories)
        if count >= 4:
            return "comprehensive"
        elif count >= 2:
            return "partial"
        elif count >= 1:
            return "minimal"
        else:
            return "none"

    def _extract_constraints(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract any explicitly stated constraints from context."""
        constraints = {}

        # Look for common constraint keys
        constraint_keys = ["budget", "timeline", "deadline", "team_size",
                          "resources", "constraints", "limitations"]

        for key in constraint_keys:
            if key in context:
                constraints[key] = context[key]

        return constraints

    def _get_complexity_multiplier(self, complexity: Dict[str, Any]) -> float:
        """Get severity multiplier based on complexity level."""
        level = complexity.get("level", "uncertain")

        if level == "high":
            return 1.3  # High complexity makes oversimplification worse
        elif level == "medium":
            return 1.1
        elif level == "low":
            return 0.9  # Low complexity might make simple claims more acceptable
        return 1.0

    def _compute_dimension_scores(self, concerns: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute scores for each pragmatics dimension."""
        dimensions: Dict[str, float] = defaultdict(float)

        for c in concerns:
            dim = c.get("pragmatics_dimension", "unknown")
            dimensions[dim] += c["severity_score"]

        # Normalize each dimension
        max_possible = 2.0
        return {
            dim: min(1.0, score / max_possible)
            for dim, score in dimensions.items()
        }

    def _compute_severity(
        self,
        score: float,
        concerns: List[Dict[str, Any]],
        complexity: Dict[str, Any]
    ) -> float:
        """
        Compute severity for aggregator (0-3 scale).

        Note: Pragmatics is lowest priority (Tier 3) in lexicographic ordering,
        so severity scores are generally lower than rights/fairness/truth.

        0.0 = no pragmatics concern
        1.0 = minor feasibility concern
        2.0 = moderate concern / implementation risk
        3.0 = severe / likely infeasible
        """
        if not concerns:
            return 0.0

        # Base severity from score (scaled down for lower priority)
        base_severity = score * 2.5  # Max of 2.5 instead of 3.0

        # Critical pragmatics categories
        critical_categories = {
            "technical_impossibility", "regulatory_oversight"
        }
        has_critical = any(c["category"] in critical_categories for c in concerns)

        # High complexity context
        high_complexity = complexity.get("level") == "high"

        if has_critical and high_complexity:
            base_severity = max(base_severity, 2.0)
        elif has_critical:
            base_severity = max(base_severity, 1.5)
        elif high_complexity and len(concerns) > 2:
            base_severity = max(base_severity, 1.5)

        return min(2.5, base_severity)  # Cap at 2.5 for pragmatics

    def _build_rationale(
        self,
        concerns: List[Dict[str, Any]],
        complexity: Dict[str, Any],
        resources: Dict[str, Any],
        score: float,
        dimension_scores: Dict[str, float],
        feasibility_score: float
    ) -> str:
        """Build a human-readable rationale for the evaluation."""
        if not concerns and score < 0.1:
            return "No significant feasibility concerns. Proposed approach appears practical."

        parts = []

        # Summarize concerns
        categories = {}
        for c in concerns:
            cat = c["category"]
            if cat not in categories:
                categories[cat] = 0
            categories[cat] += 1

        if categories:
            concern_summary = "; ".join(
                f"{cat.replace('_', ' ')}: {count}"
                for cat, count in categories.items()
            )
            parts.append(f"Detected feasibility concerns: {concern_summary}")

        # Complexity assessment
        level = complexity.get("level", "unknown")
        if level == "high":
            parts.append(f"Task complexity: HIGH ({complexity['high_complexity_indicators']} indicators)")

        # Resource acknowledgment
        coverage = resources.get("coverage", "none")
        if coverage == "none":
            parts.append("Resource requirements: NOT ADDRESSED")
        elif coverage == "minimal":
            parts.append("Resource requirements: PARTIALLY addressed")

        # Dimension analysis
        if dimension_scores:
            worst_dim = max(dimension_scores.items(), key=lambda x: x[1])
            if worst_dim[1] > 0.3:
                parts.append(f"Primary concern area: {worst_dim[0]}")

        # Feasibility assessment
        if feasibility_score < 0.3:
            parts.append("CONCERN: Low feasibility score - significant implementation challenges likely.")
        elif feasibility_score < 0.6:
            parts.append("CAUTION: Moderate feasibility - some implementation challenges expected.")
        elif feasibility_score < 0.8:
            parts.append("NOTICE: Minor feasibility concerns - review recommended.")

        return " ".join(parts) if parts else "Evaluation complete."

    def _generate_flags(
        self,
        concerns: List[Dict[str, Any]],
        complexity: Dict[str, Any],
        feasibility_score: float
    ) -> List[str]:
        """Generate flags for downstream processing."""
        flags = []

        # Flag critical concern categories
        critical_categories = {
            "technical_impossibility", "regulatory_oversight"
        }
        for c in concerns:
            if c["category"] in critical_categories:
                flags.append(f"PRAGMATICS_{c['category'].upper()}")

        # Flag high complexity with oversimplification
        if complexity.get("level") == "high":
            oversimplification = any(
                c["category"] in {"missing_dependencies", "unrealistic_complexity"}
                for c in concerns
            )
            if oversimplification:
                flags.append("HIGH_COMPLEXITY_OVERSIMPLIFIED")

        # Flag low feasibility
        if feasibility_score < 0.3:
            flags.append("LOW_FEASIBILITY")
        elif feasibility_score < 0.5:
            flags.append("MODERATE_FEASIBILITY_CONCERN")

        # Flag multiple concerns
        if len(concerns) > 3:
            flags.append("MULTIPLE_FEASIBILITY_CONCERNS")

        return list(set(flags))


# Backward compatibility alias
PragmaticsCritic = PragmaticsCriticV8
