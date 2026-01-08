"""
ELEANOR V8 — Business Metrics
------------------------------

Business-level metrics for understanding system behavior and decisions.
"""

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Histogram, Gauge
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = None
    Histogram = None
    Gauge = None


# Business Metrics
if PROMETHEUS_AVAILABLE:
    DECISIONS_TOTAL = Counter(
        "eleanor_decisions_total",
        "Total decisions made",
        ["decision_type", "final_assessment"],
    )
    
    ESCALATIONS_TOTAL = Counter(
        "eleanor_escalations_total",
        "Total escalations",
        ["tier", "critic", "reason"],
    )
    
    CRITIC_AGREEMENT = Histogram(
        "eleanor_critic_agreement",
        "Critic agreement score (0-1, higher = more agreement)",
        buckets=(0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0),
    )
    
    UNCERTAINTY_DISTRIBUTION = Histogram(
        "eleanor_uncertainty_score",
        "Uncertainty scores (0-1, higher = more uncertain)",
        buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    )
    
    SEVERITY_DISTRIBUTION = Histogram(
        "eleanor_severity_score",
        "Severity scores (0-3, higher = more severe)",
        buckets=(0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
    )
    
    CRITIC_DISAGREEMENT = Histogram(
        "eleanor_critic_disagreement",
        "Critic disagreement score (0-1, higher = more disagreement)",
        buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    )
    
    ACTIVE_TRACES = Gauge(
        "eleanor_active_traces",
        "Number of active traces being processed",
    )
    
    DEGRADED_COMPONENTS = Gauge(
        "eleanor_degraded_components",
        "Number of degraded components",
    )
else:
    DECISIONS_TOTAL = None
    ESCALATIONS_TOTAL = None
    CRITIC_AGREEMENT = None
    UNCERTAINTY_DISTRIBUTION = None
    SEVERITY_DISTRIBUTION = None
    CRITIC_DISAGREEMENT = None
    ACTIVE_TRACES = None
    DEGRADED_COMPONENTS = None


def record_decision(decision_type: str, final_assessment: str) -> None:
    """Record a decision metric."""
    if DECISIONS_TOTAL:
        DECISIONS_TOTAL.labels(
            decision_type=decision_type,
            final_assessment=final_assessment,
        ).inc()


def record_escalation(tier: int, critic: Optional[str] = None, reason: Optional[str] = None) -> None:
    """Record an escalation metric."""
    if ESCALATIONS_TOTAL:
        ESCALATIONS_TOTAL.labels(
            tier=str(tier),
            critic=critic or "unknown",
            reason=reason or "unknown",
        ).inc()


def record_critic_agreement(agreement_score: float) -> None:
    """Record critic agreement score."""
    if CRITIC_AGREEMENT:
        CRITIC_AGREEMENT.observe(agreement_score)


def record_uncertainty(uncertainty_score: float) -> None:
    """Record uncertainty score."""
    if UNCERTAINTY_DISTRIBUTION:
        UNCERTAINTY_DISTRIBUTION.observe(uncertainty_score)


def record_severity(severity_score: float) -> None:
    """Record severity score."""
    if SEVERITY_DISTRIBUTION:
        SEVERITY_DISTRIBUTION.observe(severity_score)


def record_critic_disagreement(disagreement_score: float) -> None:
    """Record critic disagreement score."""
    if CRITIC_DISAGREEMENT:
        CRITIC_DISAGREEMENT.observe(disagreement_score)


def set_active_traces(count: int) -> None:
    """Set number of active traces."""
    if ACTIVE_TRACES:
        ACTIVE_TRACES.set(count)


def set_degraded_components(count: int) -> None:
    """Set number of degraded components."""
    if DEGRADED_COMPONENTS:
        DEGRADED_COMPONENTS.set(count)


def record_engine_result(result: Dict[str, Any]) -> None:
    """Record comprehensive metrics from engine result."""
    # Record decision
    decision = result.get("final_decision") or result.get("aggregated", {}).get("decision", "unknown")
    assessment = result.get("final_assessment", "unknown")
    record_decision(decision, assessment)
    
    # Record uncertainty
    uncertainty = result.get("uncertainty") or {}
    if isinstance(uncertainty, dict):
        uncertainty_score = uncertainty.get("overall_uncertainty", 0.0)
        if uncertainty_score is not None:
            record_uncertainty(float(uncertainty_score))
    
    # Record severity
    aggregated = result.get("aggregated") or {}
    if isinstance(aggregated, dict):
        severity = aggregated.get("severity") or aggregated.get("final_severity")
        if severity is not None:
            record_severity(float(severity))
    
    # Record degraded components
    degraded = result.get("degraded_components") or []
    if isinstance(degraded, list):
        set_degraded_components(len(degraded))
    
    # Record critic disagreement if available
    critic_outputs = result.get("critic_outputs") or result.get("critic_findings") or {}
    if isinstance(critic_outputs, dict) and len(critic_outputs) > 1:
        # Calculate disagreement
        severities = []
        for output in critic_outputs.values():
            if isinstance(output, dict):
                severity = output.get("severity") or output.get("score", 0.0)
                severities.append(float(severity))
        
        if len(severities) > 1:
            # Calculate standard deviation as disagreement measure
            mean_severity = sum(severities) / len(severities)
            variance = sum((s - mean_severity) ** 2 for s in severities) / len(severities)
            std_dev = variance ** 0.5
            # Normalize to 0-1 range (assuming max severity is 3.0)
            disagreement = min(1.0, std_dev / 3.0)
            record_critic_disagreement(disagreement)
            
            # Calculate agreement (inverse of disagreement)
            agreement = 1.0 - disagreement
            record_critic_agreement(agreement)


__all__ = [
    "record_decision",
    "record_escalation",
    "record_critic_agreement",
    "record_uncertainty",
    "record_severity",
    "record_critic_disagreement",
    "set_active_traces",
    "set_degraded_components",
    "record_engine_result",
]
