"""
ELEANOR V8 — Temporal Precedent Evolution Tracking
---------------------------------------------------

Tracks how precedents change over time and detects evolutionary patterns.
Provides lifecycle management, drift detection, and evolution analytics.

Features:
- Version history for precedents
- Lifecycle state management (active, deprecated, superseded)
- Temporal drift detection
- Evolution analytics and reporting
- Automatic deprecation recommendations
"""

import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class PrecedentLifecycleState(str, Enum):
    """Lifecycle states for precedents."""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"


@dataclass
class PrecedentVersion:
    """Represents a version of a precedent."""
    version_id: str
    case_id: str
    timestamp: float
    decision: str
    aggregate_score: float
    values: List[str]
    rationale: str
    critic_outputs: Dict[str, Any]
    metadata: Dict[str, Any]
    created_by: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PrecedentEvolution:
    """Tracks the evolution of a precedent over time."""
    case_id: str
    versions: List[PrecedentVersion] = field(default_factory=list)
    lifecycle_state: PrecedentLifecycleState = PrecedentLifecycleState.ACTIVE
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    deprecated_at: Optional[float] = None
    superseded_by: Optional[str] = None
    evolution_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["lifecycle_state"] = self.lifecycle_state.value
        return d
    
    def add_version(self, version: PrecedentVersion) -> None:
        """Add a new version to the evolution history."""
        self.versions.append(version)
        self.versions.sort(key=lambda v: v.timestamp)
        self.updated_at = time.time()
    
    def get_latest_version(self) -> Optional[PrecedentVersion]:
        """Get the most recent version."""
        if not self.versions:
            return None
        return max(self.versions, key=lambda v: v.timestamp)
    
    def get_version_count(self) -> int:
        """Get the total number of versions."""
        return len(self.versions)


class TemporalPrecedentEvolutionTracker:
    """
    Tracks temporal evolution of precedents with lifecycle management.
    """
    
    def __init__(self, store_backend: Optional[Any] = None):
        """
        Initialize the temporal evolution tracker.
        
        Args:
            store_backend: Optional backend for persisting evolution data
        """
        self._evolutions: Dict[str, PrecedentEvolution] = {}
        self._store_backend = store_backend
        self._drift_threshold: float = 0.3  # Configurable threshold for drift detection
    
    def track_precedent_update(
        self,
        case_id: str,
        decision: str,
        aggregate_score: float,
        values: List[str],
        rationale: str,
        critic_outputs: Dict[str, Any],
        metadata: Dict[str, Any],
        created_by: Optional[str] = None,
    ) -> PrecedentVersion:
        """
        Track an update to a precedent, creating a new version.
        
        Args:
            case_id: Unique identifier for the precedent
            decision: The decision (allow, deny, escalate, etc.)
            aggregate_score: Aggregate critic score
            values: List of values invoked
            rationale: Decision rationale
            critic_outputs: Outputs from all critics
            metadata: Additional metadata
            created_by: Optional identifier for who/what created this version
            
        Returns:
            The created PrecedentVersion
        """
        version_id = f"{case_id}_v{int(time.time() * 1000)}"
        version = PrecedentVersion(
            version_id=version_id,
            case_id=case_id,
            timestamp=time.time(),
            decision=decision,
            aggregate_score=aggregate_score,
            values=values,
            rationale=rationale,
            critic_outputs=critic_outputs,
            metadata=metadata,
            created_by=created_by,
        )
        
        # Get or create evolution record
        if case_id not in self._evolutions:
            evolution = PrecedentEvolution(case_id=case_id)
            self._evolutions[case_id] = evolution
        else:
            evolution = self._evolutions[case_id]
        
        # Add version and detect changes
        evolution.add_version(version)
        
        # Check for decision drift
        if len(evolution.versions) > 1:
            drift_signal = self._detect_decision_drift(evolution)
            if drift_signal.get("drift_detected", False):
                logger.warning(
                    f"Decision drift detected for precedent {case_id}: "
                    f"{drift_signal.get('message', 'Unknown')}"
                )
                evolution.evolution_metadata["drift_alerts"] = (
                    evolution.evolution_metadata.get("drift_alerts", []) + [drift_signal]
                )
        
        # Persist if backend available
        if self._store_backend:
            try:
                self._persist_evolution(evolution)
            except Exception as e:
                logger.error(f"Failed to persist evolution for {case_id}: {e}")
        
        return version
    
    def get_evolution(self, case_id: str) -> Optional[PrecedentEvolution]:
        """Get the evolution record for a precedent."""
        return self._evolutions.get(case_id)
    
    def set_lifecycle_state(
        self,
        case_id: str,
        state: PrecedentLifecycleState,
        superseded_by: Optional[str] = None,
    ) -> bool:
        """
        Update the lifecycle state of a precedent.
        
        Args:
            case_id: Identifier of the precedent
            state: New lifecycle state
            superseded_by: If superseded, the ID of the superseding precedent
            
        Returns:
            True if successful, False if precedent not found
        """
        if case_id not in self._evolutions:
            logger.warning(f"Cannot update lifecycle state: precedent {case_id} not found")
            return False
        
        evolution = self._evolutions[case_id]
        evolution.lifecycle_state = state
        evolution.updated_at = time.time()
        
        if state == PrecedentLifecycleState.DEPRECATED:
            evolution.deprecated_at = time.time()
        elif state == PrecedentLifecycleState.SUPERSEDED:
            evolution.deprecated_at = time.time()
            evolution.superseded_by = superseded_by
        
        # Persist if backend available
        if self._store_backend:
            try:
                self._persist_evolution(evolution)
            except Exception as e:
                logger.error(f"Failed to persist lifecycle update for {case_id}: {e}")
        
        logger.info(f"Updated lifecycle state for {case_id} to {state.value}")
        return True
    
    def detect_temporal_drift(self, case_id: str) -> Dict[str, Any]:
        """
        Detect temporal drift in a precedent's evolution.
        
        Returns:
            Dictionary with drift metrics and signals
        """
        evolution = self.get_evolution(case_id)
        if not evolution or len(evolution.versions) < 2:
            return {
                "drift_detected": False,
                "message": "Insufficient history for drift detection",
                "versions_analyzed": len(evolution.versions) if evolution else 0,
            }
        
        versions = sorted(evolution.versions, key=lambda v: v.timestamp)
        
        # Analyze score trends
        scores = [v.aggregate_score for v in versions]
        score_variance = self._calculate_variance(scores)
        score_trend = self._calculate_trend(scores)
        
        # Analyze decision consistency
        decisions = [v.decision for v in versions]
        decision_changes = sum(1 for i in range(1, len(decisions)) if decisions[i] != decisions[i-1])
        decision_consistency = 1.0 - (decision_changes / len(decisions))
        
        # Calculate overall drift score
        drift_score = (
            (score_variance * 0.5) +
            ((1.0 - decision_consistency) * 0.3) +
            (abs(score_trend) * 0.2)
        )
        
        drift_detected = drift_score > self._drift_threshold
        
        return {
            "drift_detected": drift_detected,
            "drift_score": float(drift_score),
            "score_variance": float(score_variance),
            "score_trend": float(score_trend),
            "decision_consistency": float(decision_consistency),
            "decision_changes": decision_changes,
            "versions_analyzed": len(versions),
            "message": (
                f"Drift detected (score: {drift_score:.3f})" if drift_detected
                else f"Stable evolution (score: {drift_score:.3f})"
            ),
        }
    
    def get_evolution_analytics(self, case_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get analytics on precedent evolution.
        
        Args:
            case_id: If provided, analytics for specific precedent; otherwise aggregate
            
        Returns:
            Dictionary with evolution analytics
        """
        if case_id:
            evolution = self.get_evolution(case_id)
            if not evolution:
                return {"error": f"Precedent {case_id} not found"}
            
            drift_info = self.detect_temporal_drift(case_id)
            
            return {
                "case_id": case_id,
                "lifecycle_state": evolution.lifecycle_state.value,
                "version_count": evolution.get_version_count(),
                "created_at": evolution.created_at,
                "updated_at": evolution.updated_at,
                "deprecated_at": evolution.deprecated_at,
                "superseded_by": evolution.superseded_by,
                "drift_metrics": drift_info,
                "latest_version": (
                    evolution.get_latest_version().to_dict()
                    if evolution.get_latest_version() else None
                ),
            }
        
        # Aggregate analytics
        total_precedents = len(self._evolutions)
        active_count = sum(
            1 for e in self._evolutions.values()
            if e.lifecycle_state == PrecedentLifecycleState.ACTIVE
        )
        deprecated_count = sum(
            1 for e in self._evolutions.values()
            if e.lifecycle_state == PrecedentLifecycleState.DEPRECATED
        )
        superseded_count = sum(
            1 for e in self._evolutions.values()
            if e.lifecycle_state == PrecedentLifecycleState.SUPERSEDED
        )
        
        # Average versions per precedent
        total_versions = sum(e.get_version_count() for e in self._evolutions.values())
        avg_versions = total_versions / total_precedents if total_precedents > 0 else 0
        
        # Drift detection across all precedents
        drift_counts = {"high": 0, "medium": 0, "low": 0, "none": 0}
        for evo in self._evolutions.values():
            drift_info = self.detect_temporal_drift(evo.case_id)
            drift_score = drift_info.get("drift_score", 0.0)
            if drift_score > 0.5:
                drift_counts["high"] += 1
            elif drift_score > 0.3:
                drift_counts["medium"] += 1
            elif drift_score > 0.1:
                drift_counts["low"] += 1
            else:
                drift_counts["none"] += 1
        
        return {
            "total_precedents": total_precedents,
            "lifecycle_distribution": {
                "active": active_count,
                "deprecated": deprecated_count,
                "superseded": superseded_count,
                "archived": total_precedents - active_count - deprecated_count - superseded_count,
            },
            "average_versions_per_precedent": float(avg_versions),
            "total_versions": total_versions,
            "drift_distribution": drift_counts,
        }
    
    def recommend_deprecations(self, min_versions: int = 3) -> List[Dict[str, Any]]:
        """
        Recommend precedents that should be deprecated based on evolution patterns.
        
        Args:
            min_versions: Minimum number of versions required for recommendation
            
        Returns:
            List of deprecation recommendations
        """
        recommendations = []
        
        for case_id, evolution in self._evolutions.items():
            if evolution.lifecycle_state != PrecedentLifecycleState.ACTIVE:
                continue
            
            if evolution.get_version_count() < min_versions:
                continue
            
            drift_info = self.detect_temporal_drift(case_id)
            
            # Recommend deprecation if:
            # 1. High drift detected
            # 2. Many decision changes
            # 3. High score variance
            should_deprecate = False
            reason = []
            
            if drift_info.get("drift_score", 0.0) > 0.5:
                should_deprecate = True
                reason.append("High drift score")
            
            if drift_info.get("decision_changes", 0) > len(evolution.versions) * 0.5:
                should_deprecate = True
                reason.append("High decision inconsistency")
            
            if drift_info.get("score_variance", 0.0) > 0.4:
                should_deprecate = True
                reason.append("High score variance")
            
            if should_deprecate:
                recommendations.append({
                    "case_id": case_id,
                    "reason": "; ".join(reason),
                    "drift_metrics": drift_info,
                    "version_count": evolution.get_version_count(),
                    "recommended_action": "deprecate",
                })
        
        return sorted(recommendations, key=lambda r: r["drift_metrics"].get("drift_score", 0.0), reverse=True)
    
    def _detect_decision_drift(self, evolution: PrecedentEvolution) -> Dict[str, Any]:
        """Internal method to detect decision drift between versions."""
        if len(evolution.versions) < 2:
            return {"drift_detected": False}
        
        versions = sorted(evolution.versions, key=lambda v: v.timestamp)
        latest = versions[-1]
        previous = versions[-2]
        
        # Check for decision change
        decision_changed = latest.decision != previous.decision
        
        # Check for significant score change
        score_delta = abs(latest.aggregate_score - previous.aggregate_score)
        significant_score_change = score_delta > 0.3
        
        # Check for value set changes
        values_changed = set(latest.values) != set(previous.values)
        
        drift_detected = decision_changed or significant_score_change or values_changed
        
        return {
            "drift_detected": drift_detected,
            "decision_changed": decision_changed,
            "score_delta": float(score_delta),
            "significant_score_change": significant_score_change,
            "values_changed": values_changed,
            "message": (
                f"Decision changed: {previous.decision} -> {latest.decision}"
                if decision_changed
                else f"Score changed by {score_delta:.3f}"
                if significant_score_change
                else "Values changed"
                if values_changed
                else "No significant drift"
            ),
        }
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values."""
        if not values or len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate linear trend (slope) of values over time."""
        if not values or len(values) < 2:
            return 0.0
        
        n = len(values)
        x = list(range(n))
        y = values
        
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        denominator = (n * sum_x2) - (sum_x ** 2)
        if denominator == 0:
            return 0.0
        
        slope = ((n * sum_xy) - (sum_x * sum_y)) / denominator
        return slope
    
    def _persist_evolution(self, evolution: PrecedentEvolution) -> None:
        """Persist evolution data to backend store."""
        if not self._store_backend:
            return
        
        # Store evolution record (implementation depends on backend)
        # This is a placeholder - actual implementation would use the store backend
        pass
