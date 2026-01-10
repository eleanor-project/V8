"""
ELEANOR V8 — Adaptive Critic Weighting
--------------------------------------

Learns optimal critic weights from historical decisions using meta-learning.
Adapts weights based on:
- Historical accuracy of critic predictions
- Human review corrections
- Decision quality metrics
- Precedent alignment

Features:
- Online learning from feedback
- Multi-armed bandit approach for exploration
- Gradient-based optimization for weight updates
- Decay mechanisms for temporal relevance
- Integration with aggregation pipeline
"""

import time
import math
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class WeightUpdate:
    """Represents a weight update for a critic."""
    critic_name: str
    old_weight: float
    new_weight: float
    update_reason: str
    confidence: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class CriticPerformanceMetrics:
    """Tracks performance metrics for a critic."""
    critic_name: str
    total_evaluations: int = 0
    correct_predictions: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    average_severity_accuracy: float = 0.0
    human_override_count: int = 0
    precedent_alignment_score: float = 0.0
    last_updated: float = field(default_factory=time.time)
    
    def accuracy(self) -> float:
        """Calculate prediction accuracy."""
        if self.total_evaluations == 0:
            return 0.5  # Default neutral accuracy
        return self.correct_predictions / self.total_evaluations
    
    def precision(self) -> float:
        """Calculate precision (true positives / (true positives + false positives))."""
        tp_plus_fp = self.correct_predictions + self.false_positives
        if tp_plus_fp == 0:
            return 0.5
        return self.correct_predictions / tp_plus_fp
    
    def recall(self) -> float:
        """Calculate recall (true positives / (true positives + false negatives))."""
        tp_plus_fn = self.correct_predictions + self.false_negatives
        if tp_plus_fn == 0:
            return 0.5
        return self.correct_predictions / tp_plus_fn
    
    def f1_score(self) -> float:
        """Calculate F1 score."""
        prec = self.precision()
        rec = self.recall()
        if prec + rec == 0:
            return 0.0
        return 2 * (prec * rec) / (prec + rec)


class AdaptiveCriticWeighting:
    """
    Learns optimal critic weights from historical performance.
    Uses meta-learning to adapt weights over time.
    """
    
    def __init__(
        self,
        initial_weights: Optional[Dict[str, float]] = None,
        learning_rate: float = 0.1,
        exploration_rate: float = 0.05,
        decay_factor: float = 0.95,
        min_weight: float = 0.1,
        max_weight: float = 2.0,
        min_samples: int = 10,
    ):
        """
        Initialize adaptive weighting system.
        
        Args:
            initial_weights: Initial weights for critics (defaults to uniform 1.0)
            learning_rate: Learning rate for weight updates (default 0.1)
            exploration_rate: Rate of exploration vs exploitation (default 0.05)
            decay_factor: Temporal decay factor for old observations (default 0.95)
            min_weight: Minimum allowed weight (default 0.1)
            max_weight: Maximum allowed weight (default 2.0)
            min_samples: Minimum samples required before adjusting weights (default 10)
        """
        self.weights: Dict[str, float] = initial_weights or {}
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.decay_factor = decay_factor
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.min_samples = min_samples
        
        # Performance tracking
        self.metrics: Dict[str, CriticPerformanceMetrics] = {}
        self.feedback_history: deque = deque(maxlen=10000)  # Last 10k feedback events
        
        # Weight update history
        self.weight_history: List[WeightUpdate] = []
        
        # Default uniform weights if not specified
        if not self.weights:
            self._initialize_default_weights()
    
    def _initialize_default_weights(self) -> None:
        """Initialize with default uniform weights for known critics."""
        default_critics = [
            "rights", "autonomy", "fairness", "truth", "risk", "dignity",
            "pragmatics", "precedent", "operations"
        ]
        for critic in default_critics:
            self.weights[critic] = 1.0
            self.metrics[critic] = CriticPerformanceMetrics(critic_name=critic)
    
    def get_weights(self, critic_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Get current weights for critics.
        
        Args:
            critic_names: Optional list of critic names. If None, returns all weights.
            
        Returns:
            Dictionary mapping critic names to weights
        """
        if critic_names is None:
            return self.weights.copy()
        
        return {name: self.weights.get(name, 1.0) for name in critic_names}
    
    def get_weighted_scores(
        self,
        critic_results: Dict[str, Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Apply adaptive weights to critic scores.
        
        Args:
            critic_results: Dictionary of critic_name -> critic_result dict
            
        Returns:
            Dictionary of critic_name -> weighted_score
        """
        weighted = {}
        
        for critic_name, result in critic_results.items():
            weight = self.weights.get(critic_name, 1.0)
            raw_score = result.get("score", 0.5) if isinstance(result, dict) else 0.5
            
            # Apply weight
            weighted_score = raw_score * weight
            
            # Ensure within bounds
            weighted[critic_name] = max(0.0, min(1.0, weighted_score))
        
        return weighted
    
    def record_feedback(
        self,
        trace_id: str,
        critic_results: Dict[str, Dict[str, Any]],
        final_decision: str,
        human_review_decision: Optional[str] = None,
        human_corrected: bool = False,
        precedent_alignment: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record feedback for learning from a decision.
        
        Args:
            trace_id: Trace identifier
            critic_results: Critic evaluation results
            final_decision: Final system decision (allow/deny/escalate)
            human_review_decision: Optional human reviewer decision
            human_corrected: Whether human overrode the system decision
            precedent_alignment: Optional precedent alignment data
        """
        feedback = {
            "trace_id": trace_id,
            "timestamp": time.time(),
            "critic_results": critic_results,
            "final_decision": final_decision,
            "human_review_decision": human_review_decision,
            "human_corrected": human_corrected,
            "precedent_alignment": precedent_alignment,
        }
        
        self.feedback_history.append(feedback)
        
        # Update metrics for each critic
        for critic_name, result in critic_results.items():
            if not isinstance(result, dict):
                continue
            
            # Ensure metrics exist
            if critic_name not in self.metrics:
                self.metrics[critic_name] = CriticPerformanceMetrics(critic_name=critic_name)
            
            metrics = self.metrics[critic_name]
            metrics.total_evaluations += 1
            metrics.last_updated = time.time()
            
            # Update accuracy based on human feedback
            if human_review_decision:
                # Determine if critic was "correct" based on human decision
                critic_severity = result.get("severity", 0.0)
                critic_violations = result.get("violations", [])
                
                # Critic predicted violation if severity > 1.0 or violations exist
                critic_predicted_violation = critic_severity > 1.0 or len(critic_violations) > 0
                
                # Human decision indicates violation if deny/escalate
                human_indicated_violation = human_review_decision.lower() in ["deny", "escalate"]
                
                if critic_predicted_violation == human_indicated_violation:
                    metrics.correct_predictions += 1
                elif critic_predicted_violation and not human_indicated_violation:
                    metrics.false_positives += 1
                else:
                    metrics.false_negatives += 1
                
                if human_corrected:
                    metrics.human_override_count += 1
            
            # Update precedent alignment score
            if precedent_alignment:
                alignment_score = precedent_alignment.get("alignment_score", 0.0)
                # Moving average
                metrics.precedent_alignment_score = (
                    (metrics.precedent_alignment_score * 0.9) + (alignment_score * 0.1)
                )
    
    def update_weights(self) -> List[WeightUpdate]:
        """
        Update critic weights based on performance metrics.
        
        Returns:
            List of weight updates applied
        """
        updates = []
        
        for critic_name, metrics in self.metrics.items():
            if metrics.total_evaluations < self.min_samples:
                continue  # Need more data
            
            old_weight = self.weights.get(critic_name, 1.0)
            new_weight = old_weight
            
            # Calculate performance score
            accuracy = metrics.accuracy()
            f1 = metrics.f1_score()
            alignment = metrics.precedent_alignment_score
            
            # Composite performance score (0-1)
            performance_score = (
                (accuracy * 0.4) +
                (f1 * 0.3) +
                ((alignment + 1.0) / 2.0 * 0.2) +  # Normalize alignment -1 to 1 -> 0 to 1
                ((1.0 - metrics.human_override_count / max(1, metrics.total_evaluations)) * 0.1)
            )
            
            # Adjust weight based on performance
            # Better performance -> higher weight (up to max_weight)
            # Poor performance -> lower weight (down to min_weight)
            if performance_score > 0.7:
                # High performer - increase weight
                weight_delta = self.learning_rate * (1.0 - performance_score) * (self.max_weight - old_weight)
                new_weight = min(self.max_weight, old_weight + weight_delta)
                update_reason = "high_performance"
            elif performance_score < 0.4:
                # Low performer - decrease weight
                weight_delta = self.learning_rate * (0.4 - performance_score) * (old_weight - self.min_weight)
                new_weight = max(self.min_weight, old_weight - weight_delta)
                update_reason = "low_performance"
            else:
                # Moderate performance - slight adjustment
                target_weight = 1.0 + (performance_score - 0.5) * 0.5
                weight_delta = self.learning_rate * (target_weight - old_weight)
                new_weight = max(self.min_weight, min(self.max_weight, old_weight + weight_delta))
                update_reason = "performance_adjustment"
            
            # Apply temporal decay (reduce impact of old observations)
            if metrics.last_updated < time.time() - (30 * 24 * 3600):  # 30 days
                # Decay old metrics influence
                new_weight = old_weight * self.decay_factor + new_weight * (1 - self.decay_factor)
            
            # Exploration: occasionally try different weights
            if self.exploration_rate > 0 and len(self.weight_history) % 10 == 0:
                exploration_adjustment = (self.exploration_rate * (2.0 * (time.time() % 1.0) - 1.0))
                new_weight = max(self.min_weight, min(self.max_weight, new_weight + exploration_adjustment))
                if abs(exploration_adjustment) > 0.01:
                    update_reason += "_exploration"
            
            # Only update if change is significant
            if abs(new_weight - old_weight) > 0.01:
                self.weights[critic_name] = new_weight
                
                update = WeightUpdate(
                    critic_name=critic_name,
                    old_weight=old_weight,
                    new_weight=new_weight,
                    update_reason=update_reason,
                    confidence=abs(new_weight - old_weight),
                )
                updates.append(update)
                self.weight_history.append(update)
                
                logger.info(
                    f"Updated weight for {critic_name}: {old_weight:.3f} -> {new_weight:.3f} "
                    f"({update_reason}, performance: {performance_score:.3f})"
                )
        
        return updates
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate a performance report for all critics."""
        report = {
            "current_weights": self.weights.copy(),
            "critic_metrics": {},
            "summary": {
                "total_feedback_samples": len(self.feedback_history),
                "total_weight_updates": len(self.weight_history),
                "critics_tracked": len(self.metrics),
            },
        }
        
        for critic_name, metrics in self.metrics.items():
            report["critic_metrics"][critic_name] = {
                "weight": self.weights.get(critic_name, 1.0),
                "total_evaluations": metrics.total_evaluations,
                "accuracy": metrics.accuracy(),
                "precision": metrics.precision(),
                "recall": metrics.recall(),
                "f1_score": metrics.f1_score(),
                "false_positives": metrics.false_positives,
                "false_negatives": metrics.false_negatives,
                "human_override_count": metrics.human_override_count,
                "precedent_alignment_score": metrics.precedent_alignment_score,
                "last_updated": metrics.last_updated,
            }
        
        return report
    
    def reset_weights(self, weights: Optional[Dict[str, float]] = None) -> None:
        """
        Reset weights to specified values or default uniform weights.
        
        Args:
            weights: Optional dictionary of critic_name -> weight. If None, resets to uniform 1.0.
        """
        if weights:
            self.weights = weights.copy()
        else:
            self._initialize_default_weights()
        
        logger.info(f"Reset weights: {self.weights}")
    
    def export_state(self) -> Dict[str, Any]:
        """Export current state for persistence."""
        return {
            "weights": self.weights,
            "metrics": {
                name: {
                    "total_evaluations": m.total_evaluations,
                    "correct_predictions": m.correct_predictions,
                    "false_positives": m.false_positives,
                    "false_negatives": m.false_negatives,
                    "average_severity_accuracy": m.average_severity_accuracy,
                    "human_override_count": m.human_override_count,
                    "precedent_alignment_score": m.precedent_alignment_score,
                    "last_updated": m.last_updated,
                }
                for name, m in self.metrics.items()
            },
            "config": {
                "learning_rate": self.learning_rate,
                "exploration_rate": self.exploration_rate,
                "decay_factor": self.decay_factor,
                "min_weight": self.min_weight,
                "max_weight": self.max_weight,
                "min_samples": self.min_samples,
            },
        }
    
    def import_state(self, state: Dict[str, Any]) -> None:
        """Import state from persistence."""
        self.weights = state.get("weights", {})
        
        metrics_data = state.get("metrics", {})
        for name, data in metrics_data.items():
            self.metrics[name] = CriticPerformanceMetrics(
                critic_name=name,
                total_evaluations=data.get("total_evaluations", 0),
                correct_predictions=data.get("correct_predictions", 0),
                false_positives=data.get("false_positives", 0),
                false_negatives=data.get("false_negatives", 0),
                average_severity_accuracy=data.get("average_severity_accuracy", 0.0),
                human_override_count=data.get("human_override_count", 0),
                precedent_alignment_score=data.get("precedent_alignment_score", 0.0),
                last_updated=data.get("last_updated", time.time()),
            )
        
        config = state.get("config", {})
        self.learning_rate = config.get("learning_rate", self.learning_rate)
        self.exploration_rate = config.get("exploration_rate", self.exploration_rate)
        self.decay_factor = config.get("decay_factor", self.decay_factor)
        self.min_weight = config.get("min_weight", self.min_weight)
        self.max_weight = config.get("max_weight", self.max_weight)
        self.min_samples = config.get("min_samples", self.min_samples)
