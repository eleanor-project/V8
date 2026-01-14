"""
ELEANOR V8 — Streaming Governance
----------------------------------

Provides incremental governance decisions during streaming execution.
Enables real-time governance evaluation as pipeline stages complete,
allowing early decisions and faster feedback loops.

Features:
- Incremental governance evaluation
- Early decision signals (allow/deny/escalate)
- Confidence-based progressive decisions
- Integration with WebSocket streaming
- Real-time escalation detection
"""

import time
from typing import Dict, Any, List, Optional, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class GovernanceSignal(str, Enum):
    """Governance decision signals."""
    PRELIMINARY_ALLOW = "preliminary_allow"
    PRELIMINARY_DENY = "preliminary_deny"
    PRELIMINARY_ESCALATE = "preliminary_escalate"
    CONFIRMED_ALLOW = "confirmed_allow"
    CONFIRMED_DENY = "confirmed_deny"
    CONFIRMED_ESCALATE = "confirmed_escalate"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass
class IncrementalDecision:
    """Represents an incremental governance decision."""
    signal: GovernanceSignal
    confidence: float  # 0.0 to 1.0
    stage: str  # e.g., "critics_partial", "precedent_alignment", "aggregation"
    trace_id: str
    timestamp: float
    rationale: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    requires_confirmation: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal": self.signal.value,
            "confidence": self.confidence,
            "stage": self.stage,
            "trace_id": self.trace_id,
            "timestamp": self.timestamp,
            "rationale": self.rationale,
            "evidence": self.evidence,
            "requires_confirmation": self.requires_confirmation,
        }


class StreamingGovernance:
    """
    Provides incremental governance decisions during streaming execution.
    """
    
    def __init__(
        self,
        early_decision_threshold: float = 0.85,
        deny_threshold: float = 0.7,
        escalation_threshold: float = 0.6,
    ):
        """
        Initialize streaming governance.
        
        Args:
            early_decision_threshold: Confidence threshold for early decisions (default 0.85)
            deny_threshold: Threshold for denying based on partial evidence (default 0.7)
            escalation_threshold: Threshold for escalating based on partial evidence (default 0.6)
        """
        self.early_decision_threshold = early_decision_threshold
        self.deny_threshold = deny_threshold
        self.escalation_threshold = escalation_threshold
        self._partial_decisions: Dict[str, List[IncrementalDecision]] = {}
    
    async def evaluate_incremental(
        self,
        trace_id: str,
        stage: str,
        current_data: Dict[str, Any],
        previous_decisions: Optional[List[IncrementalDecision]] = None,
    ) -> Optional[IncrementalDecision]:
        """
        Evaluate governance incrementally based on current stage data.
        
        Args:
            trace_id: Trace identifier
            stage: Current pipeline stage (e.g., "critics_partial", "precedent_alignment")
            current_data: Current stage data (critic results, precedent alignment, etc.)
            previous_decisions: Previous incremental decisions for this trace
            
        Returns:
            IncrementalDecision if a decision can be made, None otherwise
        """
        previous = previous_decisions or []
        
        # Track decisions for this trace
        if trace_id not in self._partial_decisions:
            self._partial_decisions[trace_id] = []
        
        decision = None
        
        # Stage-specific evaluation
        if stage == "critics_partial":
            decision = self._evaluate_critics_partial(trace_id, current_data, previous)
        elif stage == "critics_complete":
            decision = self._evaluate_critics_complete(trace_id, current_data, previous)
        elif stage == "precedent_alignment":
            decision = self._evaluate_precedent_alignment(trace_id, current_data, previous)
        elif stage == "aggregation":
            decision = self._evaluate_aggregation(trace_id, current_data, previous)
        
        if decision:
            self._partial_decisions[trace_id].append(decision)
        
        return decision
    
    def _evaluate_critics_partial(
        self,
        trace_id: str,
        data: Dict[str, Any],
        previous: List[IncrementalDecision],
    ) -> Optional[IncrementalDecision]:
        """Evaluate governance based on partial critic results."""
        critic_results = data.get("critic_results", {})
        
        if not critic_results:
            return None
        
        # Calculate aggregate violation score from available critics
        total_score = 0.0
        violation_count = 0
        total_rules = 0
        
        for critic_name, result in critic_results.items():
            if isinstance(result, dict):
                violations = result.get("violations", [])
                evaluated_rules = result.get("evaluated_rules", 0)
                
                violation_count += len(violations)
                total_rules += evaluated_rules
                
                # Weight by critic importance (could be configurable)
                score = result.get("score", 0.0)
                total_score += score
        
        # Early denial if strong violations detected
        if violation_count > 0 and total_rules > 0:
            violation_ratio = violation_count / total_rules
            avg_score = total_score / len(critic_results) if critic_results else 0.0
            
            # High confidence deny if violation ratio is high or score is very low
            if violation_ratio > 0.5 or avg_score < 0.3:
                confidence = min(0.95, 0.6 + (violation_ratio * 0.35))
                if confidence >= self.deny_threshold:
                    return IncrementalDecision(
                        signal=GovernanceSignal.PRELIMINARY_DENY,
                        confidence=confidence,
                        stage="critics_partial",
                        trace_id=trace_id,
                        timestamp=time.time(),
                        rationale=f"Strong violation signals detected: {violation_count} violations across {len(critic_results)} critics",
                        evidence={
                            "violation_count": violation_count,
                            "total_rules": total_rules,
                            "violation_ratio": violation_ratio,
                            "avg_critic_score": avg_score,
                        },
                        requires_confirmation=True,
                    )
        
        # Early escalation if uncertainty is high
        if len(critic_results) >= 2:
            # Check for significant disagreement among critics
            scores = [
                r.get("score", 0.5) for r in critic_results.values()
                if isinstance(r, dict)
            ]
            if scores:
                score_variance = self._calculate_variance(scores)
                if score_variance > 0.3:  # High disagreement
                    return IncrementalDecision(
                        signal=GovernanceSignal.PRELIMINARY_ESCALATE,
                        confidence=0.65,
                        stage="critics_partial",
                        trace_id=trace_id,
                        timestamp=time.time(),
                        rationale="High disagreement among critics - requires human review",
                        evidence={
                            "critic_count": len(critic_results),
                            "score_variance": score_variance,
                            "scores": scores,
                        },
                        requires_confirmation=True,
                    )
        
        return None
    
    def _evaluate_critics_complete(
        self,
        trace_id: str,
        data: Dict[str, Any],
        previous: List[IncrementalDecision],
    ) -> Optional[IncrementalDecision]:
        """Evaluate governance after all critics complete."""
        critic_results = data.get("critic_results", {})
        
        if not critic_results:
            return None
        
        # Calculate comprehensive violation analysis
        total_violations = 0
        total_rules = 0
        scores = []
        critical_violations = []
        
        for critic_name, result in critic_results.items():
            if isinstance(result, dict):
                violations = result.get("violations", [])
                evaluated_rules = result.get("evaluated_rules", 0)
                score = result.get("score", 0.5)
                
                total_violations += len(violations)
                total_rules += evaluated_rules
                scores.append(score)
                
                # Identify critical violations (could be enhanced with severity)
                if len(violations) > 0:
                    critical_violations.append({
                        "critic": critic_name,
                        "violation_count": len(violations),
                        "violations": violations[:3],  # Limit for efficiency
                    })
        
        if total_rules == 0:
            return None
        
        avg_score = sum(scores) / len(scores) if scores else 0.5
        violation_ratio = total_violations / total_rules
        
        # Strong denial signal
        if violation_ratio > 0.4 or avg_score < 0.4:
            confidence = min(0.95, 0.7 + (violation_ratio * 0.25))
            return IncrementalDecision(
                signal=GovernanceSignal.PRELIMINARY_DENY,
                confidence=confidence,
                stage="critics_complete",
                trace_id=trace_id,
                timestamp=time.time(),
                rationale=f"Comprehensive violation analysis: {total_violations} violations detected with average score {avg_score:.2f}",
                evidence={
                    "total_violations": total_violations,
                    "total_rules": total_rules,
                    "violation_ratio": violation_ratio,
                    "avg_score": avg_score,
                    "critical_violations": critical_violations[:5],
                },
                requires_confirmation=True,
            )
        
        # Escalation for high uncertainty or disagreement
        if len(scores) > 1:
            score_variance = self._calculate_variance(scores)
            if score_variance > 0.25 or (0.4 <= avg_score <= 0.6):
                return IncrementalDecision(
                    signal=GovernanceSignal.PRELIMINARY_ESCALATE,
                    confidence=0.7,
                    stage="critics_complete",
                    trace_id=trace_id,
                    timestamp=time.time(),
                    rationale=f"High uncertainty: score variance {score_variance:.3f}, avg score {avg_score:.2f}",
                    evidence={
                        "score_variance": score_variance,
                        "avg_score": avg_score,
                        "scores": scores,
                    },
                    requires_confirmation=True,
                )
        
        # Preliminary allow (requires confirmation after aggregation)
        if avg_score > 0.75 and violation_ratio < 0.1:
            confidence = min(0.85, 0.5 + (avg_score * 0.35))
            return IncrementalDecision(
                signal=GovernanceSignal.PRELIMINARY_ALLOW,
                confidence=confidence,
                stage="critics_complete",
                trace_id=trace_id,
                timestamp=time.time(),
                rationale=f"Strong positive signals: avg score {avg_score:.2f}, low violation rate",
                evidence={
                    "avg_score": avg_score,
                    "violation_ratio": violation_ratio,
                    "total_violations": total_violations,
                },
                requires_confirmation=True,
            )
        
        return None
    
    def _evaluate_precedent_alignment(
        self,
        trace_id: str,
        data: Dict[str, Any],
        previous: List[IncrementalDecision],
    ) -> Optional[IncrementalDecision]:
        """Evaluate governance based on precedent alignment."""
        alignment_score = data.get("alignment_score")
        similar_cases = data.get("similar_cases", [])
        
        if alignment_score is None:
            return None
        
        # Check if precedent alignment reinforces or contradicts previous signals
        previous_signal = None
        if previous:
            previous_signal = previous[-1].signal
        
        # Strong misalignment suggests deny
        if alignment_score < 0.3:
            confidence = 0.75
            if previous_signal == GovernanceSignal.PRELIMINARY_DENY:
                confidence = 0.9  # Reinforced
            
            return IncrementalDecision(
                signal=GovernanceSignal.PRELIMINARY_DENY,
                confidence=confidence,
                stage="precedent_alignment",
                trace_id=trace_id,
                timestamp=time.time(),
                rationale=f"Low precedent alignment ({alignment_score:.2f}) - inconsistent with historical decisions",
                evidence={
                    "alignment_score": alignment_score,
                    "similar_cases_count": len(similar_cases),
                    "previous_signal": previous_signal.value if previous_signal else None,
                },
                requires_confirmation=True,
            )
        
        # Very high alignment might confirm allow
        if alignment_score > 0.85:
            confidence = 0.8
            if previous_signal == GovernanceSignal.PRELIMINARY_ALLOW:
                confidence = 0.92  # Reinforced
            
            return IncrementalDecision(
                signal=GovernanceSignal.PRELIMINARY_ALLOW,
                confidence=confidence,
                stage="precedent_alignment",
                trace_id=trace_id,
                timestamp=time.time(),
                rationale=f"High precedent alignment ({alignment_score:.2f}) - consistent with historical decisions",
                evidence={
                    "alignment_score": alignment_score,
                    "similar_cases_count": len(similar_cases),
                    "previous_signal": previous_signal.value if previous_signal else None,
                },
                requires_confirmation=True,
            )
        
        return None
    
    def _evaluate_aggregation(
        self,
        trace_id: str,
        data: Dict[str, Any],
        previous: List[IncrementalDecision],
    ) -> Optional[IncrementalDecision]:
        """Evaluate governance after aggregation completes."""
        decision = data.get("decision")
        aggregated_score = data.get("aggregated_score")
        confidence_metrics = data.get("confidence", {})
        
        if decision is None:
            return None
        
        # This is the final stage - convert to confirmed signal
        signal_map = {
            "allow": GovernanceSignal.CONFIRMED_ALLOW,
            "deny": GovernanceSignal.CONFIRMED_DENY,
            "escalate": GovernanceSignal.CONFIRMED_ESCALATE,
            "constrained_allow": GovernanceSignal.CONFIRMED_ALLOW,
        }
        
        confirmed_signal = signal_map.get(decision.lower(), GovernanceSignal.PRELIMINARY_ESCALATE)
        
        # Calculate final confidence from aggregation metrics
        confidence = confidence_metrics.get("overall", 0.8)
        if aggregated_score is not None:
            # Blend aggregated score into confidence
            confidence = (confidence * 0.6) + (aggregated_score * 0.4)
        
        # Check if previous decisions align
        previous_signal = None
        if previous:
            previous_signal = previous[-1].signal
        
        # If previous signal aligns, increase confidence
        if previous_signal:
            aligns = (
                (previous_signal == GovernanceSignal.PRELIMINARY_ALLOW and confirmed_signal == GovernanceSignal.CONFIRMED_ALLOW) or
                (previous_signal == GovernanceSignal.PRELIMINARY_DENY and confirmed_signal == GovernanceSignal.CONFIRMED_DENY) or
                (previous_signal == GovernanceSignal.PRELIMINARY_ESCALATE and confirmed_signal == GovernanceSignal.CONFIRMED_ESCALATE)
            )
            if aligns:
                confidence = min(0.98, confidence + 0.1)
        
        rationale = f"Final aggregation decision: {decision}"
        if aggregated_score is not None:
            rationale += f" (score: {aggregated_score:.2f})"
        if previous_signal:
            rationale += f" - {'reinforces' if aligns else 'overrides'} previous {previous_signal.value}"
        
        return IncrementalDecision(
            signal=confirmed_signal,
            confidence=confidence,
            stage="aggregation",
            trace_id=trace_id,
            timestamp=time.time(),
            rationale=rationale,
            evidence={
                "decision": decision,
                "aggregated_score": aggregated_score,
                "confidence_metrics": confidence_metrics,
                "previous_signals": [p.signal.value for p in previous],
            },
            requires_confirmation=False,  # Final decision
        )
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values."""
        if not values or len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance
    
    def get_decision_history(self, trace_id: str) -> List[IncrementalDecision]:
        """Get the decision history for a trace."""
        return self._partial_decisions.get(trace_id, [])
    
    def clear_history(self, trace_id: str) -> None:
        """Clear decision history for a trace (after completion)."""
        self._partial_decisions.pop(trace_id, None)
    
    async def stream_governance_decisions(
        self,
        trace_id: str,
        event_stream: AsyncGenerator[Dict[str, Any], None],
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream governance decisions alongside pipeline events.
        
        Args:
            trace_id: Trace identifier
            event_stream: Async generator of pipeline events
            
        Yields:
            Original events with governance decision annotations
        """
        async for event in event_stream:
            event_type = event.get("event")
            
            # Extract stage data and evaluate
            stage_data = {}
            stage_name = None
            
            if event_type == "critic_result":
                # Collect partial critic results
                stage_name = "critics_partial"
                # In a real implementation, we'd collect all critic results here
                stage_data = {"critic_results": {event.get("critic"): event}}
            elif event_type == "critics_complete":
                stage_name = "critics_complete"
                # Would need access to all critic results - simplified for now
                stage_data = {"critic_results": {}}
            elif event_type == "precedent_alignment":
                stage_name = "precedent_alignment"
                stage_data = event.get("data", {})
            elif event_type == "aggregation":
                stage_name = "aggregation"
                stage_data = event.get("data", {})
            
            # Evaluate incremental governance if we have stage data
            if stage_name and stage_data:
                previous = self.get_decision_history(trace_id)
                decision = await self.evaluate_incremental(trace_id, stage_name, stage_data, previous)
                
                if decision:
                    # Yield governance decision event
                    yield {
                        "event": "governance_decision",
                        "trace_id": trace_id,
                        "decision": decision.to_dict(),
                    }
            
            # Yield original event
            yield event
        
        # Clear history after completion
        self.clear_history(trace_id)
