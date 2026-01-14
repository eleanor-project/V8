#!/usr/bin/env python3
"""
ELEANOR V8 — Streaming Governance Demo
---------------------------------------

Demonstrates the Streaming Governance feature:
- Incremental governance decisions during pipeline execution
- Progressive decision signals (preliminary → confirmed)
- Confidence-based early decision making
- Real-time governance evaluation

Usage:
    python examples/streaming_governance_demo.py
"""

import asyncio
import time
import sys
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.governance.streaming import StreamingGovernance, GovernanceSignal


async def simulate_pipeline_stages(streaming_gov: StreamingGovernance, trace_id: str):
    """Simulate a pipeline execution with incremental governance evaluation."""
    
    decisions = []
    
    # Stage 1: Partial critic results (early signals)
    print("\n[Stage 1] Critics Partial - First few critic results available")
    print("-" * 70)
    
    partial_critics = {
        "critic_results": {
            "rights": {
                "score": 0.25,
                "violations": ["violation_1", "violation_2"],
                "evaluated_rules": 5,
            },
            "fairness": {
                "score": 0.30,
                "violations": ["violation_1"],
                "evaluated_rules": 4,
            },
        }
    }
    
    decision1 = await streaming_gov.evaluate_incremental(
        trace_id=trace_id,
        stage="critics_partial",
        current_data=partial_critics,
        previous_decisions=decisions,
    )
    
    if decision1:
        decisions.append(decision1)
        print(f"  Decision: {decision1.signal.value}")
        print(f"  Confidence: {decision1.confidence:.3f}")
        print(f"  Stage: {decision1.stage}")
        print(f"  Rationale: {decision1.rationale}")
        print(f"  Evidence: {decision1.evidence}")
    else:
        print("  No decision yet - insufficient evidence")
    
    await asyncio.sleep(0.5)
    
    # Stage 2: All critics complete
    print("\n[Stage 2] Critics Complete - All critic evaluations finished")
    print("-" * 70)
    
    complete_critics = {
        "critic_results": {
            "rights": {
                "score": 0.25,
                "violations": ["violation_1", "violation_2"],
                "evaluated_rules": 5,
            },
            "fairness": {
                "score": 0.30,
                "violations": ["violation_1"],
                "evaluated_rules": 4,
            },
            "autonomy": {
                "score": 0.85,
                "violations": [],
                "evaluated_rules": 3,
            },
            "risk": {
                "score": 0.40,
                "violations": ["high_risk"],
                "evaluated_rules": 6,
            },
        }
    }
    
    decision2 = await streaming_gov.evaluate_incremental(
        trace_id=trace_id,
        stage="critics_complete",
        current_data=complete_critics,
        previous_decisions=decisions,
    )
    
    if decision2:
        decisions.append(decision2)
        print(f"  Decision: {decision2.signal.value}")
        print(f"  Confidence: {decision2.confidence:.3f}")
        print(f"  Rationale: {decision2.rationale}")
        if decision2.evidence:
            print(f"  Evidence:")
            for key, value in decision2.evidence.items():
                if key != "critical_violations":
                    print(f"    {key}: {value}")
    
    await asyncio.sleep(0.5)
    
    # Stage 3: Precedent alignment
    print("\n[Stage 3] Precedent Alignment - Historical precedent analysis")
    print("-" * 70)
    
    precedent_data = {
        "alignment_score": 0.35,  # Low alignment - contradicts current decision
        "similar_cases": [
            {"case_id": "prev_001", "similarity": 0.85, "decision": "deny"},
            {"case_id": "prev_002", "similarity": 0.80, "decision": "deny"},
        ],
    }
    
    decision3 = await streaming_gov.evaluate_incremental(
        trace_id=trace_id,
        stage="precedent_alignment",
        current_data=precedent_data,
        previous_decisions=decisions,
    )
    
    if decision3:
        decisions.append(decision3)
        print(f"  Decision: {decision3.signal.value}")
        print(f"  Confidence: {decision3.confidence:.3f}")
        print(f"  Rationale: {decision3.rationale}")
        print(f"  Previous Signal: {decisions[-2].signal.value if len(decisions) > 1 else 'None'}")
        print(f"  Reinforcement: {decision3.signal == decisions[-2].signal if len(decisions) > 1 else 'N/A'}")
    else:
        print("  No change from previous decision")
    
    await asyncio.sleep(0.5)
    
    # Stage 4: Final aggregation
    print("\n[Stage 4] Aggregation - Final constitutional decision")
    print("-" * 70)
    
    aggregation_data = {
        "decision": "deny",
        "aggregated_score": 0.42,
        "confidence": {
            "overall": 0.88,
            "critics": 0.85,
            "precedent": 0.90,
        },
    }
    
    decision4 = await streaming_gov.evaluate_incremental(
        trace_id=trace_id,
        stage="aggregation",
        current_data=aggregation_data,
        previous_decisions=decisions,
    )
    
    if decision4:
        decisions.append(decision4)
        print(f"  Final Decision: {decision4.signal.value}")
        print(f"  Confidence: {decision4.confidence:.3f}")
        print(f"  Requires Confirmation: {decision4.requires_confirmation}")
        print(f"  Rationale: {decision4.rationale}")
    
    return decisions


async def simulate_allow_scenario(streaming_gov: StreamingGovernance, trace_id: str):
    """Simulate a scenario that results in an allow decision."""
    
    print("\n" + "=" * 70)
    print("SCENARIO 2: Strong Allow Decision")
    print("=" * 70)
    
    decisions = []
    
    # Strong positive signals
    complete_critics = {
        "critic_results": {
            "rights": {"score": 0.95, "violations": [], "evaluated_rules": 5},
            "fairness": {"score": 0.90, "violations": [], "evaluated_rules": 4},
            "autonomy": {"score": 0.92, "violations": [], "evaluated_rules": 3},
        }
    }
    
    decision = await streaming_gov.evaluate_incremental(
        trace_id=trace_id,
        stage="critics_complete",
        current_data=complete_critics,
        previous_decisions=[],
    )
    
    if decision:
        print(f"\nEarly Decision: {decision.signal.value}")
        print(f"Confidence: {decision.confidence:.3f}")
        print(f"Rationale: {decision.rationale}")
    
    # High precedent alignment
    precedent_data = {"alignment_score": 0.92, "similar_cases": []}
    decision2 = await streaming_gov.evaluate_incremental(
        trace_id=trace_id,
        stage="precedent_alignment",
        current_data=precedent_data,
        previous_decisions=[decision] if decision else [],
    )
    
    if decision2:
        print(f"\nAfter Precedent: {decision2.signal.value}")
        print(f"Confidence: {decision2.confidence:.3f}")
        print(f"Rationale: {decision2.rationale}")


async def simulate_escalation_scenario(streaming_gov: StreamingGovernance, trace_id: str):
    """Simulate a scenario that requires escalation."""
    
    print("\n" + "=" * 70)
    print("SCENARIO 3: Escalation Required (High Uncertainty)")
    print("=" * 70)
    
    # High disagreement among critics
    complete_critics = {
        "critic_results": {
            "rights": {"score": 0.95, "violations": [], "evaluated_rules": 5},
            "fairness": {"score": 0.30, "violations": ["discrimination"], "evaluated_rules": 4},
            "autonomy": {"score": 0.90, "violations": [], "evaluated_rules": 3},
            "risk": {"score": 0.35, "violations": ["high_risk"], "evaluated_rules": 6},
        }
    }
    
    decision = await streaming_gov.evaluate_incremental(
        trace_id=trace_id,
        stage="critics_complete",
        current_data=complete_critics,
        previous_decisions=[],
    )
    
    if decision:
        print(f"\nDecision: {decision.signal.value}")
        print(f"Confidence: {decision.confidence:.3f}")
        print(f"Rationale: {decision.rationale}")
        if decision.evidence:
            print(f"Score Variance: {decision.evidence.get('score_variance', 0):.3f}")
            print(f"Scores: {decision.evidence.get('scores', [])}")


async def main():
    print("=" * 70)
    print("ELEANOR V8 — Streaming Governance Demo")
    print("=" * 70)
    
    # Initialize streaming governance
    streaming_gov = StreamingGovernance(
        early_decision_threshold=0.85,
        deny_threshold=0.7,
        escalation_threshold=0.6,
    )
    
    # Scenario 1: Strong denial with early detection
    print("\nSCENARIO 1: Strong Denial with Early Detection")
    print("=" * 70)
    
    trace_id_1 = "demo_trace_001"
    decisions = await simulate_pipeline_stages(streaming_gov, trace_id_1)
    
    print("\n" + "=" * 70)
    print("Decision History Summary")
    print("-" * 70)
    for i, decision in enumerate(decisions, 1):
        print(f"\n{i}. {decision.stage}:")
        print(f"   Signal: {decision.signal.value}")
        print(f"   Confidence: {decision.confidence:.3f}")
        print(f"   Timestamp: {time.ctime(decision.timestamp)}")
    
    # Scenario 2: Strong allow
    trace_id_2 = "demo_trace_002"
    await simulate_allow_scenario(streaming_gov, trace_id_2)
    
    # Scenario 3: Escalation
    trace_id_3 = "demo_trace_003"
    await simulate_escalation_scenario(streaming_gov, trace_id_3)
    
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  - Streaming governance provides early decision signals")
    print("  - Confidence increases as more evidence accumulates")
    print("  - Preliminary decisions can guide user expectations")
    print("  - Final confirmed decisions incorporate all evidence")
    print("  - Perfect for real-time applications and WebSocket streaming")


if __name__ == "__main__":
    asyncio.run(main())
