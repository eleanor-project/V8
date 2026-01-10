#!/usr/bin/env python3
"""
ELEANOR V8 — Temporal Precedent Evolution Demo
-----------------------------------------------

Demonstrates the Temporal Precedent Evolution Tracking feature:
- Track precedent updates over time
- Detect temporal drift
- Manage precedent lifecycle
- Get evolution analytics
- Receive deprecation recommendations

Usage:
    python examples/temporal_evolution_demo.py
"""

import asyncio
import time
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.precedent.temporal_evolution import (
    TemporalPrecedentEvolutionTracker,
    PrecedentLifecycleState,
)
from engine.precedent.store import PrecedentCase


async def main():
    print("=" * 70)
    print("ELEANOR V8 — Temporal Precedent Evolution Demo")
    print("=" * 70)
    print()

    # Initialize tracker (in production, this would use a persistent store backend)
    tracker = TemporalPrecedentEvolutionTracker()

    # Simulate creating and updating a precedent over time
    case_id = "demo_case_001"
    
    print("1. Tracking Precedent Updates Over Time")
    print("-" * 70)
    
    # Initial version - strong allow decision
    print(f"\nInitial version (t=0): Strong allow decision")
    version1 = tracker.track_precedent_update(
        case_id=case_id,
        decision="allow",
        aggregate_score=0.92,
        values=["autonomy", "fairness"],
        rationale="Strong autonomy considerations, no violations detected",
        critic_outputs={
            "rights": {"score": 0.95, "violations": []},
            "fairness": {"score": 0.90, "violations": []},
        },
        metadata={"source": "demo", "iteration": 1},
    )
    print(f"   Created version: {version1.version_id}")
    
    await asyncio.sleep(0.1)  # Simulate time passing
    
    # Second version - slightly lower score
    print(f"\nUpdate (t=1): Slightly lower score, same decision")
    version2 = tracker.track_precedent_update(
        case_id=case_id,
        decision="allow",
        aggregate_score=0.88,
        values=["autonomy", "fairness"],
        rationale="Strong autonomy, minor fairness concerns",
        critic_outputs={
            "rights": {"score": 0.93, "violations": []},
            "fairness": {"score": 0.85, "violations": []},
        },
        metadata={"source": "demo", "iteration": 2},
    )
    print(f"   Created version: {version2.version_id}")
    
    await asyncio.sleep(0.1)
    
    # Third version - decision changes to constrained_allow
    print(f"\nUpdate (t=2): Decision changes to constrained_allow")
    version3 = tracker.track_precedent_update(
        case_id=case_id,
        decision="constrained_allow",
        aggregate_score=0.75,
        values=["autonomy", "fairness", "risk"],
        rationale="Autonomy present but risk concerns require constraints",
        critic_outputs={
            "rights": {"score": 0.88, "violations": []},
            "fairness": {"score": 0.80, "violations": []},
            "risk": {"score": 0.65, "violations": ["potential_harm"]},
        },
        metadata={"source": "demo", "iteration": 3},
    )
    print(f"   Created version: {version3.version_id}")
    
    await asyncio.sleep(0.1)
    
    # Fourth version - decision changes to deny
    print(f"\nUpdate (t=3): Decision changes to deny (major drift)")
    version4 = tracker.track_precedent_update(
        case_id=case_id,
        decision="deny",
        aggregate_score=0.45,
        values=["risk", "fairness"],
        rationale="Risk concerns outweigh autonomy, denial required",
        critic_outputs={
            "rights": {"score": 0.70, "violations": []},
            "fairness": {"score": 0.60, "violations": ["discrimination_risk"]},
            "risk": {"score": 0.35, "violations": ["high_risk", "harm_potential"]},
        },
        metadata={"source": "demo", "iteration": 4},
    )
    print(f"   Created version: {version4.version_id}")
    
    print("\n" + "=" * 70)
    print("2. Evolution Analytics")
    print("-" * 70)
    
    # Get analytics for this specific case
    analytics = tracker.get_evolution_analytics(case_id=case_id)
    print(f"\nCase Analytics for {case_id}:")
    print(f"  Lifecycle State: {analytics['lifecycle_state']}")
    print(f"  Version Count: {analytics['version_count']}")
    print(f"  Created At: {time.ctime(analytics['created_at'])}")
    print(f"  Updated At: {time.ctime(analytics['updated_at'])}")
    
    if analytics.get('drift_metrics'):
        drift = analytics['drift_metrics']
        print(f"\n  Drift Metrics:")
        print(f"    Drift Detected: {drift.get('drift_detected', False)}")
        print(f"    Drift Score: {drift.get('drift_score', 0):.3f}")
        print(f"    Score Variance: {drift.get('score_variance', 0):.3f}")
        print(f"    Score Trend: {drift.get('score_trend', 0):.3f}")
        print(f"    Decision Consistency: {drift.get('decision_consistency', 0):.3f}")
        print(f"    Decision Changes: {drift.get('decision_changes', 0)}")
    
    print("\n" + "=" * 70)
    print("3. Temporal Drift Detection")
    print("-" * 70)
    
    drift_info = tracker.detect_temporal_drift(case_id)
    print(f"\nDrift Analysis for {case_id}:")
    print(f"  Drift Detected: {drift_info['drift_detected']}")
    print(f"  Drift Score: {drift_info['drift_score']:.3f}")
    print(f"  Message: {drift_info['message']}")
    print(f"  Versions Analyzed: {drift_info['versions_analyzed']}")
    
    print("\n" + "=" * 70)
    print("4. Lifecycle Management")
    print("-" * 70)
    
    # Get evolution record
    evolution = tracker.get_evolution(case_id)
    if evolution:
        print(f"\nCurrent State: {evolution.lifecycle_state.value}")
        
        # Deprecate the precedent due to drift
        print(f"\nDeprecating precedent due to high drift...")
        success = tracker.set_lifecycle_state(
            case_id=case_id,
            state=PrecedentLifecycleState.DEPRECATED
        )
        print(f"  Success: {success}")
        
        # Check updated state
        updated_evolution = tracker.get_evolution(case_id)
        if updated_evolution:
            print(f"  New State: {updated_evolution.lifecycle_state.value}")
            print(f"  Deprecated At: {time.ctime(updated_evolution.deprecated_at) if updated_evolution.deprecated_at else 'N/A'}")
    
    print("\n" + "=" * 70)
    print("5. Aggregate Analytics")
    print("-" * 70)
    
    # Create a few more precedents for aggregate analytics
    for i in range(2, 5):
        other_case_id = f"demo_case_{i:03d}"
        tracker.track_precedent_update(
            case_id=other_case_id,
            decision="allow" if i % 2 == 0 else "deny",
            aggregate_score=0.7 + (i * 0.05),
            values=["autonomy"],
            rationale=f"Demo case {i}",
            critic_outputs={"rights": {"score": 0.75}},
            metadata={"source": "demo"},
        )
    
    aggregate_analytics = tracker.get_evolution_analytics()
    print(f"\nAggregate Analytics:")
    print(f"  Total Precedents: {aggregate_analytics['total_precedents']}")
    print(f"  Average Versions: {aggregate_analytics['average_versions_per_precedent']:.2f}")
    print(f"  Total Versions: {aggregate_analytics['total_versions']}")
    
    if aggregate_analytics.get('lifecycle_distribution'):
        print(f"\n  Lifecycle Distribution:")
        for state, count in aggregate_analytics['lifecycle_distribution'].items():
            print(f"    {state}: {count}")
    
    if aggregate_analytics.get('drift_distribution'):
        print(f"\n  Drift Distribution:")
        for level, count in aggregate_analytics['drift_distribution'].items():
            print(f"    {level}: {count}")
    
    print("\n" + "=" * 70)
    print("6. Deprecation Recommendations")
    print("-" * 70)
    
    recommendations = tracker.recommend_deprecations(min_versions=3)
    print(f"\nFound {len(recommendations)} deprecation recommendations:")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n  Recommendation {i}:")
        print(f"    Case ID: {rec['case_id']}")
        print(f"    Reason: {rec['reason']}")
        print(f"    Drift Score: {rec['drift_metrics']['drift_score']:.3f}")
        print(f"    Version Count: {rec['version_count']}")
    
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  - Temporal evolution tracking maintains complete version history")
    print("  - Drift detection identifies when precedents deviate from original patterns")
    print("  - Lifecycle management enables structured precedent governance")
    print("  - Analytics provide insights into precedent quality and stability")
    print("  - Automatic recommendations help identify precedents needing review")


if __name__ == "__main__":
    asyncio.run(main())
