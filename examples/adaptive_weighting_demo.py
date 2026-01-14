#!/usr/bin/env python3
"""
ELEANOR V8 — Adaptive Critic Weighting Demo
-------------------------------------------

Demonstrates the Adaptive Critic Weighting feature:
- Learn optimal critic weights from historical performance
- Track performance metrics (accuracy, precision, recall, F1)
- Update weights based on feedback
- Generate performance reports

Usage:
    python examples/adaptive_weighting_demo.py
"""

import asyncio
import time
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.aggregator.adaptive_weighting import AdaptiveCriticWeighting


async def simulate_feedback_cycles(adaptive_weighting: AdaptiveCriticWeighting, num_cycles: int = 20):
    """Simulate multiple decision cycles with feedback."""
    
    print(f"\nSimulating {num_cycles} decision cycles with feedback...")
    print("-" * 70)
    
    # Simulate scenarios where different critics perform differently
    scenarios = [
        # Scenario: Rights critic is highly accurate
        {
            "critic_results": {
                "rights": {"score": 0.85, "violations": []},
                "fairness": {"score": 0.60, "violations": []},
            },
            "final_decision": "allow",
            "human_decision": "allow",  # Human confirms
        },
        # Scenario: Fairness critic catches discrimination
        {
            "critic_results": {
                "rights": {"score": 0.75, "violations": []},
                "fairness": {"score": 0.25, "violations": ["discrimination"]},
            },
            "final_decision": "deny",
            "human_decision": "deny",  # Human confirms
        },
        # Scenario: Rights critic makes false positive
        {
            "critic_results": {
                "rights": {"score": 0.30, "violations": ["false_positive"]},
                "fairness": {"score": 0.80, "violations": []},
            },
            "final_decision": "deny",
            "human_decision": "allow",  # Human overrides - rights was wrong
        },
        # Scenario: Autonomy critic highly accurate
        {
            "critic_results": {
                "rights": {"score": 0.70, "violations": []},
                "autonomy": {"score": 0.90, "violations": []},
                "fairness": {"score": 0.65, "violations": []},
            },
            "final_decision": "allow",
            "human_decision": "allow",
        },
    ]
    
    for cycle in range(num_cycles):
        scenario = scenarios[cycle % len(scenarios)]
        trace_id = f"demo_trace_{cycle:04d}"
        
        # Record feedback
        adaptive_weighting.record_feedback(
            trace_id=trace_id,
            critic_results=scenario["critic_results"],
            final_decision=scenario["final_decision"],
            human_review_decision=scenario["human_decision"],
            human_corrected=(scenario["final_decision"] != scenario["human_decision"]),
            precedent_alignment={"alignment_score": 0.75 + (cycle % 4) * 0.05},
        )
        
        if (cycle + 1) % 5 == 0:
            print(f"  Completed {cycle + 1} cycles...")
    
    print(f"✓ Completed {num_cycles} feedback cycles")


async def main():
    print("=" * 70)
    print("ELEANOR V8 — Adaptive Critic Weighting Demo")
    print("=" * 70)
    
    # Initialize adaptive weighting
    adaptive_weighting = AdaptiveCriticWeighting(
        learning_rate=0.15,      # Higher learning rate for demo
        exploration_rate=0.05,
        decay_factor=0.95,
        min_weight=0.1,
        max_weight=2.0,
        min_samples=5,           # Lower threshold for demo
    )
    
    print("\n1. Initial Weights")
    print("-" * 70)
    initial_weights = adaptive_weighting.get_weights()
    print("\nInitial critic weights (uniform):")
    for critic, weight in sorted(initial_weights.items()):
        print(f"  {critic:20s}: {weight:.3f}")
    
    # Simulate feedback cycles
    print("\n" + "=" * 70)
    print("2. Simulating Feedback Cycles")
    print("=" * 70)
    
    await simulate_feedback_cycles(adaptive_weighting, num_cycles=25)
    
    # Check performance before weight update
    print("\n" + "=" * 70)
    print("3. Performance Metrics (Before Weight Update)")
    print("-" * 70)
    
    report_before = adaptive_weighting.get_performance_report()
    
    if report_before.get("critic_metrics"):
        print("\nCritic Performance Metrics:")
        for critic, metrics in sorted(report_before["critic_metrics"].items()):
            print(f"\n  {critic}:")
            print(f"    Total Evaluations: {metrics['total_evaluations']}")
            print(f"    Accuracy:          {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
            print(f"    Precision:         {metrics['precision']:.3f}")
            print(f"    Recall:            {metrics['recall']:.3f}")
            print(f"    F1 Score:          {metrics['f1_score']:.3f}")
            print(f"    False Positives:   {metrics['false_positives']}")
            print(f"    False Negatives:   {metrics['false_negatives']}")
            print(f"    Human Overrides:   {metrics['human_override_count']}")
    
    # Update weights
    print("\n" + "=" * 70)
    print("4. Updating Weights Based on Performance")
    print("-" * 70)
    
    updates = adaptive_weighting.update_weights()
    
    if updates:
        print(f"\nApplied {len(updates)} weight updates:")
        for update in updates:
            change = update.new_weight - update.old_weight
            change_pct = (change / update.old_weight) * 100 if update.old_weight > 0 else 0
            print(f"\n  {update.critic_name}:")
            print(f"    Old Weight:    {update.old_weight:.3f}")
            print(f"    New Weight:    {update.new_weight:.3f}")
            print(f"    Change:        {change:+.3f} ({change_pct:+.1f}%)")
            print(f"    Reason:        {update.update_reason}")
            print(f"    Confidence:    {update.confidence:.3f}")
    else:
        print("\nNo weight updates applied (need more samples or weights already optimal)")
    
    # Show updated weights
    print("\n" + "=" * 70)
    print("5. Updated Weights")
    print("-" * 70)
    
    updated_weights = adaptive_weighting.get_weights()
    print("\nCurrent critic weights (after learning):")
    for critic, weight in sorted(updated_weights.items()):
        if critic in initial_weights:
            change = weight - initial_weights[critic]
            change_pct = (change / initial_weights[critic]) * 100 if initial_weights[critic] > 0 else 0
            indicator = "↑" if change > 0.01 else "↓" if change < -0.01 else "→"
            print(f"  {critic:20s}: {weight:.3f} {indicator} ({change:+.3f}, {change_pct:+.1f}%)")
        else:
            print(f"  {critic:20s}: {weight:.3f}")
    
    # Get final performance report
    print("\n" + "=" * 70)
    print("6. Final Performance Report")
    print("-" * 70)
    
    final_report = adaptive_weighting.get_performance_report()
    
    if final_report.get("summary"):
        summary = final_report["summary"]
        print(f"\nSummary:")
        print(f"  Total Feedback Samples: {summary['total_feedback_samples']}")
        print(f"  Total Weight Updates:   {summary['total_weight_updates']}")
        print(f"  Critics Tracked:        {summary['critics_tracked']}")
    
    # Demonstrate weighted scoring
    print("\n" + "=" * 70)
    print("7. Weighted Score Calculation")
    print("-" * 70)
    
    sample_results = {
        "rights": {"score": 0.80},
        "fairness": {"score": 0.70},
        "autonomy": {"score": 0.85},
    }
    
    weighted_scores = adaptive_weighting.get_weighted_scores(sample_results)
    
    print("\nRaw vs Weighted Scores:")
    print(f"  {'Critic':20s} {'Raw Score':12s} {'Weight':12s} {'Weighted Score':15s}")
    print("  " + "-" * 60)
    
    for critic, result in sample_results.items():
        raw = result["score"]
        weight = updated_weights.get(critic, 1.0)
        weighted = weighted_scores.get(critic, raw)
        print(f"  {critic:20s} {raw:12.3f} {weight:12.3f} {weighted:15.3f}")
    
    print("\n" + "=" * 70)
    print("8. Export/Import State (Persistence)")
    print("-" * 70)
    
    # Export state
    state = adaptive_weighting.export_state()
    print(f"\nExported state (can be persisted to database/file):")
    print(f"  Weights: {len(state['weights'])} critics")
    print(f"  Metrics: {len(state['metrics'])} critics tracked")
    print(f"  Config:  {len(state['config'])} parameters")
    
    # Demonstrate reset
    print("\n" + "=" * 70)
    print("9. Reset Weights (Optional)")
    print("-" * 70)
    
    print("\nCurrent weights before reset:")
    for critic, weight in sorted(adaptive_weighting.get_weights().items()):
        print(f"  {critic:20s}: {weight:.3f}")
    
    # Uncomment to actually reset:
    # adaptive_weighting.reset_weights()
    # print("\n✓ Weights reset to uniform (1.0)")
    
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  - Adaptive weighting learns from historical performance")
    print("  - High-performing critics get higher weights over time")
    print("  - Human feedback is crucial for accurate learning")
    print("  - Weights adapt to your specific use case and context")
    print("  - Performance metrics provide insights into critic quality")
    print("  - State can be exported/imported for persistence")


if __name__ == "__main__":
    asyncio.run(main())
