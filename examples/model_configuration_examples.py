"""
ELEANOR V8 — Model Configuration Examples
------------------------------------------

Demonstrates the hybrid model configuration approach.

Three ways to configure models:
1. Explicit (preferred model in __init__)
2. Registry (centralized configuration)
3. Runtime (model passed to evaluate())

Priority: runtime > explicit > registry > error
"""

import asyncio
from engine.critics.rights import RightsCriticV8
from engine.models import ModelRegistry, ModelTier
from engine.models import cost_optimizer_strategy, priority_based_strategy


# ============================================================================
# Example 1: Explicit Model Configuration (Option 1)
# ============================================================================


async def example_explicit_model():
    """Configure critic with explicit preferred model."""
    print("=" * 60)
    print("Example 1: Explicit Model Configuration")
    print("=" * 60)

    # Mock model for demonstration
    class MockOpusModel:
        async def generate(self, text, context=None):
            return f"Opus response to: {text[:50]}"

    # Create critic with explicit model
    opus_model = MockOpusModel()
    critic = RightsCriticV8(model=opus_model)

    print(f"✓ Critic '{critic.name}' configured with explicit model")
    print(f"  Model: {opus_model.__class__.__name__}")
    print("  This critic will always use OpusModel unless overridden at runtime")
    print()


# ============================================================================
# Example 2: Registry-Based Configuration (Option 3)
# ============================================================================


async def example_registry_basic():
    """Configure critics using ModelRegistry."""
    print("=" * 60)
    print("Example 2: Registry-Based Configuration (Basic)")
    print("=" * 60)

    # Create registry with default model
    registry = ModelRegistry(default_model_id="claude-sonnet-4.5")

    # Configure specific critics
    registry.assign_model("rights", "claude-opus-4.5")
    registry.assign_model("fairness", "claude-opus-4.5")
    registry.assign_model("truth", "gpt-4")

    # Create critics with registry
    rights_critic = RightsCriticV8(registry=registry)

    # Get model for critic
    model_id = rights_critic.get_model(context={})

    print(f"✓ Registry configured with {len(registry.critic_models)} explicit assignments")
    print(f"  Rights critic will use: {model_id}")
    print(f"  Default model: {registry.default_model_id}")
    print()

    # List all assignments
    print("All critic assignments:")
    for critic_name, details in registry.list_assignments().items():
        print(f"  • {critic_name}: {details['model']} (via {details['source']})")
    print()


# ============================================================================
# Example 3: Tier-Based Configuration
# ============================================================================


async def example_registry_tiers():
    """Use performance tiers instead of explicit models."""
    print("=" * 60)
    print("Example 3: Tier-Based Configuration")
    print("=" * 60)

    registry = ModelRegistry()

    # Configure tiers
    registry.set_tier_model(ModelTier.PREMIUM, "claude-opus-4.5")
    registry.set_tier_model(ModelTier.STANDARD, "claude-sonnet-4.5")
    registry.set_tier_model(ModelTier.ECONOMY, "claude-haiku-4.0")

    # Assign critics to tiers
    registry.assign_tier("rights", ModelTier.PREMIUM)
    registry.assign_tier("fairness", ModelTier.PREMIUM)
    registry.assign_tier("truth", ModelTier.PREMIUM)
    registry.assign_tier("risk", ModelTier.STANDARD)
    registry.assign_tier("operations", ModelTier.STANDARD)

    print("✓ Configured tier-based routing:")
    print(f"  Premium tier: {registry.tier_models[ModelTier.PREMIUM]}")
    print(f"  Standard tier: {registry.tier_models[ModelTier.STANDARD]}")
    print(f"  Economy tier: {registry.tier_models[ModelTier.ECONOMY]}")
    print()

    # Get models for critics
    for critic_name in ["rights", "fairness", "truth", "risk", "operations"]:
        model_id = registry.get_model_for_critic(critic_name)
        tier = registry.critic_tiers.get(critic_name, "default")
        print(
            f"  {critic_name}: {model_id} (tier: {tier.value if hasattr(tier, 'value') else tier})"
        )
    print()


# ============================================================================
# Example 4: Load from YAML Configuration File
# ============================================================================


async def example_yaml_config():
    """Load registry configuration from YAML file."""
    print("=" * 60)
    print("Example 4: YAML Configuration File")
    print("=" * 60)

    # Load from config file
    registry = ModelRegistry.from_yaml("config/models.yaml")

    print("✓ Loaded configuration from config/models.yaml")
    print(f"  Default model: {registry.default_model_id}")
    print()

    print("Critic assignments:")
    assignments = registry.list_assignments()
    for critic_name in sorted(assignments.keys()):
        details = assignments[critic_name]
        print(f"  {critic_name}: {details['model']} (via {details['source']})")
    print()


# ============================================================================
# Example 5: Custom Routing Strategy
# ============================================================================


async def example_custom_routing():
    """Use custom routing strategy for dynamic model selection."""
    print("=" * 60)
    print("Example 5: Custom Routing Strategy")
    print("=" * 60)

    registry = ModelRegistry()

    # Set cost optimizer strategy
    registry.set_routing_strategy(cost_optimizer_strategy)

    print("✓ Using cost_optimizer_strategy:")
    print("  High-accuracy critics (rights, fairness, truth) → Premium")
    print("  Other critics → Economy")
    print()

    # Test routing
    test_critics = ["rights", "fairness", "truth", "risk", "operations"]
    for critic_name in test_critics:
        model_id = registry.get_model_for_critic(critic_name)
        print(f"  {critic_name}: {model_id}")
    print()


# ============================================================================
# Example 6: Context-Aware Routing
# ============================================================================


async def example_context_aware():
    """Route based on context (priority, budget, etc.)."""
    print("=" * 60)
    print("Example 6: Context-Aware Routing")
    print("=" * 60)

    registry = ModelRegistry()
    registry.set_routing_strategy(priority_based_strategy)

    print("✓ Using priority_based_strategy")
    print()

    # Test with different priorities
    contexts = [
        {"priority": "high"},
        {"priority": "standard"},
        {"priority": "low"},
    ]

    for context in contexts:
        model_id = registry.get_model_for_critic("rights", context=context)
        print(f"  Priority={context['priority']:8s} → {model_id}")
    print()


# ============================================================================
# Example 7: Hybrid Approach - All Three Methods Together
# ============================================================================


async def example_hybrid_all():
    """Demonstrate hybrid approach with all configuration methods."""
    print("=" * 60)
    print("Example 7: Hybrid Approach - All Methods Combined")
    print("=" * 60)

    # Setup registry
    registry = ModelRegistry()
    registry.assign_model("fairness", "claude-opus-4.5")

    # Mock models
    class MockOpusModel:
        async def generate(self, text, context=None):
            return f"Opus: {text[:30]}"

    class MockSonnetModel:
        async def generate(self, text, context=None):
            return f"Sonnet: {text[:30]}"

    opus = MockOpusModel()
    sonnet = MockSonnetModel()

    # Create critic with explicit model
    critic = RightsCriticV8(model=opus, registry=registry)

    print("Critic configuration:")
    print(f"  Explicit model: {critic._preferred_model.__class__.__name__}")
    print(f"  Registry model: {registry.get_model_for_critic('rights')}")
    print()

    print("Model selection priority:")
    print("  1. Runtime override: If model passed to evaluate()")
    print("  2. Explicit model: OpusModel (from __init__)")
    print("  3. Registry lookup: Would use registry if no explicit model")
    print()

    # Demonstrate priority
    print("Actual model used:")

    # Without runtime override - uses explicit model
    model = critic.get_model(runtime_model=None)
    print(f"  No runtime override → {model.__class__.__name__}")

    # With runtime override - uses runtime model
    model = critic.get_model(runtime_model=sonnet)
    print(f"  With runtime override → {model.__class__.__name__}")
    print()


# ============================================================================
# Example 8: Cost Estimation
# ============================================================================


async def example_cost_estimation():
    """Estimate costs for different model configurations."""
    print("=" * 60)
    print("Example 8: Cost Estimation")
    print("=" * 60)

    registry = ModelRegistry()

    # Configure some critics
    registry.assign_model("rights", "claude-opus-4.5")
    registry.assign_model("fairness", "claude-sonnet-4.5")
    registry.assign_model("risk", "claude-haiku-4.0")

    print("Cost estimates (per 1000 tokens):")

    for critic_name in ["rights", "fairness", "risk"]:
        cost = registry.get_cost_estimate(critic_name, estimated_tokens=1000)
        model_id = registry.get_model_for_critic(critic_name)
        print(f"  {critic_name:12s} ({model_id:20s}): ${cost:.4f}")
    print()

    # Total cost for all critics
    all_critics = ["rights", "fairness", "truth", "risk", "operations"]
    total_cost = sum(registry.get_cost_estimate(c, 1000) for c in all_critics)
    print(f"  Total cost for all critics: ${total_cost:.4f} per 1k tokens")
    print()


# ============================================================================
# Example 9: Monitoring and Metrics
# ============================================================================


async def example_monitoring():
    """Add monitoring/metrics callbacks."""
    print("=" * 60)
    print("Example 9: Monitoring and Metrics")
    print("=" * 60)

    registry = ModelRegistry()

    # Track metrics
    metrics = {"requests": 0, "assignments": {}}

    def metrics_callback(event, data):
        if event == "model_request":
            metrics["requests"] += 1
        elif event == "model_assigned":
            critic = data["critic"]
            if critic not in metrics["assignments"]:
                metrics["assignments"][critic] = []
            metrics["assignments"][critic].append(data["model"])

    registry.set_metrics_callback(metrics_callback)

    # Make some requests
    registry.get_model_for_critic("rights")
    registry.get_model_for_critic("fairness")
    registry.get_model_for_critic("rights")

    print("✓ Metrics collected:")
    print(f"  Total requests: {metrics['requests']}")
    print("  Assignments:")
    for critic, models in metrics["assignments"].items():
        print(f"    {critic}: {len(models)} assignments")
    print()


# ============================================================================
# Example 10: Save/Load Configuration
# ============================================================================


async def example_save_load():
    """Save and load registry configuration."""
    print("=" * 60)
    print("Example 10: Save/Load Configuration")
    print("=" * 60)

    # Create and configure registry
    registry = ModelRegistry()
    registry.assign_model("rights", "claude-opus-4.5")
    registry.assign_model("fairness", "gpt-4")
    registry.assign_tier("risk", ModelTier.STANDARD)

    # Save to file
    registry.save_yaml("config/models_saved.yaml")
    print("✓ Saved configuration to config/models_saved.yaml")

    # Load from file
    loaded_registry = ModelRegistry.from_yaml("config/models_saved.yaml")
    print("✓ Loaded configuration from file")
    print()

    print("Loaded assignments:")
    for critic, details in loaded_registry.list_assignments().items():
        print(f"  {critic}: {details['model']}")
    print()


# ============================================================================
# Run All Examples
# ============================================================================


async def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "ELEANOR V8 Model Configuration Examples" + " " * 9 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    await example_explicit_model()
    await example_registry_basic()
    await example_registry_tiers()
    await example_yaml_config()
    await example_custom_routing()
    await example_context_aware()
    await example_hybrid_all()
    await example_cost_estimation()
    await example_monitoring()
    await example_save_load()

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
