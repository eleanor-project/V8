# ELEANOR V8 — Model Configuration Guide

**Hybrid Model Configuration System**

## Overview

ELEANOR V8 supports flexible model configuration for critics through a **hybrid approach** that combines the best of explicit configuration and centralized management.

### Three Configuration Methods

1. **Explicit (Preferred Model)** - Configure model directly in `__init__`
2. **Registry (Centralized)** - Use ModelRegistry for centralized config
3. **Runtime (Override)** - Pass model at evaluation time

**Priority**: `runtime > explicit > registry > error`

---

## Quick Start

### Method 1: Explicit Model (Simple)

```python
from engine.critics.rights import RightsCriticV8

# Configure critic with explicit model
critic = RightsCriticV8(model=OpusModel())

# Model is always used unless overridden
result = await critic.evaluate(model=None, input_text="...", context={})
```

### Method 2: Registry (Centralized)

```python
from engine.models import ModelRegistry
from engine.critics.rights import RightsCriticV8

# Create registry
registry = ModelRegistry()
registry.assign_model("rights", "claude-opus-4.5")

# Configure critic with registry
critic = RightsCriticV8(registry=registry)

# Model is selected from registry
result = await critic.evaluate(model=None, input_text="...", context={})
```

### Method 3: YAML Configuration (Recommended for Production)

```yaml
# config/models.yaml
default_model: claude-sonnet-4.5

critics:
  rights:
    model: claude-opus-4.5
  fairness:
    tier: premium
  risk:
    tier: standard
```

```python
from engine.models import ModelRegistry

# Load configuration
registry = ModelRegistry.from_yaml("config/models.yaml")

# All critics configured centrally
rights = RightsCriticV8(registry=registry)
fairness = FairnessCriticV8(registry=registry)
```

---

## ModelRegistry API

### Basic Configuration

```python
from engine.models import ModelRegistry, ModelTier

registry = ModelRegistry(default_model_id="claude-sonnet-4.5")

# Explicit model assignment
registry.assign_model("rights", "claude-opus-4.5")
registry.assign_model("fairness", "gpt-4")

# Tier-based assignment
registry.assign_tier("truth", ModelTier.PREMIUM)
registry.assign_tier("risk", ModelTier.STANDARD)

# Get model for critic
model_id = registry.get_model_for_critic("rights")
# Returns: "claude-opus-4.5"
```

### Performance Tiers

```python
from engine.models import ModelTier

# Define tier mappings
registry.set_tier_model(ModelTier.PREMIUM, "claude-opus-4.5")
registry.set_tier_model(ModelTier.STANDARD, "claude-sonnet-4.5")
registry.set_tier_model(ModelTier.ECONOMY, "claude-haiku-4.0")

# Assign critics to tiers
registry.assign_tier("rights", ModelTier.PREMIUM)    # High accuracy
registry.assign_tier("fairness", ModelTier.PREMIUM)  # High accuracy
registry.assign_tier("operations", ModelTier.ECONOMY)  # Fast, cheap
```

---

## Advanced Features

### Custom Routing Strategies

```python
def custom_router(critic_name, context, registry):
    """
    Custom routing logic.

    Returns:
        model_id string or None to fall through to default routing
    """
    # Route based on budget
    if context.get("budget") == "low":
        return "claude-haiku-4.0"

    # High-accuracy critics get premium models
    if critic_name in ["rights", "fairness", "truth"]:
        return registry.tier_models[ModelTier.PREMIUM]

    # Use default routing
    return None

registry.set_routing_strategy(custom_router)
```

### Predefined Strategies

```python
from engine.models import (
    cost_optimizer_strategy,
    priority_based_strategy,
    budget_aware_strategy
)

# Cost optimizer: Premium for accuracy-critical, Economy for others
registry.set_routing_strategy(cost_optimizer_strategy)

# Priority-based: Route based on context["priority"]
registry.set_routing_strategy(priority_based_strategy)
model = registry.get_model_for_critic("rights", context={"priority": "high"})

# Budget-aware: Route based on context["budget"]
registry.set_routing_strategy(budget_aware_strategy)
```

### Context-Aware Routing

```python
# Different models based on context
contexts = [
    {"priority": "high", "budget": "unlimited"},
    {"priority": "low", "budget": "limited"},
]

for ctx in contexts:
    model_id = registry.get_model_for_critic("rights", context=ctx)
    print(f"Context {ctx} → {model_id}")

# Output:
# Context {'priority': 'high'} → claude-opus-4.5
# Context {'priority': 'low'} → claude-haiku-4.0
```

### Cost Estimation

```python
# Estimate cost for a critic
cost = registry.get_cost_estimate("rights", estimated_tokens=1000)
print(f"Rights critic: ${cost:.4f} per 1k tokens")

# Estimate total cost for all critics
all_critics = ["rights", "fairness", "truth", "risk", "operations"]
total_cost = sum(registry.get_cost_estimate(c, 1000) for c in all_critics)
print(f"Total: ${total_cost:.4f}")

# Output:
# Rights critic: $0.0150 per 1k tokens
# Total: $0.0248 per 1k tokens
```

### Monitoring and Metrics

```python
# Track model usage
metrics = {"requests": 0, "by_critic": {}}

def metrics_callback(event, data):
    if event == "model_request":
        metrics["requests"] += 1
    elif event == "model_assigned":
        critic = data["critic"]
        metrics["by_critic"][critic] = data["model"]

registry.set_metrics_callback(metrics_callback)

# Make requests
registry.get_model_for_critic("rights")
registry.get_model_for_critic("fairness")

print(f"Total requests: {metrics['requests']}")
print(f"Assignments: {metrics['by_critic']}")
```

---

## Configuration Files

### YAML Format

```yaml
# config/models.yaml

# Default model (fallback)
default_model: claude-sonnet-4.5

# Tier definitions
tiers:
  premium: claude-opus-4.5
  standard: claude-sonnet-4.5
  economy: claude-haiku-4.0

# Per-critic configuration
critics:
  # Explicit model assignment
  rights:
    model: claude-opus-4.5

  truth:
    model: claude-opus-4.5

  # Tier-based assignment
  fairness:
    tier: premium

  risk:
    tier: standard

  operations:
    tier: economy

  # No configuration = uses default_model
```

### Loading Configuration

```python
# From YAML
registry = ModelRegistry.from_yaml("config/models.yaml")

# From JSON
registry = ModelRegistry.from_json("config/models.json")

# Save current configuration
registry.save_yaml("config/models_backup.yaml")
```

---

## Hybrid Approach in Practice

### Example: All Three Methods Together

```python
# Setup registry
registry = ModelRegistry.from_yaml("config/models.yaml")

# Create critics with different configurations
rights = RightsCriticV8(
    model=OpusModel(),      # Explicit (highest priority)
    registry=registry       # Registry (fallback)
)

fairness = FairnessCriticV8(
    registry=registry       # Registry only
)

truth = TruthCriticV8()     # No configuration (runtime only)

# Usage
await rights.evaluate(model=None, ...)        # Uses explicit OpusModel
await rights.evaluate(model=SonnetModel(), ...)  # Runtime override

await fairness.evaluate(model=None, ...)      # Uses registry model
await fairness.evaluate(model=HaikuModel(), ...)  # Runtime override

await truth.evaluate(model=SonnetModel(), ...) # Must provide model
```

### Example: Cost-Optimized Production Setup

```python
# Production configuration
registry = ModelRegistry()

# Critical accuracy critics → Premium
for critic in ["rights", "fairness", "truth"]:
    registry.assign_tier(critic, ModelTier.PREMIUM)

# Standard critics → Standard
for critic in ["risk", "operations"]:
    registry.assign_tier(critic, ModelTier.STANDARD)

# Fast checks → Economy
for critic in ["autonomy"]:
    registry.assign_tier(critic, ModelTier.ECONOMY)

# Cost estimate
total_cost = sum(
    registry.get_cost_estimate(c, 1000)
    for c in all_critics
)
print(f"Cost per deliberation: ${total_cost:.4f}")
```

---

## Best Practices

### 1. Use YAML for Production

```python
# ✅ Good: Centralized, version-controlled configuration
registry = ModelRegistry.from_yaml("config/models.yaml")

# ❌ Avoid: Hardcoded model assignments scattered in code
critic = RightsCriticV8(model=OpusModel())
```

### 2. Tier-Based for Flexibility

```python
# ✅ Good: Easy to swap tier models
registry.assign_tier("rights", ModelTier.PREMIUM)
# Later: Change premium tier from Opus to GPT-4
registry.set_tier_model(ModelTier.PREMIUM, "gpt-4")

# ❌ Avoid: Explicit models require changing every assignment
registry.assign_model("rights", "claude-opus-4.5")
registry.assign_model("fairness", "claude-opus-4.5")
# Have to change all individually
```

### 3. Cost Monitoring

```python
# ✅ Good: Monitor and optimize costs
def cost_monitor(event, data):
    if event == "model_assigned":
        model_id = data["model"]
        config = registry.get_model_config(model_id)
        log_cost(data["critic"], config.cost_per_1k_tokens)

registry.set_metrics_callback(cost_monitor)
```

### 4. Environment-Specific Configs

```python
# ✅ Good: Different configs per environment
if ENV == "production":
    registry = ModelRegistry.from_yaml("config/models_prod.yaml")
elif ENV == "development":
    registry = ModelRegistry.from_yaml("config/models_dev.yaml")
else:  # testing
    registry = ModelRegistry.from_yaml("config/models_test.yaml")
```

---

## Migration Guide

### From Current System

**Before** (no model configuration):
```python
critic = RightsCriticV8()
result = await critic.evaluate(model=some_model, input_text="...", context={})
```

**After** (hybrid approach):
```python
# Option A: Keep existing behavior (runtime model)
critic = RightsCriticV8()
result = await critic.evaluate(model=some_model, input_text="...", context={})

# Option B: Configure with registry
registry = ModelRegistry.from_yaml("config/models.yaml")
critic = RightsCriticV8(registry=registry)
result = await critic.evaluate(model=None, input_text="...", context={})

# Option C: Explicit model
critic = RightsCriticV8(model=opus_model)
result = await critic.evaluate(model=None, input_text="...", context={})
```

**Backward Compatibility**: ✅ Fully backward compatible. Existing code continues to work.

---

## Troubleshooting

### Problem: "No model configured" error

```python
# ❌ Error
critic = RightsCriticV8()
result = await critic.evaluate(model=None, ...)  # Error!

# ✅ Solution 1: Provide model
result = await critic.evaluate(model=opus_model, ...)

# ✅ Solution 2: Configure registry
registry = ModelRegistry()
registry.assign_model("rights", "claude-opus-4.5")
critic = RightsCriticV8(registry=registry)
result = await critic.evaluate(model=None, ...)

# ✅ Solution 3: Explicit model
critic = RightsCriticV8(model=opus_model)
result = await critic.evaluate(model=None, ...)
```

### Problem: Model selection not as expected

```python
# Check priority order
critic = RightsCriticV8(model=explicit_model, registry=registry)

# Test each level
print(critic.get_model(runtime_model=runtime_model))  # → runtime_model
print(critic.get_model())                             # → explicit_model
print(critic.get_model() if no explicit...)          # → registry model
```

### Problem: High costs

```python
# Analyze cost breakdown
for critic_name in all_critics:
    model_id = registry.get_model_for_critic(critic_name)
    config = registry.get_model_config(model_id)
    cost = registry.get_cost_estimate(critic_name, 1000)
    print(f"{critic_name}: {model_id} → ${cost:.4f}")

# Optimize by moving to economy tier
registry.assign_tier("expensive_critic", ModelTier.ECONOMY)
```

---

## API Reference

See full API documentation: [API.md](API.md)

---

## Examples

Complete working examples: `examples/model_configuration_examples.py`

Run examples:
```bash
cd ~/Documents/GitHub/V8
PYTHONPATH=. python3 examples/model_configuration_examples.py
```

---

## Support

- GitHub Issues: https://github.com/eleanor-project/eleanor-v8/issues
- Documentation: https://github.com/eleanor-project/eleanor-v8#readme
