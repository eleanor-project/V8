"""
ELEANOR V8 — Model Registry
----------------------------

Centralized model configuration and routing for critics.

Features:
- Per-critic model assignment
- Environment-based configuration
- Cost optimization routing
- Performance monitoring hooks
- Hot-reloading support
- Default fallback strategy

Usage:
    # Simple usage
    registry = ModelRegistry()
    model = registry.get_model_for_critic("rights")

    # With custom configuration
    registry = ModelRegistry.from_yaml("config/models.yaml")

    # Dynamic routing
    model = registry.get_model_for_critic("rights", context={"priority": "high"})
"""

import yaml
import json
from typing import Any, Callable, Dict, Optional
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from engine.utils.critic_names import canonical_critic_name


class ModelTier(Enum):
    """Model performance/cost tiers."""

    PREMIUM = "premium"  # Highest accuracy, highest cost (e.g., Opus)
    STANDARD = "standard"  # Balanced (e.g., Sonnet)
    ECONOMY = "economy"  # Fast, cheap (e.g., Haiku)


@dataclass
class ModelConfig:
    """Configuration for a model instance."""

    model_id: str
    provider: str  # "anthropic", "openai", etc.
    tier: ModelTier
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout: float = 30.0
    cost_per_1k_tokens: float = 0.0  # For cost tracking
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ModelRegistry:
    """
    Centralized registry for critic-model assignments.

    Supports:
    - Explicit per-critic assignments
    - Tier-based routing (premium/standard/economy)
    - Context-aware routing (priority, budget, etc.)
    - Configuration file loading
    - Default fallback
    """

    def __init__(self, default_model_id: str = "claude-3-5-sonnet-20241022"):
        """
        Initialize registry with default model.

        Args:
            default_model_id: Model to use when no specific assignment exists
        """
        self.default_model_id = default_model_id

        # Explicit critic -> model assignments
        self.critic_models: Dict[str, str] = {}

        # Critic -> tier assignments (for tier-based routing)
        self.critic_tiers: Dict[str, ModelTier] = {}

        # Model configurations
        self.model_configs: Dict[str, ModelConfig] = {}

        # Tier -> model mappings
        self.tier_models: Dict[ModelTier, str] = {
            ModelTier.PREMIUM: "claude-3-opus-20240229",
            ModelTier.STANDARD: "claude-3-5-sonnet-20241022",
            ModelTier.ECONOMY: "claude-3-haiku-20240307",
        }

        # Routing strategies
        self.routing_strategy: Optional[
            Callable[[str, Dict[str, Any], "ModelRegistry"], Optional[str]]
        ] = None

        # Metrics/monitoring hooks
        self.metrics_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None

        # Initialize default configs
        self._init_default_configs()

    def _init_default_configs(self):
        """Initialize default model configurations."""
        self.model_configs = {
            "claude-3-opus-20240229": ModelConfig(
                model_id="claude-3-opus-20240229",
                provider="anthropic",
                tier=ModelTier.PREMIUM,
                temperature=0.1,
                cost_per_1k_tokens=0.015,
            ),
            "claude-3-5-sonnet-20241022": ModelConfig(
                model_id="claude-3-5-sonnet-20241022",
                provider="anthropic",
                tier=ModelTier.STANDARD,
                temperature=0.1,
                cost_per_1k_tokens=0.003,
            ),
            "claude-3-haiku-20240307": ModelConfig(
                model_id="claude-3-haiku-20240307",
                provider="anthropic",
                tier=ModelTier.ECONOMY,
                temperature=0.1,
                cost_per_1k_tokens=0.0008,
            ),
            "gpt-4.1": ModelConfig(
                model_id="gpt-4.1",
                provider="openai",
                tier=ModelTier.PREMIUM,
                temperature=0.1,
                cost_per_1k_tokens=0.01,
            ),
            "gpt-4o": ModelConfig(
                model_id="gpt-4o",
                provider="openai",
                tier=ModelTier.STANDARD,
                temperature=0.1,
                cost_per_1k_tokens=0.005,
            ),
            "gpt-4o-mini": ModelConfig(
                model_id="gpt-4o-mini",
                provider="openai",
                tier=ModelTier.ECONOMY,
                temperature=0.1,
                cost_per_1k_tokens=0.0015,
            ),
            "grok-2-latest": ModelConfig(
                model_id="grok-2-latest",
                provider="xai",
                tier=ModelTier.STANDARD,
                temperature=0.1,
                cost_per_1k_tokens=0.006,
            ),
        }

    def assign_model(self, critic_name: str, model_id: str) -> None:
        """
        Assign a specific model to a critic.

        Args:
            critic_name: Name of the critic
            model_id: Model identifier to use
        """
        critic_key = canonical_critic_name(critic_name)
        self.critic_models[critic_key] = model_id

    def assign_tier(self, critic_name: str, tier: ModelTier) -> None:
        """
        Assign a performance tier to a critic.

        Args:
            critic_name: Name of the critic
            tier: Performance tier (PREMIUM, STANDARD, ECONOMY)
        """
        critic_key = canonical_critic_name(critic_name)
        self.critic_tiers[critic_key] = tier

    def set_tier_model(self, tier: ModelTier, model_id: str) -> None:
        """
        Set which model to use for a given tier.

        Args:
            tier: Performance tier
            model_id: Model identifier
        """
        self.tier_models[tier] = model_id

    def get_model_for_critic(
        self, critic_name: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Get the appropriate model for a critic.

        Priority:
        1. Custom routing strategy (if set)
        2. Explicit critic-model assignment
        3. Tier-based assignment
        4. Default model

        Args:
            critic_name: Name of the critic
            context: Optional context for routing decisions

        Returns:
            Model identifier string
        """
        critic_name = canonical_critic_name(critic_name)
        context = context or {}

        # Hook for metrics/monitoring
        if self.metrics_callback:
            self.metrics_callback("model_request", {"critic": critic_name, "context": context})

        # 1. Custom routing strategy
        if self.routing_strategy:
            model_id = self.routing_strategy(critic_name, context, self)
            if model_id:
                self._record_assignment(critic_name, model_id, "routing_strategy")
                return str(model_id)

        # 2. Explicit assignment
        if critic_name in self.critic_models:
            model_id = self.critic_models[critic_name]
            self._record_assignment(critic_name, model_id, "explicit")
            return model_id

        # 3. Tier-based assignment
        if critic_name in self.critic_tiers:
            tier = self.critic_tiers[critic_name]
            model_id = self.tier_models[tier]
            self._record_assignment(critic_name, model_id, "tier")
            return model_id

        # 4. Default fallback
        self._record_assignment(critic_name, self.default_model_id, "default")
        return self.default_model_id

    def get_model_config(self, model_id: str) -> Optional[ModelConfig]:
        """
        Get configuration for a model.

        Args:
            model_id: Model identifier

        Returns:
            ModelConfig or None if not found
        """
        return self.model_configs.get(model_id)

    def _record_assignment(self, critic_name: str, model_id: str, source: str):
        """Record model assignment for observability."""
        if self.metrics_callback:
            self.metrics_callback(
                "model_assigned", {"critic": critic_name, "model": model_id, "source": source}
            )

    @classmethod
    def from_yaml(cls, config_path: str) -> "ModelRegistry":
        """
        Load registry configuration from YAML file.

        Example YAML:
            default_model: claude-sonnet-4.5

            critics:
              rights:
                model: claude-opus-4.5
              fairness:
                tier: premium
              truth:
                model: claude-sonnet-4.5
              operations:
                tier: economy

            tiers:
              premium: claude-opus-4.5
              standard: claude-sonnet-4.5
              economy: claude-haiku-4.0

        Args:
            config_path: Path to YAML configuration file

        Returns:
            Configured ModelRegistry instance
        """
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        registry = cls(default_model_id=config.get("default_model", "claude-sonnet-4.5"))

        # Load tier mappings
        if "tiers" in config:
            for tier_name, model_id in config["tiers"].items():
                tier = ModelTier(tier_name)
                registry.set_tier_model(tier, model_id)

        # Load critic assignments
        if "critics" in config:
            for critic_name, critic_config in config["critics"].items():
                critic_key = canonical_critic_name(critic_name)
                if "model" in critic_config:
                    registry.assign_model(critic_key, critic_config["model"])
                elif "tier" in critic_config:
                    tier = ModelTier(critic_config["tier"])
                    registry.assign_tier(critic_key, tier)

        return registry

    @classmethod
    def from_json(cls, config_path: str) -> "ModelRegistry":
        """Load registry configuration from JSON file."""
        with open(config_path, "r") as f:
            config = json.load(f)

        # Convert to YAML format internally (same structure)
        # Save temporarily and use from_yaml
        temp_yaml = Path(config_path).parent / "temp_config.yaml"
        with open(temp_yaml, "w") as f:
            yaml.dump(config, f)

        registry = cls.from_yaml(str(temp_yaml))
        temp_yaml.unlink()  # Clean up

        return registry

    def to_dict(self) -> Dict[str, Any]:
        """Export registry configuration as dictionary."""
        return {
            "default_model": self.default_model_id,
            "critics": {
                **{name: {"model": model} for name, model in self.critic_models.items()},
                **{name: {"tier": tier.value} for name, tier in self.critic_tiers.items()},
            },
            "tiers": {tier.value: model for tier, model in self.tier_models.items()},
        }

    def save_yaml(self, config_path: str):
        """Save current configuration to YAML file."""
        with open(config_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def set_routing_strategy(
        self,
        strategy: Callable[[str, Dict[str, Any], "ModelRegistry"], Optional[str]],
    ) -> None:
        """
        Set a custom routing strategy function.

        Strategy function signature:
            def strategy(critic_name: str, context: dict, registry: ModelRegistry) -> Optional[str]

        Return None to fall through to default routing.

        Example:
            def cost_optimizer(critic_name, context, registry):
                if context.get("budget") == "low":
                    return "claude-haiku-4.0"
                return None  # Use default routing

            registry.set_routing_strategy(cost_optimizer)
        """
        self.routing_strategy = strategy

    def set_metrics_callback(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """
        Set callback for metrics/monitoring.

        Callback signature:
            def callback(event: str, data: dict) -> None

        Events:
            - "model_request": When a model is requested
            - "model_assigned": When a model is assigned
        """
        self.metrics_callback = callback

    def get_cost_estimate(self, critic_name: str, estimated_tokens: int = 1000) -> float:
        """
        Estimate cost for running a critic.

        Args:
            critic_name: Name of the critic
            estimated_tokens: Estimated token count

        Returns:
            Estimated cost in dollars
        """
        model_id = self.get_model_for_critic(critic_name)
        config = self.get_model_config(model_id)

        if config:
            return (estimated_tokens / 1000) * config.cost_per_1k_tokens

        return 0.0

    def list_assignments(self) -> Dict[str, Dict[str, str]]:
        """
        List all current critic-model assignments.

        Returns:
            Dictionary mapping critic names to assignment details
        """
        assignments = {}

        # Get all known critics (from explicit and tier assignments)
        all_critics = set(self.critic_models.keys()) | set(self.critic_tiers.keys())

        for critic_name in all_critics:
            model_id = self.get_model_for_critic(critic_name)
            source = "unknown"

            if critic_name in self.critic_models:
                source = "explicit"
            elif critic_name in self.critic_tiers:
                source = f"tier:{self.critic_tiers[critic_name].value}"
            else:
                source = "default"

            assignments[critic_name] = {"model": model_id, "source": source}

        return assignments


# Predefined routing strategies


def cost_optimizer_strategy(
    critic_name: str, context: Dict[str, Any], registry: ModelRegistry
) -> Optional[str]:
    """
    Route based on cost optimization.

    High-accuracy critics get premium models, others get economy.
    """
    high_accuracy_critics = {"rights", "fairness", "truth"}

    if critic_name in high_accuracy_critics:
        return registry.tier_models[ModelTier.PREMIUM]

    return registry.tier_models[ModelTier.ECONOMY]


def priority_based_strategy(
    critic_name: str, context: Dict[str, Any], registry: ModelRegistry
) -> Optional[str]:
    """
    Route based on priority context.

    High priority requests get premium models.
    """
    priority = context.get("priority", "standard")

    if priority == "high":
        return registry.tier_models[ModelTier.PREMIUM]
    elif priority == "low":
        return registry.tier_models[ModelTier.ECONOMY]

    return None  # Use default routing


def budget_aware_strategy(
    critic_name: str, context: Dict[str, Any], registry: ModelRegistry
) -> Optional[str]:
    """
    Route based on budget constraints.
    """
    budget = context.get("budget", "standard")

    if budget == "unlimited":
        return registry.tier_models[ModelTier.PREMIUM]
    elif budget == "limited":
        return registry.tier_models[ModelTier.ECONOMY]

    return None  # Use default routing
