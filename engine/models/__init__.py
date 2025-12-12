"""
ELEANOR V8 — Models Module
---------------------------

Model configuration and management.
"""

from .registry import (
    ModelRegistry,
    ModelConfig,
    ModelTier,
    cost_optimizer_strategy,
    priority_based_strategy,
    budget_aware_strategy,
)

__all__ = [
    "ModelRegistry",
    "ModelConfig",
    "ModelTier",
    "cost_optimizer_strategy",
    "priority_based_strategy",
    "budget_aware_strategy",
]
