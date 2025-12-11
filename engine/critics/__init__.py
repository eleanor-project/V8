"""
ELEANOR V8  Critics Module
----------------------------

This module provides the constitutional critics that evaluate AI model outputs
against ethical principles using multi-dimensional analysis.

Critics follow a lexicographic priority ordering:
1. Rights (UDHR Articles 1, 2, 7) - Highest priority
2. Autonomy (consent, self-determination)
3. Fairness (equity, non-discrimination)
4. Truth (accuracy, honesty, transparency)
5. Risk (safety, reversibility, precaution)
6. Pragmatics (feasibility, sustainability) - Lowest priority

Each critic implements the BaseCriticV8 interface and produces evidence packages
for the aggregator to synthesize into constitutional decisions.
"""

from .base import BaseCriticV8
from .rights import RightsCriticV8, RightsCritic
from .fairness import FairnessCriticV8, FairnessCritic
from .truth import TruthCriticV8, TruthCritic
from .risk import RiskCriticV8, RiskCritic
from .pragmatics import PragmaticsCriticV8, PragmaticsCritic
from .autonomy import AutonomyCriticV8

__all__ = [
    # Base class
    "BaseCriticV8",

    # V8 Critics (preferred)
    "RightsCriticV8",
    "FairnessCriticV8",
    "TruthCriticV8",
    "RiskCriticV8",
    "PragmaticsCriticV8",
    "AutonomyCriticV8",

    # Backward compatibility aliases
    "RightsCritic",
    "FairnessCritic",
    "TruthCritic",
    "RiskCritic",
    "PragmaticsCritic",
]


def get_all_critics():
    """
    Factory function to instantiate all V8 critics.

    Returns:
        List of instantiated critic objects in priority order.
    """
    return [
        RightsCriticV8(),
        FairnessCriticV8(),
        AutonomyCriticV8(),
        TruthCriticV8(),
        RiskCriticV8(),
        PragmaticsCriticV8(),
    ]


def get_critic_by_name(name: str):
    """
    Get a critic instance by name.

    Args:
        name: Critic name (rights, fairness, truth, risk, pragmatics)

    Returns:
        Instantiated critic object.

    Raises:
        ValueError: If critic name is not recognized.
    """
    critics = {
        "rights": RightsCriticV8,
        "fairness": FairnessCriticV8,
        "autonomy": AutonomyCriticV8,
        "truth": TruthCriticV8,
        "risk": RiskCriticV8,
        "pragmatics": PragmaticsCriticV8,
    }

    if name.lower() not in critics:
        raise ValueError(f"Unknown critic: {name}. Available: {list(critics.keys())}")

    return critics[name.lower()]()
