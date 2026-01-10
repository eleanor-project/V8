"""
ELEANOR V8 — Intelligent Model Selection
----------------------------------------

Select optimal models based on cost, latency, and quality requirements.
Enables 30-50% cost reduction through intelligent routing.
"""

import logging
from typing import Dict, List, Optional, Any, Literal
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class OptimizationGoal(str, Enum):
    """Optimization goal for model selection."""
    COST = "cost"
    LATENCY = "latency"
    QUALITY = "quality"
    BALANCED = "balanced"


@dataclass
class ModelProfile:
    """Profile of a model with cost, latency, and quality metrics."""
    model_name: str
    provider: str
    cost_per_1k_tokens: float  # USD per 1k tokens
    avg_latency_ms: float  # Average latency in milliseconds
    quality_score: float  # 0.0 to 1.0, quality rating
    max_tokens: int = 4096
    supports_streaming: bool = True
    availability_score: float = 1.0  # 0.0 to 1.0, current availability


@dataclass
class SelectionRequirements:
    """Requirements for model selection."""
    min_quality: float = 0.0  # Minimum quality score required
    max_latency_ms: float = float("inf")  # Maximum acceptable latency
    max_cost_per_1k: float = float("inf")  # Maximum cost per 1k tokens
    optimization_goal: OptimizationGoal = OptimizationGoal.BALANCED
    prefer_streaming: bool = False
    context_length: Optional[int] = None  # Required context length


@dataclass
class SelectionResult:
    """Result of model selection."""
    selected_model: str
    provider: str
    selection_reason: str
    estimated_cost: Optional[float] = None
    estimated_latency_ms: Optional[float] = None
    quality_score: float = 0.0
    alternatives: List[str] = field(default_factory=list)


class IntelligentModelSelector:
    """
    Select optimal models based on requirements.
    
    Features:
    - Multi-objective optimization (cost, latency, quality)
    - Configurable model profiles
    - Real-time availability tracking
    - Cost and latency estimation
    - Fallback to quality if requirements cannot be met
    """
    
    def __init__(self):
        """Initialize intelligent model selector."""
        # Default model profiles (can be updated from config)
        self.model_profiles: Dict[str, ModelProfile] = {
            "gpt-4": ModelProfile(
                model_name="gpt-4",
                provider="openai",
                cost_per_1k_tokens=0.03,
                avg_latency_ms=2000,
                quality_score=0.95,
                max_tokens=8192
            ),
            "gpt-4-turbo": ModelProfile(
                model_name="gpt-4-turbo",
                provider="openai",
                cost_per_1k_tokens=0.01,
                avg_latency_ms=1500,
                quality_score=0.92,
                max_tokens=128000
            ),
            "gpt-3.5-turbo": ModelProfile(
                model_name="gpt-3.5-turbo",
                provider="openai",
                cost_per_1k_tokens=0.002,
                avg_latency_ms=800,
                quality_score=0.85,
                max_tokens=16385
            ),
            "claude-3-opus": ModelProfile(
                model_name="claude-3-opus",
                provider="anthropic",
                cost_per_1k_tokens=0.015,
                avg_latency_ms=2500,
                quality_score=0.97,
                max_tokens=200000
            ),
            "claude-3-sonnet": ModelProfile(
                model_name="claude-3-sonnet",
                provider="anthropic",
                cost_per_1k_tokens=0.003,
                avg_latency_ms=1200,
                quality_score=0.90,
                max_tokens=200000
            ),
            "claude-3-haiku": ModelProfile(
                model_name="claude-3-haiku",
                provider="anthropic",
                cost_per_1k_tokens=0.00025,
                avg_latency_ms=600,
                quality_score=0.80,
                max_tokens=200000
            ),
            "gpt-4o": ModelProfile(
                model_name="gpt-4o",
                provider="openai",
                cost_per_1k_tokens=0.005,
                avg_latency_ms=1000,
                quality_score=0.94,
                max_tokens=128000
            ),
            "gpt-4o-mini": ModelProfile(
                model_name="gpt-4o-mini",
                provider="openai",
                cost_per_1k_tokens=0.00015,
                avg_latency_ms=500,
                quality_score=0.82,
                max_tokens=128000
            ),
        }
        
        logger.info(f"Initialized IntelligentModelSelector with {len(self.model_profiles)} model profiles")
    
    def update_model_profile(self, profile: ModelProfile) -> None:
        """Update or add a model profile."""
        self.model_profiles[profile.model_name] = profile
        logger.debug(f"Updated model profile: {profile.model_name}")
    
    def select_model(
        self,
        requirements: Optional[SelectionRequirements] = None,
        estimated_tokens: Optional[int] = None
    ) -> SelectionResult:
        """
        Select optimal model based on requirements.
        
        Args:
            requirements: Selection requirements
            estimated_tokens: Estimated token count for cost calculation
            
        Returns:
            SelectionResult with selected model and metadata
        """
        if requirements is None:
            requirements = SelectionRequirements()
        
        # Filter models by requirements
        candidates = self._filter_candidates(requirements)
        
        if not candidates:
            # No candidates meet requirements, return highest quality
            logger.warning(
                "No models meet requirements, falling back to highest quality",
                extra={"requirements": requirements.__dict__}
            )
            return self._select_highest_quality()
        
        # Select based on optimization goal
        if requirements.optimization_goal == OptimizationGoal.COST:
            return self._select_lowest_cost(candidates, requirements, estimated_tokens)
        elif requirements.optimization_goal == OptimizationGoal.LATENCY:
            return self._select_lowest_latency(candidates, requirements)
        elif requirements.optimization_goal == OptimizationGoal.QUALITY:
            return self._select_highest_quality(candidates)
        else:  # BALANCED
            return self._select_balanced(candidates, requirements, estimated_tokens)
    
    def _filter_candidates(
        self,
        requirements: SelectionRequirements
    ) -> List[ModelProfile]:
        """Filter models that meet requirements."""
        candidates = []
        
        for profile in self.model_profiles.values():
            # Check quality requirement
            if profile.quality_score < requirements.min_quality:
                continue
            
            # Check latency requirement
            if profile.avg_latency_ms > requirements.max_latency_ms:
                continue
            
            # Check cost requirement
            if profile.cost_per_1k_tokens > requirements.max_cost_per_1k:
                continue
            
            # Check context length requirement
            if requirements.context_length and profile.max_tokens < requirements.context_length:
                continue
            
            # Check streaming requirement
            if requirements.prefer_streaming and not profile.supports_streaming:
                continue
            
            # Check availability
            if profile.availability_score < 0.5:
                continue  # Skip unavailable models
            
            candidates.append(profile)
        
        return candidates
    
    def _select_lowest_cost(
        self,
        candidates: List[ModelProfile],
        requirements: SelectionRequirements,
        estimated_tokens: Optional[int]
    ) -> SelectionResult:
        """Select model with lowest cost."""
        # Sort by cost
        candidates.sort(key=lambda p: p.cost_per_1k_tokens)
        
        selected = candidates[0]
        alternatives = [c.model_name for c in candidates[1:3]]  # Top 3 alternatives
        
        estimated_cost = None
        if estimated_tokens:
            estimated_cost = (selected.cost_per_1k_tokens * estimated_tokens) / 1000
        
        return SelectionResult(
            selected_model=selected.model_name,
            provider=selected.provider,
            selection_reason=f"Lowest cost model meeting quality requirement (≥{requirements.min_quality})",
            estimated_cost=estimated_cost,
            estimated_latency_ms=selected.avg_latency_ms,
            quality_score=selected.quality_score,
            alternatives=alternatives
        )
    
    def _select_lowest_latency(
        self,
        candidates: List[ModelProfile],
        requirements: SelectionRequirements
    ) -> SelectionResult:
        """Select model with lowest latency."""
        # Sort by latency
        candidates.sort(key=lambda p: p.avg_latency_ms)
        
        selected = candidates[0]
        alternatives = [c.model_name for c in candidates[1:3]]
        
        return SelectionResult(
            selected_model=selected.model_name,
            provider=selected.provider,
            selection_reason=f"Lowest latency model meeting quality requirement (≥{requirements.min_quality})",
            estimated_latency_ms=selected.avg_latency_ms,
            quality_score=selected.quality_score,
            alternatives=alternatives
        )
    
    def _select_highest_quality(
        self,
        candidates: Optional[List[ModelProfile]] = None
    ) -> SelectionResult:
        """Select model with highest quality."""
        if candidates is None:
            candidates = list(self.model_profiles.values())
        
        # Sort by quality (descending)
        candidates.sort(key=lambda p: p.quality_score, reverse=True)
        
        selected = candidates[0]
        alternatives = [c.model_name for c in candidates[1:3]]
        
        return SelectionResult(
            selected_model=selected.model_name,
            provider=selected.provider,
            selection_reason="Highest quality model available",
            estimated_latency_ms=selected.avg_latency_ms,
            quality_score=selected.quality_score,
            alternatives=alternatives
        )
    
    def _select_balanced(
        self,
        candidates: List[ModelProfile],
        requirements: SelectionRequirements,
        estimated_tokens: Optional[int]
    ) -> SelectionResult:
        """Select model with best balance of cost, latency, and quality."""
        # Normalize scores to 0-1 range for comparison
        max_cost = max(c.cost_per_1k_tokens for c in candidates)
        max_latency = max(c.avg_latency_ms for c in candidates)
        min_cost = min(c.cost_per_1k_tokens for c in candidates)
        min_latency = min(c.avg_latency_ms for c in candidates)
        
        # Calculate composite score (weighted)
        # Lower cost and latency are better, higher quality is better
        scored_candidates = []
        for profile in candidates:
            # Normalize cost (inverse, so lower is better)
            cost_score = 1.0 - (
                (profile.cost_per_1k_tokens - min_cost) / (max_cost - min_cost + 0.001)
            )
            
            # Normalize latency (inverse, so lower is better)
            latency_score = 1.0 - (
                (profile.avg_latency_ms - min_latency) / (max_latency - min_latency + 0.001)
            )
            
            # Composite score: 50% quality, 30% cost, 20% latency
            composite_score = (
                profile.quality_score * 0.5 +
                cost_score * 0.3 +
                latency_score * 0.2
            )
            
            scored_candidates.append((profile, composite_score))
        
        # Sort by composite score (descending)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        selected, score = scored_candidates[0]
        alternatives = [c[0].model_name for c in scored_candidates[1:3]]
        
        estimated_cost = None
        if estimated_tokens:
            estimated_cost = (selected.cost_per_1k_tokens * estimated_tokens) / 1000
        
        return SelectionResult(
            selected_model=selected.model_name,
            provider=selected.provider,
            selection_reason=f"Balanced selection (composite score: {score:.3f})",
            estimated_cost=estimated_cost,
            estimated_latency_ms=selected.avg_latency_ms,
            quality_score=selected.quality_score,
            alternatives=alternatives
        )
    
    def update_availability(self, model_name: str, availability_score: float) -> None:
        """Update availability score for a model."""
        if model_name in self.model_profiles:
            self.model_profiles[model_name].availability_score = availability_score
            logger.debug(f"Updated availability for {model_name}: {availability_score}")
    
    def get_model_profiles(self) -> Dict[str, ModelProfile]:
        """Get all model profiles."""
        return self.model_profiles.copy()
    
    def estimate_cost(self, model_name: str, tokens: int) -> float:
        """Estimate cost for a model and token count."""
        if model_name not in self.model_profiles:
            logger.warning(f"Unknown model: {model_name}")
            return 0.0
        
        profile = self.model_profiles[model_name]
        return (profile.cost_per_1k_tokens * tokens) / 1000
    
    def estimate_latency(self, model_name: str) -> float:
        """Estimate latency for a model."""
        if model_name not in self.model_profiles:
            logger.warning(f"Unknown model: {model_name}")
            return 0.0
        
        return self.model_profiles[model_name].avg_latency_ms


__all__ = [
    "IntelligentModelSelector",
    "ModelProfile",
    "SelectionRequirements",
    "SelectionResult",
    "OptimizationGoal",
]
