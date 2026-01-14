"""
ELEANOR V8 — Feature Integration
--------------------------------

Wire optional features into the engine based on feature flags configuration.
This module conditionally initializes and integrates new features when enabled.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def integrate_optional_features(engine: Any, settings: Any) -> None:
    """
    Integrate optional features into engine based on feature flags.
    
    Args:
        engine: Engine instance to enhance
        settings: Configuration settings with feature flags
    """
    if not settings:
        logger.warning("No settings provided, skipping feature integration")
        return
    
    # Explainable Governance
    if getattr(settings, "enable_explainable_governance", False):
        try:
            from engine.governance.explainable import ExplainableGovernance
            explainer = ExplainableGovernance()
            setattr(engine, "explainable_governance", explainer)
            logger.info("Explainable Governance enabled and integrated")
        except Exception as exc:
            logger.error(f"Failed to integrate Explainable Governance: {exc}", exc_info=True)
    
    # Semantic Cache
    if getattr(settings, "enable_semantic_cache", False):
        try:
            from engine.cache.semantic_cache import SemanticCache
            
            # Get existing cache manager if available
            cache_manager = getattr(engine, "cache_manager", None)
            if cache_manager:
                semantic_cache = SemanticCache(
                    similarity_threshold=0.85,
                    max_size=10000
                )
                # Wrap cache manager's get/set methods to use semantic cache
                original_get = cache_manager.get
                original_set = cache_manager.set
                
                async def semantic_get(key: Any, *args, **kwargs):
                    # Try semantic cache first if it's a string key
                    if isinstance(key, str):
                        result = await semantic_cache.get(key)
                        if result:
                            return result[0]  # Return just the result
                    # Fallback to original cache
                    return await original_get(key, *args, **kwargs)
                
                async def semantic_set(key: Any, value: Any, *args, **kwargs):
                    # Also store in semantic cache if it's a string key
                    if isinstance(key, str):
                        await semantic_cache.set(key, value)
                    # Store in original cache
                    return await original_set(key, value, *args, **kwargs)
                
                cache_manager.get = semantic_get
                cache_manager.set = semantic_set
                setattr(engine, "semantic_cache", semantic_cache)
                logger.info("Semantic Cache enabled and integrated")
            else:
                logger.warning("Cache manager not found, cannot integrate semantic cache")
        except Exception as exc:
            logger.error(f"Failed to integrate Semantic Cache: {exc}", exc_info=True)
    
    # Intelligent Model Selection
    if getattr(settings, "enable_intelligent_model_selection", False):
        try:
            from engine.router.intelligent_selector import IntelligentModelSelector
            selector = IntelligentModelSelector()
            setattr(engine, "intelligent_model_selector", selector)
            logger.info("Intelligent Model Selection enabled and integrated")
            
            # If router exists, integrate selector
            router = getattr(engine, "router", None)
            if router:
                # Store selector on router for use in model selection
                setattr(router, "intelligent_selector", selector)
        except Exception as exc:
            logger.error(f"Failed to integrate Intelligent Model Selection: {exc}", exc_info=True)
    
    # Anomaly Detection
    if getattr(settings, "enable_anomaly_detection", False):
        try:
            from engine.observability.anomaly_detector import AnomalyDetector
            detector = AnomalyDetector(
                use_ml=True,  # Use ML if available
                contamination=0.1,
                z_threshold=3.0
            )
            setattr(engine, "anomaly_detector", detector)
            logger.info("Anomaly Detection enabled and integrated")
            
            # Note: Training would happen with historical data
            # This would typically be done separately or on startup with historical metrics
        except Exception as exc:
            logger.error(f"Failed to integrate Anomaly Detection: {exc}", exc_info=True)
    
    # Streaming Governance
    if getattr(settings, "enable_streaming_governance", False):
        try:
            from engine.governance.streaming import StreamingGovernance
            streaming_gov = StreamingGovernance(
                early_decision_threshold=0.85,
                deny_threshold=0.7,
                escalation_threshold=0.6
            )
            setattr(engine, "streaming_governance", streaming_gov)
            logger.info("Streaming Governance enabled and integrated")
        except Exception as exc:
            logger.error(f"Failed to integrate Streaming Governance: {exc}", exc_info=True)
    
    # Adaptive Critic Weighting
    if getattr(settings, "enable_adaptive_critic_weighting", False):
        try:
            from engine.aggregator.adaptive_weighting import AdaptiveCriticWeighting
            adaptive_weighting = AdaptiveCriticWeighting(
                learning_rate=0.1,
                exploration_rate=0.05,
                decay_factor=0.95,
                min_weight=0.1,
                max_weight=2.0,
                min_samples=10
            )
            setattr(engine, "adaptive_weighting", adaptive_weighting)
            
            # Integrate with aggregator if available
            aggregator = getattr(engine, "aggregator", None)
            if aggregator:
                # Store reference for use during aggregation
                setattr(aggregator, "adaptive_weighting", adaptive_weighting)
            
            logger.info("Adaptive Critic Weighting enabled and integrated")
        except Exception as exc:
            logger.error(f"Failed to integrate Adaptive Critic Weighting: {exc}", exc_info=True)
    
    # Temporal Precedent Evolution
    if getattr(settings, "enable_temporal_precedent_evolution", False):
        try:
            from engine.precedent.temporal_evolution import TemporalPrecedentEvolutionTracker
            
            # Get precedent store if available
            precedent_store = getattr(engine, "precedent_store", None)
            if not precedent_store:
                # Try to get from precedent_retriever
                precedent_retriever = getattr(engine, "precedent_retriever", None)
                if precedent_retriever:
                    precedent_store = getattr(precedent_retriever, "store", None)
            
            evolution_tracker = TemporalPrecedentEvolutionTracker(store_backend=precedent_store)
            setattr(engine, "temporal_evolution_tracker", evolution_tracker)
            logger.info("Temporal Precedent Evolution Tracking enabled and integrated")
        except Exception as exc:
            logger.error(f"Failed to integrate Temporal Precedent Evolution: {exc}", exc_info=True)


def get_explanation_for_result(engine: Any, result: Dict[str, Any], detail_level: str = "summary") -> Optional[Dict[str, Any]]:
    """
    Get explainable governance explanation for a result if enabled.
    
    Args:
        engine: Engine instance
        result: Engine result dictionary
        detail_level: Level of detail (summary, detailed, interactive)
        
    Returns:
        Explanation dictionary or None if not enabled
    """
    explainer = getattr(engine, "explainable_governance", None)
    if not explainer:
        return None
    
    try:
        from engine.governance.explainable import DetailLevel
        detail_enum = DetailLevel(detail_level.lower()) if detail_level else DetailLevel.SUMMARY
        explanation = explainer.explain_decision(result, detail_level=detail_enum)
        
        # Convert to dict for JSON serialization
        return {
            "decision": explanation.decision,
            "primary_factors": [
                {
                    "factor_name": f.factor_name,
                    "influence_score": f.influence_score,
                    "description": f.description,
                    "source": f.source
                }
                for f in explanation.primary_factors
            ],
            "counterfactuals": [
                {
                    "scenario": cf.scenario,
                    "required_changes": cf.required_changes,
                    "hypothetical_decision": cf.hypothetical_decision,
                    "confidence": cf.confidence
                }
                for cf in explanation.counterfactuals
            ],
            "critic_contributions": [
                {
                    "critic_name": c.critic_name,
                    "contribution_score": c.contribution_score,
                    "severity_weight": c.severity_weight,
                    "violations_count": c.violations_count,
                    "was_decisive": c.was_decisive,
                    "rationale": c.rationale
                }
                for c in explanation.critic_contributions
            ],
            "precedent_influence": explanation.precedent_influence,
            "uncertainty_impact": explanation.uncertainty_impact,
            "human_readable": explanation.human_readable,
            "interactive_data": explanation.interactive_data
        }
    except Exception as exc:
        logger.error(f"Failed to generate explanation: {exc}", exc_info=True)
        return None


__all__ = [
    "integrate_optional_features",
    "get_explanation_for_result",
]
