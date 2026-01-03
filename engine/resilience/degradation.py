"""
ELEANOR V8 — Graceful Degradation Strategies

Fallback behaviors when components fail.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class DegradationStrategy:
    """
    Defines fallback behaviors for component failures.

    Each method returns a degraded result that allows the pipeline
    to continue operating with reduced functionality.
    """

    @staticmethod
    async def precedent_fallback(
        error: Exception, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Fallback when precedent retrieval fails.

        Returns empty precedent data indicating novelty.
        """
        payload = {
            "error": str(error),
            "error_type": type(error).__name__,
            "fallback": "no_precedents",
        }
        if context:
            payload.update(context)
        logger.warning("precedent_retrieval_failed", extra=payload)

        return {
            "cases": [],
            "alignment_score": 0.0,
            "novel": True,
            "degraded": True,
            "degradation_reason": "precedent_store_unavailable",
            "error": str(error),
        }

    @staticmethod
    async def uncertainty_fallback(
        error: Exception, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Fallback when uncertainty engine fails.

        Returns conservative high uncertainty estimate.
        """
        payload = {
            "error": str(error),
            "error_type": type(error).__name__,
            "fallback": "high_uncertainty",
        }
        if context:
            payload.update(context)
        logger.warning("uncertainty_engine_failed", extra=payload)

        return {
            "overall_uncertainty": 0.8,  # Conservative estimate
            "needs_escalation": True,
            "degraded": True,
            "degradation_reason": "uncertainty_engine_unavailable",
            "recommendation": "human_review_recommended",
            "error": str(error),
        }

    @staticmethod
    async def router_fallback(
        error: Exception,
        default_model: str = "llama3.2:3b",
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Fallback when router fails.

        Returns default model selection.
        """
        payload = {
            "error": str(error),
            "error_type": type(error).__name__,
            "fallback": "default_model",
            "default_model": default_model,
        }
        if context:
            payload.update(context)
        logger.warning("router_failed", extra=payload)

        return {
            "model_name": default_model,
            "model_version": "fallback",
            "router_selection_reason": "router_unavailable_using_default",
            "degraded": True,
            "degradation_reason": "router_unavailable",
            "error": str(error),
        }

    @staticmethod
    async def critic_fallback(
        critic_name: str, error: Exception, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Fallback when single critic fails.

        Returns empty critic result indicating failure.
        """
        payload = {
            "critic": critic_name,
            "error": str(error),
            "error_type": type(error).__name__,
            "fallback": "empty_result",
        }
        if context:
            payload.update(context)
        logger.warning("critic_failed", extra=payload)

        return {
            "critic_name": critic_name,
            "violations": [],
            "severity": 0.0,
            "degraded": True,
            "degradation_reason": f"critic_{critic_name}_unavailable",
            "error": str(error),
            "note": "Critic evaluation skipped due to failure",
        }

    @staticmethod
    async def detector_fallback(
        detector_name: str, error: Exception, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Fallback when detector fails.

        Returns empty signals indicating no detection.
        """
        payload = {
            "detector": detector_name,
            "error": str(error),
            "error_type": type(error).__name__,
            "fallback": "no_signals",
        }
        if context:
            payload.update(context)
        logger.warning("detector_failed", extra=payload)

        return {
            "detector_name": detector_name,
            "signals": [],
            "confidence": 0.0,
            "degraded": True,
            "degradation_reason": f"detector_{detector_name}_unavailable",
            "error": str(error),
        }


__all__ = ["DegradationStrategy"]
