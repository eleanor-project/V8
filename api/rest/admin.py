from __future__ import annotations

import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from api.middleware.auth import require_role
from api.rest.deps import get_engine
from api.rest.admin_write import require_admin_write_enabled, admin_write_enabled
from engine.logging_config import get_logger
from engine.utils.dependency_tracking import get_dependency_metrics

logger = get_logger(__name__)
router = APIRouter(prefix="/admin", tags=["Admin"])

ADMIN_ROLE = os.getenv("ADMIN_ROLE", "admin")

from api.rest.admin_router import router as admin_small_router
router.include_router(admin_small_router)


@router.get("/write-enabled")
@require_role(ADMIN_ROLE)
async def get_admin_write_enabled():
    """
    Reports whether admin write operations are enabled.
    Controlled by ELEANOR_ENABLE_ADMIN_WRITE (default false).
    """
    enabled = admin_write_enabled()
    return {
        "admin_write_enabled": enabled,
        "env_var": "ELEANOR_ENABLE_ADMIN_WRITE",
        "default": False,
    }


def _resolve_environment() -> str:
    return os.getenv("ELEANOR_ENVIRONMENT") or os.getenv("ELEANOR_ENV") or "development"


@router.get("/config/health")
@require_role(ADMIN_ROLE)
async def config_health():
    """Expose configuration validation status for operators."""
    try:
        from engine.config import ConfigManager

        manager = ConfigManager()
        validation = manager.validate()
        return {
            "environment": manager.settings.environment,
            "valid": validation.get("valid", False),
            "warnings": validation.get("warnings", []),
            "errors": validation.get("errors", []),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Config validation failed: {exc}",
        )


@router.get("/cache/health")
@require_role(ADMIN_ROLE)
async def cache_health(engine=Depends(get_engine)):
    """Expose cache statistics and adaptive concurrency status."""
    cache_manager = getattr(engine, "cache_manager", None)
    router_cache = getattr(engine, "router_cache", None)
    concurrency = getattr(engine, "adaptive_concurrency", None)

    if not cache_manager:
        return {"enabled": False}

    return {
        "enabled": True,
        "redis_enabled": cache_manager.redis is not None,
        "stats": cache_manager.get_stats(),
        "router_cache": router_cache.stats() if router_cache else None,
        "concurrency": concurrency.get_stats() if concurrency else None,
    }


@router.get("/gpu/health")
@require_role(ADMIN_ROLE)
async def gpu_health(engine=Depends(get_engine)):
    """Expose GPU status and memory metrics."""
    gpu_manager = getattr(engine, "gpu_manager", None)
    gpu_enabled = bool(getattr(engine, "gpu_enabled", False))
    from engine.gpu.monitoring import collect_gpu_metrics

    metrics = collect_gpu_metrics(gpu_manager)
    metrics["configured"] = gpu_enabled
    return metrics


@router.get("/resilience/health")
@require_role(ADMIN_ROLE)
async def resilience_health(engine=Depends(get_engine)):
    """Expose circuit breaker states for engine components."""
    registry = getattr(engine, "circuit_breakers", None)
    if registry is None:
        return {"enabled": False}

    components = registry.get_all_status()
    open_count = sum(1 for status in components.values() if status.get("state") == "open")
    if not components:
        overall = "unknown"
    elif open_count == 0:
        overall = "healthy"
    elif open_count < len(components) * 0.3:
        overall = "degraded"
    else:
        overall = "unhealthy"

    return {
        "enabled": True,
        "overall_health": overall,
        "components": components,
    }


@router.get("/dependencies")
@require_role(ADMIN_ROLE)
async def dependency_health():
    """Expose counts of dependencies that failed to load."""
    metrics = get_dependency_metrics()
    failures = metrics.get("failures", {})
    has_failures = bool(failures)
    status_label = "degraded" if has_failures else "healthy"
    return {
        "failures": failures,
        "has_failures": has_failures,
        "status": status_label,
        "total_failures": metrics.get("total_failures", 0),
        "tracked_dependencies": metrics.get("tracked_dependencies", 0),
        "last_checked": metrics.get("last_checked"),
    }


@router.get("/router/health")
@require_role(ADMIN_ROLE)
async def router_health(engine=Depends(get_engine)):
    router = getattr(engine, "router", None)
    if not router:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Router not available"
        )
    health = getattr(router, "health", {})
    breakers = (
        router.get_circuit_breaker_status() if hasattr(router, "get_circuit_breaker_status") else {}
    )
    return {"health": health, "circuit_breakers": breakers, "policy": getattr(router, "policy", {})}


@router.post("/router/reset_breakers")
@require_role(ADMIN_ROLE)
async def reset_breakers(
    _write_enabled: None = Depends(require_admin_write_enabled),
    engine=Depends(get_engine),
):
    router = getattr(engine, "router", None)
    if not router:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Router not available"
        )
    if hasattr(router, "_circuit_breakers"):
        router._circuit_breakers.reset_all()
    return {"status": "ok"}


# Binding and adapter registration routes live in api/rest/admin_router.py


class FeatureFlagsResponse(BaseModel):
    """Feature flags configuration response."""

    explainable_governance: bool
    semantic_cache: bool
    intelligent_model_selection: bool
    anomaly_detection: bool
    streaming_governance: bool
    adaptive_critic_weighting: bool
    temporal_precedent_evolution: bool
    reflection: bool
    drift_check: bool
    precedent_analysis: bool


class FeatureFlagsUpdate(BaseModel):
    """Feature flags update request."""

    explainable_governance: Optional[bool] = None
    semantic_cache: Optional[bool] = None
    intelligent_model_selection: Optional[bool] = None
    anomaly_detection: Optional[bool] = None
    streaming_governance: Optional[bool] = None
    adaptive_critic_weighting: Optional[bool] = None
    temporal_precedent_evolution: Optional[bool] = None
    reflection: Optional[bool] = None
    drift_check: Optional[bool] = None
    precedent_analysis: Optional[bool] = None


@router.get("/feature-flags", response_model=FeatureFlagsResponse)
@require_role(ADMIN_ROLE)
async def get_feature_flags():
    """Get current feature flags configuration."""
    try:
        from engine.config.settings import get_settings

        settings = get_settings()
        return FeatureFlagsResponse(
            explainable_governance=settings.enable_explainable_governance,
            semantic_cache=settings.enable_semantic_cache,
            intelligent_model_selection=settings.enable_intelligent_model_selection,
            anomaly_detection=settings.enable_anomaly_detection,
            streaming_governance=settings.enable_streaming_governance,
            adaptive_critic_weighting=settings.enable_adaptive_critic_weighting,
            temporal_precedent_evolution=settings.enable_temporal_precedent_evolution,
            reflection=settings.enable_reflection,
            drift_check=settings.enable_drift_check,
            precedent_analysis=settings.enable_precedent_analysis,
        )
    except Exception as exc:
        logger.error(f"Failed to get feature flags: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get feature flags: {exc}",
        )


@router.post("/feature-flags", response_model=FeatureFlagsResponse)
@require_role(ADMIN_ROLE)
async def update_feature_flags(
    flags: FeatureFlagsUpdate,
    _write_enabled: None = Depends(require_admin_write_enabled),
):
    """Update feature flags configuration."""
    try:
        from engine.config.settings import get_settings, reload_settings
        from engine.security.audit import audit_log

        env_updates = {}
        if flags.explainable_governance is not None:
            env_updates["ELEANOR_ENABLE_EXPLAINABLE_GOVERNANCE"] = str(flags.explainable_governance).lower()
        if flags.semantic_cache is not None:
            env_updates["ELEANOR_ENABLE_SEMANTIC_CACHE"] = str(flags.semantic_cache).lower()
        if flags.intelligent_model_selection is not None:
            env_updates["ELEANOR_ENABLE_INTELLIGENT_MODEL_SELECTION"] = str(flags.intelligent_model_selection).lower()
        if flags.anomaly_detection is not None:
            env_updates["ELEANOR_ENABLE_ANOMALY_DETECTION"] = str(flags.anomaly_detection).lower()
        if flags.streaming_governance is not None:
            env_updates["ELEANOR_ENABLE_STREAMING_GOVERNANCE"] = str(flags.streaming_governance).lower()
        if flags.adaptive_critic_weighting is not None:
            env_updates["ELEANOR_ENABLE_ADAPTIVE_CRITIC_WEIGHTING"] = str(flags.adaptive_critic_weighting).lower()
        if flags.temporal_precedent_evolution is not None:
            env_updates["ELEANOR_ENABLE_TEMPORAL_PRECEDENT_EVOLUTION"] = str(flags.temporal_precedent_evolution).lower()
        if flags.reflection is not None:
            env_updates["ELEANOR_ENABLE_REFLECTION"] = str(flags.reflection).lower()
        if flags.drift_check is not None:
            env_updates["ELEANOR_ENABLE_DRIFT_CHECK"] = str(flags.drift_check).lower()
        if flags.precedent_analysis is not None:
            env_updates["ELEANOR_ENABLE_PRECEDENT_ANALYSIS"] = str(flags.precedent_analysis).lower()

        for key, value in env_updates.items():
            os.environ[key] = value

        settings = reload_settings(validate=False)

        logger.info(
            "feature_flags_updated",
            extra={"flags": {k: v for k, v in flags.model_dump().items() if v is not None}},
        )
        audit_log(
            "feature_flags_updated",
            extra={"flags": {k: v for k, v in flags.model_dump().items() if v is not None}},
        )

        return FeatureFlagsResponse(
            explainable_governance=settings.enable_explainable_governance,
            semantic_cache=settings.enable_semantic_cache,
            intelligent_model_selection=settings.enable_intelligent_model_selection,
            anomaly_detection=settings.enable_anomaly_detection,
            streaming_governance=settings.enable_streaming_governance,
            adaptive_critic_weighting=settings.enable_adaptive_critic_weighting,
            temporal_precedent_evolution=settings.enable_temporal_precedent_evolution,
            reflection=settings.enable_reflection,
            drift_check=settings.enable_drift_check,
            precedent_analysis=settings.enable_precedent_analysis,
        )
    except Exception as exc:
        logger.error(f"Failed to update feature flags: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update feature flags: {exc}",
        )


@router.get("/precedent-evolution/analytics")
@require_role(ADMIN_ROLE)
async def get_precedent_evolution_analytics(case_id: Optional[str] = None, engine=Depends(get_engine)):
    """Get analytics on precedent evolution."""
    tracker = getattr(engine, "temporal_evolution_tracker", None)
    if not tracker:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Temporal Evolution Tracking not enabled. Enable via feature flag.",
        )
    return tracker.get_evolution_analytics(case_id=case_id)


@router.get("/precedent-evolution/{case_id}/drift")
@require_role(ADMIN_ROLE)
async def detect_precedent_drift(case_id: str, engine=Depends(get_engine)):
    """Detect temporal drift for a specific precedent."""
    tracker = getattr(engine, "temporal_evolution_tracker", None)
    if not tracker:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Temporal Evolution Tracking not enabled. Enable via feature flag.",
        )
    return tracker.detect_temporal_drift(case_id)


@router.get("/precedent-evolution/recommendations")
@require_role(ADMIN_ROLE)
async def get_deprecation_recommendations(min_versions: int = 3, engine=Depends(get_engine)):
    """Get deprecation recommendations for precedents."""
    tracker = getattr(engine, "temporal_evolution_tracker", None)
    if not tracker:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Temporal Evolution Tracking not enabled. Enable via feature flag.",
        )
    recommendations = tracker.recommend_deprecations(min_versions=min_versions)
    return {"recommendations": recommendations, "count": len(recommendations)}


@router.post("/precedent-evolution/{case_id}/lifecycle")
@require_role(ADMIN_ROLE)
async def update_precedent_lifecycle(
    case_id: str,
    state: str,
    superseded_by: Optional[str] = None,
    _write_enabled: None = Depends(require_admin_write_enabled),
    engine=Depends(get_engine),
):
    """Update the lifecycle state of a precedent."""
    tracker = getattr(engine, "temporal_evolution_tracker", None)
    if not tracker:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Temporal Evolution Tracking not enabled. Enable via feature flag.",
        )

    from engine.precedent.temporal_evolution import PrecedentLifecycleState

    try:
        lifecycle_state = PrecedentLifecycleState(state.lower())
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid state: {state}. Must be one of: active, deprecated, superseded, archived",
        )

    success = tracker.set_lifecycle_state(
        case_id=case_id,
        state=lifecycle_state,
        superseded_by=superseded_by if lifecycle_state == PrecedentLifecycleState.SUPERSEDED else None,
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Precedent {case_id} not found",
        )

    return {"success": True, "case_id": case_id, "new_state": state}


@router.get("/adaptive-weighting/performance")
@require_role(ADMIN_ROLE)
async def get_adaptive_weighting_performance(engine=Depends(get_engine)):
    """Get performance report for adaptive critic weighting."""
    adaptive_weighting = getattr(engine, "adaptive_weighting", None)
    if not adaptive_weighting:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Adaptive Critic Weighting not enabled. Enable via feature flag.",
        )

    report = adaptive_weighting.get_performance_report()
    return report


@router.get("/adaptive-weighting/weights")
@require_role(ADMIN_ROLE)
async def get_critic_weights(engine=Depends(get_engine)):
    """Get current critic weights."""
    adaptive_weighting = getattr(engine, "adaptive_weighting", None)
    if not adaptive_weighting:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Adaptive Critic Weighting not enabled. Enable via feature flag.",
        )

    weights = adaptive_weighting.get_weights()
    return {"weights": weights}


@router.post("/adaptive-weighting/update")
@require_role(ADMIN_ROLE)
async def update_critic_weights(
    _write_enabled: None = Depends(require_admin_write_enabled),
    engine=Depends(get_engine),
):
    """Trigger weight update based on recent feedback."""
    adaptive_weighting = getattr(engine, "adaptive_weighting", None)
    if not adaptive_weighting:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Adaptive Critic Weighting not enabled. Enable via feature flag.",
        )

    updates = adaptive_weighting.update_weights()
    return {
        "success": True,
        "updates_count": len(updates),
        "updates": [
            {
                "critic_name": u.critic_name,
                "old_weight": u.old_weight,
                "new_weight": u.new_weight,
                "update_reason": u.update_reason,
                "confidence": u.confidence,
            }
            for u in updates
        ],
    }


@router.post("/adaptive-weighting/reset")
@require_role(ADMIN_ROLE)
async def reset_critic_weights(
    _write_enabled: None = Depends(require_admin_write_enabled),
    engine=Depends(get_engine),
):
    """Reset critic weights to default uniform weights."""
    adaptive_weighting = getattr(engine, "adaptive_weighting", None)
    if not adaptive_weighting:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Adaptive Critic Weighting not enabled. Enable via feature flag.",
        )

    adaptive_weighting.reset_weights()
    return {"success": True, "weights": adaptive_weighting.get_weights()}
