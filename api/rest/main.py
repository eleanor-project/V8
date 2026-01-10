"""
ELEANOR V8 — FastAPI REST Server
------------------------------------

This service exposes the unified deliberation engine via:

1. POST /deliberate
   → Runs a full V8 deliberation pipeline

2. GET /trace/{trace_id}
   → Retrieves an audit record (from evidence recorder)

3. POST /governance/preview
   → Runs governance (OPA) evaluation without full deliberation

4. GET /health
   → Comprehensive system-integrity check

Security Features:
- JWT authentication (configurable)
- Rate limiting
- CORS with configurable origins
- Input validation via Pydantic
- Secure error handling
"""

import os
import uuid
import asyncio
from datetime import datetime, timezone
from typing import Optional, Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

# Schemas and middleware
from api.schemas import (
    DeliberationRequest,
    GovernancePreviewRequest,
    HealthResponse,
    ErrorResponse,
)
from api.middleware.auth import (
    get_current_user,
    get_auth_config,
    require_role,
    require_authenticated_user,
)
from api.middleware.rate_limit import (
    RateLimitMiddleware,
    get_rate_limiter,
    RateLimitConfig,
    check_rate_limit,
)
from api.middleware.security_headers import SecurityHeadersMiddleware
from api.middleware.user_rate_limit import UserRateLimiter

# Logging
from engine.logging_config import configure_logging, get_logger

# Enhanced observability
try:
    from engine.observability.business_metrics import record_engine_result, record_decision
    from engine.observability.correlation import CorrelationContext, get_correlation_id
    from engine.observability.cost_tracking import record_llm_call, extract_token_usage
    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False
    record_engine_result = None
    record_decision = None
    CorrelationContext = None
    get_correlation_id = None
    record_llm_call = None
    extract_token_usage = None

# Engine bootstrap
from api.bootstrap import (
    build_engine,
    evaluate_opa,
    load_constitutional_config,
    GOVERNANCE_SCHEMA_VERSION,
)
from api.replay_store import ReplayStore
from api.rest.metrics import (
    DELIB_LATENCY,
    DELIB_REQUESTS,
    OPA_CALLS,
    CONTENT_TYPE_LATEST,
    generate_latest,
)
from engine.router.adapters import (
    GPTAdapter,
    ClaudeAdapter,
    GrokAdapter,
    LlamaHFAdapter,
    OllamaAdapter,
)
from engine.security.secrets import (
    EnvironmentSecretProvider,
    build_secret_provider_from_settings,
    get_llm_api_key,
)
from api.websocket.websocket_server import ws_router
from pydantic import BaseModel
from pathlib import Path
from engine.exceptions import InputValidationError
from engine.utils.critic_names import canonical_critic_name
from engine.utils.dependency_tracking import get_dependency_metrics

# Initialize logging
configure_logging()
logger = get_logger(__name__)

ADMIN_ROLE = os.getenv("ADMIN_ROLE", "admin")


# ---------------------------------------------
# Optional Observability (OTEL + Prometheus middleware)
# ---------------------------------------------
def enable_tracing(app: FastAPI):
    """Enable OpenTelemetry tracing if configured."""
    if os.getenv("ENABLE_OTEL", "").lower() not in ("1", "true", "yes"):
        return
    try:
        from opentelemetry import trace  # type: ignore[import-not-found]
        from opentelemetry.sdk.resources import Resource  # type: ignore[import-not-found]
        from opentelemetry.sdk.trace import TracerProvider  # type: ignore[import-not-found]
        from opentelemetry.sdk.trace.export import BatchSpanProcessor  # type: ignore[import-not-found]
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # type: ignore[import-not-found]
            OTLPSpanExporter,
        )
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor  # type: ignore[import-not-found]

        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318/v1/traces")
        resource = Resource.create(
            {"service.name": os.getenv("OTEL_SERVICE_NAME", "eleanor-v8-api")}
        )
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)
        logger.info(f"OpenTelemetry tracing enabled at {endpoint}")
    except Exception as exc:
        logger.warning(f"Failed to enable OpenTelemetry: {exc}")


def enable_prometheus_middleware(app: FastAPI):
    """Enable Prometheus instrumentation middleware if available."""
    if os.getenv("ENABLE_PROMETHEUS_MIDDLEWARE", "").lower() not in ("1", "true", "yes"):
        return
    try:
        from prometheus_fastapi_instrumentator import Instrumentator  # type: ignore[import-not-found]

        Instrumentator().instrument(app).expose(app, include_in_schema=False)
        logger.info("Prometheus middleware instrumentation enabled")
    except Exception as exc:
        logger.warning(f"Failed to enable Prometheus middleware: {exc}")


# ---------------------------------------------
# Configuration
# ---------------------------------------------


def _resolve_environment() -> str:
    return os.getenv("ELEANOR_ENVIRONMENT") or os.getenv("ELEANOR_ENV") or "development"


def get_cors_origins() -> list:
    """Get allowed CORS origins from environment."""
    origins_str = os.getenv("CORS_ORIGINS", "")
    if not origins_str:
        # Default to localhost in development
        env = _resolve_environment()
        if env == "development":
            return ["http://localhost:3000", "http://localhost:8080", "http://127.0.0.1:3000"]
        # In production, require explicit configuration
        return []
    return [origin.strip() for origin in origins_str.split(",") if origin.strip()]


def get_constitutional_config() -> dict:
    """Load constitutional configuration from YAML file."""
    try:
        from engine.config import ConfigManager

        config_path = ConfigManager().settings.constitutional_config_path
    except Exception:
        config_path = os.getenv("CONSTITUTIONAL_CONFIG_PATH", "governance/constitutional.yaml")
    return load_constitutional_config(config_path)


def _ensure_writable_path(path: str | Path):
    """Check that a path is writable by attempting an atomic write."""
    Path(path).mkdir(parents=True, exist_ok=True)
    probe = Path(path) / ".write_probe"
    probe.write_text("ok")
    probe.unlink(missing_ok=True)


def run_readiness_checks() -> Dict[str, str]:
    """
    Run startup readiness checks for security and storage dependencies.
    Raises RuntimeError if a required check fails.
    """
    env = _resolve_environment()
    results: Dict[str, str] = {}
    issues = []

    # Auth configuration
    try:
        auth_config = get_auth_config()
        results["auth"] = "enabled" if auth_config.enabled else "disabled"
        if env != "development" and not auth_config.enabled:
            issues.append("auth_disabled_in_production")
    except Exception as exc:
        results["auth"] = f"error:{exc}"
        issues.append(f"auth_error:{exc}")

    # Rate limiting
    try:
        rl_config = RateLimitConfig.from_env()
        results["rate_limit"] = "enabled" if rl_config.enabled else "disabled"
    except Exception as exc:
        results["rate_limit"] = f"error:{exc}"
        issues.append(f"rate_limit_error:{exc}")

    # Configuration validation
    try:
        from engine.config import ConfigManager

        config_manager = ConfigManager()
        validation = config_manager.validate()
        results["config"] = "ok" if validation.get("valid", False) else "invalid"
        if not validation.get("valid", False):
            issues.append("config_invalid")
    except Exception as exc:
        results["config"] = f"error:{exc}"
        issues.append(f"config_error:{exc}")

    # CORS configuration
    try:
        cors_origins = get_cors_origins()
        results["cors"] = "configured" if cors_origins else "missing"
        if env != "development" and not cors_origins:
            issues.append("cors_not_configured")
            logger.error(
                "CORS not configured in production. Set CORS_ORIGINS environment variable.",
                extra={"environment": env},
            )
    except Exception as exc:
        results["cors"] = f"error:{exc}"
        issues.append(f"cors_error:{exc}")

    # Precedent backend
    try:
        precedent_backend = os.getenv("PRECEDENT_BACKEND", "memory").lower()
        results["precedent_backend"] = precedent_backend
        if env != "development" and precedent_backend == "memory":
            issues.append("precedent_backend_memory")
    except Exception as exc:
        results["precedent_backend"] = f"error:{exc}"
        issues.append(f"precedent_backend_error:{exc}")

    # Storage write checks for governance audit
    try:
        # replay_store has per-instance log path; packet/review dirs are module-level constants
        from api.replay_store import REVIEW_PACKET_DIR, REVIEW_RECORD_DIR

        _ensure_writable_path(replay_store.path.parent)
        _ensure_writable_path(REVIEW_PACKET_DIR)
        _ensure_writable_path(REVIEW_RECORD_DIR)
        exec_audit_path = Path(os.getenv("ELEANOR_EXEC_AUDIT_PATH", "logs/execution_audit.jsonl"))
        _ensure_writable_path(exec_audit_path.parent)
        results["storage"] = "ok"
    except Exception as exc:
        results["storage"] = f"error:{exc}"
        issues.append(f"storage_error:{exc}")

    if issues:
        logger.error(
            "Readiness checks failed",
            extra={"issues": issues, "results": results},
        )
        raise RuntimeError(f"Readiness checks failed: {issues}")

    logger.info("Readiness checks passed", extra={"results": results})
    return results


# ---------------------------------------------
# Engine initialization
# ---------------------------------------------

# Global engine instance
engine = None
replay_store = ReplayStore(path=os.getenv("REPLAY_LOG_PATH", "replay_log.jsonl"))
secret_provider = None


def initialize_engine():
    """Initialize the ELEANOR engine."""
    global engine, secret_provider

    try:
        constitution = get_constitutional_config()
        engine = build_engine(constitution)

        # Allow explicitly disabling OPA via env for local/dev setups
        if os.getenv("ELEANOR_DISABLE_OPA", "").lower() in ("1", "true", "yes"):
            setattr(engine, "opa_callback", None)
            logger.info("OPA disabled via ELEANOR_DISABLE_OPA")

        settings = getattr(engine, "settings", None)
        if settings is None:
            try:
                from engine.config import ConfigManager

                settings = ConfigManager().settings
            except Exception:
                settings = None

        if settings is not None:
            try:
                secret_provider = build_secret_provider_from_settings(settings)
            except Exception as exc:
                if settings.environment == "production":
                    raise
                secret_provider = EnvironmentSecretProvider(
                    cache_ttl=settings.security.secrets_cache_ttl
                )
                logger.warning(
                    "Secret provider setup failed; using environment provider",
                    extra={"error": str(exc)},
                )
        else:
            secret_provider = EnvironmentSecretProvider(cache_ttl=300)

        logger.info("ELEANOR V8 engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize engine: {e}")
        raise



async def _secret_refresh_loop(provider):
    refresh_interval = max(getattr(provider, "cache_ttl", 300), 60)
    while True:
        await asyncio.sleep(refresh_interval)
        try:
            await provider.refresh_secrets()
            logger.info("Secret cache refreshed")
        except Exception as exc:
            logger.error("Secret refresh failed", extra={"error": str(exc)})


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting ELEANOR V8 API server...")
    initialize_engine()
    skip_readiness = os.getenv("ELEANOR_SKIP_READINESS", "").lower() in ("1", "true", "yes")
    if not skip_readiness:
        run_readiness_checks()

    # Expose runtime objects to routers via app.state
    app.state.engine = engine
    app.state.replay_store = replay_store
    app.state.secret_provider = secret_provider
    
    # Ensure engine uses async context manager for proper resource initialization
    if engine is not None:
        if hasattr(engine, "__aenter__"):
            await engine.__aenter__()
        elif hasattr(engine, "_setup_resources"):
            await engine._setup_resources()
        else:
            # Fallback: manually initialize resources
            from engine.runtime.lifecycle import setup_resources
            await setup_resources(engine)
    
    refresh_task = None
    if secret_provider is not None:
        refresh_task = asyncio.create_task(_secret_refresh_loop(secret_provider))
    yield
    # Shutdown with timeout
    shutdown_timeout = float(os.getenv("SHUTDOWN_TIMEOUT_SECONDS", "30.0"))
    if engine is not None:
        if hasattr(engine, "shutdown"):
            try:
                await asyncio.wait_for(engine.shutdown(), timeout=shutdown_timeout)
            except asyncio.TimeoutError:
                logger.error(
                    "engine_shutdown_timeout",
                    extra={"timeout": shutdown_timeout},
                )
        elif hasattr(engine, "__aexit__"):
            # Use context manager if available
            try:
                await asyncio.wait_for(engine.__aexit__(None, None, None), timeout=shutdown_timeout)
            except asyncio.TimeoutError:
                logger.error(
                    "engine_shutdown_timeout",
                    extra={"timeout": shutdown_timeout},
                )
    
    if refresh_task is not None:
        refresh_task.cancel()
        try:
            await asyncio.wait_for(refresh_task, timeout=5.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
    logger.info("Shutting down ELEANOR V8 API server...")


# ---------------------------------------------
# FastAPI Application
# ---------------------------------------------

app = FastAPI(
    title="ELEANOR V8 — Governance API",
    description="Responsible multi-critic constitutional deliberation engine",
    version="8.0.0",
    lifespan=lifespan,
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)

enable_tracing(app)
enable_prometheus_middleware(app)

# Serve React SPA if present
if os.path.isdir("ui"):
    ui_dir = "ui/dist" if os.path.isdir("ui/dist") else "ui"
    app.mount("/ui", StaticFiles(directory=ui_dir, html=True), name="ui")

# Register human review router
try:
    from api.rest.review import router as review_router
    from api.rest.governance import router as governance_router
    from api.rest.routes.deliberation import router as deliberation_router
    from api.rest.routes.audit import router as audit_router

    app.include_router(review_router)
    app.include_router(governance_router)
    app.include_router(deliberation_router)
    app.include_router(audit_router)
    logger.info("Human review API endpoints registered")
except Exception as e:
    logger.warning(f"Failed to register review router: {e}")

# Register websocket router
app.include_router(ws_router)


# ---------------------------------------------
# Security Headers Middleware
# ---------------------------------------------
app.add_middleware(SecurityHeadersMiddleware)


# ---------------------------------------------
# CORS Configuration (Secure)
# ---------------------------------------------

cors_origins = get_cors_origins()
if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        max_age=600,  # Cache preflight for 10 minutes
    )
else:
    logger.warning(
        "No CORS origins configured. Set CORS_ORIGINS environment variable "
        "for cross-origin requests."
    )


# ---------------------------------------------
# Rate Limiting Middleware
# ---------------------------------------------

app.add_middleware(RateLimitMiddleware, limiter=get_rate_limiter())


# ---------------------------------------------
# Exception Handlers
# ---------------------------------------------


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions without leaking internal details."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail if exc.status_code < 500 else "Internal server error",
            "trace_id": request.headers.get("X-Request-ID"),
        },
        headers=getattr(exc, "headers", None),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Handle unexpected exceptions without leaking internal details.
    
    This is a catch-all handler for exceptions that weren't handled
    by more specific handlers. All errors are logged with full context
    but sanitized responses are returned to clients.
    """
    trace_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

    # Classify error type for better handling
    error_type = type(exc).__name__
    is_known_error = isinstance(
        exc,
        (
            HTTPException,
            InputValidationError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
        ),
    )

    # Log the full error internally with context
    log_level = "warning" if is_known_error else "error"
    getattr(logger, log_level)(
        "unhandled_exception",
        extra={
            "trace_id": trace_id,
            "error": str(exc),
            "error_type": error_type,
            "path": request.url.path,
            "method": request.method,
            "is_known_error": is_known_error,
        },
        exc_info=not is_known_error,  # Full traceback for unknown errors
    )

    # Return sanitized error to client
    # Known errors get more specific messages
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail if exc.status_code < 500 else "Internal server error",
                "trace_id": trace_id,
            },
            headers=getattr(exc, "headers", None),
        )

    # Unknown errors get generic message
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred. Please try again later.",
            "trace_id": trace_id,
        },
    )


# ---------------------------------------------
# Health Endpoint
# ---------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Comprehensive health check endpoint.

    Returns status of all subsystems.
    """
    checks = {
        "api": "ok",
        "engine": "unknown",
        "opa": "unknown",
        "precedent_store": "unknown",
    }

    # Check engine
    if engine is not None:
        checks["engine"] = "ok"

        # Check OPA
        opa_cb = getattr(engine, "opa_callback", None)
        if opa_cb is None:
            checks["opa"] = "not_configured"
        else:
            try:
                opa_probe = await evaluate_opa(opa_cb, {"health": True})
                if opa_probe.get("failures"):
                    checks["opa"] = "error"
                elif opa_probe.get("escalate"):
                    checks["opa"] = "degraded"
                else:
                    checks["opa"] = "ok"
            except Exception as e:
                logger.warning(f"OPA health check failed: {e}")
                checks["opa"] = "error"

        # Check precedent store
        retriever = getattr(engine, "precedent_retriever", None)
        if retriever is None:
            checks["precedent_store"] = "not_configured"
        else:
            try:
                retriever.retrieve("health check", [])
                checks["precedent_store"] = "ok"
            except Exception as e:
                logger.warning(f"Precedent store health check failed: {e}")
                checks["precedent_store"] = "error"
    else:
        checks["engine"] = "not_initialized"

    # Security/storage readiness
    try:
        readiness = run_readiness_checks()
        checks.update({f"readiness_{k}": v for k, v in readiness.items()})
    except Exception as exc:
        checks["readiness"] = f"error:{exc}"

    # Determine overall status
    okish = {"ok", "not_configured", "disabled", "enabled"}
    if all(v in okish for v in checks.values()):
        overall_status = "healthy"
    elif checks["engine"] == "ok" and checks["api"] == "ok":
        overall_status = "degraded"
    else:
        overall_status = "unhealthy"

    return {
        "status": overall_status,
        "version": "8.0.0",
        "checks": checks,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


# ---------------------------------------------
# Deliberation routes are defined in api/rest/routes/deliberation.py
# ---------------------------------------------

# ---------------------------------------------
# Audit routes are defined in api/rest/routes/audit.py
# ---------------------------------------------


# ---------------------------------------------
# Governance Preview Endpoint
# ---------------------------------------------


@app.post("/governance/preview", tags=["Governance"])
async def governance_preview(
    payload: GovernancePreviewRequest,
    user: str = Depends(require_authenticated_user),
):
    """
    Run governance evaluation on a mock evidence bundle.

    This is useful for testing OPA policies without running
    the full deliberation pipeline.
    """
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Engine not initialized",
        )

    try:
        opa_callback = getattr(engine, "opa_callback", None) or getattr(engine, "opa", None)

        if opa_callback is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="OPA callback not configured",
            )

        payload_dict = payload.model_dump()
        payload_dict["schema_version"] = GOVERNANCE_SCHEMA_VERSION
        result = await evaluate_opa(opa_callback, payload_dict)
        if OPA_CALLS:
            opa_outcome = (
                "deny"
                if not result.get("allow", True)
                else "escalate"
                if result.get("escalate")
                else "allow"
            )
            OPA_CALLS.labels(result=opa_outcome).inc()
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Governance preview failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Governance evaluation failed",
        )


# ---------------------------------------------
# Metrics Endpoint (Optional)
# ---------------------------------------------


@app.get("/metrics", tags=["System"], include_in_schema=False)
async def metrics():
    """
    Lightweight metrics snapshot for operators.
    """
    from api.middleware.rate_limit import get_rate_limiter

    limiter = get_rate_limiter()
    router = getattr(engine, "router", None) if engine else None
    if generate_latest and DELIB_REQUESTS:
        data = generate_latest()
        return PlainTextResponse(data, media_type=CONTENT_TYPE_LATEST)

    return {
        "rate_limit": {
            "enabled": limiter.config.enabled,
            "requests_per_window": limiter.config.requests_per_window,
            "window_seconds": limiter.config.window_seconds,
        },
        "engine_status": "initialized" if engine else "not_initialized",
        "adapter_count": len(getattr(router, "adapters", {}) or {}),
        "critics": list(getattr(engine, "critics", {}).keys()) if engine else [],
    }


# ---------------------------------------------
# Admin Endpoints (router health + critic bindings)
# ---------------------------------------------


@app.get("/admin/config/health", tags=["Admin"])
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


@app.get("/admin/cache/health", tags=["Admin"])
@require_role(ADMIN_ROLE)
async def cache_health():
    """Expose cache statistics and adaptive concurrency status."""
    if engine is None:
        return {"enabled": False, "reason": "engine_not_initialized"}

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


@app.get("/admin/gpu/health", tags=["Admin"])
@require_role(ADMIN_ROLE)
async def gpu_health():
    """Expose GPU status and memory metrics."""
    if engine is None:
        return {"enabled": False, "reason": "engine_not_initialized"}

    gpu_manager = getattr(engine, "gpu_manager", None)
    gpu_enabled = bool(getattr(engine, "gpu_enabled", False))
    from engine.gpu.monitoring import collect_gpu_metrics

    metrics = collect_gpu_metrics(gpu_manager)
    metrics["configured"] = gpu_enabled
    return metrics


@app.get("/admin/resilience/health", tags=["Admin"])
@require_role(ADMIN_ROLE)
async def resilience_health():
    """Expose circuit breaker states for engine components."""
    if engine is None:
        return {"enabled": False, "reason": "engine_not_initialized"}

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


@app.get("/admin/dependencies", tags=["Admin"])
@require_role(ADMIN_ROLE)
async def dependency_health():
    """Expose counts of dependencies that failed to load."""
    metrics = get_dependency_metrics()
    failures = metrics.get("failures", {})
    has_failures = bool(failures)
    status = "degraded" if has_failures else "healthy"
    return {
        "failures": failures,
        "has_failures": has_failures,
        "status": status,
        "total_failures": metrics.get("total_failures", 0),
        "tracked_dependencies": metrics.get("tracked_dependencies", 0),
        "last_checked": metrics.get("last_checked"),
    }


@app.get("/admin/router/health", tags=["Admin"])
@require_role(ADMIN_ROLE)
async def router_health():
    if engine is None or not getattr(engine, "router", None):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Router not available"
        )
    router = engine.router
    health = getattr(router, "health", {})
    breakers = (
        router.get_circuit_breaker_status() if hasattr(router, "get_circuit_breaker_status") else {}
    )
    return {"health": health, "circuit_breakers": breakers, "policy": getattr(router, "policy", {})}


@app.post("/admin/router/reset_breakers", tags=["Admin"])
@require_role(ADMIN_ROLE)
async def reset_breakers():
    if engine is None or not getattr(engine, "router", None):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Router not available"
        )
    router = engine.router
    if hasattr(router, "_circuit_breakers"):
        router._circuit_breakers.reset_all()
    return {"status": "ok"}


@app.get("/admin/critics/bindings", tags=["Admin"])
@require_role(ADMIN_ROLE)
async def get_critic_bindings():
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Engine not initialized"
        )
    router = getattr(engine, "router", None)
    adapters = getattr(router, "adapters", {}) if router else {}
    return {
        "bindings": getattr(engine, "critic_models", {}),
        "critics": list(getattr(engine, "critics", {}).keys()),
        "available_adapters": list(adapters.keys()),
    }


class CriticBinding(BaseModel):
    critic: str
    adapter: str


@app.post("/admin/critics/bindings", tags=["Admin"])
@require_role(ADMIN_ROLE)
async def set_critic_binding(binding: CriticBinding):
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Engine not initialized"
        )
    router = getattr(engine, "router", None)
    adapters = getattr(router, "adapters", {}) if router else {}
    adapter_fn = adapters.get(binding.adapter)
    if adapter_fn is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Adapter '{binding.adapter}' not registered",
        )
    critic_key = canonical_critic_name(binding.critic)
    engine.critic_models[critic_key] = adapter_fn
    return {"status": "ok", "binding": {critic_key: binding.adapter}}


class AdapterRegistration(BaseModel):
    name: str
    type: str = "ollama"  # ollama | hf | openai | claude | grok
    model: Optional[str] = None
    device: Optional[str] = None
    api_key: Optional[str] = None


@app.post("/admin/router/adapters", tags=["Admin"])
@require_role(ADMIN_ROLE)
async def register_adapter(adapter: AdapterRegistration):
    if engine is None or not getattr(engine, "router", None):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Router not available"
        )

    router = engine.router
    adapters = getattr(router, "adapters", {}) if router else {}

    factory = adapter.type.lower()
    try:
        if factory == "ollama":
            adapters[adapter.name] = OllamaAdapter(model=adapter.model or adapter.name)
        elif factory == "hf":
            adapters[adapter.name] = LlamaHFAdapter(
                model_path=adapter.model or adapter.name, device=adapter.device or "cpu"
            )
        elif factory == "openai":
            api_key = adapter.api_key
            if api_key is None and secret_provider is not None:
                api_key = await get_llm_api_key("openai", secret_provider)
            if api_key is None and _resolve_environment() != "production":
                api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
            if api_key is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail="OpenAI API key not available"
                )
            adapters[adapter.name] = GPTAdapter(
                model=adapter.model or "gpt-4o-mini", api_key=api_key
            )
        elif factory == "claude":
            api_key = adapter.api_key
            if api_key is None and secret_provider is not None:
                api_key = await get_llm_api_key("anthropic", secret_provider)
            if api_key is None and _resolve_environment() != "production":
                api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_KEY")
            if api_key is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Anthropic API key not available",
                )
            adapters[adapter.name] = ClaudeAdapter(
                model=adapter.model or "claude-3-5-sonnet-20241022", api_key=api_key
            )
        elif factory == "grok":
            api_key = adapter.api_key
            if api_key is None and secret_provider is not None:
                api_key = await get_llm_api_key("xai", secret_provider)
            if api_key is None and _resolve_environment() != "production":
                api_key = os.getenv("XAI_API_KEY") or os.getenv("XAI_KEY")
            if api_key is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail="xAI API key not available"
                )
            adapters[adapter.name] = GrokAdapter(
                model=adapter.model or "grok-beta", api_key=api_key
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown adapter type '{adapter.type}'",
            )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to register adapter: {exc}"
        )

    # ensure fallback order includes the adapter
    policy = getattr(router, "policy", {}) or {}
    fallback = policy.get("fallback_order") or []
    if adapter.name not in fallback and adapter.name != policy.get("primary"):
        fallback.append(adapter.name)
    policy["fallback_order"] = fallback
    router.policy = policy

    return {
        "status": "ok",
        "registered": adapter.name,
        "type": adapter.type,
        "available_adapters": list(adapters.keys()),
        "policy": router.policy,
    }


# ---------------------------------------------
# Explainable Governance Endpoints
# ---------------------------------------------


@app.get("/explanation/{trace_id}", tags=["Governance"])
async def get_explanation(
    trace_id: str,
    detail_level: str = "summary",
    user: str = Depends(require_authenticated_user),
):
    """
    Get explainable governance explanation for a trace.
    
    Args:
        trace_id: Trace ID to get explanation for
        detail_level: Level of detail (summary, detailed, interactive)
    
    Returns:
        Explanation of the governance decision for the trace
    """
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Engine not initialized"
        )
    
    # Check if explainable governance is enabled
    if not hasattr(engine, "explainable_governance"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Explainable Governance is not enabled. Enable via feature flags."
        )
    
    # Retrieve trace data
    try:
        trace_data = await replay_store.get_async(trace_id)
        if not trace_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Trace {trace_id} not found"
            )
        
        response_data = trace_data.get("response") or trace_data
        
        # Generate explanation
        from engine.core.feature_integration import get_explanation_for_result
        explanation = get_explanation_for_result(
            engine,
            response_data,
            detail_level=detail_level
        )
        
        if not explanation:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate explanation"
            )
        
        return explanation
    
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "explanation_retrieval_failed",
            extra={"trace_id": trace_id, "error": str(exc)},
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve explanation: {str(exc)}"
        )


# ---------------------------------------------
# Feature Flags Management
# ---------------------------------------------


class FeatureFlagsResponse(BaseModel):
    """Feature flags configuration response."""
    explainable_governance: bool
    semantic_cache: bool
    intelligent_model_selection: bool
    anomaly_detection: bool
    streaming_governance: bool
    adaptive_critic_weighting: bool
    temporal_precedent_evolution: bool
    # Legacy feature flags
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


@app.get("/admin/feature-flags", tags=["Admin"], response_model=FeatureFlagsResponse)
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


@app.post("/admin/feature-flags", tags=["Admin"], response_model=FeatureFlagsResponse)
@require_role(ADMIN_ROLE)
async def update_feature_flags(flags: FeatureFlagsUpdate):
    """Update feature flags configuration."""
    try:
        from engine.config.settings import get_settings, reload_settings
        import os
        from engine.security.audit import audit_log
        
        # Update environment variables (persists across restarts if .env file exists)
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
        
        # Update environment variables in current process
        for key, value in env_updates.items():
            os.environ[key] = value
        
        # Reload settings to pick up changes
        settings = reload_settings(validate=False)
        
        # Note: Runtime changes only affect current process
        # To persist, the .env file should be updated or settings should be stored in a database
        logger.info(
            "feature_flags_updated",
            extra={"flags": {k: v for k, v in flags.model_dump().items() if v is not None}}
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


# ---------------------------------------------
# Advanced Features Endpoints
# ---------------------------------------------


@app.get("/admin/precedent-evolution/analytics", tags=["Admin"])
@require_role(ADMIN_ROLE)
async def get_precedent_evolution_analytics(case_id: Optional[str] = None):
    """Get analytics on precedent evolution."""
    try:
        if engine is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Engine not available"
            )
        
        tracker = getattr(engine, "temporal_evolution_tracker", None)
        if not tracker:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Temporal Evolution Tracking not enabled. Enable via feature flag."
            )
        
        analytics = tracker.get_evolution_analytics(case_id=case_id)
        return analytics
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Failed to get precedent evolution analytics: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get analytics: {exc}",
        )


@app.get("/admin/precedent-evolution/{case_id}/drift", tags=["Admin"])
@require_role(ADMIN_ROLE)
async def detect_precedent_drift(case_id: str):
    """Detect temporal drift for a specific precedent."""
    try:
        if engine is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Engine not available"
            )
        
        tracker = getattr(engine, "temporal_evolution_tracker", None)
        if not tracker:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Temporal Evolution Tracking not enabled. Enable via feature flag."
            )
        
        drift_info = tracker.detect_temporal_drift(case_id)
        return drift_info
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Failed to detect drift: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to detect drift: {exc}",
        )


@app.get("/admin/precedent-evolution/recommendations", tags=["Admin"])
@require_role(ADMIN_ROLE)
async def get_deprecation_recommendations(min_versions: int = 3):
    """Get deprecation recommendations for precedents."""
    try:
        if engine is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Engine not available"
            )
        
        tracker = getattr(engine, "temporal_evolution_tracker", None)
        if not tracker:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Temporal Evolution Tracking not enabled. Enable via feature flag."
            )
        
        recommendations = tracker.recommend_deprecations(min_versions=min_versions)
        return {"recommendations": recommendations, "count": len(recommendations)}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Failed to get recommendations: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get recommendations: {exc}",
        )


@app.post("/admin/precedent-evolution/{case_id}/lifecycle", tags=["Admin"])
@require_role(ADMIN_ROLE)
async def update_precedent_lifecycle(
    case_id: str,
    state: str,  # "active", "deprecated", "superseded", "archived"
    superseded_by: Optional[str] = None,
):
    """Update the lifecycle state of a precedent."""
    try:
        if engine is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Engine not available"
            )
        
        tracker = getattr(engine, "temporal_evolution_tracker", None)
        if not tracker:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Temporal Evolution Tracking not enabled. Enable via feature flag."
            )
        
        from engine.precedent.temporal_evolution import PrecedentLifecycleState
        
        try:
            lifecycle_state = PrecedentLifecycleState(state.lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid state: {state}. Must be one of: active, deprecated, superseded, archived"
            )
        
        success = tracker.set_lifecycle_state(
            case_id=case_id,
            state=lifecycle_state,
            superseded_by=superseded_by if lifecycle_state == PrecedentLifecycleState.SUPERSEDED else None
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Precedent {case_id} not found"
            )
        
        return {"success": True, "case_id": case_id, "new_state": state}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Failed to update lifecycle: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update lifecycle: {exc}",
        )


@app.get("/admin/adaptive-weighting/performance", tags=["Admin"])
@require_role(ADMIN_ROLE)
async def get_adaptive_weighting_performance():
    """Get performance report for adaptive critic weighting."""
    try:
        if engine is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Engine not available"
            )
        
        adaptive_weighting = getattr(engine, "adaptive_weighting", None)
        if not adaptive_weighting:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Adaptive Critic Weighting not enabled. Enable via feature flag."
            )
        
        report = adaptive_weighting.get_performance_report()
        return report
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Failed to get performance report: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get performance report: {exc}",
        )


@app.get("/admin/adaptive-weighting/weights", tags=["Admin"])
@require_role(ADMIN_ROLE)
async def get_critic_weights():
    """Get current critic weights."""
    try:
        if engine is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Engine not available"
            )
        
        adaptive_weighting = getattr(engine, "adaptive_weighting", None)
        if not adaptive_weighting:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Adaptive Critic Weighting not enabled. Enable via feature flag."
            )
        
        weights = adaptive_weighting.get_weights()
        return {"weights": weights}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Failed to get weights: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get weights: {exc}",
        )


@app.post("/admin/adaptive-weighting/update", tags=["Admin"])
@require_role(ADMIN_ROLE)
async def update_critic_weights():
    """Trigger weight update based on recent feedback."""
    try:
        if engine is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Engine not available"
            )
        
        adaptive_weighting = getattr(engine, "adaptive_weighting", None)
        if not adaptive_weighting:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Adaptive Critic Weighting not enabled. Enable via feature flag."
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
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Failed to update weights: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update weights: {exc}",
        )


@app.post("/admin/adaptive-weighting/reset", tags=["Admin"])
@require_role(ADMIN_ROLE)
async def reset_critic_weights():
    """Reset critic weights to default uniform weights."""
    try:
        if engine is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Engine not available"
            )
        
        adaptive_weighting = getattr(engine, "adaptive_weighting", None)
        if not adaptive_weighting:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Adaptive Critic Weighting not enabled. Enable via feature flag."
            )
        
        adaptive_weighting.reset_weights()
        return {"success": True, "weights": adaptive_weighting.get_weights()}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Failed to reset weights: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset weights: {exc}",
        )


# ---------------------------------------------
# Development Server Entry Point
# ---------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")

    uvicorn.run(
        "api.rest.main:app",
        host=host,
        port=port,
        reload=_resolve_environment() == "development",
    )
