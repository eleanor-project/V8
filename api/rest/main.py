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
from typing import Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

# Schemas and middleware
from api.schemas import (
    HealthResponse,
    ErrorResponse,
)
from api.middleware.auth import (
    get_current_user,
    get_auth_config,
)
from api.middleware.rate_limit import (
    RateLimitMiddleware,
    get_rate_limiter,
    RateLimitConfig,
    check_rate_limit,
)
from api.middleware.admin_headers import AdminWriteHeaderMiddleware
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
)
from api.replay_store import ReplayStore
from api.rest.metrics import (
    DELIB_REQUESTS,
    CONTENT_TYPE_LATEST,
    generate_latest,
)
from api.rest.services.deliberation_utils import (
    resolve_execution_decision,
    apply_execution_gate,
)
from engine.security.secrets import (
    EnvironmentSecretProvider,
    build_secret_provider_from_settings,
)
from api.websocket.websocket_server import ws_router
from pathlib import Path
from engine.exceptions import InputValidationError
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
    from api.rest.admin import router as admin_router
    from api.rest.explainable import router as explainable_router

    app.include_router(review_router)
    app.include_router(governance_router)
    app.include_router(deliberation_router)
    app.include_router(audit_router)
    from api.rest.routes.config_proposals import router as config_proposals_router
    app.include_router(config_proposals_router)
    app.include_router(admin_router)
    app.include_router(explainable_router)
    logger.info("Human review API endpoints registered")
except Exception as e:
    logger.warning(f"Failed to register review router: {e}")

# Register websocket router
app.include_router(ws_router)


# ---------------------------------------------
# Security Headers Middleware
# ---------------------------------------------
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(AdminWriteHeaderMiddleware)


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


# Governance preview route lives in api/rest/governance.py


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


# Governance, admin, explainable governance routes live in dedicated routers


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
