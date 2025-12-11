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
import json
import uuid
import time
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Schemas and middleware
from api.schemas import (
    DeliberationRequest,
    GovernancePreviewRequest,
    DeliberationResponse,
    HealthResponse,
    ErrorResponse,
)
from api.middleware.auth import verify_token, get_current_user, TokenPayload
from api.middleware.rate_limit import check_rate_limit, RateLimitMiddleware, get_rate_limiter

# Logging
from engine.logging_config import configure_logging, get_logger, log_request, log_deliberation

# Engine Builder
from engine.core import build_eleanor_engine_v8

# Initialize logging
configure_logging()
logger = get_logger(__name__)


# ---------------------------------------------
# Configuration
# ---------------------------------------------

def get_cors_origins() -> list:
    """Get allowed CORS origins from environment."""
    origins_str = os.getenv("CORS_ORIGINS", "")
    if not origins_str:
        # Default to localhost in development
        env = os.getenv("ELEANOR_ENV", "development")
        if env == "development":
            return ["http://localhost:3000", "http://localhost:8080", "http://127.0.0.1:3000"]
        # In production, require explicit configuration
        return []
    return [origin.strip() for origin in origins_str.split(",") if origin.strip()]


def get_constitutional_config() -> dict:
    """Load constitutional configuration from YAML file."""
    import yaml

    config_path = os.getenv("CONSTITUTIONAL_CONFIG_PATH", "governance/constitutional.yaml")
    path = Path(config_path)

    if not path.exists():
        logger.error(f"Constitutional config not found at {config_path}")
        raise FileNotFoundError(f"Constitutional config not found: {config_path}")

    if not path.is_file():
        raise ValueError(f"Constitutional config path is not a file: {config_path}")

    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------
# Placeholder implementations
# (Replace with real implementations in production)
# ---------------------------------------------

class DummyStore:
    """Placeholder precedent store for development."""

    def search(self, q: str, top_k: int = 5) -> list:
        logger.debug(f"DummyStore.search called with q={q[:50]}..., top_k={top_k}")
        return []


def opa_eval_stub(payload: dict) -> dict:
    """Placeholder OPA evaluator for development."""
    logger.debug("OPA eval stub called")
    return {"allow": True, "failures": [], "escalate": False}


def llm_adapter(prompt: str) -> str:
    """Placeholder LLM adapter for development."""
    logger.debug(f"LLM adapter called with prompt length: {len(prompt)}")
    return json.dumps({
        "score": 1.0,
        "violation": False,
        "details": {"notes": "Placeholder LLM response"}
    })


# ---------------------------------------------
# Engine initialization
# ---------------------------------------------

# Global engine instance
engine = None


def initialize_engine():
    """Initialize the ELEANOR engine."""
    global engine

    try:
        constitution = get_constitutional_config()

        router_adapters = {
            "primary": llm_adapter,
            "backup": llm_adapter
        }

        router_policy = {
            "primary": "primary",
            "fallback_order": ["backup"],
            "max_retries": 2
        }

        precedent_store = DummyStore()

        engine = build_eleanor_engine_v8(
            llm_fn=llm_adapter,
            constitutional_config=constitution,
            router_adapters=router_adapters,
            router_policy=router_policy,
            precedent_store=precedent_store,
            opa_callback=opa_eval_stub,
        )

        logger.info("ELEANOR V8 engine initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize engine: {e}")
        raise


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting ELEANOR V8 API server...")
    initialize_engine()
    yield
    # Shutdown
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
    }
)


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
    """Handle unexpected exceptions without leaking internal details."""
    trace_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

    # Log the full error internally
    logger.error(
        "Unhandled exception",
        trace_id=trace_id,
        error=str(exc),
        error_type=type(exc).__name__,
        path=request.url.path,
    )

    # Return sanitized error to client
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

        # Check OPA (stub for now)
        try:
            test_result = engine.opa_callback({"test": True})
            checks["opa"] = "ok" if "allow" in test_result else "degraded"
        except Exception as e:
            logger.warning(f"OPA health check failed: {e}")
            checks["opa"] = "error"

        # Check precedent store
        try:
            engine.precedent_retriever.retrieve("health check")
            checks["precedent_store"] = "ok"
        except Exception as e:
            logger.warning(f"Precedent store health check failed: {e}")
            checks["precedent_store"] = "error"
    else:
        checks["engine"] = "not_initialized"

    # Determine overall status
    if all(v == "ok" for v in checks.values()):
        overall_status = "healthy"
    elif checks["engine"] == "ok" and checks["api"] == "ok":
        overall_status = "degraded"
    else:
        overall_status = "unhealthy"

    return {
        "status": overall_status,
        "version": "8.0.0",
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


# ---------------------------------------------
# Main Deliberation Endpoint
# ---------------------------------------------

@app.post("/deliberate", tags=["Deliberation"])
async def deliberate(
    request: Request,
    payload: DeliberationRequest,
    user: Optional[str] = Depends(get_current_user),
    _rate_limit: None = Depends(check_rate_limit),
):
    """
    Run the full ELEANOR V8 deliberation pipeline.

    This endpoint evaluates the input through all constitutional critics,
    retrieves relevant precedents, computes uncertainty, and returns
    a governance decision.

    Returns:
        Deliberation result including decision, critic outputs, and audit trail.
    """
    start_time = time.time()
    trace_id = payload.trace_id or str(uuid.uuid4())

    logger.info(
        "deliberation_started",
        trace_id=trace_id,
        user_id=user,
        input_length=len(payload.input),
    )

    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Engine not initialized",
        )

    try:
        # Run deliberation (async if available)
        if asyncio.iscoroutinefunction(engine.deliberate):
            result = await engine.deliberate(payload.input)
        else:
            result = engine.deliberate(payload.input)

        duration_ms = (time.time() - start_time) * 1000

        # Log deliberation result
        log_deliberation(
            logger,
            trace_id=result.get("trace_id", trace_id),
            decision=result.get("final_decision", "unknown"),
            model_used=result.get("model_used", "unknown"),
            duration_ms=duration_ms,
            uncertainty=result.get("uncertainty", {}).get("overall_uncertainty"),
            escalated=result.get("final_decision") == "escalate",
        )

        return result

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000

        logger.error(
            "deliberation_failed",
            trace_id=trace_id,
            error=str(e),
            error_type=type(e).__name__,
            duration_ms=duration_ms,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Deliberation failed",
        )


# ---------------------------------------------
# Retrieve Evidence Bundle by Trace ID
# ---------------------------------------------

@app.get("/trace/{trace_id}", tags=["Audit"])
async def get_trace(
    trace_id: str,
    user: Optional[str] = Depends(get_current_user),
):
    """
    Retrieve an audit record by trace ID.

    Returns the full evidence bundle for a previous deliberation.
    """
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Engine not initialized",
        )

    # Validate trace_id format (should be UUID)
    try:
        uuid.UUID(trace_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid trace ID format",
        )

    recorder = getattr(engine, 'recorder', None) or getattr(engine, 'evidence_recorder', None)

    if recorder is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Evidence recorder not available",
        )

    storage_mode = getattr(recorder, 'storage_mode', 'memory')

    if storage_mode == "memory":
        memory_log = getattr(recorder, 'get_memory_log', lambda: [])()
        for item in memory_log:
            if item.get("trace_id") == trace_id:
                return item
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Trace ID not found",
        )

    elif storage_mode == "jsonl":
        file_path = getattr(recorder, 'file_path', None)
        if not file_path:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Audit log path not configured",
            )

        try:
            with open(file_path, "r") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        if item.get("trace_id") == trace_id:
                            return item
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Audit log not found",
            )

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Trace ID not found",
        )

    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported storage mode: {storage_mode}",
        )


# ---------------------------------------------
# Governance Preview Endpoint
# ---------------------------------------------

@app.post("/governance/preview", tags=["Governance"])
async def governance_preview(
    payload: GovernancePreviewRequest,
    user: Optional[str] = Depends(get_current_user),
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
        opa_callback = getattr(engine, 'opa_callback', None) or getattr(engine, 'opa', None)

        if opa_callback is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="OPA callback not configured",
            )

        result = opa_callback(payload.model_dump())
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
    Prometheus-compatible metrics endpoint.

    Returns metrics in Prometheus text format.
    """
    # Placeholder - integrate with prometheus_client in production
    from api.middleware.rate_limit import get_rate_limiter

    limiter = get_rate_limiter()

    return {
        "rate_limit": {
            "enabled": limiter.config.enabled,
            "requests_per_window": limiter.config.requests_per_window,
            "window_seconds": limiter.config.window_seconds,
        },
        "engine_status": "initialized" if engine else "not_initialized",
    }


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
        reload=os.getenv("ELEANOR_ENV", "development") == "development",
    )
