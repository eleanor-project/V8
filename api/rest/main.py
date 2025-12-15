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
from datetime import datetime
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

# Schemas and middleware
from api.schemas import (
    DeliberationRequest,
    GovernancePreviewRequest,
    DeliberationResponse,
    HealthResponse,
    ErrorResponse,
)
from api.middleware.auth import verify_token, get_current_user, TokenPayload, get_auth_config
from api.middleware.rate_limit import (
    check_rate_limit,
    RateLimitMiddleware,
    get_rate_limiter,
    RateLimitConfig,
)

# Logging
from engine.logging_config import configure_logging, get_logger, log_deliberation

# Engine bootstrap
from api.bootstrap import (
    build_engine,
    evaluate_opa,
    load_constitutional_config,
    GOVERNANCE_SCHEMA_VERSION,
)
from api.replay_store import ReplayStore
from engine.router.adapters import GPTAdapter, ClaudeAdapter, GrokAdapter, LlamaHFAdapter, OllamaAdapter
from api.websocket.websocket_server import ws_router
from pydantic import BaseModel
from pathlib import Path

# Initialize logging
configure_logging()
logger = get_logger(__name__)

# ---------------------------------------------
# Metrics (optional Prometheus)
# ---------------------------------------------
try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

    DELIB_REQUESTS = Counter(
        "eleanor_deliberate_requests_total",
        "Total deliberate requests",
        ["outcome"],
    )
    DELIB_LATENCY = Histogram(
        "eleanor_deliberate_duration_seconds",
        "Deliberation duration in seconds",
        buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
    )
    OPA_CALLS = Counter(
        "eleanor_opa_calls_total",
        "Total OPA evaluations",
        ["result"],
    )
except Exception:
    DELIB_REQUESTS = None
    DELIB_LATENCY = None
    OPA_CALLS = None
    generate_latest = None
    CONTENT_TYPE_LATEST = None


# ---------------------------------------------
# Optional Observability (OTEL + Prometheus middleware)
# ---------------------------------------------
def enable_tracing(app: FastAPI):
    """Enable OpenTelemetry tracing if configured."""
    if os.getenv("ENABLE_OTEL", "").lower() not in ("1", "true", "yes"):
        return
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318/v1/traces")
        resource = Resource.create({"service.name": os.getenv("OTEL_SERVICE_NAME", "eleanor-v8-api")})
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
        from prometheus_fastapi_instrumentator import Instrumentator

        Instrumentator().instrument(app).expose(app, include_in_schema=False)
        logger.info("Prometheus middleware instrumentation enabled")
    except Exception as exc:
        logger.warning(f"Failed to enable Prometheus middleware: {exc}")


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


def check_content_length(request: Request, max_bytes: int = None):
    """Guardrail: reject overly large requests early."""
    max_allowed = max_bytes or int(os.getenv("MAX_REQUEST_BYTES", "1048576"))  # 1MB default
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > max_allowed:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Request too large (>{max_allowed} bytes)",
                )
        except ValueError:
            pass


def get_constitutional_config() -> dict:
    """Load constitutional configuration from YAML file."""
    config_path = os.getenv("CONSTITUTIONAL_CONFIG_PATH", "governance/constitutional.yaml")
    return load_constitutional_config(config_path)


def _ensure_writable_path(path: str):
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
    env = os.getenv("ELEANOR_ENV", "development")
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

    # Storage write checks for governance audit
    try:
        # replay_store has per-instance log path; packet/review dirs are module-level constants
        from api.replay_store import REVIEW_PACKET_DIR, REVIEW_RECORD_DIR

        _ensure_writable_path(replay_store.path.parent)
        _ensure_writable_path(REVIEW_PACKET_DIR)
        _ensure_writable_path(REVIEW_RECORD_DIR)
        results["storage"] = "ok"
    except Exception as exc:
        results["storage"] = f"error:{exc}"
        issues.append(f"storage_error:{exc}")

    if issues:
        logger.error("Readiness checks failed", issues=issues, results=results)
        raise RuntimeError(f"Readiness checks failed: {issues}")

    logger.info("Readiness checks passed", results=results)
    return results


# ---------------------------------------------
# Engine initialization
# ---------------------------------------------

# Global engine instance
engine = None
replay_store = ReplayStore(path=os.getenv("REPLAY_LOG_PATH", "replay_log.jsonl"))


def initialize_engine():
    """Initialize the ELEANOR engine."""
    global engine

    try:
        constitution = get_constitutional_config()
        engine = build_engine(constitution)
        logger.info("ELEANOR V8 engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize engine: {e}")
        raise


def resolve_final_decision(aggregator_decision: Optional[str], opa_result: Dict[str, Any]) -> str:
    """Combine aggregator decision with OPA governance outcome."""
    if not opa_result.get("allow", True):
        return "deny"
    if opa_result.get("escalate"):
        return "escalate"
    return aggregator_decision or "allow"


def normalize_engine_result(result_obj: Any, input_text: str, trace_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize EngineResult or dict into a common shape used by the API.
    """
    result = (
        result_obj.model_dump()
        if hasattr(result_obj, "model_dump")
        else result_obj.dict()
        if hasattr(result_obj, "dict")
        else result_obj
    )

    model_info = result.get("model_info") or {}
    model_used = model_info.get("model_name") if isinstance(model_info, dict) else getattr(model_info, "model_name", "unknown")

    aggregated = result.get("aggregated") or result.get("aggregator_output") or {}
    aggregated = aggregated if isinstance(aggregated, dict) else {}

    critic_findings = result.get("critic_findings") or result.get("critics") or {}
    critic_outputs = {
        k: (v.model_dump() if hasattr(v, "model_dump") else v)
        for k, v in critic_findings.items()
    }

    precedent_alignment = result.get("precedent_alignment") or result.get("precedent") or {}
    uncertainty = result.get("uncertainty") or {}

    model_output = aggregated.get("final_output") if aggregated else None
    model_output = model_output or result.get("output_text")

    normalized = {
        "trace_id": result.get("trace_id", trace_id),
        "timestamp": time.time(),
        "model_used": model_used,
        "model_output": model_output,
        "critic_outputs": critic_outputs,
        "precedent": precedent_alignment,
        "precedent_alignment": precedent_alignment,
        "uncertainty": uncertainty,
        "aggregator_output": aggregated,
        "input": input_text,
        "context": context,
    }
    return normalized


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


@app.on_event("startup")
async def startup_checks():
    """Fail fast if required security or storage dependencies are missing."""
    run_readiness_checks()
enable_tracing(app)
enable_prometheus_middleware(app)

# Serve React SPA if present
if os.path.isdir("ui"):
    ui_dir = "ui/dist" if os.path.isdir("ui/dist") else "ui"
    app.mount("/ui", StaticFiles(directory=ui_dir, html=True), name="ui")

# Register human review router
try:
    from api.rest.review import router as review_router
    app.include_router(review_router)
    logger.info("Human review API endpoints registered")
except Exception as e:
    logger.warning(f"Failed to register review router: {e}")

# Register websocket router
app.include_router(ws_router)


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
    _size_guard: None = Depends(check_content_length),
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
        run_fn = getattr(engine, "run", None)
        deliberate_fn = getattr(engine, "deliberate", None)

        if run_fn is not None:
            result_obj = await run_fn(payload.input, context=payload.context, trace_id=trace_id, detail_level=3)
        elif asyncio.iscoroutinefunction(deliberate_fn):
            result_obj = await deliberate_fn(payload.input)
        elif deliberate_fn:
            result_obj = deliberate_fn(payload.input)
        else:
            raise RuntimeError("Engine does not expose run() or deliberate()")

        normalized = normalize_engine_result(result_obj, payload.input, trace_id, payload.context)
        model_used = normalized["model_used"]
        critic_outputs = normalized["critic_outputs"]
        aggregated = normalized["aggregator_output"]
        precedent_alignment = normalized["precedent_alignment"]
        uncertainty = normalized["uncertainty"]
        model_output = normalized["model_output"]

        governance_payload = {
            "critics": critic_outputs,
            "aggregator": aggregated,
            "precedent": precedent_alignment,
            "uncertainty": uncertainty,
            "model_used": model_used,
            "input": payload.input,
            "trace_id": normalized["trace_id"],
            "timestamp": time.time(),
            "schema_version": GOVERNANCE_SCHEMA_VERSION,
        }

        governance_result = await evaluate_opa(
            getattr(engine, "opa_callback", None),
            governance_payload,
        )

        final_decision = resolve_final_decision(aggregated.get("decision"), governance_result)

        duration_ms = (time.time() - start_time) * 1000
        if DELIB_LATENCY:
            DELIB_LATENCY.observe(duration_ms / 1000.0)
        if DELIB_REQUESTS:
            DELIB_REQUESTS.labels(outcome=final_decision).inc()
        if OPA_CALLS:
            opa_outcome = (
                "deny" if not governance_result.get("allow", True) else
                "escalate" if governance_result.get("escalate") else
                "allow"
            )
            OPA_CALLS.labels(result=opa_outcome).inc()

        # Log deliberation result
        log_deliberation(
            logger,
            trace_id=normalized.get("trace_id", trace_id),
            decision=final_decision,
            model_used=model_used or "unknown",
            duration_ms=duration_ms,
            uncertainty=(uncertainty or {}).get("overall_uncertainty"),
            escalated=final_decision == "escalate",
        )

        response_payload = {
            **normalized,
            "opa_governance": governance_result,
            "governance": governance_result,
            "final_decision": final_decision,
        }

        # persist replay record
        replay_store.save({
            "trace_id": response_payload["trace_id"],
            "input": payload.input,
            "context": payload.context,
            "response": response_payload,
            "timestamp": response_payload["timestamp"],
        })

        return response_payload

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

    buffer = getattr(recorder, "buffer", None)
    if buffer:
        for record in buffer:
            record_dict = record.dict() if hasattr(record, "dict") else record
            if record_dict.get("trace_id") == trace_id:
                return record_dict

    # Fallback to JSONL sink if configured
    jsonl_path = getattr(recorder, "jsonl_path", None) or getattr(recorder, "file_path", None)
    if jsonl_path:
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if item.get("trace_id") == trace_id:
                        return item
        except FileNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Audit log not found",
            )

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Trace ID not found",
    )


# ---------------------------------------------
# Replay Endpoint
# ---------------------------------------------

@app.get("/replay/{trace_id}", tags=["Audit"])
async def replay_trace(
    trace_id: str,
    rerun: bool = False,
    user: Optional[str] = Depends(get_current_user),
):
    """
    Retrieve a stored deliberation and optionally re-run it through the current engine.
    """
    record = replay_store.get(trace_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Trace ID not found in replay log")

    if not rerun:
        return record

    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Engine not initialized",
        )

    input_text = record.get("input")
    context = record.get("context", {})
    if not input_text:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Replay record missing input")

    new_trace = str(uuid.uuid4())
    run_fn = getattr(engine, "run", None)
    deliberate_fn = getattr(engine, "deliberate", None)

    if run_fn is not None:
        result_obj = await run_fn(input_text, context=context, trace_id=new_trace, detail_level=3)
    elif asyncio.iscoroutinefunction(deliberate_fn):
        result_obj = await deliberate_fn(input_text)
    elif deliberate_fn:
        result_obj = deliberate_fn(input_text)
    else:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Engine does not expose run() or deliberate()")

    normalized = normalize_engine_result(result_obj, input_text, new_trace, context)
    governance_payload = {
        "critics": normalized["critic_outputs"],
        "aggregator": normalized["aggregator_output"],
        "precedent": normalized["precedent_alignment"],
        "uncertainty": normalized["uncertainty"],
        "model_used": normalized["model_used"],
        "input": input_text,
        "trace_id": normalized["trace_id"],
        "timestamp": time.time(),
        "schema_version": GOVERNANCE_SCHEMA_VERSION,
    }
    governance_result = await evaluate_opa(getattr(engine, "opa_callback", None), governance_payload)
    final_decision = resolve_final_decision(normalized["aggregator_output"].get("decision"), governance_result)

    rerun_payload = {
        **normalized,
        "opa_governance": governance_result,
        "governance": governance_result,
        "final_decision": final_decision,
        "replay_of": trace_id,
    }

    replay_store.save({
        "trace_id": rerun_payload["trace_id"],
        "input": input_text,
        "context": context,
        "response": rerun_payload,
        "timestamp": rerun_payload["timestamp"],
        "replay_of": trace_id,
    })

    return {
        "original": record,
        "rerun": rerun_payload,
    }


# ---------------------------------------------
# Audit Search Endpoint
# ---------------------------------------------

@app.get("/audit/search", tags=["Audit"])
async def audit_search(
    critic: Optional[str] = None,
    severity: Optional[str] = None,
    trace_id: Optional[str] = None,
    limit: int = 100,
    user: Optional[str] = Depends(get_current_user),
):
    """
    Search audit evidence by critic, severity label, or trace_id across in-memory buffer and JSONL sink.
    """
    recorder = getattr(engine, 'recorder', None) if engine else None
    if recorder is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Evidence recorder not available")

    matches: List[Dict[str, Any]] = []

    def record_matches(rec: Any) -> bool:
        if hasattr(rec, "dict"):
            rec = rec.dict()
        if trace_id and rec.get("trace_id") != trace_id:
            return False
        if critic and rec.get("critic") != critic:
            return False
        if severity and rec.get("severity") != severity:
            return False
        return True

    buffer = getattr(recorder, "buffer", [])
    for rec in reversed(buffer):
        if len(matches) >= limit:
            break
        if record_matches(rec):
            matches.append(rec.dict() if hasattr(rec, "dict") else rec)

    if len(matches) < limit:
        jsonl_path = getattr(recorder, "jsonl_path", None) or getattr(recorder, "file_path", None)
        if jsonl_path and os.path.exists(jsonl_path):
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if len(matches) >= limit:
                        break
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if record_matches(item):
                        matches.append(item)

    return {"matches": matches[:limit], "count": len(matches)}


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

        payload_dict = payload.model_dump()
        payload_dict["schema_version"] = GOVERNANCE_SCHEMA_VERSION
        result = await evaluate_opa(opa_callback, payload_dict)
        if OPA_CALLS:
            opa_outcome = (
                "deny" if not result.get("allow", True) else
                "escalate" if result.get("escalate") else
                "allow"
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

@app.get("/admin/router/health", tags=["Admin"])
async def router_health(user: Optional[str] = Depends(get_current_user)):
    if engine is None or not getattr(engine, "router", None):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Router not available")
    router = engine.router
    health = getattr(router, "health", {})
    breakers = router.get_circuit_breaker_status() if hasattr(router, "get_circuit_breaker_status") else {}
    return {"health": health, "circuit_breakers": breakers, "policy": getattr(router, "policy", {})}


@app.post("/admin/router/reset_breakers", tags=["Admin"])
async def reset_breakers(user: Optional[str] = Depends(get_current_user)):
    if engine is None or not getattr(engine, "router", None):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Router not available")
    router = engine.router
    if hasattr(router, "_circuit_breakers"):
        router._circuit_breakers.reset_all()
    return {"status": "ok"}


@app.get("/admin/critics/bindings", tags=["Admin"])
async def get_critic_bindings(user: Optional[str] = Depends(get_current_user)):
    if engine is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Engine not initialized")
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
async def set_critic_binding(binding: CriticBinding, user: Optional[str] = Depends(get_current_user)):
    if engine is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Engine not initialized")
    router = getattr(engine, "router", None)
    adapters = getattr(router, "adapters", {}) if router else {}
    adapter_fn = adapters.get(binding.adapter)
    if adapter_fn is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Adapter '{binding.adapter}' not registered")
    engine.critic_models[binding.critic] = adapter_fn
    return {"status": "ok", "binding": {binding.critic: binding.adapter}}


class AdapterRegistration(BaseModel):
    name: str
    type: str = "ollama"  # ollama | hf | openai | claude | grok
    model: Optional[str] = None
    device: Optional[str] = None
    api_key: Optional[str] = None


@app.post("/admin/router/adapters", tags=["Admin"])
async def register_adapter(adapter: AdapterRegistration, user: Optional[str] = Depends(get_current_user)):
    if engine is None or not getattr(engine, "router", None):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Router not available")

    router = engine.router
    adapters = getattr(router, "adapters", {}) if router else {}

    factory = adapter.type.lower()
    try:
        if factory == "ollama":
            adapters[adapter.name] = OllamaAdapter(model=adapter.model or adapter.name)
        elif factory == "hf":
            adapters[adapter.name] = LlamaHFAdapter(model_path=adapter.model or adapter.name, device=adapter.device or "cpu")
        elif factory == "openai":
            adapters[adapter.name] = GPTAdapter(model=adapter.model or "gpt-4o-mini", api_key=adapter.api_key or os.getenv("OPENAI_API_KEY"))
        elif factory == "claude":
            adapters[adapter.name] = ClaudeAdapter(model=adapter.model or "claude-3-5-sonnet-20241022", api_key=adapter.api_key or os.getenv("ANTHROPIC_API_KEY"))
        elif factory == "grok":
            adapters[adapter.name] = GrokAdapter(model=adapter.model or "grok-beta", api_key=adapter.api_key or os.getenv("XAI_API_KEY"))
        else:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unknown adapter type '{adapter.type}'")
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to register adapter: {exc}")

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
