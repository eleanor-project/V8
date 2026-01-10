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
import hashlib
import uuid
import time
import asyncio
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Literal
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

# Schemas and middleware
from api.schemas import (
    DeliberationRequest,
    GovernancePreviewRequest,
    EvaluateRequest,
    EvaluateResponse,
    EvidenceBundle,
    EvidenceIntegrity,
    EvidenceProvenance,
    CriticEvidence,
    PrecedentTrace,
    PolicyTrace,
    RoutingDecision,
    UncertaintyEnvelope,
    EvaluateError,
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
from api.middleware.fingerprinting import RequestFingerprinter

# Logging
from engine.logging_config import configure_logging, get_logger, log_deliberation

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
from engine.execution.human_review import enforce_human_review
from engine.schemas.escalation import AggregationResult, HumanAction, ExecutableDecision
from engine.exceptions import InputValidationError
from engine.version import ELEANOR_VERSION
from engine.utils.critic_names import canonical_critic_name, canonicalize_critic_map
from engine.utils.validation import sanitize_for_logging
from engine.utils.dependency_tracking import get_dependency_metrics

# Initialize logging
configure_logging()
logger = get_logger(__name__)

# ---------------------------------------------
# Metrics (optional Prometheus)
# ---------------------------------------------
try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST  # type: ignore[import-not-found]

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


def check_content_length(request: Request, max_bytes: Optional[int] = None):
    """
    Guardrail: reject overly large requests early.
    
    In production, enforces stricter limits (512KB default).
    In development, allows larger requests (1MB default).
    """
    env = _resolve_environment()
    is_production = env == "production"
    
    # Stricter limits in production
    default_max = 524288 if is_production else 1048576  # 512KB vs 1MB
    max_allowed = max_bytes or int(os.getenv("MAX_REQUEST_BYTES", str(default_max)))
    
    # Additional production check
    if is_production and max_allowed > 1048576:  # 1MB
        logger.warning(
            "MAX_REQUEST_BYTES exceeds 1MB in production, using 1MB limit",
            extra={"configured": max_allowed},
        )
        max_allowed = 1048576
    
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            length = int(content_length)
            if length > max_allowed:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Request too large ({length} bytes, max: {max_allowed} bytes)",
                )
        except ValueError:
            logger.warning("Invalid content-length header; continuing without size enforcement")


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


def resolve_final_decision(aggregator_decision: Optional[str], opa_result: Dict[str, Any]) -> str:
    """Combine aggregator decision with OPA governance outcome."""
    # Fail-closed posture: if OPA denies, we deny unless OPA explicitly requests escalation.
    if opa_result.get("allow") is False:
        if opa_result.get("escalate") is True:
            return "escalate"
        return "deny"
    if opa_result.get("escalate") is True:
        return "escalate"
    return aggregator_decision or "allow"


def map_assessment_label(decision: Optional[str]) -> str:
    """
    Map internal decision labels to non-coercive assessment language.
    """
    if not decision:
        return "requires_human_review"
    mapping = {
        "allow": "aligned",
        "constrained_allow": "aligned_with_constraints",
        "deny": "misaligned",
        "escalate": "requires_human_review",
    }
    normalized = str(decision).lower()
    return mapping.get(normalized, normalized)


def normalize_engine_result(
    result_obj: Any, input_text: str, trace_id: str, context: Dict[str, Any]
) -> Dict[str, Any]:
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
    model_used = (
        model_info.get("model_name")
        if isinstance(model_info, dict)
        else getattr(model_info, "model_name", "unknown")
    )

    aggregated = result.get("aggregated") or result.get("aggregator_output") or {}
    aggregated = aggregated if isinstance(aggregated, dict) else {}

    critic_findings = result.get("critic_findings") or result.get("critics") or {}
    critic_outputs = {
        k: (v.model_dump() if hasattr(v, "model_dump") else v) for k, v in critic_findings.items()
    }
    critic_outputs = canonicalize_critic_map(critic_outputs)

    precedent_alignment = result.get("precedent_alignment") or result.get("precedent") or {}
    uncertainty = result.get("uncertainty") or {}

    model_output = aggregated.get("final_output") if aggregated else None
    model_output = model_output or result.get("output_text")

    degraded_components = (
        result.get("degraded_components") or aggregated.get("degraded_components") or []
    )
    is_degraded = result.get("is_degraded")
    if is_degraded is None:
        is_degraded = aggregated.get("is_degraded", False)

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
        "degraded_components": degraded_components,
        "is_degraded": bool(is_degraded),
    }
    return normalized


def _safe_json_dumps(payload: Any) -> str:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
    )


def _hash_payload(payload: Any) -> str:
    raw = _safe_json_dumps(payload).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _serialize_model_output(model_output: Any) -> str:
    if isinstance(model_output, str):
        return model_output
    return json.dumps(model_output, ensure_ascii=True, default=str)


def _severity_score(data: Dict[str, Any]) -> float:
    for key in ("final_severity", "severity"):
        if data.get(key) is not None:
            try:
                return float(data[key])
            except (TypeError, ValueError):
                continue
    if data.get("score") is not None:
        try:
            return float(data["score"]) * 3.0
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _critic_verdict(severity: float) -> Literal["PASS", "WARN", "FAIL"]:
    if severity >= 2.0:
        return "FAIL"
    if severity >= 1.0:
        return "WARN"
    return "PASS"


def _build_uncertainty_envelope(uncertainty: Dict[str, Any]) -> UncertaintyEnvelope:
    overall = 0.0
    try:
        overall = float(uncertainty.get("overall_uncertainty", 0.0))
    except (TypeError, ValueError):
        overall = 0.0

    if overall >= 0.6:
        level: Literal["LOW", "MEDIUM", "HIGH"] = "HIGH"
    elif overall >= 0.3:
        level = "MEDIUM"
    else:
        level = "LOW"

    reasons = []
    explanation = uncertainty.get("explanation")
    if explanation:
        reasons.append(str(explanation))
    if uncertainty.get("needs_escalation"):
        reasons.append("Uncertainty threshold exceeded")

    return UncertaintyEnvelope(level=level, reasons=reasons)


def _confidence_from_uncertainty(uncertainty: Dict[str, Any]) -> float:
    try:
        overall = float(uncertainty.get("overall_uncertainty", 0.0))
    except (TypeError, ValueError):
        overall = 0.0
    return max(0.0, min(1.0, 1.0 - overall))


def _map_decision(
    final_decision: Optional[str],
) -> Literal["ALLOW", "ALLOW_WITH_CONSTRAINTS", "ABSTAIN", "ESCALATE", "DENY"]:
    if not final_decision:
        return "ABSTAIN"
    decision = str(final_decision).lower()
    if decision == "allow":
        return "ALLOW"
    if decision == "constrained_allow":
        return "ALLOW_WITH_CONSTRAINTS"
    if decision == "deny":
        return "DENY"
    if decision == "escalate":
        return "ESCALATE"
    return "ABSTAIN"


def _build_constraints(critic_details: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    advisories = []
    for name, data in critic_details.items():
        severity = _severity_score(data)
        if severity < 1.0:
            continue
        rationale = data.get("justification") or data.get("rationale") or ""
        if not rationale and data.get("violations"):
            rationale = str(data.get("violations")[0])
        advisories.append({"critic": name, "severity": severity, "note": rationale})

    if not advisories:
        return None
    return {"advisories": advisories}


def _build_precedent_trace(precedent_alignment: Dict[str, Any]) -> List[PrecedentTrace]:
    retrieval = precedent_alignment.get("retrieval") or {}
    cases = retrieval.get("precedent_cases") or retrieval.get("cases") or []
    traces: List[PrecedentTrace] = []
    for case in cases:
        meta = case.get("metadata") or {}
        case_id = (
            meta.get("id")
            or meta.get("case_id")
            or case.get("id")
            or case.get("case_id")
            or meta.get("title")
        )
        if not case_id:
            case_id = "precedent"
        traces.append(
            PrecedentTrace(
                id=str(case_id),
                type=meta.get("type") or meta.get("category") or "internal_precedent",
                applied_as=meta.get("applied_as") or "supporting",
                note=meta.get("summary") or meta.get("note"),
            )
        )
    return traces


def _build_policy_trace(governance_result: Dict[str, Any]) -> List[PolicyTrace]:
    failures = governance_result.get("failures") or []
    if not failures:
        return []
    allow = governance_result.get("allow", True)
    escalate = governance_result.get("escalate", False)
    effect = "deny" if not allow else "escalate" if escalate else "allow"
    traces: List[PolicyTrace] = []
    for failure in failures:
        traces.append(
            PolicyTrace(
                rule_id=str(failure.get("policy") or "policy"),
                effect=effect,
                matched_on=failure.get("matched_on"),
                note=failure.get("reason"),
            )
        )
    return traces


def _build_critic_evidence(critic_details: Dict[str, Any]) -> List[CriticEvidence]:
    outputs: List[CriticEvidence] = []
    for name, data in critic_details.items():
        severity = _severity_score(data)
        score = data.get("score")
        if score is None:
            score = max(0.0, min(1.0, severity / 3.0))
        rationale = data.get("justification") or data.get("rationale") or ""
        precedents = data.get("precedent_refs") or data.get("precedents") or []
        if not precedents:
            raw = data.get("raw") or {}
            precedents = raw.get("precedent_refs") or raw.get("precedents") or []
        policy_rules = data.get("policy_rules") or []
        if not policy_rules:
            raw = data.get("raw") or {}
            policy_rules = raw.get("policy_rules") or []
        signals = data.get("evidence") if isinstance(data.get("evidence"), dict) else None
        if signals is None and data.get("raw"):
            raw = data.get("raw") or {}
            if isinstance(raw.get("evidence"), dict):
                signals = raw.get("evidence")

        outputs.append(
            CriticEvidence(
                critic=name,
                verdict=_critic_verdict(severity),
                score=score,
                rationale=rationale,
                precedents=[str(p) for p in precedents] if isinstance(precedents, list) else [],
                policy_rules=[str(p) for p in policy_rules]
                if isinstance(policy_rules, list)
                else [],
                signals=signals,
            )
        )
    return outputs


def _build_evidence_bundle(
    *,
    decision: str,
    confidence: float,
    critic_details: Dict[str, Any],
    precedent_alignment: Dict[str, Any],
    governance_result: Dict[str, Any],
    provenance_inputs: Dict[str, Any],
) -> EvidenceBundle:
    critic_outputs = _build_critic_evidence(critic_details)
    precedent_trace = _build_precedent_trace(precedent_alignment)
    policy_trace = _build_policy_trace(governance_result)
    summary = f"Decision {decision} with confidence {confidence:.2f}."

    provenance = EvidenceProvenance(inputs=provenance_inputs)
    bundle_payload = {
        "summary": summary,
        "critic_outputs": [c.model_dump(mode="json") for c in critic_outputs],
        "precedent_trace": [p.model_dump(mode="json") for p in precedent_trace],
        "policy_trace": [p.model_dump(mode="json") for p in policy_trace],
        "provenance": provenance.model_dump(mode="json"),
    }
    bundle_hash = f"sha256:{_hash_payload(bundle_payload)}"
    integrity = EvidenceIntegrity(hash=bundle_hash)

    return EvidenceBundle(
        summary=summary,
        critic_outputs=critic_outputs,
        precedent_trace=precedent_trace,
        policy_trace=policy_trace,
        provenance=provenance,
        integrity=integrity,
    )


def resolve_execution_decision(
    aggregated: Dict[str, Any],
    human_action: Optional[HumanAction],
) -> Optional[ExecutableDecision]:
    aggregation_payload = (
        aggregated.get("aggregation_result") if isinstance(aggregated, dict) else None
    )
    if not aggregation_payload:
        return None
    try:
        aggregation_result = AggregationResult.model_validate(aggregation_payload)
    except Exception as exc:
        logger.error(
            "Invalid aggregation_result payload",
            extra={"error": str(exc)},
        )
        raise RuntimeError("Invalid aggregation_result payload") from exc
    return enforce_human_review(aggregation_result=aggregation_result, human_action=human_action)


def apply_execution_gate(
    final_decision: str, execution_decision: Optional[ExecutableDecision]
) -> str:
    if execution_decision and not execution_decision.executable and final_decision != "deny":
        return "escalate"
    return final_decision


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

    app.include_router(review_router)
    app.include_router(governance_router)
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
# Main Deliberation Endpoint
# ---------------------------------------------


@app.post("/deliberate", tags=["Deliberation"])
async def deliberate(
    request: Request,
    payload: DeliberationRequest,
    user: str = Depends(require_authenticated_user),
    _size_guard: None = Depends(check_content_length),
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
    
    # Set correlation ID first, then use it as fallback for trace_id
    correlation_id = None
    if CorrelationContext:
        correlation_id = get_correlation_id()
        CorrelationContext.set(correlation_id)
    
    # Use trace_id from payload, or correlation_id, or generate new UUID
    trace_id = payload.trace_id or correlation_id or str(uuid.uuid4())
    
    # Request fingerprinting
    fingerprinter = RequestFingerprinter()
    fingerprint = fingerprinter.fingerprint(request)
    fingerprint_components = fingerprinter.get_fingerprint_components(request)

    logger.info(
        "deliberation_started",
        extra={
            "trace_id": trace_id,
            "user_id": user,
            "input_length": len(payload.input),
        },
    )

    if engine is None:
        logger.error(
            "engine_not_initialized",
            extra={"trace_id": trace_id, "user_id": user},
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Engine not initialized",
        )

    # Input validation with security hardening
    from engine.security.input_validation import InputValidator
    validator = InputValidator(strict_mode=True)
    
    # Validate input text
    try:
        validated_input = validator.validate_string(
            payload.input,
            field_name="input",
            allow_empty=False,
            sanitize=True,
        )
    except ValueError as validation_error:
        logger.warning(
            "input_validation_failed",
            extra={
                "trace_id": trace_id,
                "user_id": user,
                "error": str(validation_error),
            },
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Input validation failed: {str(validation_error)}",
        )
    
    # Validate context
    try:
        validated_context = validator.validate_dict(
            payload.context.model_dump(mode="json") if payload.context else {},
            field_name="context",
        )
    except ValueError as validation_error:
        logger.warning(
            "context_validation_failed",
            extra={
                "trace_id": trace_id,
                "user_id": user,
                "error": str(validation_error),
            },
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Context validation failed: {str(validation_error)}",
        )

    try:
        run_fn = getattr(engine, "run", None)
        deliberate_fn = getattr(engine, "deliberate", None)

        if run_fn is not None:
            result_obj = await run_fn(
                validated_input, context=validated_context, trace_id=trace_id, detail_level=3
            )
        elif asyncio.iscoroutinefunction(deliberate_fn):
            result_obj = await deliberate_fn(validated_input)
        elif deliberate_fn:
            result_obj = deliberate_fn(validated_input)
        else:
            raise RuntimeError("Engine does not expose run() or deliberate()")

        normalized = normalize_engine_result(result_obj, validated_input, trace_id, validated_context)
        model_used = normalized["model_used"]
        critic_outputs = normalized["critic_outputs"]
        aggregated = normalized["aggregator_output"]
        precedent_alignment = normalized["precedent_alignment"]
        uncertainty = normalized["uncertainty"]

        governance_payload = {
            "critics": critic_outputs,
            "aggregator": aggregated,
            "precedent": precedent_alignment,
            "uncertainty": uncertainty,
            "model_used": model_used,
            "input": validated_input,
            "trace_id": normalized["trace_id"],
            "timestamp": time.time(),
            "schema_version": GOVERNANCE_SCHEMA_VERSION,
            "policy_profile": payload.policy_profile,
            "proposed_action": payload.proposed_action.model_dump(mode="json"),
            "context": validated_context,
            "evidence_inputs": payload.evidence_inputs.model_dump(mode="json") if payload.evidence_inputs else {},
            "model_metadata": payload.model_metadata.model_dump(mode="json") if payload.model_metadata else {},
        }

        governance_result = await evaluate_opa(
            getattr(engine, "opa_callback", None),
            governance_payload,
        )

        final_decision = resolve_final_decision(aggregated.get("decision"), governance_result)

        execution_decision = None
        try:
            execution_decision = resolve_execution_decision(aggregated, payload.human_action)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

        final_decision = apply_execution_gate(final_decision, execution_decision)
        final_assessment = map_assessment_label(final_decision)

        duration_ms = (time.time() - start_time) * 1000
        if DELIB_LATENCY:
            DELIB_LATENCY.observe(duration_ms / 1000.0)
        if DELIB_REQUESTS:
            DELIB_REQUESTS.labels(outcome=final_assessment).inc()
        if OPA_CALLS:
            opa_outcome = (
                "deny"
                if not governance_result.get("allow", True)
                else "escalate"
                if governance_result.get("escalate")
                else "allow"
            )
            OPA_CALLS.labels(result=opa_outcome).inc()

        # Log deliberation result
        log_deliberation(
            logger,
            trace_id=normalized.get("trace_id", trace_id),
            decision=final_assessment,
            model_used=model_used or "unknown",
            duration_ms=duration_ms,
            uncertainty=(uncertainty or {}).get("overall_uncertainty"),
            escalated=final_assessment == "requires_human_review",
            raw_decision=final_decision,
        )

        response_payload = {
            **normalized,
            "opa_governance": governance_result,
            "governance": governance_result,
            "final_decision": final_assessment,
            "execution_decision": execution_decision.model_dump(mode="json")
            if execution_decision
            else None,
        }
        
        # Record business metrics (after response_payload is defined)
        if OBSERVABILITY_AVAILABLE and record_engine_result:
            try:
                record_engine_result(response_payload)
                record_decision(final_decision, final_assessment)
            except Exception as exc:
                logger.debug(f"Failed to record metrics: {exc}")

        # persist replay record
        await replay_store.save_async(
            {
                "trace_id": response_payload["trace_id"],
                "input": validated_input,
                "context": validated_context,
                "response": response_payload,
                "timestamp": response_payload["timestamp"],
            }
        )

        return response_payload

    except InputValidationError as exc:
        duration_ms = (time.time() - start_time) * 1000
        safe_input = sanitize_for_logging(payload.input, max_length=300)

        logger.warning(
            "input_validation_failed",
            extra={
                "trace_id": trace_id,
                "error": exc.message,
                "validation_type": exc.details.get("validation_type"),
                "field": exc.details.get("field"),
                "input_excerpt": safe_input,
                "duration_ms": duration_ms,
            },
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_input",
                "message": exc.message,
                "details": exc.details,
            },
        )
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except (ValueError, TypeError, KeyError, AttributeError) as e:
        # Known error types - log and return appropriate error
        duration_ms = (time.time() - start_time) * 1000
        logger.warning(
            "deliberation_validation_error",
            extra={
                "trace_id": trace_id,
                "error": str(e),
                "error_type": type(e).__name__,
                "duration_ms": duration_ms,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request: {str(e)}",
        )
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000

        logger.error(
            "deliberation_failed",
            extra={
                "trace_id": trace_id,
                "error": str(e),
                "error_type": type(e).__name__,
                "duration_ms": duration_ms,
            },
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Deliberation failed",
        )


# ---------------------------------------------
# Evaluate Provided Model Output
# ---------------------------------------------


@app.post("/evaluate", response_model=EvaluateResponse, tags=["Deliberation"])
async def evaluate(
    request: Request,
    payload: EvaluateRequest,
    user: str = Depends(require_authenticated_user),
    _rate_limit: None = Depends(check_rate_limit),
):
    """
    Evaluate a provided model output against the Engine contract.
    """
    if engine is None:
        logger.error(
            "engine_not_initialized",
            extra={"user_id": user, "endpoint": "/evaluate"},
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Engine not initialized",
        )

    # Input validation with security hardening
    from engine.security.input_validation import InputValidator
    validator = InputValidator(strict_mode=True)
    
    # Validate context
    try:
        context = payload.context.model_dump(mode="json") if payload.context else {}
        validated_context = validator.validate_dict(
            context,
            field_name="context",
        )
    except ValueError as validation_error:
        logger.warning(
            "context_validation_failed",
            extra={
                "user_id": user,
                "endpoint": "/evaluate",
                "error": str(validation_error),
            },
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Context validation failed: {str(validation_error)}",
        )

    model_output_text = _serialize_model_output(payload.model_output)
    user_intent = validated_context.get("user_intent")
    input_override = user_intent if isinstance(user_intent, str) else ""

    validated_context.update(
        {
            "policy_profile": payload.policy_profile,
            "proposed_action": payload.proposed_action.model_dump(mode="json"),
            "evidence_inputs": payload.evidence_inputs.model_dump(mode="json")
            if payload.evidence_inputs
            else {},
            "model_metadata": payload.model_metadata.model_dump(mode="json")
            if payload.model_metadata
            else {},
            "model_output": model_output_text,
            "skip_router": True,
            "force_model_output": True,
            "input_text_override": input_override,
        }
    )

    try:
        run_fn = getattr(engine, "run", None)
        if run_fn is None:
            raise RuntimeError("Engine does not expose run()")

        result_obj = await run_fn(
            model_output_text,
            context=validated_context,
            trace_id=payload.request_id,
            detail_level=3,
        )

        normalized = normalize_engine_result(
            result_obj, model_output_text, payload.request_id, validated_context
        )
        aggregated = normalized.get("aggregator_output") or {}
        precedent_alignment = normalized.get("precedent_alignment") or {}
        uncertainty = normalized.get("uncertainty") or {}

        governance_payload = {
            "critics": aggregated.get("critics", {}),
            "aggregator": aggregated,
            "precedent": precedent_alignment,
            "uncertainty": uncertainty,
            "model_used": normalized.get("model_used"),
            "input": model_output_text,
            "trace_id": payload.request_id,
            "timestamp": payload.timestamp.isoformat(),
            "schema_version": GOVERNANCE_SCHEMA_VERSION,
            "policy_profile": payload.policy_profile,
            "proposed_action": payload.proposed_action.model_dump(mode="json"),
            "context": validated_context,
            "evidence_inputs": payload.evidence_inputs.model_dump(mode="json")
            if payload.evidence_inputs
            else {},
        }

        governance_result = await evaluate_opa(
            getattr(engine, "opa_callback", None),
            governance_payload,
        )

        final_decision = resolve_final_decision(aggregated.get("decision"), governance_result)

        execution_decision = None
        try:
            execution_decision = resolve_execution_decision(aggregated, None)
        except Exception as exc:
            logger.warning(
                "Execution gate evaluation failed",
                extra={"error": str(exc)},
            )

        final_decision = apply_execution_gate(final_decision, execution_decision)

        decision = _map_decision(final_decision)
        uncertainty_envelope = _build_uncertainty_envelope(uncertainty)
        confidence = _confidence_from_uncertainty(uncertainty)
        critic_details = aggregated.get("critics") or {}
        constraints = (
            _build_constraints(critic_details) if decision == "ALLOW_WITH_CONSTRAINTS" else None
        )

        routing_notes = None
        if execution_decision and not execution_decision.executable:
            routing_notes = execution_decision.execution_reason

        if decision in ("ESCALATE", "ABSTAIN"):
            routing = RoutingDecision(next_step="human_required", notes=routing_notes)
        elif decision == "DENY":
            routing = RoutingDecision(next_step="review_queue", notes=routing_notes)
        elif decision == "ALLOW_WITH_CONSTRAINTS":
            routing = RoutingDecision(next_step="policy_gate_required", notes=routing_notes)
        else:
            routing = RoutingDecision(next_step="safe_to_execute", notes=routing_notes)

        provenance_inputs = {
            "model_output_hash": f"sha256:{_hash_payload(model_output_text)}",
            "context_hash": f"sha256:{_hash_payload(context)}",
            "proposed_action_hash": f"sha256:{_hash_payload(payload.proposed_action.model_dump(mode='json'))}",
            "policy_profile": payload.policy_profile,
        }
        if payload.evidence_inputs:
            provenance_inputs[
                "evidence_inputs_hash"
            ] = f"sha256:{_hash_payload(payload.evidence_inputs.model_dump(mode='json'))}"
        if payload.model_metadata:
            provenance_inputs[
                "model_metadata_hash"
            ] = f"sha256:{_hash_payload(payload.model_metadata.model_dump(mode='json'))}"

        evidence_bundle = _build_evidence_bundle(
            decision=decision,
            confidence=confidence,
            critic_details=critic_details,
            precedent_alignment=precedent_alignment,
            governance_result=governance_result,
            provenance_inputs=provenance_inputs,
        )

        return EvaluateResponse(
            request_id=payload.request_id,
            engine_version=ELEANOR_VERSION,
            decision=decision,
            confidence=confidence,
            uncertainty=uncertainty_envelope,
            constraints=constraints,
            routing=routing,
            evidence_bundle=evidence_bundle,
            errors=[],
        )

    except InputValidationError as exc:
        logger.warning(
            "evaluation_input_validation_failed",
            extra={
                "request_id": payload.request_id,
                "error": exc.message,
                "validation_type": exc.details.get("validation_type"),
                "field": exc.details.get("field"),
            },
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_input",
                "message": exc.message,
                "details": exc.details,
            },
        )
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        # Known error types - log and return appropriate error
        logger.warning(
            "evaluation_validation_error",
            extra={
                "request_id": payload.request_id,
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request: {str(exc)}",
        )
    except Exception as exc:
        logger.error(
            "evaluation_failed",
            extra={
                "request_id": payload.request_id,
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
        )

        error = EvaluateError(code="E_ENGINE_ERROR", message=str(exc))
        uncertainty = UncertaintyEnvelope(level="HIGH", reasons=[str(exc)])
        routing = RoutingDecision(
            next_step="human_required", notes="Engine error; manual review required."
        )

        bundle_payload = {
            "summary": "Engine error; abstained from decision.",
            "critic_outputs": [],
            "precedent_trace": [],
            "policy_trace": [],
            "provenance": {"inputs": {}},
        }
        bundle_hash = f"sha256:{_hash_payload(bundle_payload)}"
        evidence_bundle = EvidenceBundle(
            summary="Engine error; abstained from decision.",
            critic_outputs=[],
            precedent_trace=[],
            policy_trace=[],
            provenance=EvidenceProvenance(inputs={}),
            integrity=EvidenceIntegrity(hash=bundle_hash),
        )

        return EvaluateResponse(
            request_id=payload.request_id,
            engine_version=ELEANOR_VERSION,
            decision="ABSTAIN",
            confidence=0.0,
            uncertainty=uncertainty,
            constraints=None,
            routing=routing,
            evidence_bundle=evidence_bundle,
            errors=[error],
        )


# ---------------------------------------------
# Retrieve Evidence Bundle by Trace ID
# ---------------------------------------------


@app.get("/trace/{trace_id}", tags=["Audit"])
async def get_trace(
    trace_id: str,
    user: str = Depends(require_authenticated_user),
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

    recorder = getattr(engine, "recorder", None) or getattr(engine, "evidence_recorder", None)

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
    user: str = Depends(require_authenticated_user),
):
    """
    Retrieve a stored deliberation and optionally re-run it through the current engine.
    """
    record = await replay_store.get_async(trace_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Trace ID not found in replay log"
        )

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
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Replay record missing input"
        )

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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Engine does not expose run() or deliberate()",
        )

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
    governance_result = await evaluate_opa(
        getattr(engine, "opa_callback", None), governance_payload
    )
    final_decision = resolve_final_decision(
        normalized["aggregator_output"].get("decision"), governance_result
    )
    execution_decision = None
    try:
        execution_decision = resolve_execution_decision(normalized["aggregator_output"], None)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    final_decision = apply_execution_gate(final_decision, execution_decision)
    final_assessment = map_assessment_label(final_decision)

    rerun_payload = {
        **normalized,
        "opa_governance": governance_result,
        "governance": governance_result,
        "final_decision": final_assessment,
        "execution_decision": execution_decision.model_dump(mode="json")
        if execution_decision
        else None,
        "replay_of": trace_id,
    }

    await replay_store.save_async(
        {
            "trace_id": rerun_payload["trace_id"],
            "input": input_text,
            "context": context,
            "response": rerun_payload,
            "timestamp": rerun_payload["timestamp"],
            "replay_of": trace_id,
        }
    )

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
    user: str = Depends(require_authenticated_user),
):
    """
    Search audit evidence by critic, severity label, or trace_id across in-memory buffer and JSONL sink.
    """
    recorder = getattr(engine, "recorder", None) if engine else None
    if recorder is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Evidence recorder not available",
        )

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
