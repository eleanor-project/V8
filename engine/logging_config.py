import os
import sys
import logging
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    TYPE_CHECKING,
    cast,
)
from datetime import datetime

# Graceful import of structlog
try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False
    structlog = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover
    from structlog.stdlib import BoundLogger  # type: ignore[misc]


Processor = Callable[[Any, str, MutableMapping[str, Any]], Any]


def get_log_level() -> int:
    """Get the log level from environment."""
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    return getattr(logging, level_name, logging.INFO)


def get_log_format() -> str:
    """Get the log format from environment."""
    env = (
        os.getenv("ELEANOR_ENVIRONMENT")
        or os.getenv("ELEANOR_ENV")
        or "development"
    )
    default = "console" if env == "development" else "json"
    return os.getenv("LOG_FORMAT", default)


def configure_logging() -> None:
    """
    Configure structured logging for the application.

    Call this once at application startup.
    """
    log_level = get_log_level()
    log_format = get_log_format()

    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    if not STRUCTLOG_AVAILABLE:
        logging.warning(
            "structlog not installed. Using basic logging. "
            "Install with: pip install structlog"
        )
        return

    # Common processors
    shared_processors: List[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if log_format == "json":
        # JSON output for production
        processors: List[Processor] = [
            *shared_processors,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ]
    else:
        # Console output for development
        processors = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: Optional[str] = None) -> "BoundLogger | logging.Logger":
    """
    Get a logger instance.

    If structlog is available, returns a structlog-wrapped logger.
    Otherwise returns a standard logging logger.
    """
    if STRUCTLOG_AVAILABLE:
        return cast("BoundLogger | logging.Logger", structlog.get_logger(name))
    return logging.getLogger(name)


class LogContext:
    """
    Context manager for adding temporary context to logs.

    Usage:
        with LogContext(trace_id="abc123", user_id="user1"):
            logger.info("Processing request")
    """

    def __init__(self, **kwargs):
        self.context = kwargs
        self._token = None

    def __enter__(self):
        if STRUCTLOG_AVAILABLE:
            import structlog
            for key, value in self.context.items():
                structlog.contextvars.bind_contextvars(**{key: value})
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if STRUCTLOG_AVAILABLE:
            import structlog
            structlog.contextvars.unbind_contextvars(*self.context.keys())


def log_request(
    logger: logging.Logger,
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    trace_id: Optional[str] = None,
    user_id: Optional[str] = None,
    **extra
) -> None:
    """
    Log an HTTP request with standard fields.

    Usage:
        log_request(
            logger,
            method="POST",
            path="/deliberate",
            status_code=200,
            duration_ms=150.5,
            trace_id="abc123"
        )
    """
    logger.info(
        "http_request",
        extra={
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration_ms": round(duration_ms, 2),
            "trace_id": trace_id,
            "user_id": user_id,
            **extra,
        },
    )


def log_deliberation(
    logger: "BoundLogger | logging.Logger",
    trace_id: str,
    decision: str,
    model_used: str,
    duration_ms: float,
    uncertainty: Optional[float] = None,
    escalated: bool = False,
    **extra
) -> None:
    """
    Log a deliberation result with standard fields.

    Usage:
        log_deliberation(
            logger,
            trace_id="abc123",
            decision="allow",
            model_used="gpt-4",
            duration_ms=500.0,
            uncertainty=0.25
        )
    """
    logger.info(
        "deliberation_complete",
        extra={
            "trace_id": trace_id,
            "decision": decision,
            "model_used": model_used,
            "duration_ms": round(duration_ms, 2),
            "uncertainty": uncertainty,
            "escalated": escalated,
            **extra,
        },
    )


def log_critic_execution(
    logger: logging.Logger,
    trace_id: str,
    critic_name: str,
    severity: float,
    duration_ms: float,
    success: bool = True,
    error: Optional[str] = None,
    **extra
) -> None:
    """
    Log a critic execution result.
    """
    level = "info" if success else "warning"
    getattr(logger, level)(
        "critic_executed",
        extra={
            "trace_id": trace_id,
            "critic_name": critic_name,
            "severity": severity,
            "duration_ms": round(duration_ms, 2),
            "success": success,
            "error": error,
            **extra,
        },
    )


def log_precedent_retrieval(
    logger,
    trace_id: str,
    query_length: int,
    cases_found: int,
    alignment_score: float,
    duration_ms: float,
    **extra
) -> None:
    """
    Log a precedent retrieval operation.
    """
    logger.info(
        "precedent_retrieved",
        extra={
            "trace_id": trace_id,
            "query_length": query_length,
            "cases_found": cases_found,
            "alignment_score": alignment_score,
            "duration_ms": round(duration_ms, 2),
            **extra,
        },
    )
