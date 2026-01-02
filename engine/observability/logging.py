"""
ELEANOR V8 — Structured Logging

JSON structured logging with structlog for production environments.
"""

import sys
import logging
from typing import Optional
import structlog
from structlog.processors import (
    JSONRenderer,
    TimeStamper,
    add_log_level,
    StackInfoRenderer,
    format_exc_info,
)
from structlog.types import Processor


def configure_logging(
    log_level: str = "INFO",
    json_logs: bool = True,
    development_mode: bool = False,
) -> None:
    """
    Configure structured logging for ELEANOR.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        json_logs: Use JSON format (True for production, False for dev)
        development_mode: Use colored console output for development
    """
    # Configure stdlib logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )
    
    # Processors for all modes
    processors: list[Processor] = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        TimeStamper(fmt="iso"),
        StackInfoRenderer(),
        format_exc_info,
    ]
    
    # Add renderer based on mode
    if development_mode and not json_logs:
        # Colored console for development
        processors.append(structlog.dev.ConsoleRenderer())
    else:
        # JSON for production
        processors.append(JSONRenderer())
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: Optional[str] = None):
    """
    Get structured logger instance.
    
    Args:
        name: Logger name (defaults to caller's module)
    
    Returns:
        Structlog logger instance
    """
    return structlog.get_logger(name)


__all__ = ["configure_logging", "get_logger"]
