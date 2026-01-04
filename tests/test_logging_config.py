import logging
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from engine import logging_config


def test_get_log_level_defaults_and_invalid(monkeypatch):
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    assert logging_config.get_log_level() == logging.INFO

    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    assert logging_config.get_log_level() == logging.DEBUG

    monkeypatch.setenv("LOG_LEVEL", "NOT_A_LEVEL")
    assert logging_config.get_log_level() == logging.INFO


def test_get_log_format_defaults(monkeypatch):
    monkeypatch.delenv("LOG_FORMAT", raising=False)
    monkeypatch.setenv("ELEANOR_ENVIRONMENT", "development")
    assert logging_config.get_log_format() == "console"

    monkeypatch.setenv("ELEANOR_ENVIRONMENT", "production")
    assert logging_config.get_log_format() == "json"

    monkeypatch.setenv("LOG_FORMAT", "console")
    assert logging_config.get_log_format() == "console"


def test_configure_logging_without_structlog(monkeypatch):
    monkeypatch.setattr(logging_config, "STRUCTLOG_AVAILABLE", False)
    logging_config.configure_logging()


def test_configure_logging_with_structlog(monkeypatch):
    stub = SimpleNamespace(
        contextvars=SimpleNamespace(
            merge_contextvars=lambda *args, **kwargs: None,
            bind_contextvars=MagicMock(),
            unbind_contextvars=MagicMock(),
        ),
        stdlib=SimpleNamespace(
            add_log_level=lambda *args, **kwargs: None,
            add_logger_name=lambda *args, **kwargs: None,
            PositionalArgumentsFormatter=lambda *args, **kwargs: lambda *_a, **_k: None,
            BoundLogger=object,
            LoggerFactory=lambda *args, **kwargs: object(),
        ),
        processors=SimpleNamespace(
            TimeStamper=lambda *args, **kwargs: lambda *_a, **_k: None,
            StackInfoRenderer=lambda *args, **kwargs: lambda *_a, **_k: None,
            UnicodeDecoder=lambda *args, **kwargs: lambda *_a, **_k: None,
            JSONRenderer=lambda *args, **kwargs: lambda *_a, **_k: None,
            format_exc_info=lambda *args, **kwargs: None,
        ),
        dev=SimpleNamespace(
            ConsoleRenderer=lambda *args, **kwargs: lambda *_a, **_k: None,
        ),
        configure=MagicMock(),
        get_logger=MagicMock(return_value=MagicMock()),
    )

    monkeypatch.setattr(logging_config, "STRUCTLOG_AVAILABLE", True)
    monkeypatch.setattr(logging_config, "structlog", stub)
    monkeypatch.setenv("LOG_FORMAT", "json")

    logging_config.configure_logging()
    assert stub.configure.called


def test_get_logger_structlog_unavailable(monkeypatch):
    monkeypatch.setattr(logging_config, "STRUCTLOG_AVAILABLE", False)
    logger = logging_config.get_logger("test")
    assert isinstance(logger, logging.Logger)


def test_log_context_binds(monkeypatch):
    bind = MagicMock()
    unbind = MagicMock()
    stub = SimpleNamespace(
        contextvars=SimpleNamespace(bind_contextvars=bind, unbind_contextvars=unbind)
    )
    monkeypatch.setattr(logging_config, "STRUCTLOG_AVAILABLE", True)
    monkeypatch.setattr(logging_config, "structlog", stub)
    monkeypatch.setitem(sys.modules, "structlog", stub)

    ctx = logging_config.LogContext(trace_id="abc", user_id="user1")
    ctx.__enter__()
    ctx.__exit__(None, None, None)

    assert bind.called
    assert unbind.called


def test_log_helpers_emit_extra(caplog):
    logger = logging.getLogger("test_logging_helpers")

    with caplog.at_level(logging.INFO):
        logging_config.log_request(
            logger,
            method="POST",
            path="/test",
            status_code=200,
            duration_ms=12.3,
            trace_id="t1",
            user_id="u1",
        )

    assert caplog.records[-1].msg == "http_request"
    assert caplog.records[-1].method == "POST"

    with caplog.at_level(logging.INFO):
        logging_config.log_deliberation(
            logger,
            trace_id="t2",
            decision="allow",
            model_used="model",
            duration_ms=10.0,
            uncertainty=0.1,
            escalated=True,
        )
    assert caplog.records[-1].msg == "deliberation_complete"

    with caplog.at_level(logging.WARNING):
        logging_config.log_critic_execution(
            logger,
            trace_id="t3",
            critic_name="critic",
            severity=0.4,
            duration_ms=5.0,
            success=False,
            error="boom",
        )
    assert caplog.records[-1].msg == "critic_executed"
    assert caplog.records[-1].error == "boom"

    with caplog.at_level(logging.INFO):
        logging_config.log_precedent_retrieval(
            logger,
            trace_id="t4",
            query_length=10,
            cases_found=2,
            alignment_score=0.5,
            duration_ms=9.0,
        )
    assert caplog.records[-1].msg == "precedent_retrieved"
