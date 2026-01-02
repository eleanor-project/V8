import importlib
import sys
import types

from fastapi.testclient import TestClient


def get_app_without_startup():
    """
    Import the API and disable lifespan init to avoid hitting real backends
    during lightweight smoke tests.
    """
    import os
    os.environ.setdefault("AUTH_ENABLED", "false")
    os.environ.setdefault("ELEANOR_SKIP_READINESS", "true")
    # Stub optional heavy deps to avoid import errors
    weaviate_stub = types.ModuleType("weaviate")
    weaviate_stub.Client = None
    sys.modules.setdefault("weaviate", weaviate_stub)

    fake_sql = types.ModuleType("psycopg2.sql")
    sys.modules.setdefault("psycopg2.sql", fake_sql)

    psycopg2_stub = types.ModuleType("psycopg2")
    psycopg2_stub.connect = None
    psycopg2_stub.Error = Exception
    psycopg2_stub.sql = fake_sql
    sys.modules.setdefault("psycopg2", psycopg2_stub)

    main = importlib.import_module("api.rest.main")
    main.engine = None
    main.initialize_engine = lambda: None  # no-op
    return main.app


def test_health_endpoint():
    app = get_app_without_startup()
    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert "status" in data
        assert "checks" in data


def test_metrics_endpoint():
    app = get_app_without_startup()
    with TestClient(app) as client:
        res = client.get("/metrics")
        assert res.status_code == 200


def test_admin_bindings_requires_engine():
    app = get_app_without_startup()
    with TestClient(app) as client:
        res = client.get("/admin/critics/bindings")
        assert res.status_code == 503


def test_evaluate_requires_engine():
    app = get_app_without_startup()
    payload = {
        "request_id": "req_test_001",
        "timestamp": "2025-01-01T00:00:00Z",
        "policy_profile": "default",
        "model_output": "Test output",
        "proposed_action": {"type": "generate_advice", "params": {}},
        "context": {"domain": "test"},
    }
    with TestClient(app) as client:
        res = client.post("/evaluate", json=payload)
        assert res.status_code == 503
