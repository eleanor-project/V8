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
    # Stub optional heavy deps to avoid import errors
    sys.modules.setdefault("weaviate", types.SimpleNamespace(Client=None))
    fake_sql = types.SimpleNamespace()
    sys.modules.setdefault("psycopg2", types.SimpleNamespace(connect=None, Error=Exception, sql=fake_sql))
    sys.modules.setdefault("psycopg2.sql", fake_sql)

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
