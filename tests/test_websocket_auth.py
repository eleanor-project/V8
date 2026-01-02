import importlib
import sys
import types

from fastapi.testclient import TestClient

from api.middleware.auth import create_token


def _load_app(monkeypatch, required_role: str | None = None):
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("JWT_SECRET", "test-secret")
    monkeypatch.setenv("ELEANOR_ENV", "development")
    monkeypatch.setenv("ELEANOR_SKIP_READINESS", "true")
    if required_role:
        monkeypatch.setenv("WS_REQUIRED_ROLE", required_role)
    else:
        monkeypatch.delenv("WS_REQUIRED_ROLE", raising=False)

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

    import api.middleware.auth as auth
    auth._auth_config = None

    if "api.rest.main" in sys.modules:
        main = importlib.reload(sys.modules["api.rest.main"])
    else:
        main = importlib.import_module("api.rest.main")
    main.engine = None
    main.initialize_engine = lambda: None
    return main.app


def test_ws_requires_auth(monkeypatch):
    app = _load_app(monkeypatch, required_role="reviewer")
    with TestClient(app) as client:
        with client.websocket_connect("/ws/deliberate") as ws:
            msg = ws.receive_json()
            assert msg["event"] == "error"
            assert "Authorization" in msg["data"]["message"]


def test_ws_requires_role(monkeypatch):
    app = _load_app(monkeypatch, required_role="reviewer")
    token = create_token("user-1", roles=["user"])
    headers = {"Authorization": f"Bearer {token}"}

    with TestClient(app) as client:
        with client.websocket_connect("/ws/deliberate", headers=headers) as ws:
            msg = ws.receive_json()
            assert msg["event"] == "error"
            assert "Role" in msg["data"]["message"]


def test_ws_allows_role(monkeypatch):
    app = _load_app(monkeypatch, required_role="reviewer")
    token = create_token("reviewer-1", roles=["reviewer"])
    headers = {"Authorization": f"Bearer {token}"}

    with TestClient(app) as client:
        with client.websocket_connect("/ws/deliberate", headers=headers) as ws:
            msg = ws.receive_json()
            assert msg["event"] == "connection"
            ws.close()
