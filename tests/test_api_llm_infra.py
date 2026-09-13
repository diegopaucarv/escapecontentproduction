"""Tests del CRUD de infraestructura LLM (api_keys + session_settings)."""

import uuid

import pytest
from fastapi.testclient import TestClient

from src.api.main import app, get_session
from src.db.models import ApiKey, SessionSettings


class _FakeSession:
    """Sesión mínima para los endpoints de api-keys/settings."""

    def __init__(self, keys=None, settings=None):
        self._keys = keys or {}
        self._settings = settings or []
        self._deleted = []

    def get(self, model, ident):
        if model is ApiKey:
            return self._keys.get(ident)
        if model is SessionSettings:
            return next((s for s in self._settings if s.id == ident), None)
        return None

    def execute(self, stmt):
        class _Result:
            def __init__(self, rows):
                self._rows = rows

            def scalars(self):
                return self

            def all(self):
                return self._rows

            def first(self):
                return self._rows[0] if self._rows else None

        # select(ApiKey) o select(SessionSettings)
        if hasattr(stmt, "_where_criteria") and stmt._where_criteria:
            col = stmt._where_criteria[0].left
            if col.name == "is_active":
                return _Result([s for s in self._settings if s.is_active])
        return _Result(list(self._keys.values()))

    def add(self, obj):
        if isinstance(obj, ApiKey):
            self._keys[obj.id] = obj
        else:
            self._settings.append(obj)

    def delete(self, obj):
        if isinstance(obj, ApiKey):
            self._keys.pop(obj.id, None)
        else:
            self._settings = [s for s in self._settings if s.id != obj.id]

    def commit(self):
        pass

    def refresh(self, obj):
        pass

    def close(self):
        pass


@pytest.fixture
def client():
    return TestClient(app)


def _make_key(**overrides) -> ApiKey:
    defaults = dict(
        id=uuid.uuid4(),
        provider="together",
        key_name="together-main",
        api_key="tgp_v1_secret_key_1234567890",
        is_active=True,
    )
    defaults.update(overrides)
    return ApiKey(**defaults)


def test_create_and_list_api_keys(client):
    fake = _FakeSession()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        resp = client.post(
            "/api-keys",
            json={
                "provider": "together",
                "key_name": "together-main",
                "api_key": "tgp_v1_secret_key_1234567890",
            },
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["provider"] == "together"
        assert "api_key_masked" in body
        assert "tgp_v1_secret_key_1234567890" not in body["api_key_masked"]

        resp = client.get("/api-keys")
        assert resp.status_code == 200
        assert len(resp.json()) == 1
    finally:
        app.dependency_overrides.clear()


def test_get_api_key_masks_secret(client):
    key = _make_key()
    fake = _FakeSession(keys={key.id: key})
    app.dependency_overrides[get_session] = lambda: fake
    try:
        resp = client.get(f"/api-keys/{key.id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["api_key_masked"] == "tgp_v1_****7890"
        assert "secret" not in body["api_key_masked"]
    finally:
        app.dependency_overrides.clear()


def test_update_api_key(client):
    key = _make_key()
    fake = _FakeSession(keys={key.id: key})
    app.dependency_overrides[get_session] = lambda: fake
    try:
        resp = client.put(
            f"/api-keys/{key.id}",
            json={"key_name": "together-prod", "is_active": False},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["key_name"] == "together-prod"
        assert body["is_active"] is False
    finally:
        app.dependency_overrides.clear()


def test_delete_api_key(client):
    key = _make_key()
    fake = _FakeSession(keys={key.id: key})
    app.dependency_overrides[get_session] = lambda: fake
    try:
        resp = client.delete(f"/api-keys/{key.id}")
        assert resp.status_code == 204
        assert key.id not in fake._keys
    finally:
        app.dependency_overrides.clear()


def test_create_settings_and_get_active(client):
    key = _make_key()
    fake = _FakeSession(keys={key.id: key})
    app.dependency_overrides[get_session] = lambda: fake
    try:
        resp = client.post(
            "/settings",
            json={
                "api_key_id": str(key.id),
                "small_model": "meta-models/Muse-Glimmer-30B",
                "large_model": "deepseek-ai/DeepSeek-V4-Flash-0731",
            },
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["small_model"] == "meta-models/Muse-Glimmer-30B"
        assert body["large_model"] == "deepseek-ai/DeepSeek-V4-Flash-0731"

        resp = client.get("/settings")
        assert resp.status_code == 200
        assert resp.json()["small_model"] == "meta-models/Muse-Glimmer-30B"
    finally:
        app.dependency_overrides.clear()


def test_get_settings_404_when_none(client):
    fake = _FakeSession()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        resp = client.get("/settings")
        assert resp.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_create_settings_404_when_key_missing(client):
    fake = _FakeSession()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        resp = client.post(
            "/settings",
            json={
                "api_key_id": str(uuid.uuid4()),
                "small_model": "m1",
                "large_model": "m2",
            },
        )
        assert resp.status_code == 404
    finally:
        app.dependency_overrides.clear()
