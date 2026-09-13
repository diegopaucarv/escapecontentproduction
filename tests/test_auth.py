"""Tests del flujo RBAC real: login (POST /auth/token) y OWNER_APPROVAL
(POST /briefs/{id}/approve) — restaurados tras el rebuild.

Cubre exactamente lo que el README afirmaba y había dejado de existir:
login líder (200), login equipo (200) pero 403 al aprobar, líder
aprobando (200), password incorrecta (401), sin token (401).
"""

import uuid

import pytest
from fastapi.testclient import TestClient

from src.api.main import app, get_session
from src.auth import hash_password
from src.db.models import AppUser, ContentBrief


class _FakeSession:
    """Sesión mínima para login + approve: responde a
    session.scalar(select(AppUser)...), session.get(AppUser|ContentBrief, id)
    y session.commit()."""

    def __init__(self, users=None, briefs=None):
        self._users = users or {}
        self._briefs = briefs or {}

    def get(self, model, ident):
        if model is AppUser:
            return self._users.get(ident)
        if model is ContentBrief:
            return self._briefs.get(ident)
        return None

    def scalar(self, stmt):
        # select(AppUser).where(AppUser.email == email, AppUser.is_active.is_(True))
        email = None
        for crit in getattr(stmt, "_where_criteria", []):
            if getattr(crit.left, "name", None) == "email":
                email = crit.right.value
        if email is None:
            return None
        return next(
            (u for u in self._users.values() if u.email == email and u.is_active),
            None,
        )

    def commit(self):
        pass

    def refresh(self, obj):
        pass

    def close(self):
        pass


@pytest.fixture
def client():
    return TestClient(app)


def _make_user(**overrides) -> AppUser:
    defaults = {
        "id": uuid.uuid4(),
        "full_name": "Líder de Prueba",
        "email": "lider@ergalia.com",
        "role": "lider",
        "hashed_password": hash_password("clave-segura"),
        "is_active": True,
    }
    defaults.update(overrides)
    return AppUser(**defaults)


def _make_brief(**overrides) -> ContentBrief:
    defaults = {
        "id": uuid.uuid4(),
        "brand_objective": "ESCAPE_SOCIAL",
        "content_bucket": "difusion_cientifica",
        "resumen": "Resumen de prueba.",
        "insight_core": "Insight de prueba.",
        "segment_client": "S1",
        "status": "revision",
        "risk_level": "medio",
        "route_decision": "nueva_solucion",
        "production_route": "complete",
    }
    defaults.update(overrides)
    return ContentBrief(**defaults)


def _login(client, email: str, password: str):
    return client.post("/auth/token", data={"username": email, "password": password})


# ---------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------


def test_login_lider_returns_token(client):
    user = _make_user()
    fake = _FakeSession(users={user.id: user})
    app.dependency_overrides[get_session] = lambda: fake
    try:
        resp = _login(client, "lider@ergalia.com", "clave-segura")
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    body = resp.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]


def test_login_wrong_password_401(client):
    user = _make_user()
    fake = _FakeSession(users={user.id: user})
    app.dependency_overrides[get_session] = lambda: fake
    try:
        resp = _login(client, "lider@ergalia.com", "clave-incorrecta")
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 401


def test_login_unknown_user_401(client):
    fake = _FakeSession()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        resp = _login(client, "nadie@ergalia.com", "cualquiera")
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 401


# ---------------------------------------------------------------------
# OWNER_APPROVAL
# ---------------------------------------------------------------------


def _approve(client, brief_id, token: str | None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return client.post(f"/briefs/{brief_id}/approve", headers=headers)


def test_approve_requires_token_401(client):
    brief = _make_brief()
    fake = _FakeSession(briefs={brief.id: brief})
    app.dependency_overrides[get_session] = lambda: fake
    try:
        resp = _approve(client, brief.id, None)
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 401


def test_approve_equipo_role_403(client):
    lider = _make_user()
    equipo = _make_user(
        id=uuid.uuid4(),
        full_name="Equipo de Prueba",
        email="equipo@ergalia.com",
        role="equipo",
    )
    brief = _make_brief()
    fake = _FakeSession(
        users={lider.id: lider, equipo.id: equipo}, briefs={brief.id: brief}
    )
    app.dependency_overrides[get_session] = lambda: fake
    try:
        token = _login(client, "equipo@ergalia.com", "clave-segura").json()[
            "access_token"
        ]
        resp = _approve(client, brief.id, token)
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 403


def test_approve_lider_moves_revision_to_aprobado(client):
    lider = _make_user()
    brief = _make_brief()
    fake = _FakeSession(users={lider.id: lider}, briefs={brief.id: brief})
    app.dependency_overrides[get_session] = lambda: fake
    try:
        token = _login(client, "lider@ergalia.com", "clave-segura").json()[
            "access_token"
        ]
        resp = _approve(client, brief.id, token)
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    body = resp.json()
    assert body["brief_id"] == str(brief.id)
    assert body["status"] == "aprobado"
    assert body["approved_by"] == str(lider.id)


def test_approve_404_when_brief_missing(client):
    lider = _make_user()
    fake = _FakeSession(users={lider.id: lider})
    app.dependency_overrides[get_session] = lambda: fake
    try:
        token = _login(client, "lider@ergalia.com", "clave-segura").json()[
            "access_token"
        ]
        resp = _approve(client, uuid.uuid4(), token)
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 404


def test_approve_409_when_not_in_revision(client):
    lider = _make_user()
    brief = _make_brief(status="aprobado")
    fake = _FakeSession(users={lider.id: lider}, briefs={brief.id: brief})
    app.dependency_overrides[get_session] = lambda: fake
    try:
        token = _login(client, "lider@ergalia.com", "clave-segura").json()[
            "access_token"
        ]
        resp = _approve(client, brief.id, token)
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 409
    assert "revision" in resp.json()["detail"]
