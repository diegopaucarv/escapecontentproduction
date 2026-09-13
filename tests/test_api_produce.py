"""Tests de los endpoints POST /briefs/{id}/produce y /produce/resume (0007).

El grafo Producer-Critic se dispara con sesión real (wiring en
src/production/produce_brief.py). Se testea con una sesión falsa en
memoria (sin Postgres), inyectada vía dependency_overrides de FastAPI.

Filosofía 0007 verificada aquí:
- Sin settings LLM, el productor/crítico degradan a determinista y el
  grafo se PAUSA con requires_user_acceptance=True (nunca sigue solo).
- Con fns LLM ok (monkeypatch), el grafo auto-pasa y materializa el
  ContentArtifact.
- El resume entrega la decisión humana al thread pausado.
"""

import uuid

import pytest
from fastapi.testclient import TestClient

from src.api.main import app, get_session
from src.db.models import ContentArtifact, ContentBrief, PipelineTemplate


class _FakeSession:
    """Sesión mínima: get(ContentBrief), execute (session_settings vacío +
    pipeline_templates filtrado), add (registra objetos) y commit no-op."""

    def __init__(
        self,
        briefs: dict[uuid.UUID, ContentBrief],
        templates: list[PipelineTemplate] | None = None,
    ):
        self._briefs = briefs
        self._templates = templates or []
        self.added: list = []

    def get(self, model, ident):
        if model is ContentBrief:
            return self._briefs.get(ident)
        return None

    def execute(self, stmt):
        class _Result:
            def __init__(self, rows):
                self._rows = rows

            def scalar_one_or_none(self):
                return self._rows[0] if self._rows else None

            def scalars(self):
                return self

            def first(self):
                return self._rows[0] if self._rows else None

        # select(SessionSettings) — sin settings -> skipped (degradación).
        if hasattr(stmt, "_raw_columns") and stmt._raw_columns:
            table = getattr(stmt._raw_columns[0], "name", "")
            if table == "session_settings":
                return _Result([])

        rows = [
            t
            for t in self._templates
            if t.brand_objective == stmt._where_criteria[0].right.value
            and t.content_bucket == stmt._where_criteria[1].right.value
        ]
        return _Result(rows)

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        pass

    def close(self):
        pass


def _make_brief(**overrides) -> ContentBrief:
    defaults = {
        "id": uuid.uuid4(),
        "brand_objective": "ESCAPE_SOCIAL",
        "content_bucket": "difusion_cientifica",
        "resumen": "Resumen de prueba.",
        "insight_core": "Insight de prueba.",
        "segment_client": "S1",
        "status": "aprobado",
        "risk_level": "bajo",
        "route_decision": "repetitivo",
        "production_route": "fast",
        "artifact_type": "video_corto",
        "channel": "tiktok",
    }
    defaults.update(overrides)
    return ContentBrief(**defaults)


def _make_template(**overrides) -> PipelineTemplate:
    defaults = dict(
        id=1,
        brand_objective="ESCAPE_SOCIAL",
        content_bucket="difusion_cientifica",
        checklist=["fuente_verificable", "cta_unico"],
        novelty_weights={},
    )
    defaults.update(overrides)
    return PipelineTemplate(**defaults)


@pytest.fixture
def client():
    return TestClient(app)


def test_produce_404_when_brief_missing(client):
    app.dependency_overrides[get_session] = lambda: _FakeSession({})
    try:
        resp = client.post(f"/briefs/{uuid.uuid4()}/produce")
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 404


def test_produce_409_when_not_approved(client):
    brief = _make_brief(status="revision")
    app.dependency_overrides[get_session] = lambda: _FakeSession(
        {brief.id: brief}, [_make_template()]
    )
    try:
        resp = client.post(f"/briefs/{brief.id}/produce")
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 409
    assert "aprobado" in resp.json()["detail"]


def test_produce_404_when_no_brand_plan(client):
    brief = _make_brief()
    app.dependency_overrides[get_session] = lambda: _FakeSession({brief.id: brief}, [])
    try:
        resp = client.post(f"/briefs/{brief.id}/produce")
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 404
    assert "plan de marca" in resp.json()["detail"]


def test_produce_degraded_pauses_for_acceptance(client):
    """0007: sin settings LLM, el wiring real degrada y el grafo se PAUSA
    con requires_user_acceptance=True — nunca sigue solo."""
    brief = _make_brief()
    app.dependency_overrides[get_session] = lambda: _FakeSession(
        {brief.id: brief}, [_make_template()]
    )
    try:
        resp = client.post(f"/briefs/{brief.id}/produce")
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    body = resp.json()
    assert body["interrupted"] is True
    assert body["requires_user_acceptance"] is True
    assert body["interrupt"]["requires_user_acceptance"] is True
    assert "determinista" in body["interrupt"]["pregunta"]


def test_produce_auto_pass_creates_artifact(client, monkeypatch):
    """0007: con productor y crítico LLM ok, el grafo auto-pasa y la pieza
    se materializa como ContentArtifact."""
    import src.production.produce_brief as produce_brief_mod

    def fake_producer(brief_data, context_pack, critic_feedback=None):
        return {"status": "ok", "draft": "borrador LLM"}

    def fake_critic(draft, checklist, brand_objective):
        return {
            "status": "ok",
            "checklist_results": {item: "ok" for item in checklist},
            "reasoning": "todo ok",
        }

    monkeypatch.setattr(
        produce_brief_mod, "make_producer_fn", lambda session: fake_producer
    )
    monkeypatch.setattr(
        produce_brief_mod, "make_critic_checklist_fn", lambda session: fake_critic
    )

    brief = _make_brief()
    fake = _FakeSession({brief.id: brief}, [_make_template()])
    app.dependency_overrides[get_session] = lambda: fake
    try:
        resp = client.post(f"/briefs/{brief.id}/produce")
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    body = resp.json()
    assert body["interrupted"] is False
    assert body["verdict"] == "auto_pass"
    assert body["draft"] == "borrador LLM"
    assert body["requires_user_acceptance"] is False
    # La pieza se materializó como artefacto.
    assert len(fake.added) == 1
    artifact = fake.added[0]
    assert isinstance(artifact, ContentArtifact)
    assert artifact.brief_id == brief.id
    assert artifact.artifact_type == "video_corto"
    assert artifact.channel == "tiktok"
    assert artifact.status == "borrador"


def test_produce_resume_after_pause(client):
    """0007: tras una pausa por degradación, el resume entrega la decisión
    humana al thread pausado (mismo checkpointer, mismo thread_id)."""
    brief = _make_brief()
    fake = _FakeSession({brief.id: brief}, [_make_template()])
    app.dependency_overrides[get_session] = lambda: fake
    try:
        resp = client.post(f"/briefs/{brief.id}/produce")
        assert resp.status_code == 200
        assert resp.json()["interrupted"] is True

        resp2 = client.post(
            f"/briefs/{brief.id}/produce/resume", json={"decision": "approve"}
        )
    finally:
        app.dependency_overrides.clear()

    assert resp2.status_code == 200
    body = resp2.json()
    assert body["human_decision"] == "approve"
    assert body["interrupted"] is False


def test_produce_resume_404_when_brief_missing(client):
    app.dependency_overrides[get_session] = lambda: _FakeSession({})
    try:
        resp = client.post(
            f"/briefs/{uuid.uuid4()}/produce/resume", json={"decision": "approve"}
        )
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 404


def test_produce_resume_rejects_invalid_decision(client):
    brief = _make_brief()
    app.dependency_overrides[get_session] = lambda: _FakeSession(
        {brief.id: brief}, [_make_template()]
    )
    try:
        resp = client.post(
            f"/briefs/{brief.id}/produce/resume", json={"decision": "talvez"}
        )
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 422
