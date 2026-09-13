"""Tests del endpoint GET /briefs/{id} — la primera acción del pipeline.

Se testea con un ContentBrief en memoria (sin base de datos real),
inyectando una sesión falsa vía dependency_overrides de FastAPI.
"""

import uuid

import pytest
from fastapi.testclient import TestClient

from src.api.main import app, get_session
from src.db.models import ContentBrief, PipelineTemplate


class _FakeSession:
    """Sesión mínima que responde a session.get(ContentBrief, id) y a
    session.execute(select(PipelineTemplate)...)."""

    def __init__(
        self,
        briefs: dict[uuid.UUID, ContentBrief],
        templates: list[PipelineTemplate] | None = None,
    ):
        self._briefs = briefs
        self._templates = templates or []

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

        # Filtra por brand_objective + content_bucket (lo que usa el endpoint)
        rows = [
            t
            for t in self._templates
            if t.brand_objective == stmt._where_criteria[0].right.value
            and t.content_bucket == stmt._where_criteria[1].right.value
        ]
        return _Result(rows)

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
        "status": "revision",
        "risk_level": "medio",
        "route_decision": "nueva_solucion",
        "production_route": "complete",
    }
    defaults.update(overrides)
    return ContentBrief(**defaults)


@pytest.fixture
def client():
    return TestClient(app)


def test_get_brief_returns_full_payload(client):
    brief = _make_brief()
    app.dependency_overrides[get_session] = lambda: _FakeSession({brief.id: brief})
    try:
        resp = client.get(f"/briefs/{brief.id}")
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == str(brief.id)
    assert body["status"] == "revision"
    assert body["brand_objective"] == "ESCAPE_SOCIAL"
    assert body["content_bucket"] == "difusion_cientifica"
    assert body["resumen"] == "Resumen de prueba."
    assert body["insight_core"] == "Insight de prueba."
    assert body["segment_client"] == "S1"
    assert body["risk_level"] == "medio"
    assert body["route_decision"] == "nueva_solucion"
    assert body["production_route"] == "complete"


def test_get_brief_404_when_missing(client):
    app.dependency_overrides[get_session] = lambda: _FakeSession({})
    try:
        resp = client.get(f"/briefs/{uuid.uuid4()}")
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 404


def test_get_brief_invalid_uuid_returns_422(client):
    resp = client.get("/briefs/no-es-un-uuid")
    assert resp.status_code == 422


def _make_template(**overrides) -> PipelineTemplate:
    defaults = dict(
        id=1,
        brand_objective="ESCAPE_SOCIAL",
        content_bucket="difusion_cientifica",
        checklist=[
            "fuente_verificable",
            "gancho_15s_ok",
            "cta_unico",
            "anonimizacion_no_aplica",
        ],
        novelty_weights={
            "bucket_nuevo": 3,
            "formato_nuevo": 2,
            "canal_nuevo": 2,
            "angulo_nuevo": 3,
        },
    )
    defaults.update(overrides)
    return PipelineTemplate(**defaults)


def test_alignment_returns_semaforo(client):
    brief = _make_brief(
        evidence_source="Estudio 2026.",
        pitch_15s="¿Crees que ya no necesitas el refuerzo?",
        cta="Descarga la guía",
        risk_level="bajo",
        route_decision="repetitivo",
        production_route="fast",
    )
    template = _make_template()
    app.dependency_overrides[get_session] = lambda: _FakeSession(
        {brief.id: brief}, [template]
    )
    try:
        resp = client.post(f"/briefs/{brief.id}/alignment")
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    body = resp.json()
    assert body["semaforo"] == "🟢"
    assert body["verdict"] == "auto_pass"
    assert body["failed_items"] == []
    assert len(body["checklist"]) == 4


def test_alignment_404_when_brief_missing(client):
    app.dependency_overrides[get_session] = lambda: _FakeSession({})
    try:
        resp = client.post(f"/briefs/{uuid.uuid4()}/alignment")
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 404


def test_alignment_404_when_no_brand_plan(client):
    brief = _make_brief()
    app.dependency_overrides[get_session] = lambda: _FakeSession({brief.id: brief}, [])
    try:
        resp = client.post(f"/briefs/{brief.id}/alignment")
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 404
    assert "plan de marca" in resp.json()["detail"]
