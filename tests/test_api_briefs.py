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

            def scalars(self):
                return self

            def first(self):
                return self._rows[0] if self._rows else None

        # select(SessionSettings) — el refuerzo LLM lee la config activa.
        # Sin settings -> skipped (decisión determinista).
        if hasattr(stmt, "_raw_columns") and stmt._raw_columns:
            table = getattr(stmt._raw_columns[0], "name", "")
            if table == "session_settings":
                return _Result([])

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
    # 0007: sin settings el refuerzo LLM se omite -> decisión determinista
    # que requiere aceptación explícita del usuario.
    assert body["llm_reinforcement"]["status"] == "skipped"
    assert body["decision_source"] == "deterministic"
    assert body["requires_user_acceptance"] is True


def test_alignment_llm_reinforcement_ok_sets_decision_source(client, monkeypatch):
    """0007: con refuerzo LLM ok, la decisión viene del LLM y NO requiere
    aceptación obligatoria del usuario."""
    import src.llm.reinforcement as reinforcement_mod
    from src.llm.base import ok_result

    brief = _make_brief(
        evidence_source="Estudio 2026.",
        pitch_15s="¿Crees que ya no necesitas el refuerzo?",
        cta="Descarga la guía",
        risk_level="bajo",
        route_decision="repetitivo",
        production_route="fast",
    )
    template = _make_template()
    monkeypatch.setattr(
        reinforcement_mod,
        "reinforce_alignment",
        lambda session, brief_data, rule_verdict, checklist_results: ok_result(
            verdict="auto_pass",
            llm_verdict="auto_pass",
            reasoning="sin riesgos",
            risks_detected=[],
            model_used="small",
            fallback_used=False,
        ),
    )
    app.dependency_overrides[get_session] = lambda: _FakeSession(
        {brief.id: brief}, [template]
    )
    try:
        resp = client.post(f"/briefs/{brief.id}/alignment")
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    body = resp.json()
    assert body["llm_reinforcement"]["status"] == "ok"
    assert body["decision_source"] == "llm"
    assert body["requires_user_acceptance"] is False


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


def test_alignment_fail_moves_brief_to_generando(client):
    """Veredicto fail → retrabajo: el brief NO queda en revision (no se
    puede aprobar), va a generando (§3.9: revision --> generando: fail)."""
    brief = _make_brief(
        status="idea",
        evidence_source=None,  # checklist 'fuente_verificable' falla
        pitch_15s=None,
        cta=None,
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
    assert body["verdict"] == "fail"
    assert body["semaforo"] == "🔴"
    assert brief.status == "generando"


def test_alignment_needs_human_review_moves_to_revision(client):
    """Veredicto needs_human_review → revision: esperando al 🟨 líder."""
    brief = _make_brief(
        status="idea",
        evidence_source="Estudio 2026.",
        pitch_15s="¿Crees que ya no necesitas el refuerzo?",
        cta="Descarga la guía",
        risk_level="medio",  # riesgo medio => needs_human_review
        route_decision="nueva_solucion",
        production_route="complete",
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
    assert resp.json()["verdict"] == "needs_human_review"
    assert brief.status == "revision"


def test_alignment_degraded_persists_requires_user_acceptance(client):
    """0007: sin settings el refuerzo se omite -> decisión determinista que
    requiere aceptación del usuario; el flag se persiste en el brief."""
    brief = _make_brief(
        status="idea",
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
    assert body["requires_user_acceptance"] is True
    assert brief.requires_user_acceptance is True


def test_alignment_llm_ok_persists_false(client, monkeypatch):
    """0007: con refuerzo LLM ok, la decisión viene del LLM y el flag de
    aceptación del usuario se persiste en False."""
    import src.llm.reinforcement as reinforcement_mod
    from src.llm.base import ok_result

    brief = _make_brief(
        status="idea",
        evidence_source="Estudio 2026.",
        pitch_15s="¿Crees que ya no necesitas el refuerzo?",
        cta="Descarga la guía",
        risk_level="bajo",
        route_decision="repetitivo",
        production_route="fast",
    )
    template = _make_template()
    monkeypatch.setattr(
        reinforcement_mod,
        "reinforce_alignment",
        lambda session, brief_data, rule_verdict, checklist_results: ok_result(
            verdict="auto_pass",
            llm_verdict="auto_pass",
            reasoning="sin riesgos",
            risks_detected=[],
            model_used="small",
            fallback_used=False,
        ),
    )
    app.dependency_overrides[get_session] = lambda: _FakeSession(
        {brief.id: brief}, [template]
    )
    try:
        resp = client.post(f"/briefs/{brief.id}/alignment")
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    body = resp.json()
    assert body["requires_user_acceptance"] is False
    assert brief.requires_user_acceptance is False


def test_get_brief_surfaces_requires_user_acceptance(client):
    """0007: el flag de aceptación del usuario se expone en GET /briefs/{id}."""
    brief = _make_brief(requires_user_acceptance=True)
    app.dependency_overrides[get_session] = lambda: _FakeSession({brief.id: brief})
    try:
        resp = client.get(f"/briefs/{brief.id}")
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    body = resp.json()
    assert body["requires_user_acceptance"] is True
