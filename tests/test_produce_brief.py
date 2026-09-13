"""Tests del wiring Producer-Critic a sesiones reales (0007).

Se testea con una sesión falsa (sin Postgres): session.execute(select(
SessionSettings)...) devuelve vacío -> load_settings -> None -> el
productor/crítico degradan a skipped (decisión determinista) y el grafo
se pausa pidiendo aceptación del usuario (0007: la degradación SIEMPRE
requiere aceptación humana).

Cada test usa su propio brief (uuid4) como thread_id para no contaminar
el checkpointer singleton del módulo.
"""

import uuid

from src.db.models import ContentBrief, PipelineTemplate
from src.production.produce_brief import build_state, resume_produce, run_produce


class _FakeSession:
    """Sesión mínima: responde a session.execute(select(SessionSettings)...)
    con vacío para que load_settings -> None y el productor/crítico degraden
    a skipped (decisión determinista)."""

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

        # select(SessionSettings) — sin settings -> skipped (determinista).
        if hasattr(stmt, "_raw_columns") and stmt._raw_columns:
            table = getattr(stmt._raw_columns[0], "name", "")
            if table == "session_settings":
                return _Result([])
        return _Result([])


def _make_brief(**overrides) -> ContentBrief:
    defaults = {
        "id": uuid.uuid4(),
        "brand_objective": "ESCAPE_SOCIAL",
        "content_bucket": "difusion_cientifica",
        "risk_level": "bajo",
        "route_decision": "repetitivo",
        "production_route": None,
        "segment_client": "S1",
        "insight_core": "insight de prueba",
        "resumen": "Resumen de prueba.",
        "evidence_source": "Estudio 2026.",
        "prior_attempts": None,
        "repurpose_plan": [],
        "artifact_type": "video_corto",
        "channel": "tiktok",
    }
    defaults.update(overrides)
    return ContentBrief(**defaults)


def _make_template(**overrides) -> PipelineTemplate:
    defaults = {
        "id": 1,
        "brand_objective": "ESCAPE_SOCIAL",
        "content_bucket": "difusion_cientifica",
        "checklist": ["fuente_verificable", "cta_unico"],
    }
    defaults.update(overrides)
    return PipelineTemplate(**defaults)


def test_build_state_maps_brief_fields():
    brief = _make_brief()
    template = _make_template()
    state = build_state(brief, template)

    assert state["brand_objective"] == "ESCAPE_SOCIAL"
    assert state["checklist_template"] == ["fuente_verificable", "cta_unico"]
    assert state["insight_core"] == "insight de prueba"
    assert state["context_pack"]["resumen"] == "Resumen de prueba."
    assert state["requires_user_acceptance"] is False
    assert state["brief_id"] == str(brief.id)


def test_run_produce_auto_pass_with_fake_fns(monkeypatch):
    """0007: con productor y crítico LLM ok (fns falsas), el grafo auto-pasa
    sin pausar."""
    import src.production.produce_brief as produce_brief_mod

    brief = _make_brief()
    template = _make_template()
    session = _FakeSession()

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

    result = run_produce(session, brief, template)

    assert result["verdict"] == "auto_pass"
    assert "__interrupt__" not in result
    assert result["draft"] == "borrador LLM"


def test_run_produce_degraded_pauses_for_acceptance():
    """0007: sin settings (fns reales + sesión falsa) -> productor y crítico
    degradan a skipped -> decisión determinista que SIEMPRE pausa para que
    el humano acepte la degradación."""
    brief = _make_brief()
    template = _make_template()
    session = _FakeSession()

    result = run_produce(session, brief, template)

    assert "__interrupt__" in result
    assert result["__interrupt__"][0].value["requires_user_acceptance"] is True
    assert result["requires_user_acceptance"] is True


def test_resume_produce_after_pause():
    """Reanuda el grafo pausado (degradado) con la decisión humana."""
    brief = _make_brief()
    template = _make_template()
    session = _FakeSession()

    first = run_produce(session, brief, template)
    assert "__interrupt__" in first

    result = resume_produce(session, str(brief.id), "approve")

    assert result["human_decision"] == "approve"
