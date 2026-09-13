"""Tests del crítico LLM del checklist (src/llm/critic.py) — sin red."""

import json
import uuid
from types import SimpleNamespace

from src.llm.critic import _interpreted_items, run_critic_checklist


class _FakeSession:
    """Sesión mínima: responde a select(SessionSettings) y
    select(PromptTemplate) distinguiendo por nombre de tabla."""

    def __init__(self, settings=None, template=None):
        self._settings = settings or []
        self._template = template

    def execute(self, stmt):
        class _Result:
            def __init__(self, rows):
                self._rows = rows

            def scalars(self):
                return self

            def first(self):
                return self._rows[0] if self._rows else None

        table = ""
        if hasattr(stmt, "_raw_columns") and stmt._raw_columns:
            table = getattr(stmt._raw_columns[0], "name", "")
        if table == "session_settings":
            return _Result(self._settings)
        if table == "prompt_templates":
            return _Result([self._template] if self._template else [])
        return _Result([])


def _settings(**overrides):
    defaults = dict(
        id=uuid.uuid4(),
        api_key_id=uuid.uuid4(),
        small_model="meta-models/Muse-Glimmer-30B",
        large_model="deepseek-ai/DeepSeek-V4-Flash-0731",
        fallback_model=None,
        llm_retries=3,
        is_active=True,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _template(**overrides):
    defaults = dict(
        id=uuid.uuid4(),
        task_key="critic_checklist",
        version="1.0",
        intent="Eres el critico.",
        rules=[
            "fuente_verificable: el claim principal cita una fuente verificable",
            "cta_unico: exactamente un CTA",
        ],
        input_schema={},
        output_schema={},
        few_shot=[],
        is_active=True,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _patch_compiler(monkeypatch, artifact):
    import src.llm.compiler as compiler_mod

    monkeypatch.setattr(
        compiler_mod,
        "get_active_prompt",
        lambda session, model, task: artifact,
    )


def _patch_complete(monkeypatch, text):
    import src.llm.together as together_mod

    def fake_complete(
        session, prompt, model_size="small", system=None, response_format=None, **kwargs
    ):
        return text

    monkeypatch.setattr(together_mod, "complete", fake_complete)


# ---------------------------------------------------------------------
# _interpreted_items
# ---------------------------------------------------------------------


def test_interpreted_items_matches_prefix_convention():
    rules = [
        "fuente_verificable: cita una fuente",
        "cta_unico: un solo CTA",
        "regla sin prefijo de item",
    ]
    covered = _interpreted_items(
        rules, ["fuente_verificable", "cta_unico", "revision_legal"]
    )
    assert covered == {"fuente_verificable", "cta_unico"}


# ---------------------------------------------------------------------
# run_critic_checklist — flujo completo
# ---------------------------------------------------------------------


def test_critic_skipped_without_settings():
    result = run_critic_checklist(
        _FakeSession([]), "draft", ["fuente_verificable"], "obj"
    )
    assert result["status"] == "skipped"
    assert result["reason"] == "no_settings"
    # 0007: sin LLM la decisión es determinista y requiere aceptación.
    assert result["decision_source"] == "deterministic"
    assert result["requires_user_acceptance"] is True


def test_critic_skipped_without_artifact(monkeypatch):
    _patch_compiler(monkeypatch, None)
    result = run_critic_checklist(
        _FakeSession([_settings()], _template()), "draft", ["fuente_verificable"], "obj"
    )
    assert result["status"] == "skipped"
    assert result["reason"] == "not_compiled"
    assert result["decision_source"] == "deterministic"
    assert result["requires_user_acceptance"] is True


def test_critic_ok_all_items_evaluated(monkeypatch):
    _patch_compiler(monkeypatch, SimpleNamespace(prompt_text="Eres el critico."))
    _patch_complete(
        monkeypatch,
        json.dumps(
            {
                "checklist_results": [
                    {"item": "fuente_verificable", "status": "ok"},
                    {"item": "cta_unico", "status": "ok"},
                ],
                "verdict": "auto_pass",
                "reasoning": "todo ok",
            }
        ),
    )
    result = run_critic_checklist(
        _FakeSession([_settings()], _template()),
        "draft",
        ["fuente_verificable", "cta_unico"],
        "obj",
    )
    assert result["status"] == "ok"
    assert result["checklist_results"] == {
        "fuente_verificable": "ok",
        "cta_unico": "ok",
    }
    assert result["verdict"] == "auto_pass"
    assert result["model_used"] == "small"
    # 0007: decisión del LLM — revisable, sin aceptación obligatoria.
    assert result["decision_source"] == "llm"
    assert result["requires_user_acceptance"] is False


def test_critic_item_without_interpretation_is_no_evaluado(monkeypatch):
    """Regla defensiva: ítem sin interpretación en rules -> no_evaluado,
    aunque el LLM diga ok."""
    _patch_compiler(monkeypatch, SimpleNamespace(prompt_text="Eres el critico."))
    _patch_complete(
        monkeypatch,
        json.dumps(
            {
                "checklist_results": [
                    {"item": "fuente_verificable", "status": "ok"},
                    {"item": "revision_legal", "status": "ok"},  # sin interpretación
                ],
                "verdict": "auto_pass",
            }
        ),
    )
    result = run_critic_checklist(
        _FakeSession([_settings()], _template()),
        "draft",
        ["fuente_verificable", "revision_legal"],
        "obj",
    )
    assert result["checklist_results"] == {
        "fuente_verificable": "ok",
        "revision_legal": "no_evaluado",
    }
    assert result["decision_source"] == "llm"
    assert result["requires_user_acceptance"] is False


def test_critic_item_missing_from_llm_output_is_no_evaluado(monkeypatch):
    _patch_compiler(monkeypatch, SimpleNamespace(prompt_text="Eres el critico."))
    _patch_complete(
        monkeypatch,
        json.dumps(
            {
                "checklist_results": [{"item": "fuente_verificable", "status": "ok"}],
                "verdict": "auto_pass",
            }
        ),
    )
    result = run_critic_checklist(
        _FakeSession([_settings()], _template()),
        "draft",
        ["fuente_verificable", "cta_unico"],
        "obj",
    )
    assert result["checklist_results"]["cta_unico"] == "no_evaluado"
    assert result["decision_source"] == "llm"
    assert result["requires_user_acceptance"] is False


def test_critic_degraded_when_llm_unavailable(monkeypatch):
    _patch_compiler(monkeypatch, SimpleNamespace(prompt_text="Eres el critico."))

    import src.llm.together as together_mod

    def boom(
        session, prompt, model_size="small", system=None, response_format=None, **kwargs
    ):
        raise RuntimeError("red caída")

    monkeypatch.setattr(together_mod, "complete", boom)
    result = run_critic_checklist(
        _FakeSession([_settings()], _template()),
        "draft",
        ["fuente_verificable"],
        "obj",
    )
    assert result["status"] == "degraded"
    assert result["reason"] == "llm_unavailable"
    assert result["checklist_results"] == {"fuente_verificable": "no_evaluado"}
    # 0007: degradación a determinista SIEMPRE requiere aceptación.
    assert result["decision_source"] == "deterministic"
    assert result["requires_user_acceptance"] is True


def test_critic_invalid_output_is_no_evaluado(monkeypatch):
    _patch_compiler(monkeypatch, SimpleNamespace(prompt_text="Eres el critico."))
    _patch_complete(monkeypatch, "no es json")
    result = run_critic_checklist(
        _FakeSession([_settings()], _template()),
        "draft",
        ["fuente_verificable"],
        "obj",
    )
    assert result["status"] == "ok"
    assert result["checklist_results"] == {"fuente_verificable": "no_evaluado"}
    assert result["verdict"] is None
    assert result["decision_source"] == "llm"
    assert result["requires_user_acceptance"] is False


# ---------------------------------------------------------------------
# Integración con el grafo Producer-Critic (critic_node)
# ---------------------------------------------------------------------


def _base_state(**overrides):
    from src.agents.producer_critic import ProducerCriticState

    state: ProducerCriticState = {
        "brief_id": "brief-test",
        "brand_objective": "ESCAPE_SOCIAL",
        "content_bucket": "difusion_cientifica",
        "risk_level": "bajo",
        "route_decision": "repetitivo",
        "production_route": None,
        "segment_client": "S1",
        "checklist_template": ["fuente_verificable", "cta_unico"],
        "draft": "borrador",
        "iteration": 0,
        "checklist_results": {},
        "verdict": None,
        "human_decision": None,
        # 0007: contexto de generación y estado de aceptación.
        "insight_core": "insight de prueba",
        "context_pack": {"contexto": "pack de prueba"},
        "artifact_type": "video_corto",
        "channel": "tiktok",
        "requires_user_acceptance": False,
        "critic_feedback": None,
    }
    state.update(overrides)
    return state


def test_critic_node_uses_llm_checklist():
    from src.agents.producer_critic import build_graph

    def fake_producer(brief_data, context_pack, critic_feedback=None):
        return {"status": "ok", "draft": "borrador LLM"}

    def fake_critic(draft, checklist, brand_objective):
        return {
            "status": "ok",
            "checklist_results": {
                "fuente_verificable": "ok",
                "cta_unico": "fail",
            },
            "reasoning": "el CTA no es único",
        }

    graph = build_graph(producer_fn=fake_producer, critic_checklist_fn=fake_critic)
    result = graph.invoke(
        _base_state(), config={"configurable": {"thread_id": "t-critic-1"}}
    )
    assert result["checklist_results"] == {
        "fuente_verificable": True,
        "cta_unico": False,
    }
    assert result["verdict"] == "fail"
    # 0007: productor ok -> el crítico ok conserva la bandera en False.
    assert result["requires_user_acceptance"] is False
    assert result["critic_feedback"] == "el CTA no es único"


def test_critic_node_no_evaluado_blocks():
    from src.agents.producer_critic import build_graph

    def fake_producer(brief_data, context_pack, critic_feedback=None):
        return {"status": "ok", "draft": "borrador LLM"}

    def fake_critic(draft, checklist, brand_objective):
        return {
            "status": "ok",
            "checklist_results": {
                "fuente_verificable": "ok",
                "cta_unico": "no_evaluado",
            },
            "reasoning": "CTA sin interpretación",
        }

    graph = build_graph(producer_fn=fake_producer, critic_checklist_fn=fake_critic)
    result = graph.invoke(
        _base_state(), config={"configurable": {"thread_id": "t-critic-2"}}
    )
    assert result["checklist_results"] == {
        "fuente_verificable": True,
        "cta_unico": False,
    }
    assert result["verdict"] == "fail"
    assert result["requires_user_acceptance"] is False
    assert result["critic_feedback"] == "CTA sin interpretación"


def test_critic_node_degraded_requires_human_review():
    """0007: crítico degradado -> NADA evaluado, needs_human_review y el
    grafo se PAUSA en human_review_node (nunca auto_pass simulado)."""
    from src.agents.producer_critic import build_graph

    def fake_producer(brief_data, context_pack, critic_feedback=None):
        return {"status": "ok", "draft": "borrador LLM"}

    def fake_critic(draft, checklist, brand_objective):
        return {"status": "degraded", "reason": "llm_unavailable"}

    graph = build_graph(producer_fn=fake_producer, critic_checklist_fn=fake_critic)
    result = graph.invoke(
        _base_state(), config={"configurable": {"thread_id": "t-critic-3"}}
    )
    assert result["checklist_results"] == {
        "fuente_verificable": False,
        "cta_unico": False,
    }
    assert result["verdict"] == "needs_human_review"
    assert result["requires_user_acceptance"] is True
    assert "__interrupt__" in result


def test_critic_node_raising_critic_requires_human_review():
    """0007: crítico con excepción -> mismo tratamiento conservador:
    nada evaluado, needs_human_review y pausa para aceptación humana."""
    from src.agents.producer_critic import build_graph

    def fake_producer(brief_data, context_pack, critic_feedback=None):
        return {"status": "ok", "draft": "borrador LLM"}

    def boom(draft, checklist, brand_objective):
        raise RuntimeError("critico caído")

    graph = build_graph(producer_fn=fake_producer, critic_checklist_fn=boom)
    result = graph.invoke(
        _base_state(), config={"configurable": {"thread_id": "t-critic-4"}}
    )
    assert result["checklist_results"] == {
        "fuente_verificable": False,
        "cta_unico": False,
    }
    assert result["verdict"] == "needs_human_review"
    assert result["requires_user_acceptance"] is True
    assert "__interrupt__" in result
