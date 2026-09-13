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


def test_critic_skipped_without_artifact(monkeypatch):
    _patch_compiler(monkeypatch, None)
    result = run_critic_checklist(
        _FakeSession([_settings()], _template()), "draft", ["fuente_verificable"], "obj"
    )
    assert result["status"] == "skipped"
    assert result["reason"] == "not_compiled"


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
    }
    state.update(overrides)
    return state


def test_critic_node_uses_llm_checklist():
    from src.agents.producer_critic import build_graph

    def fake_critic(draft, checklist, brand_objective):
        return {
            "status": "ok",
            "checklist_results": {
                "fuente_verificable": "ok",
                "cta_unico": "fail",
            },
        }

    graph = build_graph(critic_checklist_fn=fake_critic)
    result = graph.invoke(
        _base_state(), config={"configurable": {"thread_id": "t-critic-1"}}
    )
    assert result["checklist_results"] == {
        "fuente_verificable": True,
        "cta_unico": False,
    }
    assert result["verdict"] == "fail"


def test_critic_node_no_evaluado_blocks():
    from src.agents.producer_critic import build_graph

    def fake_critic(draft, checklist, brand_objective):
        return {
            "status": "ok",
            "checklist_results": {
                "fuente_verificable": "ok",
                "cta_unico": "no_evaluado",
            },
        }

    graph = build_graph(critic_checklist_fn=fake_critic)
    result = graph.invoke(
        _base_state(), config={"configurable": {"thread_id": "t-critic-2"}}
    )
    assert result["checklist_results"] == {
        "fuente_verificable": True,
        "cta_unico": False,
    }
    assert result["verdict"] == "fail"


def test_critic_node_degraded_falls_back_to_simulated():
    from src.agents.producer_critic import build_graph

    def fake_critic(draft, checklist, brand_objective):
        return {"status": "degraded", "reason": "llm_unavailable"}

    graph = build_graph(critic_checklist_fn=fake_critic)
    result = graph.invoke(
        _base_state(), config={"configurable": {"thread_id": "t-critic-3"}}
    )
    assert result["checklist_results"] == {
        "fuente_verificable": True,
        "cta_unico": True,
    }
    assert result["verdict"] == "auto_pass"


def test_critic_node_raising_critic_falls_back():
    from src.agents.producer_critic import build_graph

    def boom(draft, checklist, brand_objective):
        raise RuntimeError("critico caído")

    graph = build_graph(critic_checklist_fn=boom)
    result = graph.invoke(
        _base_state(), config={"configurable": {"thread_id": "t-critic-4"}}
    )
    assert result["verdict"] == "auto_pass"
