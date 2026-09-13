"""Tests del productor LLM (src/llm/producer.py) — sin red."""

import json
import uuid
from types import SimpleNamespace

from src.llm.producer import (
    DETERMINISTIC_DRAFT,
    make_producer_fn,
    run_producer_generation,
)


class _FakeSession:
    """Sesión mínima: responde a select(SessionSettings)."""

    def __init__(self, settings=None):
        self._settings = settings or []

    def execute(self, stmt):
        class _Result:
            def __init__(self, rows):
                self._rows = rows

            def scalars(self):
                return self

            def first(self):
                return self._rows[0] if self._rows else None

        return _Result(self._settings)


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


def _artifact(prompt_text="Eres el productor editorial."):
    return SimpleNamespace(prompt_text=prompt_text)


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
# run_producer_generation — flujo completo
# ---------------------------------------------------------------------


def test_producer_skipped_without_settings():
    result = run_producer_generation(_FakeSession([]), {}, {})
    assert result["status"] == "skipped"
    assert result["reason"] == "no_settings"
    # 0007: sin LLM la decisión es determinista y requiere aceptación.
    assert result["decision_source"] == "deterministic"
    assert result["requires_user_acceptance"] is True
    assert result["draft"] == DETERMINISTIC_DRAFT


def test_producer_skipped_without_artifact(monkeypatch):
    _patch_compiler(monkeypatch, None)
    result = run_producer_generation(_FakeSession([_settings()]), {}, {})
    assert result["status"] == "skipped"
    assert result["reason"] == "not_compiled"
    assert result["decision_source"] == "deterministic"
    assert result["requires_user_acceptance"] is True
    assert result["draft"] == DETERMINISTIC_DRAFT


def test_producer_ok_uses_llm_draft(monkeypatch):
    _patch_compiler(monkeypatch, _artifact())
    _patch_complete(
        monkeypatch,
        json.dumps(
            {
                "draft": "borrador del LLM",
                "hook_15s": "¿Sabías que...?",
                "cta": "Descarga la guía",
                "reasoning": "tono de marca respetado",
            }
        ),
    )
    result = run_producer_generation(
        _FakeSession([_settings()]),
        {"insight_core": "x"},
        {"contexto": "y"},
        critic_feedback="mejorar el gancho",
    )
    assert result["status"] == "ok"
    # 0007: decisión del LLM — revisable, sin aceptación obligatoria.
    assert result["decision_source"] == "llm"
    assert result["requires_user_acceptance"] is False
    assert result["draft"] == "borrador del LLM"
    assert result["hook_15s"] == "¿Sabías que...?"
    assert result["cta"] == "Descarga la guía"
    assert result["model_used"] == "small"


def test_producer_degraded_when_llm_unavailable(monkeypatch):
    _patch_compiler(monkeypatch, _artifact())

    import src.llm.together as together_mod

    def boom(
        session, prompt, model_size="small", system=None, response_format=None, **kwargs
    ):
        raise RuntimeError("red caída")

    monkeypatch.setattr(together_mod, "complete", boom)
    result = run_producer_generation(_FakeSession([_settings()]), {}, {})
    assert result["status"] == "degraded"
    assert result["reason"] == "llm_unavailable"
    # 0007: degradación a determinista SIEMPRE requiere aceptación.
    assert result["decision_source"] == "deterministic"
    assert result["requires_user_acceptance"] is True
    assert result["draft"] == DETERMINISTIC_DRAFT


def test_producer_degraded_on_invalid_output(monkeypatch):
    _patch_compiler(monkeypatch, _artifact())
    _patch_complete(monkeypatch, "no es json")
    result = run_producer_generation(_FakeSession([_settings()]), {}, {})
    assert result["status"] == "degraded"
    assert result["reason"] == "invalid_output"
    assert result["decision_source"] == "deterministic"
    assert result["requires_user_acceptance"] is True
    assert result["draft"] == DETERMINISTIC_DRAFT


def test_producer_degraded_on_missing_draft(monkeypatch):
    _patch_compiler(monkeypatch, _artifact())
    _patch_complete(monkeypatch, json.dumps({"hook_15s": "sin draft"}))
    result = run_producer_generation(_FakeSession([_settings()]), {}, {})
    assert result["status"] == "degraded"
    assert result["reason"] == "invalid_output"
    assert result["decision_source"] == "deterministic"
    assert result["requires_user_acceptance"] is True
    assert result["draft"] == DETERMINISTIC_DRAFT


def test_producer_degraded_on_empty_draft(monkeypatch):
    _patch_compiler(monkeypatch, _artifact())
    _patch_complete(monkeypatch, json.dumps({"draft": "   "}))
    result = run_producer_generation(_FakeSession([_settings()]), {}, {})
    assert result["status"] == "degraded"
    assert result["reason"] == "invalid_output"
    assert result["decision_source"] == "deterministic"
    assert result["requires_user_acceptance"] is True
    assert result["draft"] == DETERMINISTIC_DRAFT


# ---------------------------------------------------------------------
# make_producer_fn — vincula la sesión
# ---------------------------------------------------------------------


def test_make_producer_fn_binds_session(monkeypatch):
    _patch_compiler(monkeypatch, _artifact())
    _patch_complete(
        monkeypatch, json.dumps({"draft": "borrador del LLM", "reasoning": "ok"})
    )
    session = _FakeSession([_settings()])
    producer_fn = make_producer_fn(session)
    result = producer_fn({"insight_core": "x"}, {"contexto": "y"}, "feedback")
    assert result["status"] == "ok"
    assert result["draft"] == "borrador del LLM"
    assert result["decision_source"] == "llm"
    assert result["requires_user_acceptance"] is False
