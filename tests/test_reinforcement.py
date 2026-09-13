"""Tests del refuerzo LLM del semáforo (src/llm/reinforcement.py) — sin red."""

import json
import uuid
from types import SimpleNamespace

import pytest

from src.llm.reinforcement import _fuse, reinforce_alignment


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


def _artifact(prompt_text="Eres el refuerzo del semaforo."):
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
# Fusión conservadora (núcleo)
# ---------------------------------------------------------------------


def test_fuse_fail_stays_fail():
    assert _fuse("fail", "auto_pass") == "fail"
    assert _fuse("fail", "needs_human_review") == "fail"
    assert _fuse("fail", "fail") == "fail"
    assert _fuse("fail", None) == "fail"


def test_fuse_needs_human_review_stays():
    assert _fuse("needs_human_review", "auto_pass") == "needs_human_review"
    assert _fuse("needs_human_review", None) == "needs_human_review"


def test_fuse_auto_pass_escalates_on_llm_risk():
    assert _fuse("auto_pass", "fail") == "needs_human_review"
    assert _fuse("auto_pass", "needs_human_review") == "needs_human_review"


def test_fuse_auto_pass_stays_when_llm_ok():
    assert _fuse("auto_pass", "auto_pass") == "auto_pass"
    assert _fuse("auto_pass", None) == "auto_pass"


# ---------------------------------------------------------------------
# reinforce_alignment — flujo completo
# ---------------------------------------------------------------------


def test_reinforce_skipped_without_settings():
    session = _FakeSession([])
    result = reinforce_alignment(session, {}, "auto_pass", [])
    assert result["status"] == "skipped"
    assert result["reason"] == "no_settings"
    # 0007: sin LLM la decisión es determinista y requiere aceptación.
    assert result["decision_source"] == "deterministic"
    assert result["requires_user_acceptance"] is True


def test_reinforce_skipped_without_artifact(monkeypatch):
    session = _FakeSession([_settings()])
    _patch_compiler(monkeypatch, None)
    result = reinforce_alignment(session, {}, "auto_pass", [])
    assert result["status"] == "skipped"
    assert result["reason"] == "not_compiled"
    # 0007: sin artefacto compilado la decisión es determinista.
    assert result["decision_source"] == "deterministic"
    assert result["requires_user_acceptance"] is True


def test_reinforce_ok_auto_pass(monkeypatch):
    session = _FakeSession([_settings()])
    _patch_compiler(monkeypatch, _artifact())
    _patch_complete(
        monkeypatch,
        json.dumps(
            {"verdict": "auto_pass", "reasoning": "sin riesgos", "risks_detected": []}
        ),
    )
    result = reinforce_alignment(session, {"resumen": "x"}, "auto_pass", [])
    assert result["status"] == "ok"
    assert result["verdict"] == "auto_pass"
    assert result["llm_verdict"] == "auto_pass"
    assert result["model_used"] == "small"
    # 0007: decisión del LLM — revisable, pero sin aceptación obligatoria.
    assert result["decision_source"] == "llm"
    assert result["requires_user_acceptance"] is False


def test_reinforce_escalates_auto_pass_to_human(monkeypatch):
    session = _FakeSession([_settings()])
    _patch_compiler(monkeypatch, _artifact())
    _patch_complete(
        monkeypatch,
        json.dumps(
            {
                "verdict": "needs_human_review",
                "reasoning": "riesgo reputacional no cubierto",
                "risks_detected": ["marca mencionada sin contexto"],
            }
        ),
    )
    result = reinforce_alignment(session, {}, "auto_pass", [])
    assert result["status"] == "ok"
    assert result["verdict"] == "needs_human_review"
    assert result["risks_detected"] == ["marca mencionada sin contexto"]
    assert result["decision_source"] == "llm"
    assert result["requires_user_acceptance"] is False


def test_reinforce_fail_stays_fail(monkeypatch):
    session = _FakeSession([_settings()])
    _patch_compiler(monkeypatch, _artifact())
    _patch_complete(
        monkeypatch, json.dumps({"verdict": "auto_pass", "reasoning": "ok"})
    )
    result = reinforce_alignment(session, {}, "fail", [])
    assert result["verdict"] == "fail"
    assert result["decision_source"] == "llm"
    assert result["requires_user_acceptance"] is False


def test_reinforce_invalid_llm_verdict_is_conservative(monkeypatch):
    session = _FakeSession([_settings()])
    _patch_compiler(monkeypatch, _artifact())
    _patch_complete(monkeypatch, "no es json")
    result = reinforce_alignment(session, {}, "auto_pass", [])
    assert result["status"] == "ok"
    assert result["llm_verdict"] is None
    assert result["verdict"] == "auto_pass"  # conservador: no escala sin señal
    assert result["decision_source"] == "llm"
    assert result["requires_user_acceptance"] is False


def test_reinforce_degraded_when_llm_unavailable(monkeypatch):
    session = _FakeSession([_settings()])
    _patch_compiler(monkeypatch, _artifact())

    import src.llm.together as together_mod

    def boom(
        session, prompt, model_size="small", system=None, response_format=None, **kwargs
    ):
        raise RuntimeError("red caída")

    monkeypatch.setattr(together_mod, "complete", boom)
    result = reinforce_alignment(session, {}, "auto_pass", [])
    assert result["status"] == "degraded"
    assert result["reason"] == "llm_unavailable"
    assert result["verdict"] == "auto_pass"  # solo reglas
    # 0007: degradación a determinista SIEMPRE requiere aceptación.
    assert result["decision_source"] == "deterministic"
    assert result["requires_user_acceptance"] is True


def test_reinforce_uses_fallback_after_retries(monkeypatch):
    session = _FakeSession(
        [_settings(fallback_model="deepseek-ai/DeepSeek-V4-Flash-0731")]
    )
    _patch_compiler(monkeypatch, _artifact())

    import src.llm.together as together_mod

    calls = []

    def flaky(
        session, prompt, model_size="small", system=None, response_format=None, **kwargs
    ):
        calls.append(model_size)
        if model_size == "small":
            raise RuntimeError("pequeño caído")
        return json.dumps({"verdict": "auto_pass", "reasoning": "ok"})

    monkeypatch.setattr(together_mod, "complete", flaky)
    result = reinforce_alignment(session, {}, "auto_pass", [])
    assert result["status"] == "ok"
    assert result["fallback_used"] is True
    assert result["model_used"] == "deepseek-ai/DeepSeek-V4-Flash-0731"
    # 0009: los reintentos del primario los hace tenacity DENTRO de
    # complete() (src/llm/together.py::_post_with_retry); call_with_retries
    # ya NO repite el bucle externo (antes: retries × retries = 9 llamadas).
    assert calls.count("small") == 1
    assert calls.count("deepseek-ai/DeepSeek-V4-Flash-0731") == 1
    assert result["decision_source"] == "llm"
    assert result["requires_user_acceptance"] is False
