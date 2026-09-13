"""Tests de la infraestructura compartida de decisiones (src/llm/base.py) — sin red."""

import uuid
from types import SimpleNamespace

from src.llm.base import (
    degraded_result,
    load_settings,
    ok_result,
    parse_llm_output,
    skipped_result,
)

# ---------------------------------------------------------------------
# Contrato de resultado de decisión (0007)
# ---------------------------------------------------------------------


def test_ok_result_contract():
    """ok -> decisión del LLM, revisable pero sin aceptación obligatoria."""
    result = ok_result(verdict="auto_pass", reasoning="sin riesgos")
    assert result["status"] == "ok"
    assert result["decision_source"] == "llm"
    assert result["requires_user_acceptance"] is False
    # Campos extra fusionados.
    assert result["verdict"] == "auto_pass"
    assert result["reasoning"] == "sin riesgos"


def test_degraded_result_contract():
    """degraded -> determinista por construcción, SIEMPRE aceptación."""
    result = degraded_result("llm_unavailable", verdict="auto_pass")
    assert result["status"] == "degraded"
    assert result["decision_source"] == "deterministic"
    assert result["requires_user_acceptance"] is True
    assert result["reason"] == "llm_unavailable"
    # Campos extra fusionados.
    assert result["verdict"] == "auto_pass"


def test_skipped_result_contract():
    """skipped -> determinista por construcción, SIEMPRE aceptación."""
    result = skipped_result("no_settings")
    assert result["status"] == "skipped"
    assert result["decision_source"] == "deterministic"
    assert result["requires_user_acceptance"] is True
    assert result["reason"] == "no_settings"


# ---------------------------------------------------------------------
# parse_llm_output
# ---------------------------------------------------------------------


def test_parse_llm_output_valid_dict():
    assert parse_llm_output('{"verdict": "auto_pass"}') == {"verdict": "auto_pass"}


def test_parse_llm_output_invalid_returns_empty():
    assert parse_llm_output("no es json") == {}


def test_parse_llm_output_non_dict_json_returns_empty():
    assert parse_llm_output("[1, 2, 3]") == {}
    assert parse_llm_output('"texto"') == {}


# ---------------------------------------------------------------------
# load_settings
# ---------------------------------------------------------------------


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


def test_load_settings_returns_active_settings():
    settings = _settings()
    assert load_settings(_FakeSession([settings])) is settings


def test_load_settings_returns_none_without_settings():
    assert load_settings(_FakeSession([])) is None
