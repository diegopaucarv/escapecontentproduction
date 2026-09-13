"""Tests del refinador de novedad (src/llm/novelty_refinement.py) — sin red."""

import json
import uuid
from types import SimpleNamespace

from src.llm.novelty_refinement import refine_angle_novelty


class _FakeSession:
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


def test_refine_skipped_without_settings():
    result = refine_angle_novelty(_FakeSession([]), {}, [])
    assert result["status"] == "skipped"
    assert result["reason"] == "no_settings"


def test_refine_skipped_without_artifact(monkeypatch):
    _patch_compiler(monkeypatch, None)
    result = refine_angle_novelty(_FakeSession([_settings()]), {}, [])
    assert result["status"] == "skipped"
    assert result["reason"] == "not_compiled"


def test_refine_ok_angle_new(monkeypatch):
    _patch_compiler(monkeypatch, SimpleNamespace(prompt_text="Eres el refinador."))
    _patch_complete(
        monkeypatch,
        json.dumps({"angulo_nuevo": True, "reasoning": "perspectiva no vista"}),
    )
    result = refine_angle_novelty(_FakeSession([_settings()]), {}, [])
    assert result["status"] == "ok"
    assert result["angulo_nuevo"] is True
    assert result["reasoning"] == "perspectiva no vista"


def test_refine_ok_angle_covered(monkeypatch):
    _patch_compiler(monkeypatch, SimpleNamespace(prompt_text="Eres el refinador."))
    _patch_complete(
        monkeypatch,
        json.dumps({"angulo_nuevo": False, "reasoning": "ya cubierto"}),
    )
    result = refine_angle_novelty(_FakeSession([_settings()]), {}, [])
    assert result["status"] == "ok"
    assert result["angulo_nuevo"] is False


def test_refine_degraded_when_llm_unavailable(monkeypatch):
    _patch_compiler(monkeypatch, SimpleNamespace(prompt_text="Eres el refinador."))

    import src.llm.together as together_mod

    def boom(
        session, prompt, model_size="small", system=None, response_format=None, **kwargs
    ):
        raise RuntimeError("red caída")

    monkeypatch.setattr(together_mod, "complete", boom)
    result = refine_angle_novelty(_FakeSession([_settings()]), {}, [])
    assert result["status"] == "degraded"
    assert result["angulo_nuevo"] is True  # default determinista


def test_refine_degraded_on_invalid_output(monkeypatch):
    _patch_compiler(monkeypatch, SimpleNamespace(prompt_text="Eres el refinador."))
    _patch_complete(monkeypatch, "no es json")
    result = refine_angle_novelty(_FakeSession([_settings()]), {}, [])
    assert result["status"] == "degraded"
    assert result["angulo_nuevo"] is True
