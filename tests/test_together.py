"""Tests del cliente Together (src/llm/together.py) y del seed LLM.

El cliente se testea con httpx mockeado (sin red). El seed se testea
con una sesión falsa (sin base real).
"""

import uuid

import pytest

from src.db.models import ApiKey, SessionSettings
from src.llm.together import (
    TOGETHER_CHAT_URL,
    LLMConfig,
    LLMConfigError,
    complete,
    get_active_llm_config,
)


class _FakeSession:
    """Sesión mínima: responde a select(SessionSettings) y session.get(ApiKey)."""

    def __init__(self, settings: list[SessionSettings], keys: dict[uuid.UUID, ApiKey]):
        self._settings = settings
        self._keys = keys

    def execute(self, stmt):
        class _Result:
            def __init__(self, rows):
                self._rows = rows

            def scalars(self):
                return self

            def first(self):
                return self._rows[0] if self._rows else None

        return _Result(self._settings)

    def get(self, model, ident):
        if model is ApiKey:
            return self._keys.get(ident)
        return None


def _make_key(**overrides) -> ApiKey:
    defaults = dict(
        id=uuid.uuid4(),
        provider="together",
        key_name="together-main",
        api_key="tgp_v1_test_key_1234567890",
        is_active=True,
    )
    defaults.update(overrides)
    return ApiKey(**defaults)


def _make_settings(key_id: uuid.UUID, **overrides) -> SessionSettings:
    defaults = dict(
        id=uuid.uuid4(),
        api_key_id=key_id,
        small_model="meta-models/Muse-Glimmer-30B",
        large_model="deepseek-ai/DeepSeek-V4-Flash-0731",
        temperature_small=0.7,
        temperature_large=0.7,
        max_tokens_small=2048,
        max_tokens_large=4096,
        is_active=True,
    )
    defaults.update(overrides)
    return SessionSettings(**defaults)


def test_get_active_llm_config_returns_models_and_key():
    key = _make_key()
    settings = _make_settings(key.id)
    session = _FakeSession([settings], {key.id: key})

    cfg = get_active_llm_config(session)
    assert isinstance(cfg, LLMConfig)
    assert cfg.api_key == key.api_key
    assert cfg.small_model == "meta-models/Muse-Glimmer-30B"
    assert cfg.large_model == "deepseek-ai/DeepSeek-V4-Flash-0731"


def test_get_active_llm_config_raises_when_no_settings():
    session = _FakeSession([], {})
    with pytest.raises(LLMConfigError):
        get_active_llm_config(session)


def test_get_active_llm_config_raises_when_key_missing():
    key = _make_key()
    settings = _make_settings(uuid.uuid4())  # referencia a una key inexistente
    session = _FakeSession([settings], {})
    with pytest.raises(LLMConfigError):
        get_active_llm_config(session)


def test_complete_calls_together_with_small_model(monkeypatch):
    key = _make_key()
    settings = _make_settings(key.id)
    session = _FakeSession([settings], {key.id: key})

    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout

        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": [{"message": {"content": "respuesta del modelo"}}]}

        return _Resp()

    monkeypatch.setattr("src.llm.together.httpx.post", fake_post)

    text = complete(session, "Hola", model_size="small")
    assert text == "respuesta del modelo"
    assert captured["url"] == TOGETHER_CHAT_URL
    assert captured["headers"]["Authorization"] == f"Bearer {key.api_key}"
    assert captured["json"]["model"] == "meta-models/Muse-Glimmer-30B"
    assert captured["json"]["messages"] == [{"role": "user", "content": "Hola"}]
    assert captured["json"]["temperature"] == 0.7
    assert captured["json"]["max_tokens"] == 2048


def test_complete_uses_large_model_and_system_prompt(monkeypatch):
    key = _make_key()
    settings = _make_settings(key.id)
    session = _FakeSession([settings], {key.id: key})

    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["json"] = json

        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": [{"message": {"content": "ok"}}]}

        return _Resp()

    monkeypatch.setattr("src.llm.together.httpx.post", fake_post)

    complete(session, "Pregunta", model_size="large", system="Sé breve.")
    assert captured["json"]["model"] == "deepseek-ai/DeepSeek-V4-Flash-0731"
    assert captured["json"]["messages"] == [
        {"role": "system", "content": "Sé breve."},
        {"role": "user", "content": "Pregunta"},
    ]
    assert captured["json"]["max_tokens"] == 4096


def test_complete_rejects_invalid_model_size():
    key = _make_key()
    settings = _make_settings(key.id)
    session = _FakeSession([settings], {key.id: key})
    with pytest.raises(ValueError):
        complete(session, "Hola", model_size="medium")
