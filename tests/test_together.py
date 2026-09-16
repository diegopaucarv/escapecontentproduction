"""Tests del cliente Together (src/llm/together.py) y del seed LLM.

El cliente se testea con httpx mockeado (sin red). El seed se testea
con una sesión falsa (sin base real).
"""

import uuid

import pytest

from src.db.models import ApiKey, LlmModel, SessionSettings
from src.llm.together import (
    CONTINUATION_PROMPT,
    MAX_CONTINUATION_ROUNDS,
    TOGETHER_CHAT_URL,
    LLMConfig,
    LLMConfigError,
    complete,
    get_active_llm_config,
)


class _FakeSession:
    """Sesión mínima: responde a select(SessionSettings), select(LlmModel)
    y session.get(ApiKey)."""

    def __init__(
        self,
        settings: list[SessionSettings],
        keys: dict[uuid.UUID, ApiKey],
        models: list[LlmModel] | None = None,
    ):
        self._settings = settings
        self._keys = keys
        self._models = models or []

    def execute(self, stmt):
        if "llm_models" in str(stmt):
            rows = self._models
        else:
            rows = self._settings

        class _Result:
            def __init__(self, rows):
                self._rows = rows

            def scalars(self):
                return self

            def first(self):
                return self._rows[0] if self._rows else None

        return _Result(rows)

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


def test_complete_retries_without_thinking_when_content_missing(monkeypatch):
    """Modelo de razonamiento cortado (solo reasoning_content, sin content):
    se reintenta UNA vez con thinking disabled y se devuelve el content."""
    key = _make_key()
    settings = _make_settings(key.id)
    session = _FakeSession([settings], {key.id: key}, models=[_reasoning_model()])

    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(json)

        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                if len(calls) == 1:
                    # Primer intento: razonamiento agotó max_tokens, sin content.
                    return {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "reasoning_content": "pensando...",
                                },
                                "finish_reason": "length",
                            }
                        ]
                    }
                return {
                    "choices": [
                        {"message": {"role": "assistant", "content": "respuesta final"}}
                    ]
                }

        return _Resp()

    monkeypatch.setattr("src.llm.together.httpx.post", fake_post)

    text = complete(session, "Pregunta", model_size="large")
    assert text == "respuesta final"
    assert len(calls) == 2
    # El reintento desactiva el razonamiento.
    assert calls[1]["thinking"] == {"type": "disabled"}


def test_complete_raises_content_error_when_still_no_content(monkeypatch):
    """Si el reintento sin razonamiento tampoco devuelve content →
    LLMContentError (error claro, no KeyError)."""
    from src.llm.together import LLMContentError

    key = _make_key()
    settings = _make_settings(key.id)
    session = _FakeSession([settings], {key.id: key})

    def fake_post(url, headers=None, json=None, timeout=None):
        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "reasoning_content": "pensando...",
                            },
                            "finish_reason": "length",
                        }
                    ]
                }

        return _Resp()

    monkeypatch.setattr("src.llm.together.httpx.post", fake_post)

    with pytest.raises(LLMContentError):
        complete(session, "Pregunta", model_size="large")


def test_complete_catch_and_continue_on_truncated_content(monkeypatch):
    """finish_reason='length' con content parcial → catch-and-continue:
    se re-POST con el fragmento en el historial y se acumula hasta
    completar el JSON (sin descartar el trabajo ya generado)."""
    key = _make_key()
    settings = _make_settings(key.id)
    session = _FakeSession([settings], {key.id: key}, models=[_reasoning_model()])

    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(json)

        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                if len(calls) == 1:
                    # Primer intento: JSON cortado a la mitad.
                    return {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": '{"propositions": [{"id": 1',
                                },
                                "finish_reason": "length",
                            }
                        ]
                    }
                # Continuación: completa el JSON.
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": ',"text": "hola"}]}',
                            },
                            "finish_reason": "stop",
                        }
                    ]
                }

        return _Resp()

    monkeypatch.setattr("src.llm.together.httpx.post", fake_post)

    text = complete(
        session,
        "Pregunta",
        model_size="large",
        response_format={"type": "json_object"},
    )
    # El JSON acumulado parsea y se devuelve completo.
    assert text == '{"propositions": [{"id": 1,"text": "hola"}]}'
    assert len(calls) == 2
    # La continuación lleva el fragmento en el historial (memoria aditiva).
    assert calls[1]["messages"][-2]["role"] == "assistant"
    assert calls[1]["messages"][-2]["content"] == '{"propositions": [{"id": 1'
    assert calls[1]["messages"][-1]["content"] == CONTINUATION_PROMPT


def test_complete_catch_and_continue_falls_back_to_thinking_disabled(monkeypatch):
    """finish_reason='length' con content parcial pero continuaciones que
    nunca producen JSON parseable → escape hatch: reintento con thinking
    desactivado (no se lanza error directo)."""
    key = _make_key()
    settings = _make_settings(key.id)
    session = _FakeSession([settings], {key.id: key}, models=[_reasoning_model()])

    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(json)

        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                if len(calls) == 1:
                    return {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": '{"propositions": [',
                                },
                                "finish_reason": "length",
                            }
                        ]
                    }
                if len(calls) <= 1 + MAX_CONTINUATION_ROUNDS:
                    # Las continuaciones siguen truncando y nunca parsean.
                    return {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": "más texto sin cerrar",
                                },
                                "finish_reason": "length",
                            }
                        ]
                    }
                # Escape hatch: reintento con thinking desactivado → JSON válido.
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": '{"ok": true}',
                            },
                            "finish_reason": "stop",
                        }
                    ]
                }

        return _Resp()

    monkeypatch.setattr("src.llm.together.httpx.post", fake_post)

    text = complete(
        session,
        "Pregunta",
        model_size="large",
        response_format={"type": "json_object"},
    )
    assert text == '{"ok": true}'
    # 1 original + 3 continuaciones + 1 escape hatch = 5 llamadas.
    assert len(calls) == 1 + MAX_CONTINUATION_ROUNDS + 1
    # El escape hatch desactiva el razonamiento.
    assert calls[-1]["thinking"] == {"type": "disabled"}


def test_complete_non_json_natural_stop_retries_thinking_disabled(monkeypatch):
    """finish='stop' con content no-JSON y expects_json → reintento
    thinking-disabled directo (no hay parcial que continuar, no se llama
    al catch-and-continue)."""
    key = _make_key()
    settings = _make_settings(key.id)
    session = _FakeSession([settings], {key.id: key}, models=[_reasoning_model()])

    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(json)

        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                if len(calls) == 1:
                    # Stop natural pero texto no-JSON.
                    return {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": "no soy json",
                                },
                                "finish_reason": "stop",
                            }
                        ]
                    }
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": '{"ok": true}',
                            },
                            "finish_reason": "stop",
                        }
                    ]
                }

        return _Resp()

    monkeypatch.setattr("src.llm.together.httpx.post", fake_post)

    text = complete(
        session,
        "Pregunta",
        model_size="large",
        response_format={"type": "json_object"},
    )
    assert text == '{"ok": true}'
    assert len(calls) == 2
    assert calls[1]["thinking"] == {"type": "disabled"}
    # No hubo continuaciones: el historial del reintento no lleva el parcial.
    assert calls[1]["messages"][-1]["content"] != CONTINUATION_PROMPT


def test_complete_returns_content_normally_without_retry(monkeypatch):
    """Respuesta normal con content → UNA sola llamada, sin reintento."""
    key = _make_key()
    settings = _make_settings(key.id)
    session = _FakeSession([settings], {key.id: key})

    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(json)

        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "choices": [{"message": {"role": "assistant", "content": "ok"}}]
                }

        return _Resp()

    monkeypatch.setattr("src.llm.together.httpx.post", fake_post)

    text = complete(session, "Pregunta", model_size="large")
    assert text == "ok"
    assert len(calls) == 1


def _reasoning_model() -> LlmModel:
    """Modelo razonador con syntax_profile (como el DeepSeek V4 Flash)."""
    return LlmModel(
        model_name="deepseek-ai/DeepSeek-V4-Flash-0731",
        provider="together",
        model_size="large",
        syntax_profile={
            "reasoning_mode": {
                "supports_reasoning": True,
                "thinking_parameter": "thinking",
                "thinking_config": {"type": "enabled", "budget_tokens": 2048},
                "min_tokens_for_reasoning": 1024,
                "min_content_reserve": 512,
            }
        },
    )


def test_complete_thinking_false_omits_thinking_param(monkeypatch):
    """thinking=False → el body incluye thinking={"type": "disabled"}.

    Antes devolvía (None, None) = "no enviar el parámetro", lo que dejaba
    el default del modelo (razonamiento ACTIVADO) y los tokens de thinking
    se comían el presupuesto de max_tokens → JSON truncado. Ahora apaga el
    razonamiento explícitamente.
    """
    key = _make_key()
    settings = _make_settings(key.id)
    session = _FakeSession([settings], {key.id: key}, models=[_reasoning_model()])

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

    complete(session, "Pregunta", model_size="large", thinking=False)
    assert captured["json"]["thinking"] == {"type": "disabled"}


def test_complete_thinking_none_uses_model_default(monkeypatch):
    """thinking=None con modelo razonador → el body incluye la config de
    razonamiento del modelo (default del syntax_profile)."""
    key = _make_key()
    settings = _make_settings(key.id)
    session = _FakeSession([settings], {key.id: key}, models=[_reasoning_model()])

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

    complete(session, "Pregunta", model_size="large")
    assert captured["json"]["thinking"] == {"type": "enabled", "budget_tokens": 2048}


def test_complete_guardrail_disables_thinking_when_max_tokens_low(monkeypatch):
    """max_tokens bajo (< piso del modelo) → el razonamiento se desactiva
    automáticamente en el primer POST (sin reintento)."""
    key = _make_key()
    settings = _make_settings(key.id)
    session = _FakeSession([settings], {key.id: key}, models=[_reasoning_model()])

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

    complete(session, "Pregunta", model_size="large", max_tokens=100)
    assert captured["json"]["thinking"] == {"type": "disabled"}


def test_complete_thinking_enabled_when_max_tokens_high(monkeypatch):
    """max_tokens alto → el razonamiento se mantiene activo (default del
    modelo), sin degradar."""
    key = _make_key()
    settings = _make_settings(key.id)
    session = _FakeSession([settings], {key.id: key}, models=[_reasoning_model()])

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

    complete(session, "Pregunta", model_size="large", max_tokens=8192)
    assert captured["json"]["thinking"] == {"type": "enabled", "budget_tokens": 2048}


def test_call_with_retries_json_defaults_to_model_thinking(monkeypatch):
    """call_with_retries con response_format=json_object → thinking=None →
    default del modelo (razonamiento ACTIVO). Quien no quiera razonar debe
    pasar thinking=False explícitamente."""
    from src.llm.base import call_with_retries

    key = _make_key()
    settings = _make_settings(key.id)
    session = _FakeSession([settings], {key.id: key}, models=[_reasoning_model()])

    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["json"] = json

        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": [{"message": {"content": '{"ok": true}'}}]}

        return _Resp()

    monkeypatch.setattr("src.llm.together.httpx.post", fake_post)

    text, model, used_fallback = call_with_retries(
        session,
        prompt="p",
        system="s",
        model_size="large",
        response_format={"type": "json_object"},
        retries=1,
    )
    assert text == '{"ok": true}'
    assert model == "large"
    assert used_fallback is False
    assert captured["json"]["thinking"] == {"type": "enabled", "budget_tokens": 2048}


def test_complete_continues_truncated_json_content(monkeypatch):
    """finish_reason='length' con content parcial → catch-and-continue: se
    re-POST con el fragmento en el historial y se devuelve el JSON acumulado
    (sin descartar el parcial ni reintentar desde cero)."""
    key = _make_key()
    settings = _make_settings(key.id)
    session = _FakeSession([settings], {key.id: key}, models=[_reasoning_model()])

    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(json)

        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                if len(calls) == 1:
                    # Primer intento: JSON cortado por max_tokens.
                    return {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": '{"a": 1, "b": ',
                                },
                                "finish_reason": "length",
                            }
                        ]
                    }
                # Continuación: completa el JSON desde donde se cortó.
                return {
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "2}"},
                            "finish_reason": "stop",
                        }
                    ]
                }

        return _Resp()

    monkeypatch.setattr("src.llm.together.httpx.post", fake_post)

    text = complete(
        session,
        "Pregunta",
        model_size="large",
        response_format={"type": "json_object"},
    )
    # El parcial se pasa a la continuación ya .strip()-eado (sin el espacio
    # final), así que el merge es '{"a": 1, "b":' + '2}' — JSON válido.
    assert text == '{"a": 1, "b":2}'
    assert len(calls) == 2
    # La continuación lleva el fragmento parcial en el historial (memoria
    # aditiva) + el prompt de continuación. El fragmento va .strip()-eado,
    # igual que el que se pasó a _continue_truncated_content.
    assert calls[1]["messages"][-2:] == [
        {"role": "assistant", "content": '{"a": 1, "b":'},
        {"role": "user", "content": CONTINUATION_PROMPT},
    ]


def test_complete_falls_back_to_thinking_disabled_when_continuations_fail(monkeypatch):
    """finish_reason='length' con parcial y continuaciones que nunca parsean
    → escape hatch: reintento thinking-disabled (original + continuación
    fallida + reintento disabled)."""
    key = _make_key()
    settings = _make_settings(key.id)
    session = _FakeSession([settings], {key.id: key}, models=[_reasoning_model()])

    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(json)

        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                if len(calls) == 1:
                    # Primer intento: JSON cortado por max_tokens.
                    return {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": '{"a": ',
                                },
                                "finish_reason": "length",
                            }
                        ]
                    }
                if len(calls) == 2:
                    # Continuación: termina naturalmente pero no parsea.
                    return {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": "esto no es json",
                                },
                                "finish_reason": "stop",
                            }
                        ]
                    }
                # Escape hatch: reintento thinking-disabled → JSON válido.
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": '{"ok": true}',
                            },
                            "finish_reason": "stop",
                        }
                    ]
                }

        return _Resp()

    monkeypatch.setattr("src.llm.together.httpx.post", fake_post)

    text = complete(
        session,
        "Pregunta",
        model_size="large",
        response_format={"type": "json_object"},
    )
    assert text == '{"ok": true}'
    assert len(calls) == 3
    # El reintento (3ª llamada) desactiva el razonamiento.
    assert calls[2]["thinking"] == {"type": "disabled"}


def test_complete_retries_thinking_disabled_when_json_not_parseable(monkeypatch):
    """finish_reason='stop' con content no-JSON y expects_json → reintento
    thinking-disabled directo, SIN catch-and-continue (no hay parcial que
    continuar)."""
    key = _make_key()
    settings = _make_settings(key.id)
    session = _FakeSession([settings], {key.id: key}, models=[_reasoning_model()])

    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(json)

        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                if len(calls) == 1:
                    return {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": "esto no es json",
                                },
                                "finish_reason": "stop",
                            }
                        ]
                    }
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": '{"ok": true}',
                            },
                            "finish_reason": "stop",
                        }
                    ]
                }

        return _Resp()

    monkeypatch.setattr("src.llm.together.httpx.post", fake_post)

    text = complete(
        session,
        "Pregunta",
        model_size="large",
        response_format={"type": "json_object"},
    )
    assert text == '{"ok": true}'
    assert len(calls) == 2
    # El reintento desactiva el razonamiento y NO lleva el prompt de
    # continuación (no hubo catch-and-continue).
    assert calls[1]["thinking"] == {"type": "disabled"}
    assert calls[1]["messages"] == [{"role": "user", "content": "Pregunta"}]
