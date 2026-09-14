"""Tests del cliente de embeddings (src/embeddings.py) — sin red ni modelo real.

Nota: NO se importa torch/transformers reales a nivel de módulo (importar
torch tarda ~30s en esta máquina). El test de _get_model inyecta módulos
falsos en sys.modules para que la carga perezosa nunca toque los reales.
"""

import sys
import threading
import uuid
from types import SimpleNamespace

import numpy as np
import pytest

from src.embeddings import (
    _active_embedding_config,
    _get_model,
    _prompt_name,
    embed_text,
    embed_texts,
)


class _FakeSession:
    """Sesión mínima: responde a select(EmbeddingSetting) y session.get()."""

    def __init__(self, setting=None, key=None, model=None):
        self._setting = setting
        self._key = key
        self._model = model

    def execute(self, stmt):
        class _Result:
            def __init__(self, rows):
                self._rows = rows

            def scalars(self):
                return self

            def first(self):
                return self._rows[0] if self._rows else None

        return _Result([self._setting] if self._setting else [])

    def get(self, model, ident):
        name = getattr(model, "__name__", "")
        if name == "ApiKey":
            return self._key
        if name == "LlmModel":
            return self._model
        return None


def _setting(**overrides):
    defaults = dict(
        id=uuid.uuid4(),
        api_key_id=uuid.uuid4(),
        llm_model_id=uuid.uuid4(),
        dimension=768,
        is_active=True,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _key(**overrides):
    defaults = dict(id=uuid.uuid4(), api_key="hf_xxx", is_active=True)
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _model(**overrides):
    defaults = dict(
        id=uuid.uuid4(),
        model_name="jinaai/jina-embeddings-v5-text-nano",
        is_active=True,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------
# _prompt_name — mapeo de input_type a prompt_name
# ---------------------------------------------------------------------


def test_prompt_name_maps_query_and_document():
    assert _prompt_name("query") == "query"
    assert _prompt_name("document") == "document"
    assert _prompt_name("otro") == "document"  # default conservador


# ---------------------------------------------------------------------
# _active_embedding_config
# ---------------------------------------------------------------------


def test_config_reads_key_model_dimension():
    session = _FakeSession(_setting(), _key(), _model())
    api_key, model_name, dimension = _active_embedding_config(session)
    assert api_key == "hf_xxx"
    assert model_name == "jinaai/jina-embeddings-v5-text-nano"
    assert dimension == 768


def test_config_raises_without_setting():
    with pytest.raises(RuntimeError, match="embedding_settings"):
        _active_embedding_config(_FakeSession(None, _key(), _model()))


def test_config_raises_with_inactive_key():
    session = _FakeSession(_setting(), _key(is_active=False), _model())
    with pytest.raises(RuntimeError, match="api_key"):
        _active_embedding_config(session)


def test_config_raises_with_inactive_model():
    session = _FakeSession(_setting(), _key(), _model(is_active=False))
    with pytest.raises(RuntimeError, match="llm_model"):
        _active_embedding_config(session)


# ---------------------------------------------------------------------
# _get_model — carga perezosa, una sola vez, con token y trust_remote_code
# ---------------------------------------------------------------------


def test_get_model_loads_once(monkeypatch):
    import src.embeddings as embeddings_mod

    calls = []

    class _FakeModel:
        def encode(self, **kw):
            return [[0.1]]

        def to(self, device):
            return self

    class _FakeAutoModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.append(kwargs)
            return _FakeModel()

    class _FakeTorch:
        float32 = "float32"
        bfloat16 = "bfloat16"

        class cuda:
            @staticmethod
            def is_available():
                return False

            @staticmethod
            def is_bf16_supported():
                return False

        @staticmethod
        def device(s):
            return SimpleNamespace(type="cpu")

    class _FakeTransformers:
        AutoModel = _FakeAutoModel

    monkeypatch.setattr(embeddings_mod, "_model", None)
    monkeypatch.setattr(embeddings_mod, "_model_lock", threading.Lock())
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch)
    monkeypatch.setitem(sys.modules, "transformers", _FakeTransformers)

    m1 = _get_model("hf_xxx", "jinaai/jina-embeddings-v5-text-nano")
    m2 = _get_model("hf_xxx", "jinaai/jina-embeddings-v5-text-nano")

    assert m1 is m2  # singleton: no recarga
    assert len(calls) == 1
    assert calls[0]["trust_remote_code"] is True
    assert calls[0]["token"] == "hf_xxx"
    assert calls[0]["dtype"] == "float32"  # CPU -> float32


# ---------------------------------------------------------------------
# embed_text — sin modelo real (mockeando _get_model)
# ---------------------------------------------------------------------


def test_embed_text_uses_local_model(monkeypatch):
    import src.embeddings as embeddings_mod

    captured = {}

    class _FakeModel:
        def encode(self, texts=None, task=None, prompt_name=None):
            captured["texts"] = texts
            captured["task"] = task
            captured["prompt_name"] = prompt_name
            return [[0.1, 0.2, 0.3]]

    monkeypatch.setattr(
        embeddings_mod, "_get_model", lambda api_key, model_name: _FakeModel()
    )
    monkeypatch.setattr(
        embeddings_mod,
        "_active_embedding_config",
        lambda session=None: ("hf_xxx", "jinaai/jina-embeddings-v5-text-nano", 768),
    )

    result = embed_text("texto de prueba", input_type="query")
    assert result == [0.1, 0.2, 0.3]
    assert captured["texts"] == ["texto de prueba"]
    assert captured["task"] == "retrieval"
    assert captured["prompt_name"] == "query"


def test_embed_text_document_default(monkeypatch):
    import src.embeddings as embeddings_mod

    captured = {}

    class _FakeModel:
        def encode(self, texts=None, task=None, prompt_name=None):
            captured["prompt_name"] = prompt_name
            return [[0.1, 0.2, 0.3]]

    monkeypatch.setattr(
        embeddings_mod, "_get_model", lambda api_key, model_name: _FakeModel()
    )
    monkeypatch.setattr(
        embeddings_mod,
        "_active_embedding_config",
        lambda session=None: ("hf_xxx", "jinaai/jina-embeddings-v5-text-nano", 768),
    )

    result = embed_text("texto de prueba")  # default document
    assert result == [0.1, 0.2, 0.3]
    assert captured["prompt_name"] == "document"


def test_embed_text_handles_tensor(monkeypatch):
    """Si .encode() devuelve un torch.Tensor, se aplana a list[float]."""

    class _FakeTensor:
        def detach(self):
            return self

        def cpu(self):
            return self

        def float(self):
            return self

        def numpy(self):
            return np.array([0.1, 0.2, 0.3])

    class _FakeModel:
        def encode(self, texts=None, task=None, prompt_name=None):
            return [_FakeTensor()]

    import src.embeddings as embeddings_mod

    monkeypatch.setattr(
        embeddings_mod, "_get_model", lambda api_key, model_name: _FakeModel()
    )
    monkeypatch.setattr(
        embeddings_mod,
        "_active_embedding_config",
        lambda session=None: ("hf_xxx", "jinaai/jina-embeddings-v5-text-nano", 768),
    )

    result = embed_text("texto", input_type="document")
    assert result == [0.1, 0.2, 0.3]


# ---------------------------------------------------------------------
# embed_texts — versión batch (una sola llamada a model.encode para N textos)
# ---------------------------------------------------------------------


def test_embed_texts_batch(monkeypatch):
    import src.embeddings as embeddings_mod

    captured = {}

    class _FakeModel:
        def encode(self, texts=None, task=None, prompt_name=None):
            captured["texts"] = texts
            captured["task"] = task
            captured["prompt_name"] = prompt_name
            return [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

    monkeypatch.setattr(
        embeddings_mod, "_get_model", lambda api_key, model_name: _FakeModel()
    )
    monkeypatch.setattr(
        embeddings_mod,
        "_active_embedding_config",
        lambda session=None: ("hf_xxx", "jinaai/jina-embeddings-v5-text-nano", 768),
    )

    result = embed_texts(["a", "b", "c"], input_type="document")
    assert result == [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    assert captured["texts"] == ["a", "b", "c"]
    assert captured["task"] == "retrieval"
    assert captured["prompt_name"] == "document"


def test_embed_texts_empty_returns_empty(monkeypatch):
    import src.embeddings as embeddings_mod

    def fake_model(api_key, model_name):
        raise AssertionError("no debe cargar el modelo con lista vacía")

    monkeypatch.setattr(embeddings_mod, "_get_model", fake_model)
    assert embed_texts([]) == []


def test_embed_texts_handles_tensors(monkeypatch):
    """Si .encode() devuelve tensores, se aplanan a list[float] (bfloat16→f32)."""

    class _FakeTensor:
        def detach(self):
            return self

        def cpu(self):
            return self

        def float(self):
            return self

        def numpy(self):
            return np.array([0.1, 0.2])

    class _FakeModel:
        def encode(self, texts=None, task=None, prompt_name=None):
            return [_FakeTensor(), _FakeTensor()]

    import src.embeddings as embeddings_mod

    monkeypatch.setattr(
        embeddings_mod, "_get_model", lambda api_key, model_name: _FakeModel()
    )
    monkeypatch.setattr(
        embeddings_mod,
        "_active_embedding_config",
        lambda session=None: ("hf_xxx", "jinaai/jina-embeddings-v5-text-nano", 768),
    )

    result = embed_texts(["a", "b"])
    assert result == [[0.1, 0.2], [0.1, 0.2]]
