"""Tests del CRUD de infraestructura de IA modular (0004) — sin base real."""

import uuid
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.api.main import app, get_session


class _FakeSession:
    """Sesión mínima para los endpoints de llm-models/prompt-templates/etc."""

    def __init__(self):
        self.models = {}  # id -> obj
        self.templates = {}  # id -> obj
        self.artifacts = {}  # id -> obj
        self.embedding = []  # list
        self.keys = {}
        self._deleted = []

    def get(self, model, ident):
        name = getattr(model, "__name__", "")
        if name == "LlmModel":
            return self.models.get(ident)
        if name == "PromptTemplate":
            return self.templates.get(ident)
        if name == "PromptArtifact":
            return self.artifacts.get(ident)
        if name == "EmbeddingSetting":
            return next((s for s in self.embedding if s.id == ident), None)
        if name == "ApiKey":
            return self.keys.get(ident)
        return None

    def execute(self, stmt):
        class _Result:
            def __init__(self, rows):
                self._rows = rows

            def scalars(self):
                return self

            def all(self):
                return self._rows

            def first(self):
                return self._rows[0] if self._rows else None

            def scalar_one_or_none(self):
                return self._rows[0] if self._rows else None

        # update(EmbeddingSetting).where(is_active).values(is_active=False)
        if hasattr(stmt, "_values") and hasattr(stmt, "_where_criteria"):
            for s in self.embedding:
                if s.is_active:
                    for col, val in stmt._values.items():
                        if getattr(col, "name", None) == "is_active":
                            s.is_active = getattr(val, "value", val)
            return _Result([])

        # select(LlmModel) / select(PromptTemplate) / select(PromptArtifact)
        # / select(EmbeddingSetting) con where por clave única o is_active.
        if hasattr(stmt, "_where_criteria") and stmt._where_criteria:
            col = stmt._where_criteria[0].left
            if col.name == "model_name":
                return _Result(
                    [
                        m
                        for m in self.models.values()
                        if m.model_name == stmt._where_criteria[0].right.value
                    ]
                )
            if col.name == "task_key":
                return _Result(
                    [
                        t
                        for t in self.templates.values()
                        if t.task_key == stmt._where_criteria[0].right.value
                    ]
                )
            if col.name == "is_active":
                return _Result([s for s in self.embedding if s.is_active])
        # select sin where: lista de la clase (el raw_column es el nombre de tabla)
        if hasattr(stmt, "_raw_columns") and stmt._raw_columns:
            table = getattr(stmt._raw_columns[0], "name", "")
            if table == "llm_models":
                return _Result(list(self.models.values()))
            if table == "prompt_templates":
                return _Result(list(self.templates.values()))
            if table == "prompt_artifacts":
                return _Result(list(self.artifacts.values()))
            if table == "embedding_settings":
                return _Result(list(self.embedding))
        return _Result([])

    def add(self, obj):
        name = obj.__class__.__name__
        if getattr(obj, "id", None) is None:
            obj.id = uuid.uuid4()
        if name == "LlmModel":
            self.models[obj.id] = obj
        elif name == "PromptTemplate":
            self.templates[obj.id] = obj
        elif name == "PromptArtifact":
            self.artifacts[obj.id] = obj
        elif name == "EmbeddingSetting":
            self.embedding.append(obj)

    def delete(self, obj):
        name = obj.__class__.__name__
        if name == "LlmModel":
            self.models.pop(obj.id, None)
        elif name == "PromptTemplate":
            self.templates.pop(obj.id, None)
        elif name == "PromptArtifact":
            self.artifacts.pop(obj.id, None)
        elif name == "EmbeddingSetting":
            self.embedding = [s for s in self.embedding if s.id != obj.id]

    def commit(self):
        pass

    def refresh(self, obj):
        pass

    def close(self):
        pass


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def fake():
    return _FakeSession()


@contextmanager
def _use(client, fake):
    app.dependency_overrides[get_session] = lambda: fake
    try:
        yield
    finally:
        app.dependency_overrides.clear()


def _model(**overrides):
    defaults = dict(
        id=uuid.uuid4(),
        model_name="meta-models/Muse-Glimmer-30B",
        provider="together",
        model_size="small",
        context_window=32768,
        max_output_tokens=8192,
        temperature_default=0.7,
        strengths=["rapido"],
        weaknesses=["contexto limitado"],
        prompt_style="directo",
        syntax_profile={"api_style": "openai_chat"},
        is_active=True,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _template(**overrides):
    defaults = dict(
        id=uuid.uuid4(),
        task_key="alignment_reinforcement",
        version="1.0",
        intent="Eres el refuerzo.",
        rules=["Nunca bajar la severidad"],
        input_schema={"brief": "object"},
        output_schema={"verdict": "string"},
        few_shot=[],
        is_active=True,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _artifact(**overrides):
    defaults = dict(
        id=uuid.uuid4(),
        llm_model_id=uuid.uuid4(),
        task_key="alignment_reinforcement",
        spec_version="1.0",
        artifact_version=1,
        prompt_text="prompt compilado",
        content_hash="a" * 64,
        compiled_by="compiler",
        is_active=True,
        created_at=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _embedding(**overrides):
    defaults = dict(
        id=uuid.uuid4(),
        api_key_id=uuid.uuid4(),
        llm_model_id=uuid.uuid4(),
        dimension=1024,
        is_active=True,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------
# /llm-models
# ---------------------------------------------------------------------


def test_llm_models_crud_roundtrip(client, fake):
    with _use(client, fake):
        resp = client.post(
            "/llm-models",
            json={
                "model_name": "meta-models/Muse-Glimmer-30B",
                "provider": "together",
                "model_size": "small",
            },
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["model_name"] == "meta-models/Muse-Glimmer-30B"
        mid = body["id"]

        resp = client.get("/llm-models")
        assert resp.status_code == 200
        assert len(resp.json()) == 1

        resp = client.get(f"/llm-models/{mid}")
        assert resp.status_code == 200
        assert resp.json()["model_size"] == "small"

        resp = client.put(f"/llm-models/{mid}", json={"model_size": "large"})
        assert resp.status_code == 200
        assert resp.json()["model_size"] == "large"

        resp = client.delete(f"/llm-models/{mid}")
        assert resp.status_code == 204
        assert mid not in fake.models


def test_llm_models_409_duplicate(client, fake):
    m = _model()
    fake.models[m.id] = m
    with _use(client, fake):
        resp = client.post(
            "/llm-models",
            json={
                "model_name": m.model_name,
                "provider": "together",
                "model_size": "small",
            },
        )
        assert resp.status_code == 409


def test_llm_models_404_missing(client, fake):
    with _use(client, fake):
        resp = client.get(f"/llm-models/{uuid.uuid4()}")
        assert resp.status_code == 404


# ---------------------------------------------------------------------
# /prompt-templates
# ---------------------------------------------------------------------


def test_prompt_templates_crud_roundtrip(client, fake):
    with _use(client, fake):
        resp = client.post(
            "/prompt-templates",
            json={
                "task_key": "alignment_reinforcement",
                "intent": "Eres el refuerzo.",
                "rules": ["Nunca bajar la severidad"],
            },
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["task_key"] == "alignment_reinforcement"
        tid = body["id"]

        resp = client.get("/prompt-templates")
        assert resp.status_code == 200
        assert len(resp.json()) == 1

        resp = client.put(f"/prompt-templates/{tid}", json={"version": "1.1"})
        assert resp.status_code == 200
        assert resp.json()["version"] == "1.1"

        resp = client.delete(f"/prompt-templates/{tid}")
        assert resp.status_code == 204
        assert tid not in fake.templates


def test_prompt_templates_409_duplicate(client, fake):
    t = _template()
    fake.templates[t.id] = t
    with _use(client, fake):
        resp = client.post(
            "/prompt-templates",
            json={"task_key": t.task_key, "intent": "otro"},
        )
        assert resp.status_code == 409


def test_critic_checklist_422_when_missing_interpretation(client, fake, monkeypatch):
    import src.llm.compiler as compiler_mod

    monkeypatch.setattr(
        compiler_mod,
        "validate_critic_spec",
        lambda template, items: ["revision_legal"],
    )
    with _use(client, fake):
        resp = client.post(
            "/prompt-templates",
            json={
                "task_key": "critic_checklist",
                "intent": "Eres el critico.",
                "rules": ["fuente_verificable: cita fuente"],
                "checklist_items": ["fuente_verificable", "revision_legal"],
            },
        )
        assert resp.status_code == 422
        assert "revision_legal" in resp.json()["detail"]


def test_critic_checklist_ok_without_items(client, fake):
    with _use(client, fake):
        resp = client.post(
            "/prompt-templates",
            json={
                "task_key": "critic_checklist",
                "intent": "Eres el critico.",
                "rules": ["fuente_verificable: cita fuente"],
            },
        )
        assert resp.status_code == 201


# ---------------------------------------------------------------------
# /prompt-artifacts
# ---------------------------------------------------------------------


def test_prompt_artifacts_list_hides_text(client, fake):
    a = _artifact()
    fake.artifacts[a.id] = a
    with _use(client, fake):
        resp = client.get("/prompt-artifacts")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 1
        assert "prompt_text" not in body[0]
        assert body[0]["content_hash"] == a.content_hash


def test_prompt_artifacts_detail_includes_text(client, fake):
    a = _artifact()
    fake.artifacts[a.id] = a
    with _use(client, fake):
        resp = client.get(f"/prompt-artifacts/{a.id}")
        assert resp.status_code == 200
        assert resp.json()["prompt_text"] == "prompt compilado"


# ---------------------------------------------------------------------
# /embedding-settings
# ---------------------------------------------------------------------


def test_embedding_settings_singleton(client, fake):
    key_id = uuid.uuid4()
    model_id = uuid.uuid4()
    fake.keys[key_id] = SimpleNamespace(id=key_id, is_active=True)
    fake.models[model_id] = _model(id=model_id)
    with _use(client, fake):
        resp = client.post(
            "/embedding-settings",
            json={"api_key_id": str(key_id), "llm_model_id": str(model_id)},
        )
        assert resp.status_code == 201
        first_id = resp.json()["id"]

        # Segunda activa: desactiva la primera (singleton).
        resp = client.post(
            "/embedding-settings",
            json={"api_key_id": str(key_id), "llm_model_id": str(model_id)},
        )
        assert resp.status_code == 201
        assert fake.embedding[0].is_active is False

        resp = client.get("/embedding-settings")
        assert resp.status_code == 200
        assert resp.json()["id"] != first_id


def test_embedding_settings_404_when_none(client, fake):
    with _use(client, fake):
        resp = client.get("/embedding-settings")
        assert resp.status_code == 404


def test_embedding_settings_404_when_key_missing(client, fake):
    with _use(client, fake):
        resp = client.post(
            "/embedding-settings",
            json={"api_key_id": str(uuid.uuid4()), "llm_model_id": str(uuid.uuid4())},
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------
# /settings/ensure-prompts
# ---------------------------------------------------------------------


def test_ensure_prompts_returns_compile_summary(client, fake, monkeypatch):
    import src.llm.compiler as compiler_mod

    monkeypatch.setattr(
        compiler_mod,
        "compile_prompts",
        lambda session: {"compiled": 8, "skipped": 0, "warnings": [], "artifacts": []},
    )
    with _use(client, fake):
        resp = client.post("/settings/ensure-prompts")
        assert resp.status_code == 200
        assert resp.json()["compiled"] == 8


# ---------------------------------------------------------------------
# /settings — nuevos campos fallback_model / llm_retries
# ---------------------------------------------------------------------


def test_settings_create_includes_fallback_and_retries(client, fake):
    key_id = uuid.uuid4()
    fake.keys[key_id] = SimpleNamespace(id=key_id, is_active=True)
    with _use(client, fake):
        resp = client.post(
            "/settings",
            json={
                "api_key_id": str(key_id),
                "small_model": "m1",
                "large_model": "m2",
                "fallback_model": "m3",
                "llm_retries": 5,
            },
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["fallback_model"] == "m3"
        assert body["llm_retries"] == 5
