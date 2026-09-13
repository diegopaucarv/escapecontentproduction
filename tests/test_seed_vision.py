"""Tests del seed de visión (src/db/seed_vision.py) y del soporte de
visión en compilador/cliente — sin base real ni red."""

import uuid
from types import SimpleNamespace

from src.db.seed_vision import (
    VISION_API_KEY,
    VISION_MODELS,
    VISION_TEMPLATES,
    seed,
)
from src.llm.compiler import compile_prompts
from src.llm.together import TOGETHER_CHAT_URL, complete_vision


class _FakeSession:
    """Sesión mínima: guarda api_keys por (provider, key_name), modelos por
    model_name y templates por task_key."""

    def __init__(self):
        self.keys = {}  # (provider, key_name) -> obj
        self.models = {}  # model_name -> obj
        self.templates = {}  # task_key -> obj
        self.added = []

    def execute(self, stmt):
        class _Result:
            def __init__(self, rows):
                self._rows = rows

            def scalars(self):
                return self

            def first(self):
                return self._rows[0] if self._rows else None

        if hasattr(stmt, "_where_criteria") and stmt._where_criteria:
            pairs = {}
            for c in stmt._where_criteria:
                if hasattr(c.left, "name"):
                    v = getattr(c.right, "value", c.right)
                    v = getattr(v, "value", v)  # desenvuelve enum si aplica
                    pairs[c.left.name] = v
            if "provider" in pairs and "key_name" in pairs:
                return _Result([self.keys.get((pairs["provider"], pairs["key_name"]))])
            if "model_name" in pairs:
                return _Result([self.models.get(pairs["model_name"])])
            if "task_key" in pairs:
                return _Result([self.templates.get(pairs["task_key"])])
        return _Result([])

    def add(self, obj):
        self.added.append(obj)

    def flush(self):
        # Simula la asignación de id en el INSERT.
        for obj in self.added:
            if getattr(obj, "id", None) is None:
                obj.id = uuid.uuid4()
            if obj.__class__.__name__ == "ApiKey":
                self.keys[(obj.provider, obj.key_name)] = obj
            elif obj.__class__.__name__ == "LlmModel":
                self.models[obj.model_name] = obj
            elif obj.__class__.__name__ == "PromptTemplate":
                self.templates[obj.task_key] = obj
        self.added = []

    def commit(self):
        self.flush()


# ---------------------------------------------------------------------
# Datos del seed
# ---------------------------------------------------------------------


def test_vision_api_key_is_placeholder():
    assert VISION_API_KEY["provider"] == "together"
    assert VISION_API_KEY["key_name"] == "together_vision"
    assert VISION_API_KEY["api_key"] == ""  # placeholder, se llena vía PATCH
    assert VISION_API_KEY["is_active"] is True


def test_vision_models_have_vision_size_and_openai_profile():
    assert len(VISION_MODELS) == 2
    for m in VISION_MODELS:
        assert m["model_size"] == "vision"
        assert m["provider"] == "together"
        assert m["syntax_profile"]["api_style"] == "openai_chat"
        assert m["syntax_profile"]["structured_output"]["mode"] == "json_object"
        assert m["context_window"] > 0
        assert m["max_output_tokens"] > 0


def test_vision_templates_have_schemas():
    assert len(VISION_TEMPLATES) == 2
    keys = [t["task_key"] for t in VISION_TEMPLATES]
    assert keys == ["vision_describe_asset", "vision_prompt_visual"]
    for t in VISION_TEMPLATES:
        assert t["version"] == "1.0"
        assert t["intent"]
        assert isinstance(t["rules"], list) and t["rules"]
        assert t["input_schema"]  # no vacío
        assert t["output_schema"]  # no vacío


# ---------------------------------------------------------------------
# seed() con sesión falsa
# ---------------------------------------------------------------------


def test_seed_inserts_key_models_and_templates():
    fake = _FakeSession()
    result = seed(session=fake)
    assert len(fake.keys) == 1
    assert len(fake.models) == 2
    assert len(fake.templates) == 2
    key = fake.keys[("together", "together_vision")]
    assert key.api_key == ""
    assert result["model_names"] == [m["model_name"] for m in VISION_MODELS]
    assert result["task_keys"] == [t["task_key"] for t in VISION_TEMPLATES]


def test_seed_is_idempotent():
    fake = _FakeSession()
    result1 = seed(session=fake)
    result2 = seed(session=fake)
    assert len(fake.keys) == 1
    assert len(fake.models) == 2
    assert len(fake.templates) == 2
    assert result2["model_names"] == result1["model_names"]
    assert result2["task_keys"] == result1["task_keys"]


def test_seed_returns_summary_with_ids():
    fake = _FakeSession()
    result = seed(session=fake)
    uuid.UUID(result["api_key_id"])  # no lanza
    assert len(result["models"]) == 2
    assert len(result["templates"]) == 2
    for mid in result["models"]:
        uuid.UUID(mid)  # no lanza
    for tid in result["templates"]:
        uuid.UUID(tid)


# ---------------------------------------------------------------------
# Compilador: incluye modelos de visión
# ---------------------------------------------------------------------


def _spec(**overrides):
    defaults = dict(
        task_key="vision_describe_asset",
        version="1.0",
        intent="Eres el analista visual.",
        rules=["Salida JSON estricta"],
        input_schema={"image_url": "string"},
        output_schema={"descripcion": "string"},
        few_shot=[],
        is_active=True,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _model(**overrides):
    defaults = dict(
        id=uuid.uuid4(),
        model_name="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
        model_size="vision",
        is_active=True,
        temperature_default=0.4,
        max_output_tokens=2048,
        syntax_profile={
            "api_style": "openai_chat",
            "system_role_name": "system",
            "instruction_formatting": {"style": "markdown"},
            "structured_output": {"mode": "json_object"},
            "tool_calling": {"mode": "none", "supports_strict_tools": False},
            "prompt_caching": {"supports_prefix_caching": True},
            "reasoning_mode": {"supports_reasoning": False},
        },
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class _CompileSession:
    """Sesión falsa para compile_prompts: devuelve modelos y templates."""

    def __init__(self, models, templates):
        self._models = models
        self._templates = templates
        self.added = []

    def execute(self, stmt):
        class _Result:
            def __init__(self, rows):
                self._rows = rows

            def scalars(self):
                return self

            def all(self):
                return self._rows

        # select(LlmModel) -> modelos; select(PromptTemplate) -> templates.
        if hasattr(stmt, "_where_criteria") and stmt._where_criteria:
            col = stmt._where_criteria[0].left
            table = getattr(col, "table", None)
            table_name = getattr(table, "name", "")
            if table_name == "llm_models":
                return _Result(self._models)
            if table_name == "prompt_templates":
                return _Result(self._templates)
        return _Result([])

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        pass


def test_compile_prompts_includes_vision_models():
    model = _model()
    spec = _spec()
    session = _CompileSession([model], [spec])

    result = compile_prompts(session)

    assert result["compiled"] == 1
    assert result["artifacts"][0]["model"] == model.model_name
    assert result["artifacts"][0]["task"] == spec.task_key
    # El artefacto se insertó para el modelo de visión.
    assert len(session.added) == 1
    artifact = session.added[0]
    assert artifact.llm_model_id == model.id
    assert artifact.task_key == spec.task_key
    assert artifact.is_active is True


# ---------------------------------------------------------------------
# complete_vision: body multimodal
# ---------------------------------------------------------------------


def test_complete_vision_builds_multimodal_body(monkeypatch):
    from src.db.models import ApiKey, SessionSettings

    key = ApiKey(
        id=uuid.uuid4(),
        provider="together",
        key_name="together_vision",
        api_key="tgp_v1_test_key_1234567890",
        is_active=True,
    )
    settings = SessionSettings(
        id=uuid.uuid4(),
        api_key_id=key.id,
        small_model="meta-models/Muse-Glimmer-30B",
        large_model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
        temperature_small=0.7,
        temperature_large=0.4,
        max_tokens_small=2048,
        max_tokens_large=2048,
        is_active=True,
    )

    class _Session:
        def __init__(self):
            self._settings = [settings]
            self._keys = {key.id: key}

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

    session = _Session()
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
                return {"choices": [{"message": {"content": "descripción"}}]}

        return _Resp()

    monkeypatch.setattr("src.llm.together.httpx.post", fake_post)

    text = complete_vision(
        session,
        "Describe esta imagen",
        image_url="https://example.com/asset.png",
        system="Sé preciso.",
    )

    assert text == "descripción"
    assert captured["url"] == TOGETHER_CHAT_URL
    assert captured["headers"]["Authorization"] == f"Bearer {key.api_key}"
    assert captured["json"]["model"] == "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo"
    assert captured["json"]["messages"][0] == {
        "role": "system",
        "content": "Sé preciso.",
    }
    content = captured["json"]["messages"][1]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "Describe esta imagen"}
    assert content[1] == {
        "type": "image_url",
        "image_url": {"url": "https://example.com/asset.png"},
    }
    assert captured["json"]["temperature"] == 0.4
    assert captured["json"]["max_tokens"] == 2048
