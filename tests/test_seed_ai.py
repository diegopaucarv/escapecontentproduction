"""Tests del seed de IA modular (src/db/seed_ai.py) — sin base real."""

import uuid

from src.db.seed_ai import MODELS, TEMPLATES, seed


class _FakeSession:
    """Sesión mínima: guarda modelos/templates por su clave única."""

    def __init__(self):
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

        # select(LlmModel) / select(PromptTemplate) con where por clave única
        if hasattr(stmt, "_where_criteria") and stmt._where_criteria:
            col = stmt._where_criteria[0].left
            if col.name == "model_name":
                return _Result([self.models.get(stmt._where_criteria[0].right.value)])
            if col.name == "task_key":
                return _Result(
                    [self.templates.get(stmt._where_criteria[0].right.value)]
                )
        return _Result([])

    def add(self, obj):
        self.added.append(obj)

    def flush(self):
        # Simula la asignación de id en el INSERT.
        for obj in self.added:
            if getattr(obj, "id", None) is None:
                obj.id = uuid.uuid4()
            if obj.__class__.__name__ == "LlmModel":
                self.models[obj.model_name] = obj
            elif obj.__class__.__name__ == "PromptTemplate":
                self.templates[obj.task_key] = obj
        self.added = []

    def commit(self):
        self.flush()


def test_seed_data_has_3_models():
    names = [m["model_name"] for m in MODELS]
    assert names == [
        "meta-models/Muse-Glimmer-30B",
        "deepseek-ai/DeepSeek-V4-Flash-0731",
        "voyage-3-large",
    ]
    sizes = {m["model_name"]: m["model_size"] for m in MODELS}
    assert sizes["meta-models/Muse-Glimmer-30B"] == "small"
    assert sizes["deepseek-ai/DeepSeek-V4-Flash-0731"] == "large"
    assert sizes["voyage-3-large"] == "embedding"


def test_seed_data_has_4_templates():
    keys = [t["task_key"] for t in TEMPLATES]
    assert keys == [
        "alignment_reinforcement",
        "critic_checklist",
        "novelty_scoring",
        "producer_draft",
    ]
    for t in TEMPLATES:
        assert t["version"] == "1.0"
        assert t["intent"]
        assert isinstance(t["rules"], list) and t["rules"]


def test_seed_upserts_without_duplicates():
    fake = _FakeSession()
    result1 = seed(session=fake)
    assert len(result1["model_names"]) == 3
    assert len(result1["task_keys"]) == 4

    # Segunda corrida: no duplica (upsert por clave única).
    result2 = seed(session=fake)
    assert len(fake.models) == 3
    assert len(fake.templates) == 4
    assert result2["model_names"] == result1["model_names"]
    assert result2["task_keys"] == result1["task_keys"]


def test_seed_returns_summary_with_ids():
    fake = _FakeSession()
    result = seed(session=fake)
    assert "models" in result and len(result["models"]) == 3
    assert "templates" in result and len(result["templates"]) == 4
    for mid in result["models"]:
        uuid.UUID(mid)  # no lanza
    for tid in result["templates"]:
        uuid.UUID(tid)
