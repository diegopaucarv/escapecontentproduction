"""Tests del seed de producción creativa (src/db/seed_tools.py) — sin base real."""

import uuid

from src.db.models import BrandObjective
from src.db.seed_tools import (
    FORMAT_SPECS,
    TOOL_ADAPTERS,
    VISUAL_REQUIREMENTS,
    seed,
)

# Valores válidos del enum artifact_type_t (migración 0001).
VALID_ARTIFACT_TYPES = {
    "post",
    "video_corto",
    "video_largo",
    "carrusel",
    "one_pager",
    "white_paper",
    "pdf_recurso",
    "broadcast",
    "poster_qr",
    "webinar",
    "propuesta",
    "app_herramienta",
    "newsletter",
    "hilo",
}

VALID_EXECUTION_MODES = {
    "local",
    "local_orchestration_cloud_inference",
    "remote_cloud",
}

# 'producer' es el paso LLM (Producer-Critic), no un tool_adapter.
KNOWN_ADAPTERS = {a["name"] for a in TOOL_ADAPTERS} | {"producer"}


class _FakeSession:
    """Sesión mínima: guarda adapters por name y specs por (brand, artifact_type)."""

    def __init__(self):
        self.adapters = {}  # name -> obj
        self.specs = {}  # (brand, artifact_type) -> obj
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
            if "name" in pairs:
                return _Result([self.adapters.get(pairs["name"])])
            if "brand_objective" in pairs and "artifact_type" in pairs:
                return _Result(
                    [self.specs.get((pairs["brand_objective"], pairs["artifact_type"]))]
                )
        return _Result([])

    def add(self, obj):
        self.added.append(obj)

    def flush(self):
        for obj in self.added:
            if getattr(obj, "id", None) is None:
                obj.id = uuid.uuid4()
            if obj.__class__.__name__ == "ToolAdapter":
                self.adapters[obj.name] = obj
            elif obj.__class__.__name__ == "FormatSpec":
                brand = getattr(obj.brand_objective, "value", obj.brand_objective)
                self.specs[(brand, obj.artifact_type)] = obj
        self.added = []

    def commit(self):
        self.flush()


def test_tool_adapters_catalog_has_10():
    names = [a["name"] for a in TOOL_ADAPTERS]
    assert names == [
        "inkscape",
        "krita",
        "reaper",
        "resolve",
        "comfyui",
        "elevenlabs",
        "veo3",
        "canva",
        "gdrive",
        "filesystem",
    ]
    for a in TOOL_ADAPTERS:
        assert a["execution_mode"] in VALID_EXECUTION_MODES
        assert a["mcp_server_name"]


def test_format_specs_artifact_types_valid():
    types = {s["artifact_type"] for s in FORMAT_SPECS}
    assert types <= VALID_ARTIFACT_TYPES
    assert len(FORMAT_SPECS) == 8


def test_tool_chains_reference_known_adapters():
    for s in FORMAT_SPECS:
        for step in s["tool_chain"]:
            assert step in KNOWN_ADAPTERS, f"{s['artifact_type']}: {step}"


def test_tool_chains_start_with_producer():
    for s in FORMAT_SPECS:
        assert s["tool_chain"][0] == "producer"


def test_visual_requirements_cover_both_brands():
    for brand in (BrandObjective.ESCAPE_SOCIAL, BrandObjective.ERGALIA_COMERCIAL):
        visual = VISUAL_REQUIREMENTS[brand.value]
        for s in FORMAT_SPECS:
            assert s["artifact_type"] in visual


def test_seed_upserts_without_duplicates():
    fake = _FakeSession()
    result1 = seed(session=fake)
    assert len(result1["adapter_names"]) == 10
    assert result1["format_count"] == 16

    # Segunda corrida: no duplica (upsert por clave única).
    result2 = seed(session=fake)
    assert len(fake.adapters) == 10
    assert len(fake.specs) == 16
    assert result2["adapter_names"] == result1["adapter_names"]
    assert result2["format_count"] == result1["format_count"]


def test_seed_returns_summary_with_ids():
    fake = _FakeSession()
    result = seed(session=fake)
    assert len(result["adapters"]) == 10
    assert len(result["format_specs"]) == 16
    for aid in result["adapters"]:
        uuid.UUID(aid)  # no lanza
    for sid in result["format_specs"]:
        uuid.UUID(sid)
