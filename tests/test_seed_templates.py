"""Tests del seed de Fase 1.5 (src/db/seed_templates.py) — sin base real."""

import uuid

from src.db.models import BrandObjective, FormatSpec
from src.db.seed_templates import (
    FORMAT_PHASES,
    PRODUCTION_TEMPLATES,
    seed,
)
from src.db.seed_tools import FORMAT_SPECS

VALID_CONTENT_TYPES = {"audio", "video", "grafico"}
VALID_PHASES = {"preproduccion", "produccion", "postproduccion"}
VALID_TEMPLATE_FORMATS = {"json", "otio", "ass", "cube", "svg", "txt"}

# Los 17 templates del diseño §3, agrupados por fase.
EXPECTED_TEMPLATE_NAMES = [
    # Audio
    "Script_SSML_Template",
    "Audio_Track_Manifest",
    "TTS_Voice_Preset",
    "Raw_Audio_Spec",
    "Voice_FX_Chain_Preset",
    "Sidechain_Ducking_Preset",
    "Loudness_Master_Spec",
    # Video
    "Storyboard_Spec",
    "Timeline_Template",
    "ComfyUI_Workflow_Template",
    "Subtitles_Style_Template",
    "Color_Grade_LUT_Template",
    "Transitions_Rules_Spec",
    "Render_Export_Preset",
    # Gráfico
    "Canvas_Layout_Spec",
    "Inkscape_Layer_Template",
    "Image_Composite_FX_Spec",
]


class _FakeSession:
    """Sesión mínima: guarda templates por name y specs por (brand, artifact_type)."""

    def __init__(self):
        self.templates = {}  # name -> obj
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
                return _Result([self.templates.get(pairs["name"])])
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
            if obj.__class__.__name__ == "ProductionTemplate":
                self.templates[obj.name] = obj
            elif obj.__class__.__name__ == "FormatSpec":
                brand = getattr(obj.brand_objective, "value", obj.brand_objective)
                self.specs[(brand, obj.artifact_type)] = obj
        self.added = []

    def commit(self):
        self.flush()


def _prepopulate_specs(fake):
    """Crea las 16 specs (8 tipos x 2 marcas) que seed_tools insertaría."""
    for brand in (BrandObjective.ESCAPE_SOCIAL, BrandObjective.ERGALIA_COMERCIAL):
        for base in FORMAT_SPECS:
            spec = FormatSpec(
                brand_objective=brand,
                artifact_type=base["artifact_type"],
                structure=base["structure"],
                constraints=base["constraints"],
                derivation_rules=base["derivation_rules"],
                visual_requirements={},
                qa_checks=base["qa_checks"],
                tool_chain=base["tool_chain"],
            )
            fake.add(spec)
    fake.flush()


def test_17_templates_with_valid_metadata():
    assert len(PRODUCTION_TEMPLATES) == 17
    names = [t["name"] for t in PRODUCTION_TEMPLATES]
    assert names == EXPECTED_TEMPLATE_NAMES
    for t in PRODUCTION_TEMPLATES:
        assert t["content_type"] in VALID_CONTENT_TYPES, t["name"]
        assert t["phase"] in VALID_PHASES, t["name"]
        assert t["template_format"] in VALID_TEMPLATE_FORMATS, t["name"]


def test_every_template_has_non_empty_content():
    for t in PRODUCTION_TEMPLATES:
        assert isinstance(t["content"], dict), t["name"]
        assert t["content"], f"{t['name']} tiene content vacío"


def test_format_phases_cover_all_artifact_types():
    spec_types = {s["artifact_type"] for s in FORMAT_SPECS}
    assert len(FORMAT_SPECS) == 8
    assert set(FORMAT_PHASES) == spec_types


def test_format_phases_reference_existing_templates():
    template_names = {t["name"] for t in PRODUCTION_TEMPLATES}
    for artifact_type, phases in FORMAT_PHASES.items():
        for phase, spec in phases.items():
            assert phase in VALID_PHASES, f"{artifact_type}: {phase}"
            for name in spec["templates"]:
                assert name in template_names, f"{artifact_type}/{phase}: {name}"


def test_seed_upserts_without_duplicates():
    fake = _FakeSession()
    _prepopulate_specs(fake)
    result1 = seed(session=fake)
    assert len(result1["template_names"]) == 17
    assert result1["format_count"] == 16

    # Segunda corrida: no duplica (upsert por clave única).
    result2 = seed(session=fake)
    assert len(fake.templates) == 17
    assert len(fake.specs) == 16
    assert result2["template_names"] == result1["template_names"]
    assert result2["format_count"] == result1["format_count"]


def test_seed_returns_summary_with_ids():
    fake = _FakeSession()
    _prepopulate_specs(fake)
    result = seed(session=fake)
    assert len(result["templates"]) == 17
    assert len(result["format_specs"]) == 16
    for tid in result["templates"]:
        uuid.UUID(tid)  # no lanza
    for sid in result["format_specs"]:
        uuid.UUID(sid)
