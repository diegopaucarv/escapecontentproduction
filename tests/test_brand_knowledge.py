"""Tests del canon de marca como Context Pack (0012) — sin base real.

Cubre:
  1. El seed (src/db/seed_brand_knowledge.py): datos mockup marcados
     is_mock=True, con source_doc apuntando a los documentos reales que
     deben reemplazarlos (instrucción #3 del usuario).
  2. load_brand_knowledge: la consulta por brand_objective + content_bucket.
  3. build_context_pack: el canon llega al Context Pack del Producer.
"""

import uuid

from src.db.models import (
    BrandKnowledge,
    BrandObjective,
    ContentBrief,
    ContentBucket,
)
from src.db.seed_brand_knowledge import BRAND_SECTIONS, BUCKET_SECTIONS, seed
from src.production.produce_brief import build_context_pack, load_brand_knowledge


class _FakeSession:
    """Sesión mínima: responde a select(BrandKnowledge) con filas
    preconfiguradas y a select(SessionSettings) con vacío."""

    def __init__(self, brand_knowledge=None):
        self._brand_knowledge = brand_knowledge or []

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

        if hasattr(stmt, "_raw_columns") and stmt._raw_columns:
            table = getattr(stmt._raw_columns[0], "name", "")
            if table == "brand_knowledge":
                return _Result(self._brand_knowledge)
        return _Result([])


def _make_brief(**overrides):
    defaults = {
        "id": uuid.uuid4(),
        "brand_objective": "ESCAPE_SOCIAL",
        "content_bucket": "difusion_cientifica",
        "risk_level": "bajo",
        "route_decision": "repetitivo",
        "production_route": None,
        "segment_client": "S1",
        "insight_core": "insight de prueba",
        "resumen": "Resumen de prueba.",
        "evidence_source": "Estudio 2026.",
        "prior_attempts": None,
        "repurpose_plan": [],
        "artifact_type": "video_corto",
        "channel": "tiktok",
    }
    defaults.update(overrides)
    return ContentBrief(**defaults)


def _make_knowledge_row(**overrides):
    defaults = {
        "id": 1,
        "brand_objective": "ESCAPE_SOCIAL",
        "content_bucket": None,
        "section_key": "tono_y_voz",
        "section_title": "Tono y voz",
        "content": "Hablar como un divulgador riguroso pero cercano.",
        "source_doc": "docs/escape/escape_guia_contenidos_y_produccion.md",
        "is_mock": True,
    }
    defaults.update(overrides)
    return BrandKnowledge(**defaults)


# ---------------------------------------------------------------------
# Seed — datos mockup con instrucciones de reemplazo
# ---------------------------------------------------------------------


def test_seed_covers_both_brands():
    assert set(BRAND_SECTIONS) == {"ESCAPE_SOCIAL", "ERGALIA_COMERCIAL"}
    for brand, sections in BRAND_SECTIONS.items():
        assert sections, f"{brand} sin secciones"
        for s in sections:
            assert s["section_key"]
            assert s["section_title"]
            assert s["content"]
            assert s["source_doc"]


def test_seed_sections_are_marked_as_mockup():
    """Instrucción #3: los documentos reales no existen, así que el seed
    inserta mockups con is_mock=True y source_doc apuntando al documento
    real que debe reemplazarlos."""
    for brand, sections in BRAND_SECTIONS.items():
        for s in sections:
            assert "MOCKUP" in s["content"], (
                f"{brand}/{s['section_key']} sin marca MOCKUP"
            )
            assert s["source_doc"].endswith(".md"), s["source_doc"]


def test_bucket_sections_reference_existing_brands_and_buckets():
    for (brand, bucket), sections in BUCKET_SECTIONS.items():
        assert brand in BRAND_SECTIONS, f"marca {brand} sin secciones transversales"
        assert bucket in ContentBucket._value2member_map_, f"bucket inválido: {bucket}"
        assert sections, f"{brand}/{bucket} sin secciones"


def test_seed_upserts_without_duplicates():
    class _SeedFakeSession:
        def __init__(self):
            self.rows = {}
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
                        v = getattr(v, "value", v)
                        pairs[c.left.name] = v
                key = (
                    pairs.get("brand_objective"),
                    pairs.get("content_bucket"),
                    pairs.get("section_key"),
                )
                return _Result([self.rows.get(key)])
            return _Result([])

        def add(self, obj):
            self.added.append(obj)

        def flush(self):
            for obj in self.added:
                if getattr(obj, "id", None) is None:
                    obj.id = uuid.uuid4()
                key = (
                    getattr(obj.brand_objective, "value", obj.brand_objective),
                    getattr(obj.content_bucket, "value", obj.content_bucket),
                    obj.section_key,
                )
                self.rows[key] = obj
            self.added = []

        def commit(self):
            self.flush()

    fake = _SeedFakeSession()
    result1 = seed(session=fake)
    assert result1["count"] > 0
    assert result1["brands"] == ["ERGALIA_COMERCIAL", "ESCAPE_SOCIAL"]

    # Segunda corrida: no duplica (upsert por clave única).
    result2 = seed(session=fake)
    assert result2["count"] == result1["count"]
    assert len(fake.rows) == result1["count"]


# ---------------------------------------------------------------------
# load_brand_knowledge — consulta por marca + bucket
# ---------------------------------------------------------------------


def test_load_brand_knowledge_returns_brand_and_bucket_sections():
    rows = [
        _make_knowledge_row(id=1, content_bucket=None, section_key="tono_y_voz"),
        _make_knowledge_row(
            id=2,
            content_bucket="difusion_cientifica",
            section_key="bucket_objetivo",
        ),
    ]
    session = _FakeSession(rows)
    brief = _make_brief()
    result = load_brand_knowledge(session, brief)

    assert len(result) == 2
    assert result[0]["section_key"] == "tono_y_voz"
    assert result[0]["is_mock"] is True
    assert result[0]["source_doc"].endswith(".md")
    assert result[1]["section_key"] == "bucket_objetivo"


def test_load_brand_knowledge_empty_when_no_rows():
    session = _FakeSession([])
    brief = _make_brief()
    assert load_brand_knowledge(session, brief) == []


# ---------------------------------------------------------------------
# build_context_pack — el canon llega al Producer
# ---------------------------------------------------------------------


def test_build_context_pack_includes_brand_knowledge():
    rows = [
        _make_knowledge_row(id=1, content_bucket=None, section_key="tono_y_voz"),
        _make_knowledge_row(
            id=2,
            content_bucket="difusion_cientifica",
            section_key="bucket_objetivo",
        ),
    ]
    session = _FakeSession(rows)
    brief = _make_brief()
    pack = build_context_pack(session, brief)

    assert pack["resumen"] == "Resumen de prueba."
    assert "brand_knowledge" in pack
    assert len(pack["brand_knowledge"]) == 2
    assert pack["brand_knowledge"][0]["section_key"] == "tono_y_voz"


def test_build_context_pack_without_canon_stays_minimal():
    """Sin filas de canon (seed no corrido), el pack sigue siendo el mínimo
    del brief — degradación silenciosa, no bloqueante."""
    session = _FakeSession([])
    brief = _make_brief()
    pack = build_context_pack(session, brief)

    assert pack["resumen"] == "Resumen de prueba."
    assert "brand_knowledge" not in pack


def test_build_context_pack_survives_query_error():
    """Si la consulta al canon falla (tabla no existe aún), el pack no se
    rompe: el canon queda vacío y la generación continúa."""

    class _BrokenSession:
        def execute(self, stmt):
            raise RuntimeError("tabla no existe")

    brief = _make_brief()
    pack = build_context_pack(_BrokenSession(), brief)

    assert pack["resumen"] == "Resumen de prueba."
    assert "brand_knowledge" not in pack
