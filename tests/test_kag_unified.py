"""Tests de la unificación KAG — proposiciones atómicas + modo audited.

Sin DB real, sin torch/spacy/transformers. Sesiones falsas + monkeypatch
(patrón de tests/test_kag.py).
"""

from types import SimpleNamespace

import pytest

from src.kag_query import (
    _evaluate_sufficiency,
    _fts_tsquery,
    _json_dumps,
    _verify_grounding,
    assemble_context,
    propositions_for_chunks,
)


@pytest.fixture(autouse=True)
def _reset_caches():
    """Resetea los cachés module-level de src.kag_query entre tests."""
    import src.kag_query as kq

    kq._adjacency_cache = None
    kq._adjacency_version = -1
    kq._CORPUS_CACHE = {"ts": 0.0, "words": None, "langs": None}
    yield
    kq._adjacency_cache = None
    kq._adjacency_version = -1
    kq._CORPUS_CACHE = {"ts": 0.0, "words": None, "langs": None}


class _Result:
    def __init__(self, rows=None, scalar=None):
        self._rows = rows or []
        self._scalar = scalar

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def first(self):
        return self._rows[0] if self._rows else None

    def scalar(self):
        return self._scalar


# ---------------------------------------------------------------------
# propositions_for_chunks
# ---------------------------------------------------------------------


def _prop_row(prop_id, chunk_id, statement="stmt", span="span"):
    return SimpleNamespace(
        prop_id=prop_id,
        chunk_id=chunk_id,
        document_id=7,
        statement=statement,
        text_span=span,
        char_start=0,
        char_end=10,
        citation_references=["Ref 1"],
        doc_title="doc.md",
        chapter_title="## Cap",
    )


def test_propositions_for_chunks_filters_by_ids():
    import src.kag_query as kq

    class _FakeSession:
        def __init__(self, rows):
            self._rows = rows
            self.queries = 0

        def execute(self, stmt, params=None):
            self.queries += 1
            ids = params["ids"]
            return _Result([r for r in self._rows if r.chunk_id in ids])

    rows = [
        _prop_row(1, 10),
        _prop_row(2, 20),
        _prop_row(3, 10),
    ]
    session = _FakeSession(rows)
    out = propositions_for_chunks(session, [10, 20])
    assert len(out) == 3
    assert [p["chunk_id"] for p in out] == [1, 2, 3]
    assert out[0]["doc_title"] == "doc.md"
    assert out[0]["chapter_title"] == "## Cap"
    assert out[0]["document_id"] == 7
    assert out[0]["citation_references"] == ["Ref 1"]


def test_propositions_for_chunks_per_chunk_limit():
    import src.kag_query as kq

    class _FakeSession:
        def execute(self, stmt, params=None):
            return _Result([_prop_row(i, 10) for i in range(1, 11)])

    out = propositions_for_chunks(_FakeSession(), [10], per_chunk=3)
    assert len(out) == 3


def test_propositions_for_chunks_max_total():
    import src.kag_query as kq

    class _FakeSession:
        def execute(self, stmt, params=None):
            return _Result([_prop_row(i, 10) for i in range(1, 11)])

    out = propositions_for_chunks(_FakeSession(), [10], per_chunk=10, max_total=4)
    assert len(out) == 4


def test_propositions_for_chunks_empty_and_missing_table():
    import src.kag_query as kq

    class _FakeSession:
        def execute(self, stmt, params=None):
            raise Exception("relation kag_propositions does not exist")

    assert propositions_for_chunks(_FakeSession(), []) == []
    assert propositions_for_chunks(_FakeSession(), [1, 2]) == []


# ---------------------------------------------------------------------
# assemble_context con proposiciones
# ---------------------------------------------------------------------


def test_assemble_context_with_propositions_section_position():
    chunks = [
        {
            "doc_path": "a.md",
            "section_path": "## Intro",
            "chunk_index": 0,
            "content": "contenido a",
        }
    ]
    figures = [{"image_path": "images/a/fig1.png", "description": "una figura"}]
    propositions = [
        {
            "chunk_id": 1,
            "doc_title": "a.md",
            "chapter_title": "## Intro",
            "statement": "La cultura afecta a las organizaciones.",
            "text_span": "La cultura afecta a las organizaciones.",
        }
    ]
    ctx = assemble_context(
        chunks, [], figures, [], "pregunta", propositions=propositions
    )
    assert "PROPOSICIONES ATÓMICAS (capa micro)" in ctx
    assert "La cultura afecta a las organizaciones." in ctx
    # La sección de proposiciones va entre fragmentos y figuras.
    assert ctx.index("FRAGMENTOS RECUPERADOS") < ctx.index(
        "PROPOSICIONES ATÓMICAS (capa micro)"
    )
    assert ctx.index("PROPOSICIONES ATÓMICAS (capa micro)") < ctx.index("FIGURAS")


def test_assemble_context_with_propositions_empty():
    ctx = assemble_context([], [], [], [], "pregunta", propositions=[])
    assert "PROPOSICIONES ATÓMICAS (capa micro)" in ctx
    assert "sin proposiciones" in ctx


def test_assemble_context_without_propositions_omits_section():
    ctx = assemble_context([], [], [], [], "pregunta")
    assert "PROPOSICIONES ATÓMICAS (capa micro)" not in ctx


# ---------------------------------------------------------------------
# Helpers del modo audited
# ---------------------------------------------------------------------


def test_json_dumps_handles_non_serializable():
    assert _json_dumps({"a": 1}) == '{"a": 1}'
    # Objeto no serializable → repr (no lanza).
    out = _json_dumps({"x": object()})
    assert isinstance(out, str)


def test_fts_tsquery_quotes_and_escapes():
    assert _fts_tsquery(["hola mundo"]) == '"hola mundo"'
    assert _fts_tsquery(["a", "b"]) == '"a" | "b"'
    assert _fts_tsquery(['di"cho']) == '"di""cho"'
    assert _fts_tsquery(["", "  "]) == ""


def test_evaluate_sufficiency_verdicts():
    import src.kag_query as kq

    class _FakeSession:
        def execute(self, stmt, params=None):
            return _Result()

    session = _FakeSession()
    facts = [
        {
            "relevance_level": "direct_answer",
            "atomic_summary": "X",
            "verbatim_evidence": "X",
        }
    ]
    # Sin LLM (load_settings falla) → degradación determinista.
    out = _evaluate_sufficiency(session, "q", facts, [])
    assert out["verdict"] == "SUFFICIENT_FOR_SYNTHESIS"
    assert out["confidence_score"] == 0.5

    out2 = _evaluate_sufficiency(session, "q", [], [])
    assert out2["verdict"] == "INSUFFICIENT_TRIGGER_BRANCH_B"


def test_verify_grounding_exact_and_fuzzy(monkeypatch):
    import src.kag_query as kq

    class _FakeSession:
        def __init__(self, rows):
            self._rows = rows

        def execute(self, stmt, params=None):
            return _Result(self._rows)

        def rollback(self):
            pass

    # Exacto: verbatim es substring del text_span en DB.
    session = _FakeSession(
        [
            SimpleNamespace(
                text_span="La cultura afecta a las organizaciones.",
                citation_references=["Ref"],
                doc_id=1,
                doc_title="doc.md",
                chapter_title="## Cap",
            )
        ]
    )
    facts = [
        {
            "relevance_level": "direct_answer",
            "chunk_id": "5",
            "verbatim_evidence": "La cultura afecta",
            "atomic_summary": "resumen",
        }
    ]
    out = _verify_grounding(session, facts)
    assert len(out) == 1
    assert out[0]["verified_fact"] == "resumen"
    assert out[0]["document_title"] == "doc.md"

    # Fallo: verbatim no coincide (fuzzy bajo) → descartado.
    session2 = _FakeSession(
        [
            SimpleNamespace(
                text_span="texto completamente distinto",
                citation_references=[],
                doc_id=1,
                doc_title="doc.md",
                chapter_title="## Cap",
            )
        ]
    )
    out2 = _verify_grounding(session2, facts)
    assert out2 == []


def test_verify_grounding_empty_facts():
    assert _verify_grounding(None, []) == []
