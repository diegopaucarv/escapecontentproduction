"""Tests de recuperación §3.1/§3.2: search_summaries y proposition_vector_search.

Sin DB real: sesiones falsas con execute() que devuelve filas falsas (patrón
de tests/test_kag.py). Se testea la lógica pura de src/kag_query.py: RRF de
densa+FTS, filtro por level, degradación sin tabla (ProgrammingError) y la
búsqueda directa por embedding de proposiciones.
"""

import uuid
from types import SimpleNamespace

from sqlalchemy.exc import ProgrammingError

from src.kag_query import proposition_vector_search, search_summaries


class _Result:
    def __init__(self, rows=None):
        self._rows = rows if rows is not None else []

    def fetchall(self):
        return self._rows


class _PropRow:
    """Fila con _mapping (como sqlalchemy Row) para dict(r._mapping)."""

    def __init__(self, **kw):
        self._mapping = kw


class _RetrievalSession:
    """Sesión falsa: despacha por el SQL y registra las llamadas.

    - summary_index + ts_rank_cd → filas FTS
    - summary_index + embedding <=> → filas densas
    - summary_index (resto) → filas de texto (segundo SELECT por ids)
    - kag_propositions → filas de proposiciones
    - fail=True → ProgrammingError en execute (tabla ausente)
    """

    def __init__(
        self, dense_rows=None, fts_rows=None, text_rows=None, prop_rows=None, fail=False
    ):
        self.dense_rows = dense_rows or []
        self.fts_rows = fts_rows or []
        self.text_rows = text_rows or []
        self.prop_rows = prop_rows or []
        self.fail = fail
        self.calls = []
        self.rolled_back = 0

    def execute(self, stmt, params=None):
        self.calls.append((str(stmt), params))
        if self.fail:
            raise ProgrammingError("stmt", {}, Exception("relation does not exist"))
        sql = str(stmt)
        if "FROM summary_index" in sql and "ts_rank_cd" in sql:
            return _Result(self.fts_rows)
        if "FROM summary_index" in sql and "embedding <=>" in sql:
            return _Result(self.dense_rows)
        if "FROM summary_index" in sql:
            return _Result(self.text_rows)
        if "FROM kag_propositions" in sql:
            return _Result(self.prop_rows)
        return _Result()

    def rollback(self):
        self.rolled_back += 1


def _summary_row(
    sid, doc_id=1, level="section", chapter_id="s1", text="resumen", score=0.5
):
    return SimpleNamespace(
        id=sid,
        doc_id=doc_id,
        level=level,
        chapter_id=chapter_id,
        text=text,
        score=score,
    )


# ---------------------------------------------------------------------
# search_summaries
# ---------------------------------------------------------------------


def test_search_summaries_hybrid_rrf_returns_dicts_with_text():
    """Densa+FTS → RRF fusiona y devuelve dicts con text en orden de score."""
    u1, u2, u3 = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    session = _RetrievalSession(
        dense_rows=[
            _summary_row(u1, text="A", score=0.9),
            _summary_row(u2, text="B", score=0.8),
        ],
        fts_rows=[
            _summary_row(u2, text="B", score=0.7),
            _summary_row(u3, text="C", score=0.6),
        ],
        text_rows=[
            SimpleNamespace(
                id=u1, doc_id=1, level="section", chapter_id="s1", text="A"
            ),
            SimpleNamespace(
                id=u2, doc_id=1, level="section", chapter_id="s1", text="B"
            ),
            SimpleNamespace(
                id=u3, doc_id=1, level="section", chapter_id="s1", text="C"
            ),
        ],
    )
    hits = search_summaries(session, "pregunta", [0.1, 0.2], top_k=3)
    # RRF k=60: u2 (1/61+1/62) > u1 (1/61) > u3 (1/62)
    assert [h["id"] for h in hits] == [u2, u1, u3]
    assert [h["text"] for h in hits] == ["B", "A", "C"]
    assert all(
        set(h) >= {"id", "doc_id", "level", "chapter_id", "text", "score"} for h in hits
    )
    assert hits[0]["score"] > hits[1]["score"] > hits[2]["score"]


def test_search_summaries_embedding_none_uses_only_fts():
    """query_embedding=None → solo FTS (sin consulta densa)."""
    u1 = uuid.uuid4()
    session = _RetrievalSession(
        fts_rows=[_summary_row(u1, text="único", score=0.7)],
        text_rows=[
            SimpleNamespace(
                id=u1, doc_id=1, level="section", chapter_id="s1", text="único"
            )
        ],
    )
    hits = search_summaries(session, "pregunta", None, top_k=5)
    assert [h["id"] for h in hits] == [u1]
    assert hits[0]["text"] == "único"
    dense_calls = [c for c in session.calls if "embedding <=>" in c[0]]
    assert dense_calls == []


def test_search_summaries_level_filters():
    """level='document' → el filtro llega en los params de ambas capas."""
    u1 = uuid.uuid4()
    session = _RetrievalSession(
        dense_rows=[_summary_row(u1, level="document", text="D", score=0.9)],
        fts_rows=[_summary_row(u1, level="document", text="D", score=0.8)],
        text_rows=[
            SimpleNamespace(
                id=u1, doc_id=1, level="document", chapter_id=None, text="D"
            )
        ],
    )
    hits = search_summaries(session, "pregunta", [0.1, 0.2], top_k=5, level="document")
    assert [h["id"] for h in hits] == [u1]
    assert hits[0]["level"] == "document"
    for _sql, params in session.calls:
        if "FROM summary_index" in _sql and "ts_rank_cd" in _sql:
            assert params["level"] == "document"
        if "FROM summary_index" in _sql and "embedding <=>" in _sql:
            assert params["level"] == "document"


def test_search_summaries_missing_table_degrades():
    """Tabla summary_index ausente (migración 0032 no aplicada) → []."""
    session = _RetrievalSession(fail=True)
    assert search_summaries(session, "pregunta", [0.1, 0.2], top_k=5) == []
    assert session.rolled_back >= 1


def test_search_summaries_both_empty_returns_empty():
    """Sin hits densos ni FTS → [] (sin segundo SELECT)."""
    session = _RetrievalSession()
    assert search_summaries(session, "pregunta", [0.1, 0.2], top_k=5) == []
    text_calls = [c for c in session.calls if "id::text = ANY" in c[0]]
    assert text_calls == []


# ---------------------------------------------------------------------
# proposition_vector_search
# ---------------------------------------------------------------------


def test_proposition_vector_search_returns_dicts():
    """Búsqueda directa por embedding → dicts con chunk_id/statement/score."""
    session = _RetrievalSession(
        prop_rows=[
            _PropRow(
                id=1, chunk_id=10, doc_id=2, statement="La fórmula X.", score=0.91
            ),
            _PropRow(
                id=2, chunk_id=11, doc_id=2, statement="El teorema Y.", score=0.87
            ),
        ]
    )
    hits = proposition_vector_search(session, [0.1, 0.2], top_k=5)
    assert hits == [
        {
            "id": 1,
            "chunk_id": 10,
            "doc_id": 2,
            "statement": "La fórmula X.",
            "score": 0.91,
        },
        {
            "id": 2,
            "chunk_id": 11,
            "doc_id": 2,
            "statement": "El teorema Y.",
            "score": 0.87,
        },
    ]


def test_proposition_vector_search_embedding_none_returns_empty():
    """query_embedding=None → [] sin ejecutar SQL."""
    session = _RetrievalSession()
    assert proposition_vector_search(session, None, top_k=5) == []
    assert session.calls == []


def test_proposition_vector_search_missing_table_degrades():
    """Tabla kag_propositions ausente (migración 0024 no aplicada) → []."""
    session = _RetrievalSession(fail=True)
    assert proposition_vector_search(session, [0.1, 0.2], top_k=5) == []
    assert session.rolled_back >= 1
