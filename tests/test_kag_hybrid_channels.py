"""Tests de canales híbridos §3.3: paráfrasis FTS y proposiciones DENTRO de
hybrid_search, y separación de evidencias en assemble_context.

Sin DB real: sesiones falsas + monkeypatch (patrón de
tests/test_kag_retrieval.py y tests/test_kag_document_propositions.py).
`_with_own_session` se parchea con `_patch_own_session` para que los
workers del ThreadPoolExecutor usen la sesión fake del test.
"""

import sys
import types
from types import SimpleNamespace

import pytest
from sqlalchemy.exc import ProgrammingError

import src.kag_query as kq
from src.kag_query import assemble_context, hybrid_search


class _Result:
    def __init__(self, rows=None):
        self._rows = rows if rows is not None else []

    def fetchall(self):
        return self._rows


class _HybridSession:
    """Sesión falsa: despacha por el SQL y registra las llamadas.

    - paraphrase_tsv → filas del canal de paráfrasis
    - kag_propositions → filas de proposiciones
    - fail_paraphrase=True → ProgrammingError en el SQL de paráfrasis
      (columna ausente, migración 0033 no aplicada)
    """

    def __init__(self, para_rows=None, prop_rows=None, fail_paraphrase=False):
        self.para_rows = para_rows or []
        self.prop_rows = prop_rows or []
        self.fail_paraphrase = fail_paraphrase
        self.calls = []
        self.rolled_back = 0

    def execute(self, stmt, params=None):
        self.calls.append((str(stmt), params))
        sql = str(stmt)
        if "paraphrase_tsv" in sql:
            if self.fail_paraphrase:
                raise ProgrammingError(
                    "stmt", {}, Exception("column paraphrase_tsv does not exist")
                )
            return _Result(self.para_rows)
        if "FROM kag_propositions" in sql:
            return _Result(self.prop_rows)
        return _Result()

    def rollback(self):
        self.rolled_back += 1


class _FakeSession:
    """Sesión falsa mínima para ask()/_ask_audited(): execute vacío."""

    def __init__(self):
        self.rolled_back = 0

    def execute(self, stmt, params=None):
        return _Result()

    def rollback(self):
        self.rolled_back += 1


def _patch_own_session(monkeypatch, session):
    """Los workers de hybrid_search abren su propia sesión real
    (run_in_own_session, src/db/session.py) — sin DB real en tests, se
    parchea para que usen la sesión fake del test directamente."""
    monkeypatch.setattr(
        kq,
        "_with_own_session",
        lambda fn, *a, **k: fn(session, *a, **k),
    )


def _para_row(cid, score=0.7):
    return SimpleNamespace(id=cid, score=score)


class _PropRow:
    """Fila con _mapping (como sqlalchemy Row) para dict(r._mapping)."""

    def __init__(self, **kw):
        self._mapping = kw


def _config_all_on(session, name, default=None):
    return default


def _install_hybrid_mocks(monkeypatch, session, prop_hits=None, config=None):
    """Mockea vector_search/fts_search/proposition_vector_search y
    _kag_config_value para aislar los canales nuevos de hybrid_search.
    Devuelve dict con los callables capturados."""
    captured = {"prop_calls": []}

    def _fake_vector(session, q_emb, top_k):
        return [(1, 0.9), (2, 0.8)]

    def _fake_fts(session, query_text, top_k):
        return [(2, 5.0), (3, 4.0)]

    def _fake_prop_search(session, q_emb, top_k):
        captured["prop_calls"].append((q_emb, top_k))
        return prop_hits if prop_hits is not None else []

    monkeypatch.setattr(kq, "vector_search", _fake_vector)
    monkeypatch.setattr(kq, "fts_search", _fake_fts)
    monkeypatch.setattr(kq, "proposition_vector_search", _fake_prop_search)
    monkeypatch.setattr(kq, "_kag_config_value", config or _config_all_on)
    _patch_own_session(monkeypatch, session)
    return captured


# ---------------------------------------------------------------------
# hybrid_search — canal de paráfrasis (FTS sobre paraphrase_tsv)
# ---------------------------------------------------------------------


def test_hybrid_search_includes_paraphrase_channel(monkeypatch):
    """El canal de paráfrasis (FTS sobre paraphrase_tsv) entra al RRF."""
    session = _HybridSession(para_rows=[_para_row(4, 0.6), _para_row(5, 0.5)])
    _install_hybrid_mocks(monkeypatch, session)
    hits = hybrid_search(session, "pregunta", [0.1, 0.2], top_k=10)
    ids = [cid for cid, _ in hits]
    assert 4 in ids and 5 in ids
    # El SQL de paráfrasis se ejecutó con la query y el top_k.
    para_calls = [c for c in session.calls if "paraphrase_tsv" in c[0]]
    assert len(para_calls) == 1
    assert para_calls[0][1]["q"] == "pregunta"
    assert para_calls[0][1]["top_k"] == 10


def test_hybrid_search_paraphrase_channel_missing_column_degrades(monkeypatch):
    """Columna paraphrase_tsv ausente (migración 0033 no aplicada) → el canal
    degrada a [] sin romper: densa + FTS siguen en el RRF."""
    session = _HybridSession(fail_paraphrase=True)
    _install_hybrid_mocks(monkeypatch, session)
    hits = hybrid_search(session, "pregunta", [0.1, 0.2], top_k=10)
    ids = [cid for cid, _ in hits]
    assert 1 in ids and 2 in ids and 3 in ids
    assert session.rolled_back >= 1


def test_hybrid_search_paraphrase_channel_disabled_by_config(monkeypatch):
    """KAG_PARAPHRASE_CHANNEL=False apaga el canal de paráfrasis."""
    session = _HybridSession(para_rows=[_para_row(4, 0.6)])

    def _config(session, name, default=None):
        if name == "KAG_PARAPHRASE_CHANNEL":
            return False
        return default

    _install_hybrid_mocks(monkeypatch, session, config=_config)
    hits = hybrid_search(session, "pregunta", [0.1, 0.2], top_k=10)
    ids = [cid for cid, _ in hits]
    assert 4 not in ids
    para_calls = [c for c in session.calls if "paraphrase_tsv" in c[0]]
    assert para_calls == []


# ---------------------------------------------------------------------
# hybrid_search — canal de proposiciones (semántico)
# ---------------------------------------------------------------------


def test_hybrid_search_includes_proposition_channel(monkeypatch):
    """El canal de proposiciones (proposition_vector_search) entra al RRF;
    los hits sin chunk_id se descartan."""
    session = _HybridSession()
    prop_hits = [
        {"id": "p1", "chunk_id": 6, "doc_id": 1, "statement": "s", "score": 0.9},
        {"id": "p2", "chunk_id": None, "doc_id": 1, "statement": "s2", "score": 0.8},
    ]
    captured = _install_hybrid_mocks(monkeypatch, session, prop_hits=prop_hits)
    hits = hybrid_search(session, "pregunta", [0.1, 0.2], top_k=10)
    ids = [cid for cid, _ in hits]
    assert 6 in ids
    assert len(captured["prop_calls"]) == 1


def test_hybrid_search_proposition_channel_embedding_none(monkeypatch):
    """query_embedding=None → el canal de proposiciones no aporta hits
    (proposition_vector_search degrada a []); FTS + paráfrasis siguen."""
    session = _HybridSession(para_rows=[_para_row(4, 0.6)])
    captured = _install_hybrid_mocks(monkeypatch, session)
    hits = hybrid_search(session, "pregunta", None, top_k=10)
    ids = [cid for cid, _ in hits]
    assert 4 in ids  # paráfrasis sigue activa
    assert captured["prop_calls"] == [(None, 10)]


def test_hybrid_search_proposition_channel_disabled_by_config(monkeypatch):
    """KAG_PROPOSITION_CHANNEL=False apaga el canal de proposiciones."""
    session = _HybridSession()

    def _config(session, name, default=None):
        if name == "KAG_PROPOSITION_CHANNEL":
            return False
        return default

    captured = _install_hybrid_mocks(monkeypatch, session, config=_config)
    hits = hybrid_search(session, "pregunta", [0.1, 0.2], top_k=10)
    assert captured["prop_calls"] == []
    assert hits  # densa + FTS siguen


# ---------------------------------------------------------------------
# assemble_context — tres bloques de evidencia separados
# ---------------------------------------------------------------------


def _chunk(cid="c1", doc_id=1, chapter_id="s1"):
    return {
        "chunk_id": cid,
        "doc_id": doc_id,
        "chapter_id": chapter_id,
        "doc_path": "a.md",
        "chapter_title": "## Intro",
        "chunk_index": 0,
        "content": "contenido a",
        "is_anchor": True,
        "score": 0.5,
    }


def _proposition(pid="p1"):
    return {
        "chunk_id": pid,
        "doc_title": "a.md",
        "chapter_title": "## Intro",
        "statement": "Hecho atómico.",
        "text_span": "Hecho atómico.",
    }


def _summary_hit(doc_id=1, level="document", chapter_id=None, text="marco"):
    return {
        "id": f"s-{doc_id}",
        "doc_id": doc_id,
        "level": level,
        "chapter_id": chapter_id,
        "text": text,
        "score": 0.8,
    }


def test_assemble_context_three_blocks_separated():
    """Las tres capas aparecen como bloques separados con los encabezados
    exactos, en orden marco → evidencia → capa atómica."""
    ctx = assemble_context(
        [_chunk()],
        [],
        [],
        [],
        "pregunta",
        propositions=[_proposition()],
        summary_hits=[_summary_hit()],
    )
    marco = ctx.index("--- MARCO TEMÁTICO (resúmenes de documento/sección) ---")
    evidencia = ctx.index("--- EVIDENCIA TEXTUAL (chunks con cita) ---")
    atomica = ctx.index("--- CAPA ATÓMICA (proposiciones) ---")
    assert marco < evidencia < atomica
    # El marco NO se mezcla dentro del bloque de chunks.
    assert "marco" in ctx[marco:evidencia]
    assert "marco" not in ctx[evidencia:atomica]


def test_assemble_context_evidence_blocks_include_citation_fields():
    """El bloque de chunks lleva doc_id/chapter_id para citar; el de
    proposiciones lleva chunk_id y span."""
    ctx = assemble_context(
        [_chunk(cid="c1", doc_id=7, chapter_id="s2")],
        [],
        [],
        [],
        "q",
        propositions=[_proposition(pid="p9")],
    )
    assert "doc_id: 7" in ctx
    assert "chapter_id: s2" in ctx
    assert "chunk_id: p9" in ctx
    assert "cita: Hecho atómico." in ctx


def test_assemble_context_marco_placeholder_when_empty():
    """summary_hits=[] → bloque MARCO con placeholder; propositions=[] →
    bloque CAPA ATÓMICA con placeholder."""
    ctx = assemble_context([], [], [], [], "q", propositions=[], summary_hits=[])
    assert "--- MARCO TEMÁTICO (resúmenes de documento/sección) ---" in ctx
    assert "(sin resúmenes recuperados)" in ctx
    assert "--- CAPA ATÓMICA (proposiciones) ---" in ctx
    assert "(sin proposiciones)" in ctx


# ---------------------------------------------------------------------
# ask()/_ask_audited() — el rrf_merge final ya no recibe prop_channel
# ---------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fake_embeddings(monkeypatch):
    """Inyecta src.embeddings fake: ask()/_ask_audited() hacen `from
    src.embeddings import embed_text` en caliente — el módulo falso evita
    torch/transformers."""
    fake_emb = types.ModuleType("src.embeddings")
    fake_emb.embed_text = lambda query, input_type="query": [0.1, 0.2]
    fake_emb.embed_texts = lambda texts, **k: [[0.1, 0.2] for _ in texts]
    monkeypatch.setitem(sys.modules, "src.embeddings", fake_emb)


def test_ask_final_rrf_merge_has_no_prop_channel(monkeypatch):
    """ask() ya no agrega prop_channel como lista extra en el rrf_merge final
    (el canal de proposiciones vive dentro de hybrid_search)."""
    captured = {}

    def _fake_config(session, name, default=None):
        if name == "KAG_QUERY_PARALLEL":
            return False
        return default

    def _fake_rrf_merge(*lists, k=60, top_k=20):
        captured["merge_lists"] = lists
        return []

    monkeypatch.setattr(kq, "_kag_config_value", _fake_config)
    monkeypatch.setattr(kq, "rrf_merge", _fake_rrf_merge)
    monkeypatch.setattr(kq, "hybrid_search", lambda *a, **k: [])
    monkeypatch.setattr(kq, "critic_and_linking", lambda *a, **k: ([], [], []))
    monkeypatch.setattr(kq, "chunks_by_ids", lambda *a, **k: [])
    monkeypatch.setattr(kq, "figures_for_chunks", lambda *a, **k: [])
    monkeypatch.setattr(kq, "doc_summaries", lambda *a, **k: [])
    monkeypatch.setattr(kq, "propositions_for_chunks", lambda *a, **k: [])
    monkeypatch.setattr(kq, "generate_answer", lambda *a, **k: "respuesta")
    monkeypatch.setattr(
        kq,
        "classify_query_strategy",
        lambda session, query: (
            "hierarchical",
            ["dense", "fts"],
            "standard",
            "",
            False,
        ),
    )

    session = _FakeSession()
    kq.ask(session, "¿De qué trata el libro?", verbose=False)

    assert captured["merge_lists"] is not None
    # vec, regex, ppr — sin prop_channel como lista extra.
    assert len(captured["merge_lists"]) == 3


def test_ask_audited_final_rrf_merge_has_no_prop_channel(monkeypatch):
    """_ask_audited() ya no agrega prop_channel como lista extra en el
    rrf_merge final (el canal de proposiciones vive dentro de hybrid_search)."""
    captured = {}

    def _fake_config(session, name, default=None):
        if name == "KAG_QUERY_PARALLEL":
            return False
        if name == "KAG_BRANCH_B_MAX_ITERS":
            return 0
        return default

    def _fake_rrf_merge(*lists, k=60, top_k=20):
        captured["merge_lists"] = lists
        return []

    def _fake_audit(session, query, propositions, corpus_metadata, **kw):
        return (
            [],
            {"contradictions_detected": False, "analysis_cases": []},
            {"verdict": "SUFFICIENT_FOR_SYNTHESIS", "confidence_score": 0.9},
        )

    def _fake_call(
        session, *, prompt, system, model_size, response_format, retries, fallback_model
    ):
        return "respuesta", "model", False

    monkeypatch.setattr(kq, "_kag_config_value", _fake_config)
    monkeypatch.setattr(kq, "rrf_merge", _fake_rrf_merge)
    monkeypatch.setattr(kq, "hybrid_search", lambda *a, **k: [])
    monkeypatch.setattr(kq, "critic_and_linking", lambda *a, **k: ([], [], []))
    monkeypatch.setattr(kq, "chunks_by_ids", lambda *a, **k: [])
    monkeypatch.setattr(kq, "propositions_for_chunks", lambda *a, **k: [])
    monkeypatch.setattr(kq, "_audit_epistemic_fused", _fake_audit)
    monkeypatch.setattr(kq, "call_with_retries", _fake_call)
    monkeypatch.setattr(
        kq,
        "classify_query_strategy",
        lambda session, query: (
            "hierarchical",
            ["dense", "fts"],
            "standard",
            "",
            False,
        ),
    )

    session = _FakeSession()
    kq._ask_audited(session, "¿De qué trata el libro?", verbose=False)

    assert result["answer"] == "respuesta"
    assert captured["merge_lists"] is not None
    # vec, regex, ppr — sin prop_channel como lista extra.
    assert len(captured["merge_lists"]) == 3
