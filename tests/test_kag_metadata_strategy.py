"""Tests de la estrategia metadata (filtrado guiado por metadatos) en
src/kag_query.py — sin DB real, sin torch/spacy/transformers.

Patrón de tests/test_kag_query_strategy.py: sesiones falsas + monkeypatch de
call_with_retries/_get_prompt_pair. `_with_own_session` se parchea con
`_patch_own_session` para que los workers del ThreadPoolExecutor usen la
sesión fake del test.

Cubre:
  - extract_metadata_filters: filtros parseados (JSON válido), fallback a
    vacío si el LLM lanza o el JSON es inválido, KAG_METADATA_FILTER=False
    → no llama al LLM.
  - _doc_ids_for_filters: construye el SELECT correcto (verifica el SQL
    generado y el resultado con sesión fake).
  - hybrid_search con doc_ids filtra (verifica que el SQL incluye
    `doc_id = ANY`).
  - ask() con strategy "metadata" pasa doc_ids a hybrid_search.
"""

import json
import sys
import types

import src.kag_query as kq


class _Result:
    def __init__(self, rows=None, scalar=None):
        self._rows = rows or []
        self._scalar = scalar

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def scalars(self):
        return self

    def first(self):
        return self._rows[0] if self._rows else None

    def scalar(self):
        return self._scalar


class _FakeSession:
    """Sesión falsa mínima: execute devuelve _Result vacío (load_settings
    devuelve None → settings por defecto)."""

    def __init__(self):
        self.rolled_back = 0
        self.executed = []  # (sql, params) capturados

    def execute(self, stmt, params=None):
        self.executed.append((str(stmt), params))
        return _Result()

    def rollback(self):
        self.rolled_back += 1


class _DocIdsSession:
    """Sesión fake para _doc_ids_for_filters: devuelve ids de kag_documents."""

    def __init__(self, ids):
        self._ids = ids
        self.executed = []
        self.rolled_back = 0

    def execute(self, stmt, params=None):
        self.executed.append((str(stmt), params))
        return _Result(rows=[types.SimpleNamespace(id=i) for i in self._ids])

    def rollback(self):
        self.rolled_back += 1


def _patch_own_session(monkeypatch, session):
    """Los workers de ask()/_ask_audited() abren su propia sesión real
    (run_in_own_session, src/db/session.py) — sin DB real en tests, se
    parchea para que usen la sesión fake del test directamente."""
    monkeypatch.setattr(
        kq,
        "_with_own_session",
        lambda fn, *a, **k: fn(session, *a, **k),
    )


def _fake_embeddings(monkeypatch):
    """Inyecta src.embeddings fake: ask()/_ask_audited() hacen `from
    src.embeddings import embed_text` en caliente — el módulo falso evita
    torch/transformers."""
    fake_emb = types.ModuleType("src.embeddings")
    fake_emb.embed_text = lambda query, input_type="query": [0.1, 0.2]
    fake_emb.embed_texts = lambda texts, **k: [[0.1, 0.2] for _ in texts]
    monkeypatch.setitem(sys.modules, "src.embeddings", fake_emb)


def _llm_ok(filters, reason="porque sí"):
    """Fake de call_with_retries que devuelve el JSON del schema
    kag_query_metadata."""

    def fake_call(
        session, *, prompt, system, model_size, response_format, retries, fallback_model
    ):
        return json.dumps({"filters": filters, "reason": reason}), "large", False

    return fake_call


def _llm_raises(exc=None):
    def fake_call(
        session, *, prompt, system, model_size, response_format, retries, fallback_model
    ):
        raise exc or RuntimeError("LLM no disponible")

    return fake_call


# ---------------------------------------------------------------------
# extract_metadata_filters
# ---------------------------------------------------------------------


def test_extract_metadata_filters_parses_filters(monkeypatch):
    """JSON válido → filtros parseados + reason."""
    filters = {
        "authors": ["Foucault"],
        "years": ["1975"],
        "fields": ["Filosofía"],
        "works": ["Vigilar y castigar"],
        "languages": ["es"],
    }
    monkeypatch.setattr(kq, "call_with_retries", _llm_ok(filters, "por autor y año"))
    session = _FakeSession()
    out = kq.extract_metadata_filters(session, "¿qué dice Foucault en 1975?")
    assert out["filters"] == filters
    assert out["reason"] == "por autor y año"


def test_extract_metadata_filters_empty_filters(monkeypatch):
    """Sin filtros → arrays vacíos (no rompe)."""
    monkeypatch.setattr(
        kq,
        "call_with_retries",
        _llm_ok(
            {"authors": [], "years": [], "fields": [], "works": [], "languages": []}
        ),
    )
    session = _FakeSession()
    out = kq.extract_metadata_filters(session, "¿de qué trata el libro?")
    assert out["filters"] == {
        "authors": [],
        "years": [],
        "fields": [],
        "works": [],
        "languages": [],
    }


def test_extract_metadata_filters_llm_raises_degrades(monkeypatch):
    """call_with_retries lanza → {"filters": {}, "reason": ""} (nunca rompe)."""
    monkeypatch.setattr(kq, "call_with_retries", _llm_raises())
    session = _FakeSession()
    out = kq.extract_metadata_filters(session, "q")
    assert out == {"filters": {}, "reason": ""}


def test_extract_metadata_filters_invalid_json_degrades(monkeypatch):
    """JSON inválido (parse_llm_output → {}) → vacío sin romper."""
    monkeypatch.setattr(
        kq,
        "call_with_retries",
        lambda session, *, prompt, system, model_size, response_format, retries, fallback_model: (
            "esto no es json",
            "large",
            False,
        ),
    )
    session = _FakeSession()
    out = kq.extract_metadata_filters(session, "q")
    assert out == {"filters": {}, "reason": ""}


def test_extract_metadata_filters_filters_not_dict_degrades(monkeypatch):
    """filters no dict (p.ej. lista) → {} sin romper."""
    monkeypatch.setattr(
        kq,
        "call_with_retries",
        lambda session, *, prompt, system, model_size, response_format, retries, fallback_model: (
            json.dumps({"filters": ["no", "soy", "dict"], "reason": "x"}),
            "large",
            False,
        ),
    )
    session = _FakeSession()
    out = kq.extract_metadata_filters(session, "q")
    assert out == {"filters": {}, "reason": "x"}


def test_extract_metadata_filters_flag_off_skips_llm(monkeypatch):
    """KAG_METADATA_FILTER=False → vacío SIN llamar al LLM."""
    calls = []

    def _fake_call(
        session, *, prompt, system, model_size, response_format, retries, fallback_model
    ):
        calls.append(prompt)
        return (
            json.dumps({"filters": {"authors": ["X"]}, "reason": "x"}),
            "large",
            False,
        )

    def _config(session, name, default=None):
        if name == "KAG_METADATA_FILTER":
            return False
        return default

    monkeypatch.setattr(kq, "call_with_retries", _fake_call)
    monkeypatch.setattr(kq, "_kag_config_value", _config)
    session = _FakeSession()
    out = kq.extract_metadata_filters(session, "q")
    assert out == {"filters": {}, "reason": ""}
    assert calls == []


# ---------------------------------------------------------------------
# _doc_ids_for_filters
# ---------------------------------------------------------------------


def test_doc_ids_for_filters_builds_select_with_ors():
    """Construye UN SELECT con ORs sobre kag_documents (works → title ILIKE,
    authors/years → bibtex ILIKE, fields → lcsh @> O EXISTS, languages →
    language = ANY)."""
    session = _DocIdsSession([11, 22])
    filters = {
        "authors": ["Foucault"],
        "years": ["1975"],
        "fields": ["Filosofía"],
        "works": ["Vigilar y castigar"],
        "languages": ["es"],
    }
    out = kq._doc_ids_for_filters(session, filters)
    assert out == [11, 22]
    assert len(session.executed) == 1
    sql, params = session.executed[0]
    assert "FROM kag_documents" in sql
    assert "status = 'ready'" in sql
    assert "title ILIKE '%' || :w0 || '%'" in sql
    assert "ficha_jsonb->>'bibtex' ILIKE '%' || :a0 || '%'" in sql
    assert "ficha_jsonb->>'bibtex' ILIKE '%' || :a1 || '%'" in sql
    assert "ficha_jsonb->'library_of_congress'->'lcsh_terms' @> :f0" in sql
    assert "thematic_areas_iso25964" in sql
    assert "language = ANY(:langs)" in sql
    assert params["w0"] == "Vigilar y castigar"
    assert params["a0"] == "Foucault"
    assert params["a1"] == "1975"
    assert params["f0"] == ["Filosofía"]
    assert params["fl0"] == "Filosofía"
    assert params["langs"] == ["es"]


def test_doc_ids_for_filters_empty_filters_returns_empty():
    """filters vacío → [] sin SELECT."""
    session = _DocIdsSession([1])
    out = kq._doc_ids_for_filters(session, {})
    assert out == []
    assert session.executed == []


def test_doc_ids_for_filters_all_empty_lists_returns_empty():
    """filters con arrays vacíos → [] sin SELECT."""
    session = _DocIdsSession([1])
    filters = {
        "authors": [],
        "years": [],
        "fields": [],
        "works": [],
        "languages": [],
    }
    out = kq._doc_ids_for_filters(session, filters)
    assert out == []
    assert session.executed == []


def test_doc_ids_for_filters_sql_error_degrades():
    """Error SQL → rollback + [] sin romper."""

    class _BoomSession:
        def __init__(self):
            self.rolled_back = 0

        def execute(self, stmt, params=None):
            raise RuntimeError("tabla no existe")

        def rollback(self):
            self.rolled_back += 1

    session = _BoomSession()
    out = kq._doc_ids_for_filters(session, {"authors": ["Foucault"]})
    assert out == []
    assert session.rolled_back == 1


# ---------------------------------------------------------------------
# hybrid_search con doc_ids
# ---------------------------------------------------------------------


def test_hybrid_search_with_doc_ids_filters_channels(monkeypatch):
    """doc_ids no vacío → TODOS los canales SQL agregan `doc_id = ANY`."""
    session = _FakeSession()

    def _fake_config(session, name, default=None):
        if name in ("KAG_PARAPHRASE_CHANNEL", "KAG_PROPOSITION_CHANNEL"):
            return True
        return default

    monkeypatch.setattr(kq, "_kag_config_value", _fake_config)
    monkeypatch.setattr(kq, "embedding_to_sql", lambda q: "[0.1,0.2]")
    monkeypatch.setattr(kq, "rrf_merge", lambda *lists, k=60, top_k=20: [])
    _patch_own_session(monkeypatch, session)

    kq.hybrid_search(
        session, "query", [0.1, 0.2], top_k=8, verbose=False, doc_ids=[11, 22]
    )
    sqls = [sql for sql, _p in session.executed]
    assert any("doc_id = ANY(:doc_ids)" in sql for sql in sqls)
    # Los 4 canales corren: densa, FTS, paráfrasis, proposiciones.
    assert len(sqls) == 4
    for sql in sqls:
        assert "doc_id = ANY(:doc_ids)" in sql
    # El parámetro doc_ids se pasa como lista plana.
    for _sql, params in session.executed:
        assert params["doc_ids"] == [11, 22]


def test_hybrid_search_without_doc_ids_no_filter(monkeypatch):
    """doc_ids None → comportamiento actual (sin `doc_id = ANY`)."""
    session = _FakeSession()

    def _fake_config(session, name, default=None):
        if name in ("KAG_PARAPHRASE_CHANNEL", "KAG_PROPOSITION_CHANNEL"):
            return True
        return default

    monkeypatch.setattr(kq, "_kag_config_value", _fake_config)
    monkeypatch.setattr(kq, "embedding_to_sql", lambda q: "[0.1,0.2]")
    monkeypatch.setattr(kq, "rrf_merge", lambda *lists, k=60, top_k=20: [])
    _patch_own_session(monkeypatch, session)

    kq.hybrid_search(session, "query", [0.1, 0.2], top_k=8, verbose=False)
    sqls = [sql for sql, _p in session.executed]
    assert len(sqls) == 4
    for sql in sqls:
        assert "doc_id = ANY" not in sql


# ---------------------------------------------------------------------
# ask() con strategy "metadata" — wiring
# ---------------------------------------------------------------------


def test_ask_metadata_passes_doc_ids_to_hybrid_search(monkeypatch):
    """ask(): strategy 'metadata' → extract_metadata_filters +
    _doc_ids_for_filters → hybrid_search recibe doc_ids."""
    _fake_embeddings(monkeypatch)
    session = _FakeSession()
    captured = {"doc_ids": None, "hybrid_k": []}

    def _fake_config(session, name, default=None):
        if name == "KAG_QUERY_PARALLEL":
            return False
        if name == "KAG_BRANCH_B_MAX_ITERS":
            return 0
        return default

    def _fake_classify(session, query):
        return "metadata", ["dense", "fts"], "standard", "razón", False

    def _fake_extract(session, query):
        return {"filters": {"authors": ["Foucault"]}, "reason": "por autor"}

    def _fake_doc_ids(session, filters):
        return [11, 22]

    def _fake_hybrid(
        session, query, q_emb, k, verbose=False, doc_ids=None, channels=None
    ):
        captured["doc_ids"] = doc_ids
        captured["hybrid_k"].append(k)
        return []

    def _fake_rrf_merge(*lists, k=60, top_k=20):
        return []

    monkeypatch.setattr(kq, "classify_query_strategy", _fake_classify)
    monkeypatch.setattr(kq, "_kag_config_value", _fake_config)
    monkeypatch.setattr(kq, "extract_metadata_filters", _fake_extract)
    monkeypatch.setattr(kq, "_doc_ids_for_filters", _fake_doc_ids)
    monkeypatch.setattr(kq, "hybrid_search", _fake_hybrid)
    monkeypatch.setattr(kq, "critic_and_linking", lambda *a, **k: ([], [], []))
    monkeypatch.setattr(kq, "rrf_merge", _fake_rrf_merge)
    monkeypatch.setattr(kq, "chunks_by_ids", lambda *a, **k: [])
    monkeypatch.setattr(kq, "figures_for_chunks", lambda *a, **k: [])
    monkeypatch.setattr(kq, "doc_summaries", lambda *a, **k: [])
    monkeypatch.setattr(kq, "propositions_for_chunks", lambda *a, **k: [])
    monkeypatch.setattr(kq, "generate_answer", lambda *a, **k: "respuesta")
    _patch_own_session(monkeypatch, session)

    kq.ask(
        session, "¿qué dice Foucault en 1975?", top_k=8, global_top_k=20, verbose=False
    )
    assert captured["doc_ids"] == [11, 22]
    assert captured["hybrid_k"] == [8]


def test_ask_metadata_empty_filters_no_doc_ids(monkeypatch):
    """ask(): strategy 'metadata' pero sin filtros → hybrid_search sin
    doc_ids (comportamiento local normal)."""
    _fake_embeddings(monkeypatch)
    session = _FakeSession()
    captured = {"doc_ids": "unset"}

    def _fake_config(session, name, default=None):
        if name == "KAG_QUERY_PARALLEL":
            return False
        if name == "KAG_BRANCH_B_MAX_ITERS":
            return 0
        return default

    def _fake_classify(session, query):
        return "metadata", ["dense", "fts"], "standard", "razón", False

    def _fake_extract(session, query):
        return {"filters": {}, "reason": ""}

    def _fake_doc_ids(session, filters):
        return []

    def _fake_hybrid(
        session, query, q_emb, k, verbose=False, doc_ids=None, channels=None
    ):
        captured["doc_ids"] = doc_ids
        return []

    def _fake_rrf_merge(*lists, k=60, top_k=20):
        return []

    monkeypatch.setattr(kq, "classify_query_strategy", _fake_classify)
    monkeypatch.setattr(kq, "_kag_config_value", _fake_config)
    monkeypatch.setattr(kq, "extract_metadata_filters", _fake_extract)
    monkeypatch.setattr(kq, "_doc_ids_for_filters", _fake_doc_ids)
    monkeypatch.setattr(kq, "hybrid_search", _fake_hybrid)
    monkeypatch.setattr(kq, "critic_and_linking", lambda *a, **k: ([], [], []))
    monkeypatch.setattr(kq, "rrf_merge", _fake_rrf_merge)
    monkeypatch.setattr(kq, "chunks_by_ids", lambda *a, **k: [])
    monkeypatch.setattr(kq, "figures_for_chunks", lambda *a, **k: [])
    monkeypatch.setattr(kq, "doc_summaries", lambda *a, **k: [])
    monkeypatch.setattr(kq, "propositions_for_chunks", lambda *a, **k: [])
    monkeypatch.setattr(kq, "generate_answer", lambda *a, **k: "respuesta")
    _patch_own_session(monkeypatch, session)

    kq.ask(
        session, "¿qué dice Foucault en 1975?", top_k=8, global_top_k=20, verbose=False
    )
    assert captured["doc_ids"] is None
