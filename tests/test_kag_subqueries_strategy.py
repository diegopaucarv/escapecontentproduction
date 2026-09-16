"""Tests de la estrategia subqueries (descomposición en subconsultas) en
src/kag_query.py — sin DB real, sin torch/spacy/transformers.

Patrón de tests/test_kag_metadata_strategy.py: sesiones falsas + monkeypatch
de call_with_retries/_get_prompt_pair. `_with_own_session` se parchea con
`_patch_own_session` para que los workers del ThreadPoolExecutor usen la
sesión fake del test.

Cubre:
  - generate_subqueries: subconsultas parseadas (2-4), degradación a la
    original si el LLM lanza / JSON inválido / más de 4, KAG_SUBQUERIES=False
    → original sin llamar al LLM.
  - _retrieval_phase: devuelve el dict con las claves esperadas.
  - ask() con strategy "subqueries": genera subconsultas, corre
    _retrieval_phase por cada una, fusiona con rrf_merge y el flujo continúa.
  - Degradación: generate_subqueries devuelve la original → path normal.
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


def _llm_ok(subqueries, reason="porque sí"):
    """Fake de call_with_retries que devuelve el JSON del schema
    kag_query_subqueries."""

    def fake_call(
        session, *, prompt, system, model_size, response_format, retries, fallback_model
    ):
        return json.dumps({"subqueries": subqueries, "reason": reason}), "large", False

    return fake_call


def _llm_raises(exc=None):
    def fake_call(
        session, *, prompt, system, model_size, response_format, retries, fallback_model
    ):
        raise exc or RuntimeError("LLM no disponible")

    return fake_call


# ---------------------------------------------------------------------
# generate_subqueries
# ---------------------------------------------------------------------


def test_generate_subqueries_parses_subqueries(monkeypatch):
    """JSON válido → subconsultas parseadas (2-4)."""
    subs = [
        {"query": "¿qué dice Foucault sobre el poder?", "intent": "autor"},
        {"query": "¿qué dice Foucault sobre la disciplina?", "intent": "concepto"},
    ]
    monkeypatch.setattr(kq, "call_with_retries", _llm_ok(subs, "dos facetas"))
    session = _FakeSession()
    out = kq.generate_subqueries(
        session, "¿qué dice Foucault sobre poder y disciplina?"
    )
    assert out == subs


def test_generate_subqueries_single_atomic(monkeypatch):
    """Consulta atómica → UNA subconsulta (= la original)."""
    subs = [{"query": "¿qué es la epistemología?", "intent": "concepto"}]
    monkeypatch.setattr(kq, "call_with_retries", _llm_ok(subs))
    session = _FakeSession()
    out = kq.generate_subqueries(session, "¿qué es la epistemología?")
    assert out == subs


def test_generate_subqueries_llm_raises_degrades(monkeypatch):
    """call_with_retries lanza → [{"query": original, "intent": "consulta
    original"}] (nunca rompe)."""
    monkeypatch.setattr(kq, "call_with_retries", _llm_raises())
    session = _FakeSession()
    out = kq.generate_subqueries(session, "q original")
    assert out == [{"query": "q original", "intent": "consulta original"}]


def test_generate_subqueries_invalid_json_degrades(monkeypatch):
    """JSON inválido (parse_llm_output → {}) → original sin romper."""
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
    out = kq.generate_subqueries(session, "q original")
    assert out == [{"query": "q original", "intent": "consulta original"}]


def test_generate_subqueries_more_than_4_truncates(monkeypatch):
    """Más de 4 subconsultas → se truncan a 4 (no rompe)."""
    subs = [{"query": f"sub {i}", "intent": f"f{i}"} for i in range(6)]
    monkeypatch.setattr(kq, "call_with_retries", _llm_ok(subs))
    session = _FakeSession()
    out = kq.generate_subqueries(session, "q")
    assert len(out) == 4
    assert [s["query"] for s in out] == [f"sub {i}" for i in range(4)]


def test_generate_subqueries_empty_queries_degrades(monkeypatch):
    """Todas las queries vacías → original (degradación)."""
    monkeypatch.setattr(
        kq,
        "call_with_retries",
        _llm_ok([{"query": "  ", "intent": "x"}, {"query": "", "intent": "y"}]),
    )
    session = _FakeSession()
    out = kq.generate_subqueries(session, "q original")
    assert out == [{"query": "q original", "intent": "consulta original"}]


def test_generate_subqueries_flag_off_skips_llm(monkeypatch):
    """KAG_SUBQUERIES=False → original SIN llamar al LLM."""
    calls = []

    def _fake_call(
        session, *, prompt, system, model_size, response_format, retries, fallback_model
    ):
        calls.append(prompt)
        return (
            json.dumps({"subqueries": [{"query": "X", "intent": "x"}]}),
            "large",
            False,
        )

    def _config(session, name, default=None):
        if name == "KAG_SUBQUERIES":
            return False
        return default

    monkeypatch.setattr(kq, "call_with_retries", _fake_call)
    monkeypatch.setattr(kq, "_kag_config_value", _config)
    session = _FakeSession()
    out = kq.generate_subqueries(session, "q original")
    assert out == [{"query": "q original", "intent": "consulta original"}]
    assert calls == []


# ---------------------------------------------------------------------
# _retrieval_phase
# ---------------------------------------------------------------------


def test_retrieval_phase_returns_expected_dict(monkeypatch):
    """_retrieval_phase devuelve el dict con las claves esperadas."""
    session = _FakeSession()
    _patch_own_session(monkeypatch, session)

    def _fake_config(session, name, default=None):
        if name == "KAG_QUERY_PARALLEL":
            return False
        return default

    monkeypatch.setattr(kq, "_kag_config_value", _fake_config)
    monkeypatch.setattr(kq, "hybrid_search", lambda *a, **k: [(1, 0.5), (2, 0.3)])
    monkeypatch.setattr(
        kq, "critic_and_linking", lambda *a, **k: ([(3, 0.2)], ["t"], ["N"])
    )
    monkeypatch.setattr(
        kq, "apply_relevance_threshold", lambda hits, verbose=False: hits
    )
    monkeypatch.setattr(kq, "match_entities_candidates", lambda *a, **k: [[1]])
    monkeypatch.setattr(kq, "build_adjacency", lambda *a, **k: {})
    monkeypatch.setattr(kq, "disambiguate_by_cooccurrence", lambda *a, **k: [1])
    monkeypatch.setattr(kq, "personalized_pagerank_cached", lambda *a, **k: {1: 0.9})
    monkeypatch.setattr(kq, "ppr_entity_selection", lambda scores, verbose=False: [1])
    monkeypatch.setattr(
        kq, "chunks_for_entities", lambda *a, **k: [{"chunk_id": 4, "doc_id": 7}]
    )
    monkeypatch.setattr(
        kq,
        "rrf_merge",
        lambda *lists, k=60, top_k=20: [(4, 0.1), (1, 0.05), (3, 0.02)],
    )

    out = kq._retrieval_phase(session, "q", 8, [0.1, 0.2], verbose=False)
    assert set(out.keys()) == {
        "merged_hits",
        "vec_hits",
        "regex_hits",
        "ppr_chunks",
        "names",
        "groups",
        "entity_ids",
        "ppr_scores",
    }
    assert out["merged_hits"] == [(4, 0.1), (1, 0.05), (3, 0.02)]
    assert out["vec_hits"] == [(1, 0.5), (2, 0.3)]
    assert out["regex_hits"] == [(3, 0.2)]
    assert out["ppr_chunks"] == [{"chunk_id": 4, "doc_id": 7}]
    assert out["names"] == ["N"]
    assert out["groups"] == [[1]]
    assert out["entity_ids"] == [1]
    assert out["ppr_scores"] == {1: 0.9}


# ---------------------------------------------------------------------
# ask() con strategy "subqueries" — wiring
# ---------------------------------------------------------------------


def _install_ask_mocks(monkeypatch, session, strategy):
    """Mockea el flujo de ask() para aislar el wiring de la estrategia
    subqueries. Devuelve dict con los callables capturados."""
    captured = {"retrieval_queries": [], "rrf_lists": [], "subs": None}

    def _fake_config(session, name, default=None):
        if name == "KAG_QUERY_PARALLEL":
            return False
        if name == "KAG_BRANCH_B_MAX_ITERS":
            return 0
        return default

    def _fake_classify(session, query):
        return (
            strategy,
            ["dense", "fts", "paraphrase", "propositions", "graph"],
            "standard",
            "razón",
            False,
        )

    def _fake_generate(session, query):
        captured["subs"] = [
            {"query": "sub A", "intent": "f1"},
            {"query": "sub B", "intent": "f2"},
        ]
        return captured["subs"]

    def _fake_retrieval(
        session, query, k, q_emb, doc_ids=None, verbose=False, channels=None
    ):
        captured["retrieval_queries"].append(query)
        return {
            "merged_hits": [(1, 0.5)],
            "vec_hits": [(1, 0.5)],
            "regex_hits": [],
            "ppr_chunks": [],
            "names": [],
            "groups": [],
            "entity_ids": [1],
            "ppr_scores": {},
        }

    def _fake_rrf_merge(*lists, k=60, top_k=20):
        captured["rrf_lists"].append(lists)
        return [(1, 0.5)]

    def _fake_chunks_by_ids(session, chunk_ids):
        return [
            {
                "chunk_id": 1,
                "doc_id": 7,
                "chunk_index": 0,
                "content": "chunk 1",
            }
        ]

    monkeypatch.setattr(kq, "classify_query_strategy", _fake_classify)
    monkeypatch.setattr(kq, "_kag_config_value", _fake_config)
    monkeypatch.setattr(kq, "generate_subqueries", _fake_generate)
    monkeypatch.setattr(kq, "_retrieval_phase", _fake_retrieval)
    monkeypatch.setattr(kq, "rrf_merge", _fake_rrf_merge)
    monkeypatch.setattr(kq, "chunks_by_ids", _fake_chunks_by_ids)
    monkeypatch.setattr(kq, "subgraph_triples", lambda *a, **k: [])
    monkeypatch.setattr(kq, "figures_for_chunks", lambda *a, **k: [])
    monkeypatch.setattr(kq, "doc_summaries", lambda *a, **k: [])
    monkeypatch.setattr(kq, "propositions_for_chunks", lambda *a, **k: [])
    monkeypatch.setattr(kq, "generate_answer", lambda *a, **k: "respuesta")
    _patch_own_session(monkeypatch, session)
    return captured


def test_ask_subqueries_runs_retrieval_per_subquery(monkeypatch):
    """ask(): strategy 'subqueries' → generate_subqueries + _retrieval_phase
    por cada subconsulta + rrf_merge de los merged_hits + flujo continúa."""
    _fake_embeddings(monkeypatch)
    session = _FakeSession()
    captured = _install_ask_mocks(monkeypatch, session, "subqueries")
    answer = kq.ask(
        session,
        "¿qué dice Foucault sobre poder y disciplina?",
        top_k=8,
        global_top_k=20,
        verbose=False,
    )
    assert answer == "respuesta"
    assert captured["subs"] is not None
    assert captured["retrieval_queries"] == ["sub A", "sub B"]
    # rrf_merge recibe los merged_hits de cada subconsulta.
    assert len(captured["rrf_lists"]) == 1
    assert captured["rrf_lists"][0] == ([(1, 0.5)], [(1, 0.5)])


def test_ask_subqueries_degrades_to_normal_path(monkeypatch):
    """generate_subqueries devuelve la original → path normal (UNA
    _retrieval_phase con la consulta original, sin rrf_merge de subconsultas)."""
    _fake_embeddings(monkeypatch)
    session = _FakeSession()
    captured = {"retrieval_queries": [], "rrf_calls": 0}

    def _fake_config(session, name, default=None):
        if name == "KAG_QUERY_PARALLEL":
            return False
        if name == "KAG_BRANCH_B_MAX_ITERS":
            return 0
        return default

    def _fake_classify(session, query):
        return (
            "subqueries",
            ["dense", "fts", "paraphrase", "propositions", "graph"],
            "standard",
            "razón",
            False,
        )

    def _fake_generate(session, query):
        return [{"query": query, "intent": "consulta original"}]

    def _fake_retrieval(
        session, query, k, q_emb, doc_ids=None, verbose=False, channels=None
    ):
        captured["retrieval_queries"].append(query)
        return {
            "merged_hits": [(1, 0.5)],
            "vec_hits": [(1, 0.5)],
            "regex_hits": [],
            "ppr_chunks": [],
            "names": [],
            "groups": [],
            "entity_ids": [1],
            "ppr_scores": {},
        }

    def _fake_rrf_merge(*lists, k=60, top_k=20):
        captured["rrf_calls"] += 1
        return [(1, 0.5)]

    monkeypatch.setattr(kq, "classify_query_strategy", _fake_classify)
    monkeypatch.setattr(kq, "_kag_config_value", _fake_config)
    monkeypatch.setattr(kq, "generate_subqueries", _fake_generate)
    monkeypatch.setattr(kq, "_retrieval_phase", _fake_retrieval)
    monkeypatch.setattr(kq, "rrf_merge", _fake_rrf_merge)
    monkeypatch.setattr(
        kq,
        "chunks_by_ids",
        lambda *a, **k: [{"chunk_id": 1, "doc_id": 7, "chunk_index": 0}],
    )
    monkeypatch.setattr(kq, "subgraph_triples", lambda *a, **k: [])
    monkeypatch.setattr(kq, "figures_for_chunks", lambda *a, **k: [])
    monkeypatch.setattr(kq, "doc_summaries", lambda *a, **k: [])
    monkeypatch.setattr(kq, "propositions_for_chunks", lambda *a, **k: [])
    monkeypatch.setattr(kq, "generate_answer", lambda *a, **k: "respuesta")
    _patch_own_session(monkeypatch, session)

    answer = kq.ask(session, "q original", top_k=8, global_top_k=20, verbose=False)
    assert answer == "respuesta"
    # Path normal: UNA recuperación con la consulta original.
    assert captured["retrieval_queries"] == ["q original"]
    # rrf_merge NO se llama en ask() en el path normal (vive dentro de
    # _retrieval_phase); el path de subconsultas SÍ lo llama en ask().
    assert captured["rrf_calls"] == 0
