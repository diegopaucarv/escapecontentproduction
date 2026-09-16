"""Tests del channel selector: _top_k_for, hybrid_search con channels y el
gate de search_summaries en ask().

Sin DB real: sesiones falsas + monkeypatch (patrón de
tests/test_kag_document_propositions.py). `_with_own_session` se parchea
con `_patch_own_session` para que los workers del ThreadPoolExecutor usen
la sesión fake del test.
"""

import sys
import types

import pytest

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

    def execute(self, stmt, params=None):
        return _Result()

    def rollback(self):
        self.rolled_back += 1


def _patch_own_session(monkeypatch, session):
    """Los workers de ask()/hybrid_search() abren su propia sesión real
    (run_in_own_session, src/db/session.py) — sin DB real en tests, se
    parchea para que usen la sesión fake del test directamente."""
    monkeypatch.setattr(
        kq,
        "_with_own_session",
        lambda fn, *a, **k: fn(session, *a, **k),
    )


def _fake_embeddings(monkeypatch):
    """Inyecta src.embeddings fake: ask() hace `from src.embeddings import
    embed_text` en caliente — el módulo falso evita torch/transformers."""
    fake_emb = types.ModuleType("src.embeddings")
    fake_emb.embed_text = lambda query, input_type="query": [0.1, 0.2]
    fake_emb.embed_texts = lambda texts, **k: [[0.1, 0.2] for _ in texts]
    monkeypatch.setitem(sys.modules, "src.embeddings", fake_emb)


# ---------------------------------------------------------------------
# _top_k_for
# ---------------------------------------------------------------------


def test_top_k_for_narrow():
    """narrow → acotado a [3, 6]."""
    assert kq._top_k_for("graph", "narrow", 8, 20) == 6
    assert kq._top_k_for("graph", "narrow", 4, 20) == 4
    assert kq._top_k_for("graph", "narrow", 2, 20) == 3


def test_top_k_for_standard():
    """standard → top_k del caller."""
    assert kq._top_k_for("graph", "standard", 8, 20) == 8


def test_top_k_for_wide():
    """wide → global_top_k (marco temático amplio)."""
    assert kq._top_k_for("hierarchical", "wide", 8, 20) == 20


# ---------------------------------------------------------------------
# hybrid_search con channels
# ---------------------------------------------------------------------


def _install_hybrid_channel_mocks(monkeypatch, session):
    """Mockea los 4 canales de hybrid_search y rrf_merge; devuelve dict con
    los contadores de llamadas."""
    calls = {"dense": 0, "fts": 0, "para": 0, "prop": 0}

    def _fake_vector(session, embedding, top_k, **kw):
        calls["dense"] += 1
        return [(1, 0.9)]

    def _fake_fts(session, query, top_k, **kw):
        calls["fts"] += 1
        return [(2, 0.8)]

    def _fake_para(session, query, top_k, **kw):
        calls["para"] += 1
        return [(3, 0.7)]

    def _fake_prop(session, embedding, top_k, **kw):
        calls["prop"] += 1
        return [{"chunk_id": 4, "score": 0.6}]

    monkeypatch.setattr(kq, "vector_search", _fake_vector)
    monkeypatch.setattr(kq, "fts_search", _fake_fts)
    monkeypatch.setattr(kq, "_paraphrase_fts_search", _fake_para)
    monkeypatch.setattr(kq, "proposition_vector_search", _fake_prop)
    monkeypatch.setattr(kq, "rrf_merge", lambda *lists, k=60, top_k=20: list(lists))
    _patch_own_session(monkeypatch, session)
    return calls


def test_hybrid_search_channels_dense_fts_skips_others(monkeypatch):
    """channels=["dense","fts"] → NO llama a _paraphrase_fts_search ni
    proposition_vector_search."""
    session = _FakeSession()
    calls = _install_hybrid_channel_mocks(monkeypatch, session)
    kq.hybrid_search(session, "q", [0.1, 0.2], 8, channels=["dense", "fts"])
    assert calls == {"dense": 1, "fts": 1, "para": 0, "prop": 0}


def test_hybrid_search_channels_none_runs_all(monkeypatch):
    """channels=None → corre los 4 canales (comportamiento actual)."""
    session = _FakeSession()
    calls = _install_hybrid_channel_mocks(monkeypatch, session)
    kq.hybrid_search(session, "q", [0.1, 0.2], 8)
    assert calls == {"dense": 1, "fts": 1, "para": 1, "prop": 1}


# ---------------------------------------------------------------------
# ask() — gate de search_summaries por channels
# ---------------------------------------------------------------------


def _install_ask_mocks(monkeypatch, session, strategy, channels):
    """Mockea el flujo de ask() para aislar el gate de search_summaries."""
    captured = {"summary_calls": [], "hybrid_k": []}

    def _fake_config(session, name, default=None):
        if name == "KAG_QUERY_PARALLEL":
            return False
        if name == "KAG_BRANCH_B_MAX_ITERS":
            return 0
        if name == "KAG_QUERY_ALL_CHANNELS":
            return False
        return default

    def _fake_classify(session, query):
        return strategy, channels, "standard", "razón", False

    def _fake_search_summaries(
        session, query_text, query_embedding, top_k, level=None, rrf_k=60, verbose=False
    ):
        captured["summary_calls"].append(top_k)
        return []

    def _fake_hybrid(session, query, q_emb, k, verbose=False, channels=None):
        captured["hybrid_k"].append(k)
        return []

    def _fake_rrf_merge(*lists, k=60, top_k=20):
        return []

    monkeypatch.setattr(kq, "classify_query_strategy", _fake_classify)
    monkeypatch.setattr(kq, "_kag_config_value", _fake_config)
    monkeypatch.setattr(kq, "search_summaries", _fake_search_summaries)
    monkeypatch.setattr(kq, "hybrid_search", _fake_hybrid)
    monkeypatch.setattr(kq, "critic_and_linking", lambda *a, **k: ([], [], []))
    monkeypatch.setattr(kq, "rrf_merge", _fake_rrf_merge)
    monkeypatch.setattr(kq, "chunks_by_ids", lambda *a, **k: [])
    monkeypatch.setattr(kq, "figures_for_chunks", lambda *a, **k: [])
    monkeypatch.setattr(kq, "doc_summaries", lambda *a, **k: [])
    monkeypatch.setattr(kq, "propositions_for_chunks", lambda *a, **k: [])
    monkeypatch.setattr(kq, "generate_answer", lambda *a, **k: "respuesta")
    _patch_own_session(monkeypatch, session)
    return captured


def test_ask_summaries_channel_activates_search_summaries(monkeypatch):
    """ask(): "summaries" en channels → search_summaries se activa."""
    _fake_embeddings(monkeypatch)
    session = _FakeSession()
    captured = _install_ask_mocks(
        monkeypatch, session, "hierarchical", ["dense", "fts", "summaries"]
    )
    kq.ask(session, "¿de qué trata el libro?", top_k=8, global_top_k=20, verbose=False)
    assert captured["summary_calls"] == [20]


def test_ask_without_summaries_channel_skips_search_summaries(monkeypatch):
    """ask(): sin "summaries" en channels → search_summaries NO se llama."""
    _fake_embeddings(monkeypatch)
    session = _FakeSession()
    captured = _install_ask_mocks(
        monkeypatch, session, "hierarchical", ["dense", "fts"]
    )
    kq.ask(session, "¿de qué trata el libro?", top_k=8, global_top_k=20, verbose=False)
    assert captured["summary_calls"] == []
