"""Tests de la clasificación SLM de la estrategia de consulta
(src/kag_query.py::classify_query_strategy) — sin DB real, sin
torch/spacy/transformers.

Patrón de tests/test_kag_document_propositions.py: sesiones falsas +
monkeypatch de call_with_retries/_get_prompt_pair. `_with_own_session` se
parchea con `_patch_own_session` para que los workers del
ThreadPoolExecutor usen la sesión fake del test.

Cubre:
  - El SLM clasifica cada una de las 5 estrategias (JSON válido).
  - Fallback: call_with_retries lanza → heurística (global→hierarchical,
    local→graph) con used_fallback=True.
  - JSON inválido (no dict / strategy desconocida) → fallback.
  - KAG_QUERY_STRATEGY_SLM=False → fallback sin llamar al SLM.
  - ask()/_ask_audited(): "hierarchical" usa global_top_k y activa
    search_summaries; "graph" usa top_k y no activa resúmenes.
"""

import json
import sys
import types

import pytest

import src.kag_query as kq

# Canales por defecto por estrategia (validación post-parseo y fallback).
_DEFAULT_CHANNELS = {
    "subqueries": ["dense", "fts", "paraphrase", "propositions", "graph"],
    "metadata": ["dense", "fts"],
    "hierarchical": ["dense", "fts", "summaries"],
    "graph": ["dense", "fts", "paraphrase", "propositions", "graph"],
    "multidoc": ["dense", "fts", "summaries"],
}

# Etiqueta de top_k por estrategia en los mocks de ask()/_ask_audited().
_TOP_K_LABEL = {
    "hierarchical": "wide",
    "graph": "standard",
    "metadata": "standard",
    "subqueries": "standard",
    "multidoc": "standard",
}


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


def _llm_ok(strategy, reason="porque sí"):
    """Fake de call_with_retries que devuelve el JSON del schema
    kag_query_strategy."""

    def fake_call(
        session, *, prompt, system, model_size, response_format, retries, fallback_model,
        thinking=None,
    ):
        return json.dumps({"strategy": strategy, "reason": reason}), "small", False

    return fake_call


def _llm_raises(exc=None):
    def fake_call(
        session, *, prompt, system, model_size, response_format, retries, fallback_model,
        thinking=None,
    ):
        raise exc or RuntimeError("LLM no disponible")

    return fake_call


# ---------------------------------------------------------------------
# classify_query_strategy — SLM clasifica las 5 estrategias
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "strategy",
    ["subqueries", "metadata", "hierarchical", "graph", "multidoc"],
)
def test_slm_classifies_each_strategy(monkeypatch, strategy):
    """El SLM clasifica cada una de las 5 estrategias (JSON válido)."""
    monkeypatch.setattr(kq, "call_with_retries", _llm_ok(strategy, "razón"))
    session = _FakeSession()
    out = kq.classify_query_strategy(session, "¿cómo difieren X e Y en Z?")
    assert out == (strategy, _DEFAULT_CHANNELS[strategy], "standard", "razón", False)


def test_slm_reason_empty_string(monkeypatch):
    """reason ausente/vacío → str vacío (no None)."""
    monkeypatch.setattr(kq, "call_with_retries", _llm_ok("graph", ""))
    session = _FakeSession()
    out = kq.classify_query_strategy(session, "q")
    assert out == ("graph", _DEFAULT_CHANNELS["graph"], "standard", "", False)


# ---------------------------------------------------------------------
# classify_query_strategy — fallback
# ---------------------------------------------------------------------


def test_fallback_when_llm_raises_global(monkeypatch):
    """call_with_retries lanza → heurística: 'resumen' → hierarchical."""
    monkeypatch.setattr(kq, "call_with_retries", _llm_raises())
    session = _FakeSession()
    out = kq.classify_query_strategy(session, "¿de qué trata el libro? resumen")
    assert out[0] == "hierarchical"
    assert out[1] == ["dense", "fts", "summaries"]
    assert out[2] == "wide"
    assert out[3].startswith("fallback:")
    assert out[4] is True


def test_fallback_when_llm_raises_local(monkeypatch):
    """call_with_retries lanza → heurística: 'relación entre' → graph."""
    monkeypatch.setattr(kq, "call_with_retries", _llm_raises())
    session = _FakeSession()
    out = kq.classify_query_strategy(session, "¿qué relación entre X e Y?")
    assert out[0] == "graph"
    assert out[1] == ["dense", "fts", "paraphrase", "propositions", "graph"]
    assert out[2] == "standard"
    assert out[3].startswith("fallback:")
    assert out[4] is True


def test_fallback_when_json_not_dict(monkeypatch):
    """JSON válido pero no dict (p.ej. lista) → fallback."""
    monkeypatch.setattr(
        kq,
        "call_with_retries",
        lambda session, *, prompt, system, model_size, response_format, retries, fallback_model: (
            '["no", "soy", "dict"]',
            "small",
            False,
        ),
    )
    session = _FakeSession()
    out = kq.classify_query_strategy(session, "resumen del libro")
    assert out[0] == "hierarchical"
    assert out[4] is True


def test_fallback_when_strategy_unknown(monkeypatch):
    """strategy fuera del set → fallback."""
    monkeypatch.setattr(kq, "call_with_retries", _llm_ok("unknown_strategy"))
    session = _FakeSession()
    out = kq.classify_query_strategy(session, "resumen del libro")
    assert out[0] == "hierarchical"
    assert out[4] is True
    assert "inválida" in out[3]


def test_fallback_when_json_invalid(monkeypatch):
    """JSON inválido (parse_llm_output → {}) → fallback."""
    monkeypatch.setattr(
        kq,
        "call_with_retries",
        lambda session, *, prompt, system, model_size, response_format, retries, fallback_model: (
            "esto no es json",
            "small",
            False,
        ),
    )
    session = _FakeSession()
    out = kq.classify_query_strategy(session, "resumen del libro")
    assert out[0] == "hierarchical"
    assert out[4] is True


def test_flag_off_skips_slm(monkeypatch):
    """KAG_QUERY_STRATEGY_SLM=False → fallback determinista SIN llamar al SLM."""
    calls = []

    def _fake_call(
        session, *, prompt, system, model_size, response_format, retries, fallback_model,
        thinking=None,
    ):
        calls.append(prompt)
        return json.dumps({"strategy": "graph", "reason": "x"}), "small", False

    def _config(session, name, default=None):
        if name == "KAG_QUERY_STRATEGY_SLM":
            return False
        return default

    monkeypatch.setattr(kq, "call_with_retries", _fake_call)
    monkeypatch.setattr(kq, "_kag_config_value", _config)
    session = _FakeSession()
    out = kq.classify_query_strategy(session, "resumen del libro")
    assert out[0] == "hierarchical"
    assert out[4] is True
    assert calls == []


# ---------------------------------------------------------------------
# ask()/_ask_audited() — wiring de la estrategia
# ---------------------------------------------------------------------


def _install_ask_mocks(monkeypatch, session, strategy):
    """Mockea el flujo de ask()/_ask_audited() para aislar el wiring de la
    estrategia. Devuelve dict con los callables capturados."""
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
        return (
            strategy,
            _DEFAULT_CHANNELS[strategy],
            _TOP_K_LABEL[strategy],
            "razón",
            False,
        )

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

    def _fake_audit(session, query, propositions, corpus_metadata, **kw):
        return (
            [],
            {"contradictions_detected": False, "analysis_cases": []},
            {"verdict": "SUFFICIENT_FOR_SYNTHESIS", "confidence_score": 0.9},
        )

    def _fake_call(
        session, *, prompt, system, model_size, response_format, retries, fallback_model,
        thinking=None,
    ):
        return "respuesta", "model", False

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
    monkeypatch.setattr(kq, "_audit_epistemic_fused", _fake_audit)
    monkeypatch.setattr(kq, "call_with_retries", _fake_call)
    _patch_own_session(monkeypatch, session)
    return captured


def test_ask_hierarchical_uses_global_top_k_and_summaries(monkeypatch):
    """ask(): strategy 'hierarchical' → k=global_top_k y search_summaries
    activo."""
    _fake_embeddings(monkeypatch)
    session = _FakeSession()
    captured = _install_ask_mocks(monkeypatch, session, "hierarchical")
    kq.ask(session, "¿de qué trata el libro?", top_k=8, global_top_k=20, verbose=False)
    assert captured["hybrid_k"] == [20]
    assert captured["summary_calls"] == [20]


def test_ask_graph_uses_top_k_no_summaries(monkeypatch):
    """ask(): strategy 'graph' → k=top_k y search_summaries NO se llama."""
    _fake_embeddings(monkeypatch)
    session = _FakeSession()
    captured = _install_ask_mocks(monkeypatch, session, "graph")
    kq.ask(
        session, "¿qué relación entre X e Y?", top_k=8, global_top_k=20, verbose=False
    )
    assert captured["hybrid_k"] == [8]
    assert captured["summary_calls"] == []


def test_ask_metadata_uses_top_k_no_summaries(monkeypatch):
    """ask(): strategy 'metadata' → POR AHORA comportamiento 'local'."""
    _fake_embeddings(monkeypatch)
    session = _FakeSession()
    captured = _install_ask_mocks(monkeypatch, session, "metadata")
    kq.ask(
        session, "¿qué dice Foucault en 1975?", top_k=8, global_top_k=20, verbose=False
    )
    assert captured["hybrid_k"] == [8]
    assert captured["summary_calls"] == []


def test_ask_audited_hierarchical_uses_global_top_k_and_summaries(monkeypatch):
    """_ask_audited(): strategy 'hierarchical' → k=global_top_k y
    search_summaries activo."""
    _fake_embeddings(monkeypatch)
    session = _FakeSession()
    captured = _install_ask_mocks(monkeypatch, session, "hierarchical")
    result = kq._ask_audited(
        session, "¿de qué trata el libro?", top_k=8, global_top_k=20, verbose=False
    )
    assert result["answer"] == "respuesta"
    assert captured["hybrid_k"] == [20]
    assert captured["summary_calls"] == [20]


def test_ask_audited_graph_uses_top_k_no_summaries(monkeypatch):
    """_ask_audited(): strategy 'graph' → k=top_k y search_summaries NO se
    llama."""
    _fake_embeddings(monkeypatch)
    session = _FakeSession()
    captured = _install_ask_mocks(monkeypatch, session, "graph")
    result = kq._ask_audited(
        session, "¿qué relación entre X e Y?", top_k=8, global_top_k=20, verbose=False
    )
    assert result["answer"] == "respuesta"
    assert captured["hybrid_k"] == [8]
    assert captured["summary_calls"] == []
