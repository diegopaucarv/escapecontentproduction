"""Tests de enrutamiento §3.2/§3.3: canal de proposiciones en el RRF y
resúmenes temáticos como marco separado.

Sin DB real, sin torch/spacy/transformers. Sesiones falsas + monkeypatch
(patrón de tests/test_kag_retrieval.py y tests/test_kag_audited.py).
"""

import sys
import types

import pytest

from src.kag_query import (
    _audit_epistemic_fused,
    assemble_context,
    classify_query,
)


class _Result:
    def __init__(self, rows=None):
        self._rows = rows if rows is not None else []

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def first(self):
        return self._rows[0] if self._rows else None

    def scalar(self):
        return None


class _FakeSession:
    """Sesión falsa: execute devuelve _Result vacío (nada toca la DB)."""

    def __init__(self):
        self.rolled_back = 0

    def execute(self, stmt, params=None):
        return _Result()

    def rollback(self):
        self.rolled_back += 1


@pytest.fixture(autouse=True)
def _fake_embeddings(monkeypatch):
    """Inyecta src.embeddings fake: ask() hace `from src.embeddings import
    embed_text` en caliente — el módulo falso evita torch/transformers."""
    fake_emb = types.ModuleType("src.embeddings")
    fake_emb.embed_text = lambda query, input_type="query": [0.1, 0.2]
    fake_emb.embed_texts = lambda texts, **k: [[0.1, 0.2] for _ in texts]
    monkeypatch.setitem(sys.modules, "src.embeddings", fake_emb)


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


# ---------------------------------------------------------------------
# assemble_context — sección de resúmenes recuperados (marco temático)
# ---------------------------------------------------------------------


def _summary_hit(doc_id=1, level="document", chapter_id=None, text="resumen"):
    return {
        "id": f"s-{doc_id}",
        "doc_id": doc_id,
        "level": level,
        "chapter_id": chapter_id,
        "text": text,
        "score": 0.8,
    }


def test_assemble_context_summary_hits_section_with_text():
    """summary_hits → bloque MARCO TEMÁTICO con el texto."""
    ctx = assemble_context([], [], [], [], "pregunta", summary_hits=[_summary_hit()])
    assert "--- MARCO TEMÁTICO (resúmenes de documento/sección) ---" in ctx
    assert "[doc_id: 1 | nivel: document] resumen" in ctx


def test_assemble_context_summary_hits_section_level():
    """Nivel section con capítulo → formato con capítulo."""
    ctx = assemble_context(
        [],
        [],
        [],
        [],
        "pregunta",
        summary_hits=[_summary_hit(level="section", chapter_id="s1", text="sección")],
    )
    assert "[doc_id: 1 | nivel: section | capítulo: s1] sección" in ctx


def test_assemble_context_no_summary_hits_no_section():
    """summary_hits=None (default) → el bloque MARCO NO aparece."""
    ctx = assemble_context([], [], [], [], "pregunta")
    assert "MARCO TEMÁTICO" not in ctx


def test_assemble_context_empty_summary_hits_placeholder():
    """summary_hits=[] → bloque MARCO con placeholder (consistente con el resto)."""
    ctx = assemble_context([], [], [], [], "pregunta", summary_hits=[])
    assert "--- MARCO TEMÁTICO (resúmenes de documento/sección) ---" in ctx
    assert "(sin resúmenes recuperados)" in ctx


def test_assemble_context_summary_section_after_doc_summaries():
    """El bloque MARCO va PRIMERO (antes de los resúmenes de documento y de la
    evidencia textual): las tres capas (marco → chunks → proposiciones) se
    entregan separadas y etiquetadas."""
    ctx = assemble_context(
        [],
        [],
        [],
        [{"doc_path": "doc.md", "summary": "resumen doc"}],
        "pregunta",
        summary_hits=[_summary_hit()],
    )
    assert (
        ctx.index("--- MARCO TEMÁTICO (resúmenes de documento/sección) ---")
        < ctx.index("--- EVIDENCIA TEXTUAL (chunks con cita) ---")
        < ctx.index("--- RESUMENES DE DOCUMENTO (referencia secundaria) ---")
    )


# ---------------------------------------------------------------------
# ask() — canal de proposiciones en el RRF + enrutamiento temático
# ---------------------------------------------------------------------


def _install_ask_mocks(monkeypatch, prop_hits=None, summary_hits=None):
    """Mockea las dependencias de ask() para aislar los canales nuevos.

    KAG_QUERY_PARALLEL=False fuerza el camino secuencial (sin sesiones
    propias de hilo); el canal de proposiciones ya vive DENTRO de
    hybrid_search (mockeado aquí), así que ask() no lo invoca. Devuelve un
    dict con los callables capturados para aserciones.
    """
    import src.kag_query as kq

    captured = {"prop_calls": [], "summary_calls": [], "merge_lists": None}

    def _fake_config(session, name, default=None):
        if name == "KAG_QUERY_PARALLEL":
            return False
        return default

    def _fake_prop_search(session, q_emb, top_k):
        captured["prop_calls"].append((q_emb, top_k))
        return prop_hits if prop_hits is not None else []

    def _fake_summaries(session, query_text, query_embedding, top_k, **kw):
        captured["summary_calls"].append((query_text, query_embedding, top_k))
        return summary_hits if summary_hits is not None else []

    def _fake_rrf_merge(*lists, k=60, top_k=20):
        captured["merge_lists"] = lists
        return []

    monkeypatch.setattr(kq, "_kag_config_value", _fake_config)
    monkeypatch.setattr(kq, "proposition_vector_search", _fake_prop_search)
    monkeypatch.setattr(kq, "search_summaries", _fake_summaries)
    monkeypatch.setattr(kq, "rrf_merge", _fake_rrf_merge)
    monkeypatch.setattr(kq, "hybrid_search", lambda *a, **k: [])
    monkeypatch.setattr(kq, "critic_and_linking", lambda *a, **k: ([], [], []))
    monkeypatch.setattr(kq, "chunks_by_ids", lambda *a, **k: [])
    monkeypatch.setattr(kq, "figures_for_chunks", lambda *a, **k: [])
    monkeypatch.setattr(kq, "doc_summaries", lambda *a, **k: [])
    monkeypatch.setattr(kq, "propositions_for_chunks", lambda *a, **k: [])
    monkeypatch.setattr(kq, "generate_answer", lambda *a, **k: "respuesta")
    # El SLM de estrategia se mockea con el mapeo de la heurística vieja
    # (global→hierarchical, local→graph): los tests de routing verifican el
    # comportamiento de los canales, no la clasificación en sí.
    monkeypatch.setattr(
        kq,
        "classify_query_strategy",
        lambda session, query: (
            (
                "hierarchical",
                ["dense", "fts", "summaries"],
                "wide",
                "",
                False,
            )
            if classify_query(query) == "global"
            else (
                "graph",
                ["dense", "fts", "paraphrase", "propositions", "graph"],
                "standard",
                "",
                False,
            )
        ),
    )
    return captured


def test_ask_global_routes_summaries_and_propositions(monkeypatch):
    """Consulta global → search_summaries con top_k=global_top_k; el RRF final
    NO recibe prop_channel como lista extra (vive en hybrid_search)."""
    import src.kag_query as kq

    prop_hits = [
        {"id": "p1", "chunk_id": "c1", "doc_id": "d1", "statement": "s", "score": 0.9}
    ]
    summary_hits = [_summary_hit()]
    captured = _install_ask_mocks(monkeypatch, prop_hits, summary_hits)

    session = _FakeSession()
    answer = kq.ask(session, "¿De qué trata el libro?", verbose=False)

    assert answer == "respuesta"
    # search_summaries se invocó con top_k=global_top_k (20) y el embedding.
    assert len(captured["summary_calls"]) == 1
    _query, _q_emb, top_k = captured["summary_calls"][0]
    assert top_k == 20
    # ask() ya no invoca proposition_vector_search (lo hace hybrid_search).
    assert captured["prop_calls"] == []
    # RRF final con 3 listas: vec, regex, ppr — sin prop_channel.
    assert captured["merge_lists"] is not None
    assert len(captured["merge_lists"]) == 3


def test_ask_global_context_contains_summary_section(monkeypatch):
    """El contexto ensamblado (vía generate_answer) incluye el bloque MARCO
    TEMÁTICO como marco de la consulta global."""
    import src.kag_query as kq

    captured = {}

    def _fake_generate(session, context, query):
        captured["context"] = context
        return "respuesta"

    _install_ask_mocks(
        monkeypatch,
        prop_hits=[],
        summary_hits=[_summary_hit(doc_id=3, text="marco del libro")],
    )
    monkeypatch.setattr(kq, "generate_answer", _fake_generate)

    session = _FakeSession()
    kq.ask(session, "¿De qué trata el libro?", verbose=False)

    assert (
        "--- MARCO TEMÁTICO (resúmenes de documento/sección) ---" in captured["context"]
    )
    assert "[doc_id: 3 | nivel: document] marco del libro" in captured["context"]


def test_ask_local_skips_summaries_keeps_propositions(monkeypatch):
    """Consulta local → search_summaries NO se llama; el RRF final sigue sin
    prop_channel (el canal de proposiciones vive en hybrid_search)."""
    import src.kag_query as kq

    prop_hits = [
        {"id": "p1", "chunk_id": "c1", "doc_id": "d1", "statement": "s", "score": 0.9}
    ]
    captured = _install_ask_mocks(monkeypatch, prop_hits=prop_hits, summary_hits=None)

    session = _FakeSession()
    kq.ask(
        session,
        "¿Cuál es el impacto de la cultura en las organizaciones?",
        verbose=False,
    )

    assert captured["summary_calls"] == []
    assert captured["prop_calls"] == []
    assert len(captured["merge_lists"]) == 3


def test_ask_prop_channel_skips_missing_chunk_id(monkeypatch):
    """ask() ya no filtra hits de proposiciones: el canal vive en
    hybrid_search (cubierto en test_kag_hybrid_channels.py). El RRF final
    no recibe prop_channel."""
    import src.kag_query as kq

    prop_hits = [
        {"id": "p1", "chunk_id": "c1", "doc_id": "d1", "statement": "s", "score": 0.9},
        {"id": "p2", "chunk_id": None, "doc_id": "d1", "statement": "s2", "score": 0.8},
    ]
    captured = _install_ask_mocks(monkeypatch, prop_hits=prop_hits, summary_hits=None)

    session = _FakeSession()
    kq.ask(session, "¿De qué trata el libro?", verbose=False)

    assert captured["prop_calls"] == []
    assert len(captured["merge_lists"]) == 3


def test_ask_prop_channel_failure_degrades(monkeypatch):
    """ask() ya no invoca proposition_vector_search (vive en hybrid_search,
    que degrada solo); el flujo sigue intacto."""
    import src.kag_query as kq

    def _boom(session, q_emb, top_k):
        raise RuntimeError("tabla ausente")

    captured = _install_ask_mocks(monkeypatch, prop_hits=None, summary_hits=None)
    monkeypatch.setattr(kq, "proposition_vector_search", _boom)

    session = _FakeSession()
    answer = kq.ask(session, "¿De qué trata el libro?", verbose=False)

    assert answer == "respuesta"
    assert captured["prop_calls"] == []
    assert len(captured["merge_lists"]) == 3


def test_ask_summary_channel_failure_degrades(monkeypatch):
    """Si search_summaries lanza, el flujo sigue (degradación)."""
    import src.kag_query as kq

    def _boom(session, query_text, query_embedding, top_k, **kw):
        raise RuntimeError("índice ausente")

    captured = _install_ask_mocks(monkeypatch, prop_hits=None, summary_hits=None)
    monkeypatch.setattr(kq, "search_summaries", _boom)

    session = _FakeSession()
    answer = kq.ask(session, "¿De qué trata el libro?", verbose=False)

    assert answer == "respuesta"
    assert captured["summary_calls"] == []


def test_ask_channels_disabled_by_config(monkeypatch):
    """KAG_PROPOSITION_CHANNEL / KAG_SUMMARY_CHANNEL=False apagan los canales."""
    import src.kag_query as kq

    captured = {}

    def _fake_config(session, name, default=None):
        if name == "KAG_QUERY_PARALLEL":
            return False
        if name in ("KAG_PROPOSITION_CHANNEL", "KAG_SUMMARY_CHANNEL"):
            return False
        return default

    def _fake_prop_search(session, q_emb, top_k):
        captured["prop_called"] = True
        return []

    def _fake_summaries(session, query_text, query_embedding, top_k, **kw):
        captured["summary_called"] = True
        return []

    monkeypatch.setattr(kq, "_kag_config_value", _fake_config)
    monkeypatch.setattr(kq, "proposition_vector_search", _fake_prop_search)
    monkeypatch.setattr(kq, "search_summaries", _fake_summaries)
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
    answer = kq.ask(session, "¿De qué trata el libro?", verbose=False)

    assert answer == "respuesta"
    assert "prop_called" not in captured
    assert "summary_called" not in captured


# ---------------------------------------------------------------------
# _audit_epistemic_fused — marco temático en el prompt de auditoría
# ---------------------------------------------------------------------


def test_audit_fused_summary_hits_in_prompt(monkeypatch):
    """summary_hits + chunks → el prompt recibe los TRES bloques de evidencia
    separados con los mismos encabezados que assemble_context."""
    import src.kag_query as kq

    props = [
        {
            "chunk_id": 1,
            "document_id": 7,
            "statement": "La cultura afecta a las organizaciones.",
            "text_span": "La cultura afecta a las organizaciones.",
            "citation_references": [],
            "doc_title": "doc.md",
        }
    ]
    captured = {}

    def _fake_call(
        session,
        *,
        prompt,
        system,
        model_size,
        response_format,
        retries,
        fallback_model,
    ):
        captured["prompt"] = prompt
        return (
            '{"facts": [{"statement": "La cultura afecta a las organizaciones.", '
            '"verbatim_evidence": "La cultura afecta a las organizaciones.", '
            '"relevance": "direct_answer"}], "contradictions": [], '
            '"sufficiency": {"verdict": "SUFFICIENT", "confidence": 0.9}}',
            "model",
            False,
        )

    monkeypatch.setattr(kq, "call_with_retries", _fake_call)
    session = _FakeSession()
    facts, _report, evaluation = _audit_epistemic_fused(
        session,
        "¿qué es la cultura?",
        props,
        [],
        summary_hits=[_summary_hit(doc_id=2, text="marco temático")],
        chunks=[{"chunk_id": "c1", "content": "texto"}],
    )
    assert evaluation["verdict"] == "SUFFICIENT_FOR_SYNTHESIS"
    assert len(facts) == 1
    assert (
        "--- MARCO TEMÁTICO (resúmenes de documento/sección) ---" in captured["prompt"]
    )
    assert "--- EVIDENCIA TEXTUAL (chunks con cita) ---" in captured["prompt"]
    assert "--- CAPA ATÓMICA (proposiciones) ---" in captured["prompt"]
    assert "marco temático" in captured["prompt"]


def test_audit_fused_no_summary_hits_prompt_unchanged(monkeypatch):
    """Sin summary_hits ni chunks el prompt no lleva los bloques de evidencia."""
    import src.kag_query as kq

    props = [
        {
            "chunk_id": 1,
            "document_id": 7,
            "statement": "Hecho A.",
            "text_span": "Hecho A.",
            "citation_references": [],
            "doc_title": "doc.md",
        }
    ]
    captured = {}

    def _fake_call(
        session,
        *,
        prompt,
        system,
        model_size,
        response_format,
        retries,
        fallback_model,
    ):
        captured["prompt"] = prompt
        return (
            '{"facts": [{"statement": "Hecho A.", '
            '"verbatim_evidence": "Hecho A.", '
            '"relevance": "supporting_evidence"}], "contradictions": [], '
            '"sufficiency": {"verdict": "SUFFICIENT", "confidence": 0.8}}',
            "model",
            False,
        )

    monkeypatch.setattr(kq, "call_with_retries", _fake_call)
    session = _FakeSession()
    _audit_epistemic_fused(session, "q", props, [])
    assert "MARCO TEMÁTICO" not in captured["prompt"]
    assert "EVIDENCIA TEXTUAL" not in captured["prompt"]
    assert "CAPA ATÓMICA" not in captured["prompt"]


# ---------------------------------------------------------------------
# classify_query — heurística de enrutamiento
# ---------------------------------------------------------------------


def test_classify_query_global_keywords():
    assert classify_query("¿De qué trata el libro?") == "global"
    assert classify_query("resumen del documento") == "global"
    assert classify_query("¿Cuál es el impacto de X?") == "local"
