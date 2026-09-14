"""Tests de los agentes query-time del KAG proposicional (src/kag_agents.py).

Sin DB real y sin red: sesión falsa (_FakeSession/_FakeResult), LLM mockeado
(monkeypatch de call_with_retries) y embeddings mockeados. Patrón de
tests/test_kag_propositional.py.
"""

import json
from types import SimpleNamespace

import pytest

import src.kag_agents as ka

# ---------------------------------------------------------------------
# Sesión falsa
# ---------------------------------------------------------------------


class _FakeResult:
    """Resultado mínimo de session.execute: first/scalar/fetchall/scalars/fetchone."""

    def __init__(self, rows=None, scalar=None):
        self._rows = rows or []
        self._scalar = scalar

    def first(self):
        return self._rows[0] if self._rows else None

    def scalar(self):
        return self._scalar

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def scalars(self):
        return self

    def all(self):
        return self._rows


def _unwrap(value):
    """Desenvuelve bindparam(expanding=True) de SQLAlchemy a su valor."""
    return value.value if hasattr(value, "value") else value


class _FakeSession:
    """Sesión falsa que despacha por SQL y registra las queries ejecutadas.

    Almacena documents/chapters/chunks/relations en listas de dicts y
    devuelve SimpleNamespace con los atributos que cada query espera.
    """

    def __init__(self):
        self.documents = []
        self.chapters = []
        self.chunks = []
        self.relations = []
        self.executed = []  # (sql, params) para aserciones
        self.commits = 0
        self.rollbacks = 0

    def execute(self, stmt, params=None):
        sql = str(stmt)
        params = params or {}
        self.executed.append((sql, params))
        if "session_settings" in sql:
            return _FakeResult(rows=[])
        if "SELECT p.id AS prop_id" in sql:
            # Nivel 1: top-N por distancia coseno (el ORDER BY lo hace la DB).
            top = _unwrap(params.get("top_propositions")) or len(self.chunks)
            rows = sorted(self.chunks, key=lambda c: c["cosine_distance"])[:top]
            out = []
            for c in rows:
                out.append(
                    SimpleNamespace(
                        prop_id=c["id"],
                        document_id=c["document_id"],
                        chapter_id=c["chapter_id"],
                        statement=c["statement"],
                        text_span=c["text_span"],
                        char_start=c["char_start"],
                        char_end=c["char_end"],
                        citation_references=c["citation_references"],
                        chapter_title=c["chapter_title"],
                        doc_title=c["doc_title"],
                        bibtex=c["bibtex"],
                        cosine_distance=c["cosine_distance"],
                    )
                )
            return _FakeResult(rows=out)
        if "SELECT id, summary, line_start, line_end FROM document_chapters" in sql:
            ids = _unwrap(params.get("ids")) or []
            rows = [c for c in self.chapters if c["id"] in ids]
            return _FakeResult(rows=[SimpleNamespace(**c) for c in rows])
        if "SELECT id, title FROM documents" in sql:
            # Branch B (a): excluye documentos visitados.
            visited = _unwrap(params.get("visited_docs")) or []
            rows = [d for d in self.documents if d["id"] not in visited][:3]
            return _FakeResult(rows=[SimpleNamespace(**c) for c in rows])
        if "SELECT r.target_entity_id, e.name FROM kag_relations" in sql:
            # Branch B (b): vecinos de entidades semilla.
            eids = _unwrap(params.get("entity_ids")) or []
            rows = [r for r in self.relations if r["source_entity_id"] in eids][:10]
            return _FakeResult(rows=[SimpleNamespace(name=r["name"]) for r in rows])
        if "SELECT p.id AS chunk_id, p.statement, p.document_id" in sql:
            # Branch B (c): FTS — devuelve los chunks por score.
            rows = sorted(self.chunks, key=lambda c: c.get("score", 0.0), reverse=True)[
                :5
            ]
            out = []
            for c in rows:
                out.append(
                    SimpleNamespace(
                        chunk_id=c["id"],
                        statement=c["statement"],
                        document_id=c["document_id"],
                        score=c.get("score", 0.0),
                    )
                )
            return _FakeResult(rows=out)
        if (
            "SELECT text_span, char_start, char_end, citation_references, document_id"
            in sql
        ):
            # verify_claim_grounding: chunk por id.
            cid = _unwrap(params.get("cid"))
            for c in self.chunks:
                if c["id"] == cid:
                    return _FakeResult(
                        rows=[
                            SimpleNamespace(
                                text_span=c["text_span"],
                                char_start=c["char_start"],
                                char_end=c["char_end"],
                                citation_references=c["citation_references"],
                                document_id=c["document_id"],
                            )
                        ]
                    )
            return _FakeResult(rows=[])
        if "SELECT title, thematic_areas_iso25964, library_of_congress" in sql:
            # Metadatos de corpus (ask_propositional paso 5).
            rows = [d for d in self.documents if d.get("status") == "ready"]
            return _FakeResult(rows=[SimpleNamespace(**c) for c in rows])
        if "SELECT d.id AS doc_id, d.title AS doc_title, d.bibtex" in sql:
            # Metadatos de chunk para enriquecer la evidencia.
            cid = _unwrap(params.get("cid"))
            for c in self.chunks:
                if c["id"] == cid:
                    return _FakeResult(
                        rows=[
                            SimpleNamespace(
                                doc_id=c["document_id"],
                                doc_title=c["doc_title"],
                                bibtex=c["bibtex"],
                                chapter_title=c["chapter_title"],
                            )
                        ]
                    )
            return _FakeResult(rows=[])
        return _FakeResult()

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


# ---------------------------------------------------------------------
# Fixtures de datos
# ---------------------------------------------------------------------


def _make_chunk(
    cid,
    statement,
    text_span=None,
    doc_id="doc_001",
    chapter_id=1,
    cosine=0.1,
    score=1.0,
    citations=None,
):
    """Chunk en formato de almacenamiento de la sesión falsa."""
    span = text_span if text_span is not None else statement
    return {
        "id": cid,
        "document_id": doc_id,
        "chapter_id": chapter_id,
        "statement": statement,
        "text_span": span,
        "char_start": 0,
        "char_end": len(span),
        "citation_references": citations
        if citations is not None
        else ["Bourdieu, 1984, p. 52"],
        "chapter_title": "Capítulo Uno",
        "doc_title": "Documento de prueba",
        "bibtex": "@book{prueba, title={Documento de prueba}}",
        "cosine_distance": cosine,
        "score": score,
    }


def _make_document(doc_id, title="Documento de prueba", status="ready"):
    return {
        "id": doc_id,
        "title": title,
        "thematic_areas_iso25964": [{"preferred_term": "Psicología Cultural"}],
        "library_of_congress": {"lcsh_terms": [{"term": "Ethnopsychology"}]},
        "status": status,
    }


def _make_chapter(cid, summary="Resumen del capítulo.", line_start=1, line_end=60):
    return {
        "id": cid,
        "summary": summary,
        "line_start": line_start,
        "line_end": line_end,
    }


def _nivel1_chunk(cid, statement, text_span=None, doc_id="doc_001", cosine=0.1):
    """Chunk en el formato de salida de query_focused_proposition_extractor."""
    span = text_span if text_span is not None else statement
    return {
        "chunk_id": cid,
        "document_id": doc_id,
        "chapter_id": 1,
        "statement": statement,
        "text_span": span,
        "char_start": 0,
        "char_end": len(span),
        "citation_references": ["Bourdieu, 1984, p. 52"],
        "chapter_title": "Capítulo Uno",
        "doc_title": "Documento de prueba",
        "bibtex": "@book{prueba, title={Documento de prueba}}",
        "cosine_distance": cosine,
    }


# ---------------------------------------------------------------------
# LLM mockeado
# ---------------------------------------------------------------------


def _extract_chunks_from_prompt(prompt):
    """Extrae el JSON de candidate_chunks_json del prompt de síntesis."""
    marker = "Fragmentos recuperados para análisis:"
    if marker not in prompt:
        return []
    tail = prompt.split(marker, 1)[1]
    json_part = tail.split("Devuelve el JSON", 1)[0].strip()
    try:
        data = json.loads(json_part)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def _fake_call_valid(
    session,
    *,
    prompt,
    system,
    model_size,
    response_format,
    retries,
    fallback_model=None,
):
    """call_with_retries mockeado: JSON válido según el system prompt."""
    if system == ka.SYNTHESIS_SYSTEM_SHORT:
        chunks = _extract_chunks_from_prompt(prompt)
        facts = []
        for c in chunks:
            cid = c.get("chunk_id")
            chunk = next((x for x in session.chunks if str(x["id"]) == str(cid)), None)
            facts.append(
                {
                    "chunk_id": str(cid),
                    "document_id": c.get("document_id", ""),
                    "source_file": c.get("doc_title", ""),
                    "relevance_level": "direct_answer",
                    "atomic_summary": c.get("statement", ""),
                    "verbatim_evidence": (
                        chunk["text_span"] if chunk else c.get("text_span", "")
                    ),
                    "academic_citations": c.get("citation_references") or [],
                }
            )
        return (
            json.dumps(
                {
                    "query": "q",
                    "total_chunks_processed": len(chunks),
                    "synthesized_facts": facts,
                }
            ),
            "small",
            False,
        )
    if system == ka.CONTRADICTION_SYSTEM_SHORT:
        return (
            json.dumps(
                {
                    "contradictions_detected": True,
                    "analysis_cases": [
                        {
                            "conflict_type": "paradigmatic_theoretical_divergence",
                            "divergence_summary": "Dos escuelas divergen.",
                            "thesis_a": {
                                "proposition_id": "1",
                                "document_id": "doc_001",
                                "claim": "A",
                                "author_or_framework": "Escuela A",
                                "empirical_context": "ctx",
                            },
                            "thesis_b": {
                                "proposition_id": "2",
                                "document_id": "doc_001",
                                "claim": "B",
                                "author_or_framework": "Escuela B",
                                "empirical_context": "ctx",
                            },
                            "epistemic_reconciliation": "reconciliación",
                        }
                    ],
                }
            ),
            "small",
            False,
        )
    if system == ka.SUFFICIENCY_SYSTEM_SHORT:
        return (
            json.dumps(
                {
                    "verdict": "SUFFICIENT_FOR_SYNTHESIS",
                    "confidence_score": 0.9,
                    "negative_rejection_details": {},
                    "branch_b_instructions": {},
                }
            ),
            "small",
            False,
        )
    if system == ka.ANSWER_SYSTEM_SHORT:
        return ("Respuesta final de prueba.", "large", False)
    return ("{}", model_size, False)


def _boom(
    session,
    *,
    prompt,
    system,
    model_size,
    response_format,
    retries,
    fallback_model=None,
):
    """call_with_retries mockeado: LLM caído."""
    raise RuntimeError("LLM caído")


# ---------------------------------------------------------------------
# Helper puro: _fts_tsquery
# ---------------------------------------------------------------------


def test_fts_tsquery_escapes_quotes_and_skips_empty():
    assert ka._fts_tsquery(["propagación", "hacia atrás"]) == (
        '"propagación" | "hacia atrás"'
    )
    assert ka._fts_tsquery(["a", "", "b"]) == '"a" | "b"'
    assert ka._fts_tsquery([]) == ""
    # Comillas dobles internas se duplican (escape de tsquery).
    assert ka._fts_tsquery(['dijo "hola"']) == '"dijo ""hola"""'


# ---------------------------------------------------------------------
# 1. Recuperación proposicional (Nivel 1)
# ---------------------------------------------------------------------


def test_query_focused_proposition_extractor_orders_by_cosine_distance():
    session = _FakeSession()
    session.chunks = [
        _make_chunk("c1", "stmt 1", cosine=0.5),
        _make_chunk("c2", "stmt 2", cosine=0.1),
        _make_chunk("c3", "stmt 3", cosine=0.3),
    ]
    props = ka.query_focused_proposition_extractor(session, [0.1, 0.2, 0.3])
    assert [p["chunk_id"] for p in props] == ["c2", "c3", "c1"]
    assert all(isinstance(p["cosine_distance"], float) for p in props)
    assert props[0]["statement"] == "stmt 2"
    assert props[0]["doc_title"] == "Documento de prueba"
    assert props[0]["citation_references"] == ["Bourdieu, 1984, p. 52"]


def test_query_focused_proposition_extractor_respects_top_propositions():
    session = _FakeSession()
    session.chunks = [
        _make_chunk("c1", "stmt 1", cosine=0.5),
        _make_chunk("c2", "stmt 2", cosine=0.1),
        _make_chunk("c3", "stmt 3", cosine=0.3),
    ]
    props = ka.query_focused_proposition_extractor(session, [0.1], top_propositions=2)
    assert len(props) == 2
    assert [p["chunk_id"] for p in props] == ["c2", "c3"]


def test_query_focused_proposition_extractor_none_embedding_degrades():
    session = _FakeSession()
    assert ka.query_focused_proposition_extractor(session, None) == []


def test_query_focused_proposition_extractor_empty_corpus_returns_empty():
    session = _FakeSession()
    assert ka.query_focused_proposition_extractor(session, [0.1, 0.2, 0.3]) == []


# ---------------------------------------------------------------------
# 2. Escalado a contexto de capítulo (Nivel 2)
# ---------------------------------------------------------------------


def test_escalate_to_parent_context_returns_summaries_by_id():
    session = _FakeSession()
    session.chapters = [
        _make_chapter(1, summary="Resumen cap 1"),
        _make_chapter(2, summary="Resumen cap 2"),
        _make_chapter(3, summary="Resumen cap 3"),
    ]
    result = ka.escalate_to_parent_context(session, [1, 3])
    assert result == {1: "Resumen cap 1", 3: "Resumen cap 3"}


def test_escalate_to_parent_context_empty_ids_returns_empty():
    session = _FakeSession()
    assert ka.escalate_to_parent_context(session, []) == {}


# ---------------------------------------------------------------------
# 3. Agente de Síntesis
# ---------------------------------------------------------------------


def test_synthesize_chunks_produces_facts_with_llm(monkeypatch):
    session = _FakeSession()
    chunks = [_nivel1_chunk("chunk_1", "La cultura influye en la cognición humana.")]
    monkeypatch.setattr(ka, "call_with_retries", _fake_call_valid)
    facts = ka.synthesize_chunks(session, "¿Qué influye en la cognición?", chunks)
    assert len(facts) == 1
    f = facts[0]
    assert f["relevance_level"] == "direct_answer"
    assert f["atomic_summary"] == "La cultura influye en la cognición humana."
    assert f["verbatim_evidence"] == "La cultura influye en la cognición humana."
    assert f["academic_citations"] == ["Bourdieu, 1984, p. 52"]
    assert f["chunk_id"] == "chunk_1"


def test_synthesize_chunks_empty_returns_empty():
    session = _FakeSession()
    assert ka.synthesize_chunks(session, "q", []) == []


def test_synthesize_chunks_degrades_when_llm_fails(monkeypatch):
    session = _FakeSession()
    chunks = [_nivel1_chunk("chunk_1", "La cultura influye en la cognición humana.")]
    monkeypatch.setattr(ka, "call_with_retries", _boom)
    facts = ka.synthesize_chunks(session, "q", chunks)
    assert len(facts) == 1
    assert facts[0]["relevance_level"] == "supporting_evidence"
    assert facts[0]["atomic_summary"] == "La cultura influye en la cognición humana."
    assert facts[0]["verbatim_evidence"] == "La cultura influye en la cognición humana."
    assert session.rollbacks == 1


# ---------------------------------------------------------------------
# 4. Resolución de contradicciones
# ---------------------------------------------------------------------


def test_resolve_contradictions_typifies_conflicts(monkeypatch):
    session = _FakeSession()
    facts = [
        {"chunk_id": "1", "atomic_summary": "A"},
        {"chunk_id": "2", "atomic_summary": "B"},
    ]

    def fake_llm(
        session,
        *,
        prompt,
        system,
        model_size,
        response_format,
        retries,
        fallback_model=None,
    ):
        return (
            json.dumps(
                {
                    "contradictions_detected": True,
                    "analysis_cases": [
                        {
                            "conflict_type": "empirical_contextual_boundary",
                            "divergence_summary": "Resultados opuestos por geografía.",
                            "thesis_a": {
                                "proposition_id": "1",
                                "document_id": "doc_001",
                                "claim": "A",
                                "author_or_framework": "Autor A",
                                "empirical_context": "ctx",
                            },
                            "thesis_b": {
                                "proposition_id": "2",
                                "document_id": "doc_001",
                                "claim": "B",
                                "author_or_framework": "Autor B",
                                "empirical_context": "ctx",
                            },
                            "epistemic_reconciliation": "reconciliación",
                        }
                    ],
                }
            ),
            "small",
            False,
        )

    monkeypatch.setattr(ka, "call_with_retries", fake_llm)
    report = ka.resolve_contradictions(session, "q", facts)
    assert report["contradictions_detected"] is True
    assert (
        report["analysis_cases"][0]["conflict_type"] == "empirical_contextual_boundary"
    )


def test_resolve_contradictions_none_detected(monkeypatch):
    session = _FakeSession()
    facts = [{"chunk_id": "1", "atomic_summary": "A"}]

    def fake_llm(
        session,
        *,
        prompt,
        system,
        model_size,
        response_format,
        retries,
        fallback_model=None,
    ):
        return (
            json.dumps({"contradictions_detected": False, "analysis_cases": []}),
            "small",
            False,
        )

    monkeypatch.setattr(ka, "call_with_retries", fake_llm)
    report = ka.resolve_contradictions(session, "q", facts)
    assert report["contradictions_detected"] is False
    assert report["analysis_cases"] == []


def test_resolve_contradictions_empty_facts_short_circuits():
    session = _FakeSession()
    report = ka.resolve_contradictions(session, "q", [])
    assert report == {"contradictions_detected": False, "analysis_cases": []}


def test_resolve_contradictions_degrades_when_llm_fails(monkeypatch):
    session = _FakeSession()
    facts = [{"chunk_id": "1", "atomic_summary": "A"}]
    monkeypatch.setattr(ka, "call_with_retries", _boom)
    report = ka.resolve_contradictions(session, "q", facts)
    assert report == {"contradictions_detected": False, "analysis_cases": []}
    assert session.rollbacks == 1


# ---------------------------------------------------------------------
# 5. Auditoría de suficiencia
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "verdict",
    [
        "SUFFICIENT_FOR_SYNTHESIS",
        "INSUFFICIENT_TRIGGER_BRANCH_B",
        "NEGATIVE_REJECTION",
    ],
)
def test_evaluate_sufficiency_verdicts(monkeypatch, verdict):
    session = _FakeSession()
    facts = [{"relevance_level": "direct_answer", "atomic_summary": "x"}]

    def fake_llm(
        session,
        *,
        prompt,
        system,
        model_size,
        response_format,
        retries,
        fallback_model=None,
    ):
        return (
            json.dumps({"verdict": verdict, "confidence_score": 0.8}),
            "small",
            False,
        )

    monkeypatch.setattr(ka, "call_with_retries", fake_llm)
    result = ka.evaluate_sufficiency(session, "q", facts, [])
    assert result["verdict"] == verdict
    assert result["confidence_score"] == 0.8


def test_evaluate_sufficiency_degrades_when_llm_fails(monkeypatch):
    session = _FakeSession()
    monkeypatch.setattr(ka, "call_with_retries", _boom)
    # Con hechos relevantes -> SUFFICIENT.
    facts = [{"relevance_level": "direct_answer"}]
    result = ka.evaluate_sufficiency(session, "q", facts, [])
    assert result["verdict"] == "SUFFICIENT_FOR_SYNTHESIS"
    # Sin hechos relevantes -> INSUFFICIENT.
    facts2 = [{"relevance_level": "irrelevant"}]
    result2 = ka.evaluate_sufficiency(session, "q", facts2, [])
    assert result2["verdict"] == "INSUFFICIENT_TRIGGER_BRANCH_B"
    assert session.rollbacks == 2


# ---------------------------------------------------------------------
# 6. Branch B: expansión de la recuperación
# ---------------------------------------------------------------------


def test_branch_b_execute_expansion_dedups_and_excludes_visited():
    session = _FakeSession()
    session.chunks = [
        _make_chunk("chunk_1", "stmt 1", cosine=0.1, score=1.0),
        _make_chunk("chunk_2", "stmt 2", cosine=0.2, score=0.9),
        _make_chunk("chunk_3", "stmt 3", cosine=0.3, score=0.8),
    ]
    orch = ka.BranchBOrchestrator(session, max_iterations=2)
    state = {
        "iteration": 0,
        "visited_docs": ["doc_001"],
        "visited_chunks": ["chunk_1"],
        "entity_ids": [],
    }
    evaluation = {
        "branch_b_instructions": {
            "unresolved_subqueries": ["propagación hacia atrás"],
            "target_thesaurus_concepts": ["redes neuronales"],
        }
    }
    new_chunks = orch.execute_expansion(state, evaluation)
    # chunk_1 visitado -> excluido; chunk_2/chunk_3 nuevos y sin duplicados
    # pese a que el FTS corre para 2 subconsultas.
    assert [c["chunk_id"] for c in new_chunks] == ["chunk_2", "chunk_3"]
    assert all(c["chunk_id"] not in state["visited_chunks"] for c in new_chunks)


def test_branch_b_excludes_visited_documents():
    session = _FakeSession()
    session.documents = [
        _make_document("doc_001", title="Visitado"),
        _make_document("doc_002", title="Nuevo"),
    ]
    session.chunks = [_make_chunk("chunk_1", "stmt 1", doc_id="doc_002", score=1.0)]
    orch = ka.BranchBOrchestrator(session, max_iterations=2)
    state = {
        "iteration": 0,
        "visited_docs": ["doc_001"],
        "visited_chunks": [],
        "entity_ids": [],
    }
    evaluation = {
        "branch_b_instructions": {
            "unresolved_subqueries": [],
            "target_thesaurus_concepts": ["psicología"],
        }
    }
    orch.execute_expansion(state, evaluation)
    # La query de documentos se emitió excluyendo los visitados.
    doc_queries = [
        p for s, p in session.executed if "SELECT id, title FROM documents" in s
    ]
    assert doc_queries
    assert _unwrap(doc_queries[0]["visited_docs"]) == ["doc_001"]


def test_branch_b_respects_hop_limit():
    session = _FakeSession()
    orch = ka.BranchBOrchestrator(session, max_iterations=2)
    # iteration > max_iterations -> corta sin ejecutar.
    state = {"iteration": 3, "visited_docs": [], "visited_chunks": [], "entity_ids": []}
    assert orch.execute_expansion(state, {}) == []
    # En el límite (iteration == max_iterations) sí ejecuta (sin subconsultas
    # no hay chunks nuevos, pero no corta).
    state2 = {
        "iteration": 2,
        "visited_docs": [],
        "visited_chunks": [],
        "entity_ids": [],
    }
    assert orch.execute_expansion(state2, {}) == []


def test_branch_b_graph_expansion_adds_neighbor_terms():
    session = _FakeSession()
    session.relations = [
        {"source_entity_id": "e1", "target_entity_id": "e2", "name": "Vecino"}
    ]
    session.chunks = [_make_chunk("chunk_1", "stmt 1", score=1.0)]
    orch = ka.BranchBOrchestrator(session, max_iterations=2)
    state = {
        "iteration": 0,
        "visited_docs": [],
        "visited_chunks": [],
        "entity_ids": ["e1"],
    }
    evaluation = {
        "branch_b_instructions": {
            "unresolved_subqueries": [],
            "target_thesaurus_concepts": [],
        }
    }
    new_chunks = orch.execute_expansion(state, evaluation)
    # El vecino "Vecino" se añadió a las subconsultas FTS.
    fts_queries = [p for s, p in session.executed if "SELECT p.id AS chunk_id" in s]
    assert fts_queries
    assert "Vecino" in _unwrap(fts_queries[0]["tsq"])
    assert [c["chunk_id"] for c in new_chunks] == ["chunk_1"]


# ---------------------------------------------------------------------
# 7. Verificación de grounding verbatim
# ---------------------------------------------------------------------


def test_verify_claim_grounding_exact_match():
    session = _FakeSession()
    session.chunks.append(
        _make_chunk("chunk_1", "La cultura influye en la cognición humana.")
    )
    check = ka.verify_claim_grounding(
        session, "q", "chunk_1", "La cultura influye en la cognición humana."
    )
    assert check["verified"] is True
    assert check["status"] == "VERIFIED"
    assert check["match_score"] == 100.0
    assert check["academic_citations"] == ["Bourdieu, 1984, p. 52"]
    assert check["document_id"] == "doc_001"


def test_verify_claim_grounding_fuzzy_match():
    session = _FakeSession()
    session.chunks.append(
        _make_chunk("chunk_1", "La cultura influye en la cognición humana.")
    )
    # Cita con un error tipográfico (falta el acento): no es substring exacto
    # pero partial_ratio >= 95 -> verificado.
    check = ka.verify_claim_grounding(
        session, "q", "chunk_1", "La cultura influye en la cognicion humana."
    )
    assert check["verified"] is True
    assert check["status"] == "VERIFIED"
    assert 95.0 <= check["match_score"] < 100.0


def test_verify_claim_grounding_no_support_fails():
    session = _FakeSession()
    session.chunks.append(
        _make_chunk("chunk_1", "La cultura influye en la cognición humana.")
    )
    check = ka.verify_claim_grounding(
        session, "q", "chunk_1", "La teoría de la relatividad general."
    )
    assert check["verified"] is False
    assert check["status"] == "FAILED_VERBATIM_MISMATCH"
    assert check["match_score"] < 95.0


def test_verify_claim_grounding_invalid_chunk_id():
    session = _FakeSession()
    check = ka.verify_claim_grounding(session, "q", "chunk_999", "cualquier cosa")
    assert check["verified"] is False
    assert check["status"] == "UNGROUNDED_INVALID_CHUNK_ID"


# ---------------------------------------------------------------------
# 8. Ensamblaje del contexto final
# ---------------------------------------------------------------------


def test_assemble_final_context_builds_evidence_and_tensions():
    evidence = [
        {
            "claim_id": "chunk_1",
            "verified_fact": "La cultura influye en la cognición.",
            "verbatim_quote": "La cultura influye en la cognición humana.",
            "document_title": "Documento de prueba",
            "chapter_title": "Capítulo Uno",
            "bibtex_citation_key": "@book{prueba}",
        }
    ]
    tensions = [
        {
            "tension_label": "paradigmatic_theoretical_divergence",
            "divergence_description": "Dos escuelas divergen.",
            "framework_a": "Escuela A",
            "framework_b": "Escuela B",
        }
    ]
    ctx = ka.assemble_final_context("q", evidence, tensions)
    assert ctx["query"] == "q"
    assert ctx["grounded_evidence"][0]["source_metadata"]["document_title"] == (
        "Documento de prueba"
    )
    assert ctx["grounded_evidence"][0]["source_metadata"]["bibtex_citation_key"] == (
        "@book{prueba}"
    )
    assert ctx["epistemic_tensions"][0]["tension_label"] == (
        "paradigmatic_theoretical_divergence"
    )


def test_assemble_final_context_empty_inputs():
    ctx = ka.assemble_final_context("q", [], [])
    assert ctx["grounded_evidence"] == []
    assert ctx["epistemic_tensions"] == []


# ---------------------------------------------------------------------
# 9. Orquestador completo
# ---------------------------------------------------------------------


def test_ask_propositional_full_pipeline(monkeypatch):
    session = _FakeSession()
    session.documents = [_make_document("doc_001")]
    session.chunks = [
        _make_chunk(
            "chunk_1",
            "La propagación hacia atrás usa la regla de la cadena.",
            cosine=0.1,
            score=1.0,
        ),
        _make_chunk(
            "chunk_2",
            "El gradiente se propaga desde la salida.",
            cosine=0.2,
            score=0.9,
        ),
    ]
    monkeypatch.setattr(
        "src.embeddings.embed_text",
        lambda text, input_type="document": [0.1, 0.2, 0.3],
    )
    monkeypatch.setattr(ka, "call_with_retries", _fake_call_valid)

    result = ka.ask_propositional(
        session, "¿Qué fórmula usa la propagación hacia atrás?"
    )

    assert result["verdict"] == "SUFFICIENT_FOR_SYNTHESIS"
    assert result["answer"] == "Respuesta final de prueba."
    assert result["used_fallback"] is False
    assert len(result["grounded_evidence"]) == 2
    ev = result["grounded_evidence"][0]
    assert ev["document_title"] == "Documento de prueba"
    assert ev["bibtex_citation_key"] == "@book{prueba, title={Documento de prueba}}"
    assert (
        ev["verbatim_quote"] == "La propagación hacia atrás usa la regla de la cadena."
    )
    # La contradicción tipificada por el LLM fake se ensambla como tensión.
    assert len(result["epistemic_tensions"]) == 1
    assert result["epistemic_tensions"][0]["tension_label"] == (
        "paradigmatic_theoretical_divergence"
    )


def test_ask_propositional_negative_rejection(monkeypatch):
    session = _FakeSession()
    session.documents = [_make_document("doc_001")]
    session.chunks = [_make_chunk("chunk_1", "stmt 1", cosine=0.1)]
    monkeypatch.setattr(
        "src.embeddings.embed_text",
        lambda text, input_type="document": [0.1, 0.2, 0.3],
    )

    def fake_llm(
        session,
        *,
        prompt,
        system,
        model_size,
        response_format,
        retries,
        fallback_model=None,
    ):
        if system == ka.SUFFICIENCY_SYSTEM_SHORT:
            return (
                json.dumps(
                    {
                        "verdict": "NEGATIVE_REJECTION",
                        "confidence_score": 0.95,
                        "negative_rejection_details": {
                            "reason": "out_of_thematic_scope_iso25964",
                            "closest_available_topics": ["Psicología"],
                            "formal_abstention_statement": (
                                "El corpus no cubre el dominio requerido."
                            ),
                        },
                        "branch_b_instructions": {},
                    }
                ),
                "small",
                False,
            )
        return ("{}", model_size, False)

    monkeypatch.setattr(ka, "call_with_retries", fake_llm)
    result = ka.ask_propositional(session, "¿Física cuántica?")
    assert result["verdict"] == "NEGATIVE_REJECTION"
    assert result["answer"] == "El corpus no cubre el dominio requerido."
    assert result["grounded_evidence"] == []
    assert result["epistemic_tensions"] == []


def test_ask_propositional_branch_b_expansion(monkeypatch):
    session = _FakeSession()
    session.documents = [_make_document("doc_001")]
    session.chunks = [
        _make_chunk("chunk_1", "stmt 1", cosine=0.1, score=1.0),
        _make_chunk("chunk_2", "stmt 2", cosine=0.2, score=0.9),
        _make_chunk("chunk_3", "stmt 3", cosine=0.9, score=0.8),
        _make_chunk("chunk_4", "stmt 4", cosine=0.8, score=0.7),
    ]
    monkeypatch.setattr(
        "src.embeddings.embed_text",
        lambda text, input_type="document": [0.1, 0.2, 0.3],
    )
    calls = {"sufficiency": 0}

    def fake_llm(
        session,
        *,
        prompt,
        system,
        model_size,
        response_format,
        retries,
        fallback_model=None,
    ):
        if system == ka.SUFFICIENCY_SYSTEM_SHORT:
            calls["sufficiency"] += 1
            if calls["sufficiency"] == 1:
                return (
                    json.dumps(
                        {
                            "verdict": "INSUFFICIENT_TRIGGER_BRANCH_B",
                            "confidence_score": 0.4,
                            "negative_rejection_details": {},
                            "branch_b_instructions": {
                                "unresolved_subqueries": ["regla de la cadena"],
                                "target_thesaurus_concepts": ["cálculo"],
                            },
                        }
                    ),
                    "small",
                    False,
                )
            return (
                json.dumps(
                    {
                        "verdict": "SUFFICIENT_FOR_SYNTHESIS",
                        "confidence_score": 0.9,
                        "negative_rejection_details": {},
                        "branch_b_instructions": {},
                    }
                ),
                "small",
                False,
            )
        return _fake_call_valid(
            session,
            prompt=prompt,
            system=system,
            model_size=model_size,
            response_format=response_format,
            retries=retries,
            fallback_model=fallback_model,
        )

    monkeypatch.setattr(ka, "call_with_retries", fake_llm)
    result = ka.ask_propositional(
        session, "¿Cómo se propaga el gradiente?", top_propositions=2
    )
    assert calls["sufficiency"] == 2
    assert result["verdict"] == "SUFFICIENT_FOR_SYNTHESIS"
    assert result["answer"] == "Respuesta final de prueba."
    # Evidencia incluye hechos de los chunks nuevos (chunk_3/chunk_4).
    assert len(result["grounded_evidence"]) == 4


def test_ask_propositional_degrades_when_llm_fails(monkeypatch):
    session = _FakeSession()
    session.documents = [_make_document("doc_001")]
    session.chunks = [
        _make_chunk(
            "chunk_1", "La cultura influye en la cognición humana.", cosine=0.1
        ),
        _make_chunk(
            "chunk_2", "Las normas sociales determinan el comportamiento.", cosine=0.2
        ),
    ]
    monkeypatch.setattr(
        "src.embeddings.embed_text",
        lambda text, input_type="document": [0.1, 0.2, 0.3],
    )
    monkeypatch.setattr(ka, "call_with_retries", _boom)

    result = ka.ask_propositional(session, "¿Qué influye en la cognición?")

    # Degradación determinista: veredicto por hechos relevantes, evidencia
    # verificada (verbatim == text_span) y respuesta = contexto crudo JSON.
    assert result["verdict"] == "SUFFICIENT_FOR_SYNTHESIS"
    assert result["used_fallback"] is True
    assert len(result["grounded_evidence"]) == 2
    assert isinstance(result["answer"], str)
    assert result["answer"].startswith("{")


def test_ask_propositional_embedding_failure_branch_b_rescues(monkeypatch):
    session = _FakeSession()
    session.documents = [_make_document("doc_001")]
    session.chunks = [_make_chunk("chunk_1", "stmt 1", cosine=0.1, score=1.0)]

    def boom_embed(text, input_type="document"):
        raise RuntimeError("modelo de embeddings no disponible")

    monkeypatch.setattr("src.embeddings.embed_text", boom_embed)
    calls = {"sufficiency": 0}

    def fake_llm(
        session,
        *,
        prompt,
        system,
        model_size,
        response_format,
        retries,
        fallback_model=None,
    ):
        if system == ka.SUFFICIENCY_SYSTEM_SHORT:
            calls["sufficiency"] += 1
            if calls["sufficiency"] == 1:
                return (
                    json.dumps(
                        {
                            "verdict": "INSUFFICIENT_TRIGGER_BRANCH_B",
                            "confidence_score": 0.4,
                            "negative_rejection_details": {},
                            "branch_b_instructions": {
                                "unresolved_subqueries": ["stmt"],
                                "target_thesaurus_concepts": [],
                            },
                        }
                    ),
                    "small",
                    False,
                )
            return (
                json.dumps(
                    {
                        "verdict": "SUFFICIENT_FOR_SYNTHESIS",
                        "confidence_score": 0.9,
                        "negative_rejection_details": {},
                        "branch_b_instructions": {},
                    }
                ),
                "small",
                False,
            )
        return _fake_call_valid(
            session,
            prompt=prompt,
            system=system,
            model_size=model_size,
            response_format=response_format,
            retries=retries,
            fallback_model=fallback_model,
        )

    monkeypatch.setattr(ka, "call_with_retries", fake_llm)
    result = ka.ask_propositional(session, "q")
    # Sin embedding: Nivel 1 vacío -> Branch B rescata chunks vía FTS.
    assert calls["sufficiency"] == 2
    assert result["verdict"] == "SUFFICIENT_FOR_SYNTHESIS"
    assert len(result["grounded_evidence"]) == 1
