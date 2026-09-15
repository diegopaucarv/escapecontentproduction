"""Tests del modo audited optimizado — fusión epistémica, Branch B
short-circuit y poda de grounding.

Sin DB real, sin torch/spacy/transformers. Sesiones falsas + monkeypatch
(patrón de tests/test_kag_unified.py).
"""

from types import SimpleNamespace

import pytest

from src.kag_query import (
    _audit_epistemic_fused,
    _fused_facts_to_shape,
    _kag_config_value,
    _verify_grounding,
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


class _FakeSession:
    """Sesión falsa: execute devuelve _Result vacío (load_settings falla)."""

    def __init__(self, rows=None):
        self._rows = rows or []

    def execute(self, stmt, params=None):
        return _Result(self._rows)

    def rollback(self):
        pass


def _proposition(pid, stmt, span, doc_id=7, citations=None):
    return {
        "chunk_id": pid,
        "document_id": doc_id,
        "statement": stmt,
        "text_span": span,
        "citation_references": citations or [],
        "doc_title": "doc.md",
    }


# ---------------------------------------------------------------------
# 1. Fusión epistémica: mismo shape de salida
# ---------------------------------------------------------------------


def test_audit_fused_same_shape_as_separate(monkeypatch):
    import src.kag_query as kq

    props = [
        _proposition(
            1,
            "La cultura afecta a las organizaciones.",
            "La cultura afecta a las organizaciones.",
        ),
        _proposition(
            2, "El liderazgo transforma equipos.", "El liderazgo transforma equipos."
        ),
    ]
    fused_json = (
        '{"facts": ['
        '{"statement": "La cultura afecta a las organizaciones.", '
        '"verbatim_evidence": "La cultura afecta a las organizaciones.", '
        '"relevance": "direct_answer"}, '
        '{"statement": "El liderazgo transforma equipos.", '
        '"verbatim_evidence": "El liderazgo transforma equipos.", '
        '"relevance": "supporting_evidence"}], '
        '"contradictions": [{"type": "paradigmatic_theoretical_divergence", '
        '"resolution": "Ambas son compatibles en niveles distintos."}], '
        '"sufficiency": {"verdict": "SUFFICIENT", "confidence": 0.9}}'
    )

    def _fake_call(
        session,
        *,
        prompt,
        system,
        model_size,
        response_format,
        retries,
        fallback_model=None,
    ):
        return fused_json, "small", False

    monkeypatch.setattr(kq, "call_with_retries", _fake_call)
    session = _FakeSession()
    facts, report, evaluation = _audit_epistemic_fused(
        session, "¿qué es la cultura?", props, []
    )

    # Shape de facts = shape de _synthesize_facts.
    assert isinstance(facts, list) and len(facts) == 2
    for f in facts:
        assert set(f) >= {
            "chunk_id",
            "document_id",
            "source_file",
            "relevance_level",
            "atomic_summary",
            "verbatim_evidence",
            "academic_citations",
        }
    assert facts[0]["relevance_level"] == "direct_answer"
    assert facts[0]["chunk_id"] == "1"
    assert facts[0]["academic_citations"] == []

    # Shape de report = shape de _resolve_contradictions.
    assert report["contradictions_detected"] is True
    assert len(report["analysis_cases"]) == 1
    case = report["analysis_cases"][0]
    assert case["conflict_type"] == "paradigmatic_theoretical_divergence"
    assert case["divergence_summary"] == "Ambas son compatibles en niveles distintos."

    # Shape de evaluation = shape de _evaluate_sufficiency.
    assert evaluation["verdict"] == "SUFFICIENT_FOR_SYNTHESIS"
    assert evaluation["confidence_score"] == 0.9


def test_audit_fused_insufficient_triggers_branch_b(monkeypatch):
    import src.kag_query as kq

    props = [_proposition(1, "Solo contexto.", "Solo contexto.")]
    fused_json = (
        '{"facts": [{"statement": "Solo contexto.", '
        '"verbatim_evidence": "Solo contexto.", '
        '"relevance": "contextual_background"}], '
        '"contradictions": [], '
        '"sufficiency": {"verdict": "INSUFFICIENT", "confidence": 0.3}}'
    )

    def _fake_call(
        session,
        *,
        prompt,
        system,
        model_size,
        response_format,
        retries,
        fallback_model=None,
    ):
        return fused_json, "small", False

    monkeypatch.setattr(kq, "call_with_retries", _fake_call)
    session = _FakeSession()
    facts, report, evaluation = _audit_epistemic_fused(session, "q", props, [])
    assert evaluation["verdict"] == "INSUFFICIENT_TRIGGER_BRANCH_B"
    assert evaluation["confidence_score"] == 0.3
    assert report["contradictions_detected"] is False


def test_audit_fused_negative_rejection(monkeypatch):
    import src.kag_query as kq

    props = [_proposition(1, "X", "X")]
    fused_json = (
        '{"facts": [], "contradictions": [], '
        '"sufficiency": {"verdict": "NEGATIVE_REJECTION", "confidence": 0.95}}'
    )

    def _fake_call(
        session,
        *,
        prompt,
        system,
        model_size,
        response_format,
        retries,
        fallback_model=None,
    ):
        return fused_json, "small", False

    monkeypatch.setattr(kq, "call_with_retries", _fake_call)
    session = _FakeSession()
    facts, report, evaluation = _audit_epistemic_fused(session, "q", props, [])
    assert evaluation["verdict"] == "NEGATIVE_REJECTION"
    assert "formal_abstention_statement" in evaluation["negative_rejection_details"]


# ---------------------------------------------------------------------
# 2. JSON malformado → degradación sin romper
# ---------------------------------------------------------------------


def test_audit_fused_malformed_json_degrades(monkeypatch):
    import src.kag_query as kq

    props = [
        _proposition(1, "Hecho A.", "Hecho A."),
        _proposition(2, "Hecho B.", "Hecho B."),
    ]
    calls = []

    def _fake_call(
        session,
        *,
        prompt,
        system,
        model_size,
        response_format,
        retries,
        fallback_model=None,
    ):
        calls.append(model_size)
        # Primera llamada (fusionada): JSON malformado.
        if len(calls) == 1:
            return "esto no es json {", "small", False
        # Llamadas de degradación: síntesis → contradicciones → suficiencia.
        if len(calls) == 2:
            return (
                '{"synthesized_facts": [{"chunk_id": "1", "document_id": "7", '
                '"source_file": "doc.md", "relevance_level": "direct_answer", '
                '"atomic_summary": "Hecho A.", "verbatim_evidence": "Hecho A.", '
                '"academic_citations": []}]}',
                "small",
                False,
            )
        if len(calls) == 3:
            return (
                '{"contradictions_detected": false, "analysis_cases": []}',
                "small",
                False,
            )
        return (
            '{"verdict": "SUFFICIENT_FOR_SYNTHESIS", "confidence_score": 0.5}',
            "small",
            False,
        )

    monkeypatch.setattr(kq, "call_with_retries", _fake_call)
    session = _FakeSession()
    facts, report, evaluation = _audit_epistemic_fused(session, "q", props, [])
    # Degradó a las 3 llamadas separadas (4 llamadas en total).
    assert len(calls) == 4
    assert len(facts) == 1
    assert facts[0]["relevance_level"] == "direct_answer"
    assert report["contradictions_detected"] is False
    assert evaluation["verdict"] == "SUFFICIENT_FOR_SYNTHESIS"


def test_audit_fused_invalid_verdict_degrades(monkeypatch):
    import src.kag_query as kq

    props = [_proposition(1, "Hecho A.", "Hecho A.")]
    calls = []

    def _fake_call(
        session,
        *,
        prompt,
        system,
        model_size,
        response_format,
        retries,
        fallback_model=None,
    ):
        calls.append(model_size)
        if len(calls) == 1:
            # Veredicto inválido → degradación.
            return (
                '{"facts": [{"statement": "Hecho A.", "verbatim_evidence": "Hecho A.", '
                '"relevance": "direct_answer"}], "contradictions": [], '
                '"sufficiency": {"verdict": "TAL_VEZ", "confidence": 0.5}}',
                "small",
                False,
            )
        if len(calls) == 2:
            return (
                '{"synthesized_facts": [{"chunk_id": "1", "document_id": "7", '
                '"source_file": "doc.md", "relevance_level": "direct_answer", '
                '"atomic_summary": "Hecho A.", "verbatim_evidence": "Hecho A.", '
                '"academic_citations": []}]}',
                "small",
                False,
            )
        if len(calls) == 3:
            return (
                '{"contradictions_detected": false, "analysis_cases": []}',
                "small",
                False,
            )
        return (
            '{"verdict": "SUFFICIENT_FOR_SYNTHESIS", "confidence_score": 0.5}',
            "small",
            False,
        )

    monkeypatch.setattr(kq, "call_with_retries", _fake_call)
    session = _FakeSession()
    facts, report, evaluation = _audit_epistemic_fused(session, "q", props, [])
    assert len(calls) == 4
    assert evaluation["verdict"] == "SUFFICIENT_FOR_SYNTHESIS"


def test_audit_fused_empty_propositions_no_llm():
    session = _FakeSession()
    facts, report, evaluation = _audit_epistemic_fused(session, "q", [], [])
    assert facts == []
    assert report == {"contradictions_detected": False, "analysis_cases": []}
    assert evaluation["verdict"] == "INSUFFICIENT_TRIGGER_BRANCH_B"


# ---------------------------------------------------------------------
# _fused_facts_to_shape
# ---------------------------------------------------------------------


def test_fused_facts_to_shape_anchors_to_propositions():
    props = [
        _proposition(1, "Stmt A", "Span A", citations=["Ref 1"]),
        _proposition(2, "Stmt B", "Span B"),
    ]
    raw = [
        {
            "statement": "Stmt A",
            "verbatim_evidence": "Span A",
            "relevance": "direct_answer",
        },
        {
            "statement": "Stmt B",
            "verbatim_evidence": "Span B",
            "relevance": "contextual_background",
        },
        {"statement": "Stmt C", "verbatim_evidence": "Span C", "relevance": "raro"},
    ]
    out = _fused_facts_to_shape(raw, props)
    assert len(out) == 3
    assert out[0]["chunk_id"] == "1"
    assert out[0]["academic_citations"] == ["Ref 1"]
    assert out[0]["relevance_level"] == "direct_answer"
    # Relevancia inválida → supporting_evidence.
    assert out[2]["relevance_level"] == "supporting_evidence"
    # Sin proposición ancla → chunk_id vacío, no rompe.
    assert out[2]["chunk_id"] == ""


def test_fused_facts_to_shape_non_list():
    assert _fused_facts_to_shape(None, []) == []
    assert _fused_facts_to_shape("nope", []) == []


# ---------------------------------------------------------------------
# 3. Branch B short-circuit
# ---------------------------------------------------------------------


def test_branch_b_aborts_on_marginal_gain(monkeypatch):
    """Δ < 0.05 → aborta de inmediato (una sola iteración)."""
    import src.kag_query as kq

    session = _FakeSession()
    calls = {"expand": 0, "suff": 0}

    def _fake_expand(
        session, query, entity_ids, visited_chunks, top_k=8, verbose=False
    ):
        calls["expand"] += 1
        return [
            {
                "chunk_id": 100,
                "document_id": 9,
                "statement": "Nuevo hecho.",
                "text_span": "Nuevo hecho.",
                "citation_references": [],
                "doc_title": "doc2.md",
            }
        ]

    def _fake_synth(session, query, propositions, verbose=False):
        return [
            {
                "chunk_id": str(p.get("chunk_id")),
                "document_id": str(p.get("document_id") or ""),
                "source_file": p.get("doc_title", ""),
                "relevance_level": "supporting_evidence",
                "atomic_summary": p.get("statement", ""),
                "verbatim_evidence": p.get("text_span", ""),
                "academic_citations": p.get("citation_references") or [],
            }
            for p in propositions
        ]

    def _fake_suff(session, query, facts, corpus_metadata, verbose=False):
        calls["suff"] += 1
        # Confianza sube solo 0.02 → ganancia marginal.
        return {"verdict": "SUFFICIENT_FOR_SYNTHESIS", "confidence_score": 0.52}

    monkeypatch.setattr(kq, "_branch_b_expand", _fake_expand)
    monkeypatch.setattr(kq, "_synthesize_facts", _fake_synth)
    monkeypatch.setattr(kq, "_evaluate_sufficiency", _fake_suff)

    # Simula el bucle de _ask_audited con veredicto inicial INSUFFICIENT.
    verdict = "INSUFFICIENT_TRIGGER_BRANCH_B"
    facts = [
        {
            "chunk_id": "1",
            "relevance_level": "contextual_background",
            "atomic_summary": "ctx",
            "verbatim_evidence": "ctx",
        }
    ]
    visited_chunks = ["1"]
    max_iterations = 2
    iteration = 0
    prev_confidence = 0.5
    while (
        verdict == "INSUFFICIENT_TRIGGER_BRANCH_B"
        and max_iterations > 0
        and iteration < max_iterations
    ):
        extra = _fake_expand(session, "q", [], visited_chunks)
        if not extra:
            break
        new_doc_ids = {p.get("document_id") for p in extra if p.get("document_id")}
        new_entities = set()
        for p in extra:
            for ent in p.get("entities") or []:
                if ent:
                    new_entities.add(str(ent))
        if not new_doc_ids and not new_entities:
            break
        extra_facts = _fake_synth(session, "q", extra)
        facts = list(facts) + extra_facts
        evaluation = _fake_suff(session, "q", facts, [])
        verdict = evaluation.get("verdict", "SUFFICIENT_FOR_SYNTHESIS")
        new_confidence = float(evaluation.get("confidence_score") or 0.0)
        gain = new_confidence - prev_confidence
        prev_confidence = new_confidence
        iteration += 1
        visited_chunks = list(
            dict.fromkeys(visited_chunks + [p.get("chunk_id") for p in extra])
        )
        if gain < 0.05:
            break

    assert calls["expand"] == 1
    assert calls["suff"] == 1
    assert iteration == 1


def test_branch_b_aborts_without_new_doc_ids(monkeypatch):
    """Sin doc_ids ni entidades nuevos → aborta sin llamar a suficiencia."""
    import src.kag_query as kq

    session = _FakeSession()
    calls = {"expand": 0, "suff": 0}

    def _fake_expand(
        session, query, entity_ids, visited_chunks, top_k=8, verbose=False
    ):
        calls["expand"] += 1
        # document_id None y sin entities → sin novedad.
        return [
            {
                "chunk_id": 100,
                "document_id": None,
                "statement": "Nuevo.",
                "text_span": "Nuevo.",
                "citation_references": [],
                "doc_title": "doc2.md",
            }
        ]

    def _fake_suff(session, query, facts, corpus_metadata, verbose=False):
        calls["suff"] += 1
        return {"verdict": "SUFFICIENT_FOR_SYNTHESIS", "confidence_score": 0.9}

    def _fake_synth(session, query, propositions, verbose=False):
        return []

    monkeypatch.setattr(kq, "_branch_b_expand", _fake_expand)
    monkeypatch.setattr(kq, "_evaluate_sufficiency", _fake_suff)

    verdict = "INSUFFICIENT_TRIGGER_BRANCH_B"
    facts = []
    visited_chunks = ["1"]
    max_iterations = 2
    iteration = 0
    prev_confidence = 0.5
    while (
        verdict == "INSUFFICIENT_TRIGGER_BRANCH_B"
        and max_iterations > 0
        and iteration < max_iterations
    ):
        extra = _fake_expand(session, "q", [], visited_chunks)
        if not extra:
            break
        new_doc_ids = {p.get("document_id") for p in extra if p.get("document_id")}
        new_entities = set()
        for p in extra:
            for ent in p.get("entities") or []:
                if ent:
                    new_entities.add(str(ent))
        if not new_doc_ids and not new_entities:
            break
        extra_facts = _fake_synth(session, "q", extra)
        facts = list(facts) + extra_facts
        evaluation = _fake_suff(session, "q", facts, [])
        verdict = evaluation.get("verdict", "SUFFICIENT_FOR_SYNTHESIS")
        new_confidence = float(evaluation.get("confidence_score") or 0.0)
        gain = new_confidence - prev_confidence
        prev_confidence = new_confidence
        iteration += 1
        visited_chunks = list(
            dict.fromkeys(visited_chunks + [p.get("chunk_id") for p in extra])
        )
        if gain < 0.05:
            break

    assert calls["expand"] == 1
    assert calls["suff"] == 0
    assert iteration == 0


def test_branch_b_max_iters_zero_disables(monkeypatch):
    """KAG_BRANCH_B_MAX_ITERS=0 → el bucle no se ejecuta."""
    import src.kag_query as kq

    session = _FakeSession()
    calls = {"expand": 0}

    def _fake_expand(
        session, query, entity_ids, visited_chunks, top_k=8, verbose=False
    ):
        calls["expand"] += 1
        return []

    monkeypatch.setattr(kq, "_branch_b_expand", _fake_expand)

    verdict = "INSUFFICIENT_TRIGGER_BRANCH_B"
    max_iterations = 0
    iteration = 0
    while (
        verdict == "INSUFFICIENT_TRIGGER_BRANCH_B"
        and max_iterations > 0
        and iteration < max_iterations
    ):
        extra = _fake_expand(session, "q", [], [])
        if not extra:
            break
        iteration += 1

    assert calls["expand"] == 0
    assert iteration == 0


def test_kag_config_value_defaults_and_override(monkeypatch):
    session = _FakeSession()
    # Sin env ni DB → defaults.
    assert _kag_config_value(session, "KAG_BRANCH_B_MAX_ITERS", 2) == 2
    assert _kag_config_value(session, "KAG_GROUNDING_THRESHOLD", 95.0) == 95.0
    # Flag desconocida → default del caller.
    assert _kag_config_value(session, "KAG_NO_EXISTE", "x") == "x"

    # Override vía env var.
    monkeypatch.setenv("KAG_BRANCH_B_MAX_ITERS", "0")
    assert _kag_config_value(session, "KAG_BRANCH_B_MAX_ITERS", 2) == 0
    monkeypatch.delenv("KAG_BRANCH_B_MAX_ITERS")


# ---------------------------------------------------------------------
# 4. Poda de grounding: solo direct_answer/citados pasan fuzzy
# ---------------------------------------------------------------------


def _grounding_row(text_span="La cultura afecta a las organizaciones."):
    return SimpleNamespace(
        text_span=text_span,
        citation_references=["Ref"],
        doc_id=1,
        doc_title="doc.md",
        chapter_title="## Cap",
    )


def test_verify_grounding_contextual_skips_fuzzy(monkeypatch):
    """supporting_evidence sin citas (contextual) NO pasa verificación difusa."""
    import src.kag_query as kq

    session = _FakeSession([_grounding_row("texto completamente distinto")])
    facts = [
        {
            "relevance_level": "supporting_evidence",
            "chunk_id": "5",
            "verbatim_evidence": "no coincide con nada",
            "atomic_summary": "resumen",
            "academic_citations": [],
        }
    ]
    # Sin rapidfuzz disponible → el fuzzy no se aplica igualmente.
    monkeypatch.setattr(kq, "fuzz", None, raising=False)
    out = _verify_grounding(session, facts)
    # Contextual de fondo: pasa sin verificación carácter por carácter.
    assert len(out) == 1
    assert out[0]["verified_fact"] == "resumen"


def test_verify_grounding_direct_answer_still_fuzzy(monkeypatch):
    """direct_answer sin coincidencia → rechazado por fuzzy."""
    session = _FakeSession([_grounding_row("texto completamente distinto")])
    facts = [
        {
            "relevance_level": "direct_answer",
            "chunk_id": "5",
            "verbatim_evidence": "no coincide con nada",
            "atomic_summary": "resumen",
            "academic_citations": [],
        }
    ]
    out = _verify_grounding(session, facts)
    assert out == []


def test_verify_grounding_cited_fact_still_fuzzy(monkeypatch):
    """supporting_evidence CON citas → verificación difusa aplica."""
    session = _FakeSession([_grounding_row("texto completamente distinto")])
    facts = [
        {
            "relevance_level": "supporting_evidence",
            "chunk_id": "5",
            "verbatim_evidence": "no coincide con nada",
            "atomic_summary": "resumen",
            "academic_citations": ["Ref 1"],
        }
    ]
    out = _verify_grounding(session, facts)
    assert out == []


def test_verify_grounding_exact_match_still_passes():
    """Substring exacto → pasa para direct_answer y contextual."""
    session = _FakeSession([_grounding_row("La cultura afecta a las organizaciones.")])
    facts = [
        {
            "relevance_level": "direct_answer",
            "chunk_id": "5",
            "verbatim_evidence": "La cultura afecta",
            "atomic_summary": "resumen",
            "academic_citations": [],
        }
    ]
    out = _verify_grounding(session, facts)
    assert len(out) == 1
    assert out[0]["document_title"] == "doc.md"
