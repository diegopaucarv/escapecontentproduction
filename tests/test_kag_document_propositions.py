"""Tests de la Fase 5 (proposiciones + entidades + relaciones por documento
con paráfrasis) — sin DB real, sin torch/spacy/transformers.

Patrón de tests/test_kag_propositions.py: sesiones falsas + monkeypatch de
call_with_retries/_get_prompt_pair. NO se importa src.kag.segmentador a
nivel de módulo (carga torch/spacy, ~30s).

Cubre:
  - _extract_document_propositions: proposiciones asignadas por chunk_index,
    entidades+relaciones persistidas vía _store_entities_relations, fallback
    a content cuando paraphrase NULL, cache por content_hash, degradación si
    el LLM falla.
  - _extract_document_batch: chunk_index inválido → fallback a
    _locate_span_in_batch; ambos fallan → proposición descartada.
"""

import json
from types import SimpleNamespace

import src.kag_ingest as ki


class _Result:
    def __init__(self, first=None, scalar=None, fetchall=None):
        self._first = first
        self._scalar = scalar
        self._fetchall = fetchall if fetchall is not None else []

    def scalars(self):
        return self

    def first(self):
        return self._first

    def scalar(self):
        return self._scalar

    def fetchall(self):
        return self._fetchall


class _EmptySession:
    """Sesión vacía: load_settings devuelve None (settings por defecto)."""

    def execute(self, stmt, params=None):
        return _Result()


class _DocPropsSession:
    """Sesión fake para _extract_document_propositions: captura los UPDATE de
    content_hash y las llamadas a _store_entities_relations."""

    def __init__(self, chunk_rows, cached_chunk_ids):
        self._chunk_rows = chunk_rows
        self._cached = cached_chunk_ids
        self.updates = []  # params de los UPDATE kag_chunks
        self.entity_calls = []  # (chunk_id, data) de _store_entities_relations
        self.commits = 0
        self.rollbacks = 0

    def execute(self, stmt, params=None):
        sql = str(stmt)
        if "FROM kag_chunks" in sql and "ORDER BY chunk_index" in sql:
            return _Result(fetchall=self._chunk_rows)
        if "FROM kag_propositions" in sql:
            return _Result(fetchall=[(cid,) for cid in self._cached])
        if "UPDATE kag_chunks" in sql:
            self.updates.append(params)
            return _Result()
        return _Result()

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


def _chunk_row(
    cid, content, paraphrase, content_hash, chapter_id="s1", chunk_index=None
):
    return SimpleNamespace(
        id=cid,
        content=content,
        chunk_index=chunk_index if chunk_index is not None else cid,
        paraphrase=paraphrase,
        chapter_id=chapter_id,
        content_hash=content_hash,
    )


def _fake_prompt_pair(session, model_name, task_key, system_fallback, user_fallback):
    """Devuelve los fallbacks (sin artefacto compilado, como en tests sin DB)."""
    return system_fallback, user_fallback


def _patch_own_session(monkeypatch, session):
    """Los workers de _run_proposition_batches_parallel abren su propia sesión
    real (run_in_own_session, src/db/session.py) — sin DB real en tests, se
    parchea para que usen la sesión fake del test directamente."""
    monkeypatch.setattr(
        ki,
        "_with_own_session",
        lambda fn, *a, **k: fn(session, *a, **k),
    )


def _llm_ok(props, entities=None, relations=None):
    """Fake de call_with_retries que devuelve el JSON del schema Fase 5."""

    def fake_call(session, prompt, system=None, model_size=None, **kwargs):
        return (
            json.dumps(
                {
                    "propositions": props,
                    "entities": entities or [],
                    "relations": relations or [],
                }
            ),
            "model",
            False,
        )

    return fake_call


# ---------------------------------------------------------------------
# _extract_document_propositions
# ---------------------------------------------------------------------


def test_extract_document_propositions_assigns_by_chunk_index_and_stores_entities(
    monkeypatch,
):
    """Proposiciones asignadas por chunk_index; entidades+relaciones se
    persisten vía _store_entities_relations contra el primer chunk del lote."""
    h1 = ki._chunk_content_hash("aaa")
    h2 = ki._chunk_content_hash("bbb")
    rows = [
        _chunk_row(1, "aaa", "Paráfrasis A.", h1, chunk_index=0),
        _chunk_row(2, "bbb", "Paráfrasis B.", h2, chunk_index=1),
    ]
    session = _DocPropsSession(rows, cached_chunk_ids=[])
    stored_props = []
    stored_entities = []
    _patch_own_session(monkeypatch, session)

    fake_call = _llm_ok(
        props=[
            {
                "chunk_index": 0,
                "core_idea_id": "CI1",
                "argument_id": "A1",
                "statement": "Prop 1.",
                "text_span": "Paráfrasis A.",
                "citations_references": ["Ref 1"],
            },
            {
                "chunk_index": 1,
                "core_idea_id": "CI2",
                "argument_id": "A2",
                "statement": "Prop 2.",
                "text_span": "Paráfrasis B.",
                "citations_references": [],
            },
        ],
        entities=[{"name": "Entidad X", "type": "concept", "description": "Desc X"}],
        relations=[
            {
                "source": "Entidad X",
                "target": "Entidad Y",
                "type": "RELACIONA",
                "description": "Rel",
            }
        ],
    )
    monkeypatch.setattr("src.kag_ingest.call_with_retries", fake_call)
    monkeypatch.setattr(ki, "_get_prompt_pair", _fake_prompt_pair)
    monkeypatch.setattr(
        ki,
        "_store_propositions",
        lambda s, d, props, embed_fn=None: stored_props.append(props),
    )
    monkeypatch.setattr(
        ki,
        "_store_entities_relations",
        lambda s, d, cid, data, embed_fn=None: stored_entities.append((cid, data)),
    )
    ki._extract_document_propositions(
        session, doc_id=1, doc_path="doc.md", verbose=False, batch_size_tokens=250000
    )
    # Proposiciones asignadas por chunk_index (0 → chunk 1, 1 → chunk 2).
    assert [p["chunk_id"] for p in stored_props[0]] == [1, 2]
    assert stored_props[0][0]["chapter_id"] == "s1"
    assert stored_props[0][0]["citations"] == ["Ref 1"]
    # Entidades + relaciones persistidas contra el primer chunk del lote.
    assert len(stored_entities) == 1
    cid, data = stored_entities[0]
    assert cid == 1
    assert data["entities"] == [
        {"name": "Entidad X", "type": "concept", "description": "Desc X"}
    ]
    assert data["relations"][0]["source"] == "Entidad X"
    # content_hash actualizado para los chunks extraídos.
    assert {u["id"] for u in session.updates} == {1, 2}


def test_extract_document_propositions_fallback_to_content_when_paraphrase_null(
    monkeypatch,
):
    """Paraphrase NULL → el texto procesado es el content (degradación
    natural si la Fase 4 no corrió)."""
    h1 = ki._chunk_content_hash("aaa")
    rows = [_chunk_row(1, "aaa", None, h1)]  # paraphrase NULL
    session = _DocPropsSession(rows, cached_chunk_ids=[])
    seen_prompts = []
    stored = []
    _patch_own_session(monkeypatch, session)

    def fake_call(session, prompt, system=None, model_size=None, **kwargs):
        seen_prompts.append(prompt)
        return (
            json.dumps(
                {
                    "propositions": [
                        {
                            "chunk_index": 0,
                            "core_idea_id": "CI",
                            "argument_id": "A",
                            "statement": "Prop.",
                            "text_span": "aaa",
                            "citations_references": [],
                        }
                    ],
                    "entities": [],
                    "relations": [],
                }
            ),
            "model",
            False,
        )

    monkeypatch.setattr("src.kag_ingest.call_with_retries", fake_call)
    monkeypatch.setattr(ki, "_get_prompt_pair", _fake_prompt_pair)
    monkeypatch.setattr(
        ki,
        "_store_propositions",
        lambda s, d, props, embed_fn=None: stored.append(props),
    )
    ki._extract_document_propositions(
        session, doc_id=1, doc_path="doc.md", verbose=False, batch_size_tokens=250000
    )
    # El prompt contiene el content (paraphrase NULL → fallback).
    assert "aaa" in seen_prompts[0]
    assert stored[0][0]["chunk_id"] == 1


def test_extract_document_propositions_cache_skips_extracted(monkeypatch):
    """Cache por content_hash: chunks ya extraídos con hash idéntico se
    saltan (no re-extraer en re-ingestas)."""
    h1 = ki._chunk_content_hash("aaa")
    h2 = ki._chunk_content_hash("bbb")
    rows = [
        _chunk_row(1, "aaa", "Paráfrasis A.", h1),  # cache hit
        _chunk_row(2, "bbb", "Paráfrasis B.", h2),  # sin proposiciones → extraer
    ]
    session = _DocPropsSession(rows, cached_chunk_ids=[1])
    called = []
    stored = []
    _patch_own_session(monkeypatch, session)

    def fake_call(session, prompt, system=None, model_size=None, **kwargs):
        called.append(prompt)
        return (
            json.dumps(
                {
                    "propositions": [
                        {
                            "chunk_index": 1,
                            "core_idea_id": "CI",
                            "argument_id": "A",
                            "statement": "Prop.",
                            "text_span": "Paráfrasis B.",
                            "citations_references": [],
                        }
                    ],
                    "entities": [],
                    "relations": [],
                }
            ),
            "model",
            False,
        )

    monkeypatch.setattr("src.kag_ingest.call_with_retries", fake_call)
    monkeypatch.setattr(ki, "_get_prompt_pair", _fake_prompt_pair)
    monkeypatch.setattr(
        ki,
        "_store_propositions",
        lambda s, d, props, embed_fn=None: stored.append(props),
    )
    ki._extract_document_propositions(
        session, doc_id=1, doc_path="doc.md", verbose=False, batch_size_tokens=250000
    )
    # Solo el lote con el chunk 2 se extrae (el 1 está en cache).
    assert len(called) == 1
    assert [p["chunk_id"] for p in stored[0]] == [2]
    assert {u["id"] for u in session.updates} == {2}


def test_extract_document_propositions_all_cached_skips_llm(monkeypatch):
    """Todos los chunks en cache → no se llama al LLM."""
    h1 = ki._chunk_content_hash("aaa")
    rows = [_chunk_row(1, "aaa", "Paráfrasis A.", h1)]
    session = _DocPropsSession(rows, cached_chunk_ids=[1])
    called = []
    monkeypatch.setattr(
        "src.kag_ingest.call_with_retries",
        lambda *a, **k: called.append(1) or ("{}", "model", False),
    )
    monkeypatch.setattr(ki, "_get_prompt_pair", _fake_prompt_pair)
    ki._extract_document_propositions(
        session, doc_id=1, doc_path="doc.md", verbose=False, batch_size_tokens=250000
    )
    assert called == []


def test_extract_document_propositions_llm_failure_degrades(monkeypatch):
    """LLM falla → skip del lote, la ingesta sigue (sin excepción, sin
    proposiciones persistidas)."""
    h1 = ki._chunk_content_hash("aaa")
    rows = [_chunk_row(1, "aaa", "Paráfrasis A.", h1)]
    session = _DocPropsSession(rows, cached_chunk_ids=[])
    stored = []
    _patch_own_session(monkeypatch, session)

    def boom(session, prompt, system=None, model_size=None, **kwargs):
        raise RuntimeError("LLM no disponible")

    monkeypatch.setattr("src.kag_ingest.call_with_retries", boom)
    monkeypatch.setattr(ki, "_get_prompt_pair", _fake_prompt_pair)
    monkeypatch.setattr(
        ki,
        "_store_propositions",
        lambda s, d, props, embed_fn=None: stored.append(props) if props else None,
    )
    # No debe lanzar excepción.
    ki._extract_document_propositions(
        session, doc_id=1, doc_path="doc.md", verbose=False, batch_size_tokens=250000
    )
    assert stored == []  # sin proposiciones persistidas
    assert session.entity_calls == []  # sin entidades


def test_extract_document_propositions_invalid_json_degrades(monkeypatch):
    """JSON inválido del LLM → skip del lote (degradación)."""
    h1 = ki._chunk_content_hash("aaa")
    rows = [_chunk_row(1, "aaa", "Paráfrasis A.", h1)]
    session = _DocPropsSession(rows, cached_chunk_ids=[])
    stored = []
    _patch_own_session(monkeypatch, session)

    def fake_call(session, prompt, system=None, model_size=None, **kwargs):
        return "no es json", "model", False

    monkeypatch.setattr("src.kag_ingest.call_with_retries", fake_call)
    monkeypatch.setattr(ki, "_get_prompt_pair", _fake_prompt_pair)
    monkeypatch.setattr(
        ki,
        "_store_propositions",
        lambda s, d, props, embed_fn=None: stored.append(props) if props else None,
    )
    ki._extract_document_propositions(
        session, doc_id=1, doc_path="doc.md", verbose=False, batch_size_tokens=250000
    )
    assert stored == []


def test_extract_document_propositions_empty_entities_only_propositions(monkeypatch):
    """Entidades vacías → solo proposiciones (sin llamada a
    _store_entities_relations)."""
    h1 = ki._chunk_content_hash("aaa")
    rows = [_chunk_row(1, "aaa", "Paráfrasis A.", h1)]
    session = _DocPropsSession(rows, cached_chunk_ids=[])
    stored = []
    _patch_own_session(monkeypatch, session)

    fake_call = _llm_ok(
        props=[
            {
                "chunk_index": 0,
                "core_idea_id": "CI",
                "argument_id": "A",
                "statement": "Prop.",
                "text_span": "Paráfrasis A.",
                "citations_references": [],
            }
        ]
    )
    monkeypatch.setattr("src.kag_ingest.call_with_retries", fake_call)
    monkeypatch.setattr(ki, "_get_prompt_pair", _fake_prompt_pair)
    monkeypatch.setattr(
        ki,
        "_store_propositions",
        lambda s, d, props, embed_fn=None: stored.append(props),
    )
    ki._extract_document_propositions(
        session, doc_id=1, doc_path="doc.md", verbose=False, batch_size_tokens=250000
    )
    assert [p["chunk_id"] for p in stored[0]] == [1]
    assert session.entity_calls == []


# ---------------------------------------------------------------------
# _extract_document_batch
# ---------------------------------------------------------------------


def test_extract_document_batch_invalid_chunk_index_falls_back_to_span(monkeypatch):
    """chunk_index inválido → fallback a _locate_span_in_batch sobre el texto
    concatenado del lote."""
    chunks = [
        {"id": 1, "content": "Paráfrasis A.", "chunk_index": 0, "chapter_id": "s1"},
        {"id": 2, "content": "Paráfrasis B.", "chunk_index": 1, "chapter_id": "s1"},
    ]

    def fake_call(session, prompt, system=None, model_size=None, **kwargs):
        return (
            json.dumps(
                {
                    "propositions": [
                        {
                            "chunk_index": 99,  # inválido → fallback a span
                            "core_idea_id": "CI",
                            "argument_id": "A",
                            "statement": "Prop A.",
                            "text_span": "Paráfrasis A.",
                            "citations_references": [],
                        }
                    ],
                    "entities": [],
                    "relations": [],
                }
            ),
            "model",
            False,
        )

    monkeypatch.setattr("src.kag_ingest.call_with_retries", fake_call)
    monkeypatch.setattr(ki, "_get_prompt_pair", _fake_prompt_pair)
    props, entities, relations = ki._extract_document_batch(
        _EmptySession(), doc_id=1, chunks=chunks, doc_path="doc.md", verbose=False
    )
    assert len(props) == 1
    assert props[0]["chunk_id"] == 1  # asignado vía _locate_span_in_batch
    assert props[0]["char_start"] == 0
    assert props[0]["chapter_id"] == "s1"
    assert entities == []
    assert relations == []


def test_extract_document_batch_both_fail_discards_proposition(monkeypatch):
    """chunk_index inválido Y span no localizable → proposición descartada
    (chunk_id NOT NULL en kag_propositions)."""
    chunks = [
        {"id": 1, "content": "Paráfrasis A.", "chunk_index": 0, "chapter_id": "s1"},
    ]

    def fake_call(session, prompt, system=None, model_size=None, **kwargs):
        return (
            json.dumps(
                {
                    "propositions": [
                        {
                            "chunk_index": 99,  # inválido
                            "core_idea_id": "CI",
                            "argument_id": "A",
                            "statement": "Prop.",
                            "text_span": "no existe en el lote",  # no localizable
                            "citations_references": [],
                        }
                    ],
                    "entities": [],
                    "relations": [],
                }
            ),
            "model",
            False,
        )

    monkeypatch.setattr("src.kag_ingest.call_with_retries", fake_call)
    monkeypatch.setattr(ki, "_get_prompt_pair", _fake_prompt_pair)
    props, entities, relations = ki._extract_document_batch(
        _EmptySession(), doc_id=1, chunks=chunks, doc_path="doc.md", verbose=False
    )
    assert props == []
    assert entities == []
    assert relations == []


def test_extract_document_batch_llm_failure_degrades(monkeypatch):
    """LLM falla → ([], [], []) sin excepción (degradación)."""
    chunks = [
        {"id": 1, "content": "Paráfrasis A.", "chunk_index": 0, "chapter_id": "s1"},
    ]

    def boom(session, prompt, system=None, model_size=None, **kwargs):
        raise RuntimeError("LLM no disponible")

    monkeypatch.setattr("src.kag_ingest.call_with_retries", boom)
    monkeypatch.setattr(ki, "_get_prompt_pair", _fake_prompt_pair)
    props, entities, relations = ki._extract_document_batch(
        _EmptySession(), doc_id=1, chunks=chunks, doc_path="doc.md", verbose=False
    )
    assert props == []
    assert entities == []
    assert relations == []


def test_extract_document_batch_uses_large_model_by_default(monkeypatch):
    """El lote de documento usa el LLM grande por defecto (model_size large)."""
    chunks = [
        {"id": 1, "content": "Paráfrasis A.", "chunk_index": 0, "chapter_id": "s1"},
    ]
    seen = {}

    def fake_call(session, prompt, system=None, model_size=None, **kwargs):
        seen["model_size"] = model_size
        return (
            json.dumps(
                {
                    "propositions": [],
                    "entities": [],
                    "relations": [],
                }
            ),
            "model",
            False,
        )

    monkeypatch.setattr("src.kag_ingest.call_with_retries", fake_call)
    monkeypatch.setattr(ki, "_get_prompt_pair", _fake_prompt_pair)
    ki._extract_document_batch(
        _EmptySession(), doc_id=1, chunks=chunks, doc_path="doc.md", verbose=False
    )
    assert seen["model_size"] == "large"


# ---------------------------------------------------------------------
# _extract_document_fused_batch (extracción FUSIONADA, spec kag_document_extract)
# ---------------------------------------------------------------------


def _fused_llm_ok(
    paraphrases=None,
    props=None,
    entities=None,
    relations=None,
    section_summaries=None,
    document_summary="",
):
    """Fake de call_with_retries que devuelve el JSON del schema fusionado."""

    def fake_call(session, prompt, system=None, model_size=None, **kwargs):
        return (
            json.dumps(
                {
                    "paraphrases": paraphrases or [],
                    "propositions": props or [],
                    "entities": entities or [],
                    "relations": relations or [],
                    "section_summaries": section_summaries or [],
                    "document_summary": document_summary,
                }
            ),
            "model",
            False,
        )

    return fake_call


def test_extract_document_fused_batch_returns_all_layers(monkeypatch):
    """UNA llamada devuelve las 6 claves: paráfrasis, proposiciones,
    entidades, relaciones, resúmenes de sección y resumen de documento."""
    chunks = [
        {"id": 1, "content": "Paráfrasis A.", "chunk_index": 0, "chapter_id": "ch1"},
        {"id": 2, "content": "Paráfrasis B.", "chunk_index": 1, "chapter_id": "ch1"},
    ]
    fake_call = _fused_llm_ok(
        paraphrases=[
            {"chunk_index": 0, "paraphrase": "Paráfrasis A."},
            {"chunk_index": 1, "paraphrase": "Paráfrasis B."},
        ],
        props=[
            {
                "chunk_index": 0,
                "core_idea_id": "CI1",
                "argument_id": "A1",
                "statement": "Prop 1.",
                "text_span": "Paráfrasis A.",
                "citations_references": ["Ref 1"],
            }
        ],
        entities=[{"name": "Entidad X", "type": "concept", "description": "Desc"}],
        relations=[
            {
                "source": "Entidad X",
                "target": "Entidad Y",
                "type": "RELACIONA",
                "description": "Rel",
            }
        ],
        section_summaries=[{"chapter_id": "ch1", "summary": "Resumen cap 1."}],
        document_summary="Resumen del documento.",
    )
    monkeypatch.setattr("src.kag_ingest.call_with_retries", fake_call)
    monkeypatch.setattr(ki, "_get_prompt_pair", _fake_prompt_pair)
    out = ki._extract_document_fused_batch(
        _EmptySession(), doc_id=1, chunks=chunks, doc_path="doc.md", verbose=False
    )
    assert [p["chunk_index"] for p in out["paraphrases"]] == [0, 1]
    assert out["propositions"][0]["chunk_id"] == 1
    assert out["propositions"][0]["citations"] == ["Ref 1"]
    assert out["entities"][0]["name"] == "Entidad X"
    assert out["relations"][0]["source"] == "Entidad X"
    assert out["section_summaries"] == [
        {"chapter_id": "ch1", "summary": "Resumen cap 1."}
    ]
    assert out["document_summary"] == "Resumen del documento."


def test_extract_document_fused_batch_invalid_chunk_index_falls_back_to_span(
    monkeypatch,
):
    """chunk_index inválido → fallback a _locate_span_in_batch (mismo patrón
    que _extract_document_batch)."""
    chunks = [
        {"id": 1, "content": "Paráfrasis A.", "chunk_index": 0, "chapter_id": "s1"},
    ]
    fake_call = _fused_llm_ok(
        props=[
            {
                "chunk_index": 99,  # inválido → fallback a span
                "core_idea_id": "CI",
                "argument_id": "A",
                "statement": "Prop A.",
                "text_span": "Paráfrasis A.",
                "citations_references": [],
            }
        ]
    )
    monkeypatch.setattr("src.kag_ingest.call_with_retries", fake_call)
    monkeypatch.setattr(ki, "_get_prompt_pair", _fake_prompt_pair)
    out = ki._extract_document_fused_batch(
        _EmptySession(), doc_id=1, chunks=chunks, doc_path="doc.md", verbose=False
    )
    assert out["propositions"][0]["chunk_id"] == 1
    assert out["propositions"][0]["char_start"] == 0


def test_extract_document_fused_batch_llm_failure_degrades(monkeypatch):
    """LLM falla → dict vacío sin excepción (degradación)."""
    chunks = [
        {"id": 1, "content": "Paráfrasis A.", "chunk_index": 0, "chapter_id": "s1"},
    ]

    def boom(session, prompt, system=None, model_size=None, **kwargs):
        raise RuntimeError("LLM no disponible")

    monkeypatch.setattr("src.kag_ingest.call_with_retries", boom)
    monkeypatch.setattr(ki, "_get_prompt_pair", _fake_prompt_pair)
    out = ki._extract_document_fused_batch(
        _EmptySession(), doc_id=1, chunks=chunks, doc_path="doc.md", verbose=False
    )
    assert out == {}


# ---------------------------------------------------------------------
# _extract_document_fused (orquestador: paráfrasis + proposiciones +
# entidades + relaciones + resúmenes en UNA llamada por lote)
# ---------------------------------------------------------------------


def test_extract_document_fused_persists_all_layers(monkeypatch):
    """El orquestador persiste paráfrasis (UPDATE), proposiciones, entidades,
    relaciones, links y el índice temático (resumen de documento)."""
    h1 = ki._chunk_content_hash("aaa")
    h2 = ki._chunk_content_hash("bbb")
    rows = [
        _chunk_row(1, "aaa", None, h1, chunk_index=0),
        _chunk_row(2, "bbb", None, h2, chunk_index=1),
    ]
    session = _DocPropsSession(rows, cached_chunk_ids=[])
    stored_props = []
    stored_entities = []
    persisted_summaries = []
    _patch_own_session(monkeypatch, session)

    fake_call = _fused_llm_ok(
        paraphrases=[
            {"chunk_index": 0, "paraphrase": "Paráfrasis A."},
            {"chunk_index": 1, "paraphrase": "Paráfrasis B."},
        ],
        props=[
            {
                "chunk_index": 0,
                "core_idea_id": "CI1",
                "argument_id": "A1",
                "statement": "Prop 1.",
                "text_span": "Paráfrasis A.",
                "citations_references": [],
            },
            {
                "chunk_index": 1,
                "core_idea_id": "CI2",
                "argument_id": "A2",
                "statement": "Prop 2.",
                "text_span": "Paráfrasis B.",
                "citations_references": [],
            },
        ],
        entities=[{"name": "Entidad X", "type": "concept", "description": "Desc"}],
        relations=[],
        section_summaries=[{"chapter_id": "ch1", "summary": "Resumen cap 1."}],
        document_summary="Resumen del documento.",
    )
    monkeypatch.setattr("src.kag_ingest.call_with_retries", fake_call)
    monkeypatch.setattr(ki, "_get_prompt_pair", _fake_prompt_pair)
    monkeypatch.setattr(
        ki,
        "_store_propositions",
        lambda s, d, props, embed_fn=None: stored_props.append(props),
    )
    monkeypatch.setattr(
        ki,
        "_store_entities_relations",
        lambda s, d, cid, data, embed_fn=None: stored_entities.append((cid, data)),
    )
    monkeypatch.setattr(
        ki,
        "_persist_summary_index",
        lambda s, d, secs, final, section_chapter_ids=None: persisted_summaries.append(
            (secs, final, section_chapter_ids)
        ),
    )
    ki._extract_document_fused(
        session, doc_id=1, doc_path="doc.md", verbose=False, batch_size_tokens=250000
    )
    # Paráfrasis persistidas (UPDATE kag_chunks.paraphrase).
    assert {u["id"] for u in session.updates} == {1, 2}
    # Proposiciones asignadas por chunk_index.
    assert [p["chunk_id"] for p in stored_props[0]] == [1, 2]
    # Entidades persistidas contra el primer chunk del lote.
    assert stored_entities[0][0] == 1
    assert stored_entities[0][1]["entities"][0]["name"] == "Entidad X"
    # Índice temático persistido con secciones + resumen de documento.
    assert persisted_summaries == [
        (["Resumen cap 1."], "Resumen del documento.", ["ch1"])
    ]


def test_extract_document_fused_cache_skips_extracted(monkeypatch):
    """Cache por content_hash: chunks ya extraídos con hash idéntico se
    saltan (no re-extraer en re-ingestas)."""
    h1 = ki._chunk_content_hash("aaa")
    h2 = ki._chunk_content_hash("bbb")
    rows = [
        _chunk_row(1, "aaa", "Paráfrasis A.", h1, chunk_index=0),  # cache hit
        _chunk_row(2, "bbb", None, h2, chunk_index=1),  # sin proposiciones → extraer
    ]
    session = _DocPropsSession(rows, cached_chunk_ids=[1])
    called = []
    stored = []
    _patch_own_session(monkeypatch, session)

    def fake_call(session, prompt, system=None, model_size=None, **kwargs):
        called.append(prompt)
        return (
            json.dumps(
                {
                    "paraphrases": [{"chunk_index": 1, "paraphrase": "Paráfrasis B."}],
                    "propositions": [
                        {
                            "chunk_index": 1,
                            "core_idea_id": "CI",
                            "argument_id": "A",
                            "statement": "Prop.",
                            "text_span": "Paráfrasis B.",
                            "citations_references": [],
                        }
                    ],
                    "entities": [],
                    "relations": [],
                    "section_summaries": [],
                    "document_summary": "",
                }
            ),
            "model",
            False,
        )

    monkeypatch.setattr("src.kag_ingest.call_with_retries", fake_call)
    monkeypatch.setattr(ki, "_get_prompt_pair", _fake_prompt_pair)
    monkeypatch.setattr(
        ki,
        "_store_propositions",
        lambda s, d, props, embed_fn=None: stored.append(props),
    )
    ki._extract_document_fused(
        session, doc_id=1, doc_path="doc.md", verbose=False, batch_size_tokens=250000
    )
    assert len(called) == 1
    assert [p["chunk_id"] for p in stored[0]] == [2]
    assert {u["id"] for u in session.updates} == {2}


def test_extract_document_fused_llm_failure_degrades(monkeypatch):
    """LLM falla → skip del lote, la ingesta sigue (sin excepción)."""
    h1 = ki._chunk_content_hash("aaa")
    rows = [_chunk_row(1, "aaa", None, h1, chunk_index=0)]
    session = _DocPropsSession(rows, cached_chunk_ids=[])
    stored = []
    _patch_own_session(monkeypatch, session)

    def boom(session, prompt, system=None, model_size=None, **kwargs):
        raise RuntimeError("LLM no disponible")

    monkeypatch.setattr("src.kag_ingest.call_with_retries", boom)
    monkeypatch.setattr(ki, "_get_prompt_pair", _fake_prompt_pair)
    monkeypatch.setattr(
        ki,
        "_store_propositions",
        lambda s, d, props, embed_fn=None: stored.append(props) if props else None,
    )
    ki._extract_document_fused(
        session, doc_id=1, doc_path="doc.md", verbose=False, batch_size_tokens=250000
    )
    assert stored == []
    assert session.entity_calls == []
    assert session.updates == []  # sin paráfrasis ni content_hash


def test_extract_document_fused_no_document_summary_skips_persist(monkeypatch):
    """Sin resumen de documento → no se persiste el índice temático (el
    resumen final vacío degrada en _persist_summary_index)."""
    h1 = ki._chunk_content_hash("aaa")
    rows = [_chunk_row(1, "aaa", None, h1, chunk_index=0)]
    session = _DocPropsSession(rows, cached_chunk_ids=[])
    stored = []
    persisted = []
    _patch_own_session(monkeypatch, session)

    fake_call = _fused_llm_ok(
        paraphrases=[{"chunk_index": 0, "paraphrase": "Paráfrasis A."}],
        props=[],
        section_summaries=[{"chapter_id": "ch1", "summary": "Resumen cap 1."}],
        document_summary="",
    )
    monkeypatch.setattr("src.kag_ingest.call_with_retries", fake_call)
    monkeypatch.setattr(ki, "_get_prompt_pair", _fake_prompt_pair)
    monkeypatch.setattr(
        ki,
        "_store_propositions",
        lambda s, d, props, embed_fn=None: stored.append(props),
    )
    monkeypatch.setattr(
        ki,
        "_persist_summary_index",
        lambda s, d, secs, final, section_chapter_ids=None: persisted.append(1),
    )
    ki._extract_document_fused(
        session, doc_id=1, doc_path="doc.md", verbose=False, batch_size_tokens=250000
    )
    assert persisted == []  # sin document_summary → sin persistencia
    assert {u["id"] for u in session.updates} == {1}  # paráfrasis + hash sí
