"""Tests de la expansión de ingesta proposicional (Agente B) — sin DB real.

Cubre los helpers nuevos de src/kag_ingest.py:
  - _fill_prompt: replace por clave, NO rompe las llaves JSON literales.
  - _locate_span_in_chunk: hit, miss (None), normalización de espacios.
  - _extract_propositions: LLM fake OK, LLM falla -> [], JSON inválido -> [],
    span no encontrado -> spans None (degradación), truncado de contenido.
  - _store_propositions: UN solo INSERT multi-VALUES (sin executemany),
    degradación de embeddings a [None]*n, lista vacía -> no SQL.
"""

import json
from types import SimpleNamespace

import src.kag_ingest as ki

# ---------------------------------------------------------------------
# _fill_prompt
# ---------------------------------------------------------------------


def test_fill_prompt_replaces_placeholders_without_breaking_json_braces():
    template = (
        'Genera el JSON: {"propositions": [{"statement": str, '
        '"text_span": str}]} con {chunk_index} y {chapter_text_content}'
    )
    out = ki._fill_prompt(template, chunk_index=3, chapter_text_content="El texto.")
    assert "{chunk_index}" not in out
    assert "{chapter_text_content}" not in out
    assert "3" in out
    assert "El texto." in out
    # Las llaves JSON literales se conservan intactas (sin KeyError).
    assert '{"propositions": [{"statement": str, "text_span": str}]}' in out


def test_fill_prompt_missing_key_leaves_placeholder():
    """Una clave ausente deja el placeholder intacto (no lanza KeyError)."""
    out = ki._fill_prompt("Hola {nombre}", nombre="Ana")
    assert out == "Hola Ana"


# ---------------------------------------------------------------------
# _locate_span_in_chunk
# ---------------------------------------------------------------------


def test_locate_span_in_chunk_hit():
    content = "Primera línea.\nSegunda línea con el span aquí.\nTercera."
    span = "el span aquí"
    char_start, char_end, line_start, line_end = ki._locate_span_in_chunk(content, span)
    assert content[char_start:char_end] == span
    assert line_start == 2
    assert line_end == 2


def test_locate_span_in_chunk_miss_returns_none():
    content = "Una línea sin el span."
    assert ki._locate_span_in_chunk(content, "no existe") == (
        None,
        None,
        None,
        None,
    )


def test_locate_span_in_chunk_empty_span_returns_none():
    assert ki._locate_span_in_chunk("cualquier texto", "") == (
        None,
        None,
        None,
        None,
    )


def test_locate_span_in_chunk_normalizes_spaces():
    content = "El  gato   negro duerme."
    span = "El gato negro"
    char_start, char_end, line_start, line_end = ki._locate_span_in_chunk(content, span)
    # El span normalizado mapea al rango original (incluye los espacios extra).
    assert " ".join(content[char_start:char_end].split()) == span
    assert line_start == 1
    assert line_end == 1


def test_locate_span_in_chunk_multiline_span():
    content = "Línea uno.\nLínea dos con el span.\nLínea tres."
    span = "Línea dos con el span."
    char_start, char_end, line_start, line_end = ki._locate_span_in_chunk(content, span)
    assert content[char_start:char_end] == span
    assert line_start == 2
    assert line_end == 2


# ---------------------------------------------------------------------
# _extract_propositions
# ---------------------------------------------------------------------


class _EmptySession:
    """Sesión fake: get_active_prompt (select) devuelve None -> fallback a
    constantes en _get_prompt_pair."""

    def execute(self, stmt, params=None):
        class _R:
            def scalars(self):
                return self

            def first(self):
                return None

        return _R()


def test_extract_propositions_llm_ok(monkeypatch):
    monkeypatch.setattr(ki, "load_settings", lambda session: None)

    def fake_call(
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
                    "divisions": [
                        {
                            "section_path": "# Intro",
                            "propositions": [
                                {
                                    "core_idea_id": "CI-1",
                                    "argument_id": "ARG-1",
                                    "statement": "El gato duerme.",
                                    "text_span": "El gato duerme",
                                    "char_start": 0,
                                    "char_end": 14,
                                    "line_start": 1,
                                    "line_end": 1,
                                    "citations_references": ["Bourdieu, 1984"],
                                }
                            ],
                        }
                    ]
                }
            ),
            "small",
            False,
        )

    monkeypatch.setattr(ki, "call_with_retries", fake_call)
    props = ki._extract_propositions(
        _EmptySession(),
        doc_id=1,
        chunk_id=2,
        content="El gato duerme.",
        doc_path="doc.md",
        chunk_index=0,
        section_path="# Intro",
        verbose=False,
    )
    assert len(props) == 1
    p = props[0]
    assert p["doc_id"] == 1
    assert p["chunk_id"] == 2
    assert p["section_path"] == "# Intro"
    assert p["core_idea_id"] == "CI-1"
    assert p["argument_id"] == "ARG-1"
    assert p["statement"] == "El gato duerme."
    assert p["text_span"] == "El gato duerme"
    assert p["char_start"] == 0
    assert p["char_end"] == 14
    assert p["line_start"] == 1
    assert p["line_end"] == 1
    assert p["citations"] == ["Bourdieu, 1984"]


def test_extract_propositions_llm_fails_returns_empty(monkeypatch):
    monkeypatch.setattr(ki, "load_settings", lambda session: None)

    def boom(
        session,
        *,
        prompt,
        system,
        model_size,
        response_format,
        retries,
        fallback_model=None,
    ):
        raise RuntimeError("LLM no disponible")

    monkeypatch.setattr(ki, "call_with_retries", boom)
    props = ki._extract_propositions(
        _EmptySession(),
        doc_id=1,
        chunk_id=2,
        content="texto",
        doc_path="doc.md",
        chunk_index=0,
        section_path="",
        verbose=False,
    )
    assert props == []


def test_extract_propositions_invalid_json_returns_empty(monkeypatch):
    monkeypatch.setattr(ki, "load_settings", lambda session: None)

    def fake_call(
        session,
        *,
        prompt,
        system,
        model_size,
        response_format,
        retries,
        fallback_model=None,
    ):
        return ("esto no es json", "small", False)

    monkeypatch.setattr(ki, "call_with_retries", fake_call)
    props = ki._extract_propositions(
        _EmptySession(),
        doc_id=1,
        chunk_id=2,
        content="texto",
        doc_path="doc.md",
        chunk_index=0,
        section_path="",
        verbose=False,
    )
    assert props == []


def test_extract_propositions_json_without_propositions_returns_empty(monkeypatch):
    monkeypatch.setattr(ki, "load_settings", lambda session: None)

    def fake_call(
        session,
        *,
        prompt,
        system,
        model_size,
        response_format,
        retries,
        fallback_model=None,
    ):
        return (json.dumps({"otra_cosa": 1}), "small", False)

    monkeypatch.setattr(ki, "call_with_retries", fake_call)
    props = ki._extract_propositions(
        _EmptySession(),
        doc_id=1,
        chunk_id=2,
        content="texto",
        doc_path="doc.md",
        chunk_index=0,
        section_path="",
        verbose=False,
    )
    assert props == []


def test_extract_propositions_span_not_found_keeps_none_spans(monkeypatch):
    """Degradación: span no localizable -> proposición se guarda con spans NULL."""
    monkeypatch.setattr(ki, "load_settings", lambda session: None)

    def fake_call(
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
                    "divisions": [
                        {
                            "section_path": "",
                            "propositions": [
                                {
                                    "core_idea_id": "CI-1",
                                    "argument_id": "ARG-1",
                                    "statement": "El gato duerme.",
                                    "text_span": "fragmento que no está",
                                    "char_start": 0,
                                    "char_end": 5,
                                    "line_start": 1,
                                    "line_end": 1,
                                    "citations_references": [],
                                }
                            ],
                        }
                    ]
                }
            ),
            "small",
            False,
        )

    monkeypatch.setattr(ki, "call_with_retries", fake_call)
    props = ki._extract_propositions(
        _EmptySession(),
        doc_id=1,
        chunk_id=2,
        content="El gato duerme.",
        doc_path="doc.md",
        chunk_index=0,
        section_path="",
        verbose=False,
    )
    assert len(props) == 1
    assert props[0]["char_start"] is None
    assert props[0]["char_end"] is None
    assert props[0]["line_start"] is None
    assert props[0]["line_end"] is None


def test_extract_propositions_truncates_long_content(monkeypatch):
    """Contenido > MAX_CONTEXT_CHARS se trunca antes de armar el prompt."""
    monkeypatch.setattr(ki, "load_settings", lambda session: None)
    captured = {}

    def fake_call(
        session,
        *,
        prompt,
        system,
        model_size,
        response_format,
        retries,
        fallback_model=None,
    ):
        captured["prompt"] = prompt
        return (json.dumps({"propositions": []}), "small", False)

    monkeypatch.setattr(ki, "call_with_retries", fake_call)
    long_content = "x" * 20000
    ki._extract_propositions(
        _EmptySession(),
        doc_id=1,
        chunk_id=2,
        content=long_content,
        doc_path="doc.md",
        chunk_index=0,
        section_path="",
        verbose=False,
    )
    assert len(captured["prompt"]) < 20000
    assert "x" * ki.MAX_CONTEXT_CHARS in captured["prompt"]


# ---------------------------------------------------------------------
# _store_propositions
# ---------------------------------------------------------------------


def _prop(doc_id=1, chunk_id=2, i=0, citations=None):
    return {
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "section_path": "s1",
        "core_idea_id": f"CI-{i}",
        "argument_id": f"ARG-{i}",
        "statement": f"Proposición {i}.",
        "text_span": f"Proposición {i}",
        "char_start": 0,
        "char_end": 12,
        "line_start": 1,
        "line_end": 1,
        "citations": citations or [],
    }


def test_store_propositions_single_multi_values_statement():
    """UN solo INSERT multi-VALUES con placeholders :d0,:c0,... (sin executemany)."""
    sqls = []

    class _CaptureSession:
        def execute(self, stmt, params=None):
            sqls.append((str(stmt), params))

    props = [
        _prop(i=0, citations=["Bourdieu, 1984"]),
        _prop(i=1),
    ]

    def fake_embed(texts):
        return [[0.1, 0.2], [0.3, 0.4]]

    ki._store_propositions(
        _CaptureSession(), doc_id=1, propositions=props, embed_fn=fake_embed
    )
    assert len(sqls) == 1
    sql, params = sqls[0]
    assert "INSERT INTO kag_propositions" in sql
    assert "VALUES" in sql
    # Un solo statement multi-VALUES: dos filas con placeholders indexados.
    assert ":d0" in sql and ":d1" in sql
    assert ":c0" in sql and ":c1" in sql
    assert ":sp0" in sql and ":sp1" in sql
    assert ":s0" in sql and ":s1" in sql
    assert "executemany" not in sql.lower()
    assert params["d0"] == 1
    assert params["c0"] == 2
    assert params["sp0"] == "s1"
    assert params["s0"] == "Proposición 0."
    assert params["cr0"] == json.dumps(["Bourdieu, 1984"], ensure_ascii=False)
    assert params["cr1"] == "[]"
    assert params["emb0"] == "[0.1,0.2]"
    assert params["emb1"] == "[0.3,0.4]"


def test_store_propositions_embedding_failure_degrades_to_none():
    """Si embed_fn falla -> embedding NULL ([None]*n), la ingesta sigue."""
    sqls = []

    class _CaptureSession:
        def execute(self, stmt, params=None):
            sqls.append((str(stmt), params))

    props = [_prop(i=0)]

    def boom(texts):
        raise RuntimeError("embedding down")

    ki._store_propositions(
        _CaptureSession(), doc_id=1, propositions=props, embed_fn=boom
    )
    assert len(sqls) == 1
    _sql, params = sqls[0]
    assert params["emb0"] is None


def test_store_propositions_without_embed_fn_uses_none():
    sqls = []

    class _CaptureSession:
        def execute(self, stmt, params=None):
            sqls.append((str(stmt), params))

    ki._store_propositions(
        _CaptureSession(), doc_id=1, propositions=[_prop(i=0)], embed_fn=None
    )
    assert len(sqls) == 1
    _sql, params = sqls[0]
    assert params["emb0"] is None


def test_store_propositions_empty_does_nothing():
    sqls = []

    class _CaptureSession:
        def execute(self, stmt, params=None):
            sqls.append((str(stmt), params))

    ki._store_propositions(_CaptureSession(), doc_id=1, propositions=[], embed_fn=None)
    assert sqls == []


# ---------------------------------------------------------------------
# _batch_chunks_by_tokens (batching por presupuesto de tokens)
# ---------------------------------------------------------------------


def _chunk(content, section="s1", cid=0):
    return {"id": cid, "content": content, "chunk_index": cid, "section_path": section}


def test_batch_chunks_by_tokens_groups_by_budget():
    """Agrupa por presupuesto de tokens (mock de la estimación), no por nº fijo."""
    # 4 chunks de 100 tokens cada uno; presupuesto 250 → 2 lotes de 2.
    chunks = [_chunk("x" * 400, cid=i) for i in range(4)]  # 100 tokens c/u
    batches = ki._batch_chunks_by_tokens(
        chunks, batch_size_tokens=250, estimate_fn=lambda c: len(c) // 4
    )
    assert [len(b) for b in batches] == [2, 2]
    assert [b[0]["id"] for b in batches] == [0, 2]


def test_batch_chunks_by_tokens_respects_250k_limit():
    """El corte respeta el límite de 250k tokens (secciones distintas)."""
    chunks = [
        _chunk("x" * 400000, section=f"s{i}", cid=i)
        for i in range(6)  # 100k c/u
    ]
    batches = ki._batch_chunks_by_tokens(
        chunks, batch_size_tokens=250000, estimate_fn=lambda c: len(c) // 4
    )
    for b in batches:
        assert sum(len(c["content"]) // 4 for c in b) <= 250000
    assert len(batches) == 3


def test_batch_chunks_by_tokens_keeps_same_section_together():
    """Chunks consecutivos de la misma sección quedan juntos si caben."""
    chunks = [
        _chunk("x" * 400000, section="s1", cid=0),  # 100k
        _chunk("x" * 400000, section="s1", cid=1),  # 100k (misma sección)
        _chunk("x" * 400000, section="s2", cid=2),  # 100k (sección nueva)
    ]
    batches = ki._batch_chunks_by_tokens(
        chunks, batch_size_tokens=250000, estimate_fn=lambda c: len(c) // 4
    )
    # c0+c1 (misma sección) caben juntos; c2 corta por presupuesto.
    assert [len(b) for b in batches] == [2, 1]
    assert batches[0][0]["section_path"] == batches[0][1]["section_path"] == "s1"


def test_batch_chunks_by_tokens_single_chunk_over_budget():
    """Un chunk que excede el presupuesto forma su propio lote (no se parte)."""
    chunks = [_chunk("x" * 2000000, cid=0)]  # 500k > 250k
    batches = ki._batch_chunks_by_tokens(
        chunks, batch_size_tokens=250000, estimate_fn=lambda c: len(c) // 4
    )
    assert len(batches) == 1
    assert len(batches[0]) == 1


# ---------------------------------------------------------------------
# _group_chunks_by_chapter (agrupación jerárquica doc→capítulo)
# ---------------------------------------------------------------------


def test_group_chunks_by_chapter_respects_30k_threshold():
    """Capítulo >30k tokens = grupo propio; ≤30k = nivel archivo."""
    chunks = [
        _chunk("x" * 40000, section="big1", cid=0),  # 10k
        _chunk("x" * 40000, section="big1", cid=1),  # 10k (big1 total 20k ≤ 30k)
        _chunk("x" * 200000, section="huge", cid=2),  # 50k > 30k → grupo propio
        _chunk("x" * 40000, section="small", cid=3),  # 10k → nivel archivo
    ]
    groups = ki._group_chunks_by_chapter(chunks, estimate_fn=lambda c: len(c) // 4)
    assert [g["section_path"] for g in groups] == ["huge", ""]
    assert [c["id"] for c in groups[0]["chunks"]] == [2]
    assert [c["id"] for c in groups[1]["chunks"]] == [0, 1, 3]


def test_group_chunks_by_chapter_all_small_merges_to_file_level():
    """Todos los capítulos ≤30k → un único grupo a nivel de archivo."""
    chunks = [
        _chunk("x" * 40000, section="s1", cid=0),  # 10k
        _chunk("x" * 40000, section="s2", cid=1),  # 10k
    ]
    groups = ki._group_chunks_by_chapter(chunks, estimate_fn=lambda c: len(c) // 4)
    assert len(groups) == 1
    assert groups[0]["section_path"] == ""
    assert [c["id"] for c in groups[0]["chunks"]] == [0, 1]


def test_group_chunks_by_chapter_keeps_chunk_index_order():
    """Los chunks de cada grupo conservan el orden por chunk_index."""
    chunks = [
        _chunk("x" * 40000, section="s1", cid=0),  # 10k → archivo
        _chunk("x" * 200000, section="huge", cid=1),  # 50k → propio
        _chunk("x" * 40000, section="s2", cid=2),  # 10k → archivo
    ]
    groups = ki._group_chunks_by_chapter(chunks, estimate_fn=lambda c: len(c) // 4)
    assert [c["id"] for c in groups[1]["chunks"]] == [0, 2]
    assert [c["id"] for c in groups[0]["chunks"]] == [1]


# ---------------------------------------------------------------------
# _locate_span_in_batch
# ---------------------------------------------------------------------


def test_locate_span_in_batch_assigns_to_containing_chunk():
    chunks = [
        _chunk("El gato duerme.", cid=1),
        _chunk("El perro corre.", cid=2),
    ]
    sep = "\n\n---\n\n"
    batch_text = sep.join(c["content"] for c in chunks)
    offsets = []
    cursor = 0
    for c in chunks:
        start = cursor
        cursor += len(c["content"])
        offsets.append((c["id"], start, cursor))
        cursor += len(sep)
    cid, cs, ce, ls, le = ki._locate_span_in_batch(
        chunks, batch_text, offsets, "El perro corre"
    )
    assert cid == 2
    # Los offsets son relativos al chunk que contiene el span.
    assert chunks[1]["content"][cs:ce] == "El perro corre"
    assert ls == 1 and le == 1


def test_locate_span_in_batch_miss_returns_none():
    chunks = [_chunk("El gato duerme.", cid=1)]
    batch_text = chunks[0]["content"]
    offsets = [(1, 0, len(batch_text))]
    assert ki._locate_span_in_batch(chunks, batch_text, offsets, "no existe") == (
        None,
        None,
        None,
        None,
        None,
    )


# ---------------------------------------------------------------------
# _extract_propositions_batch (UNA llamada LLM por lote)
# ---------------------------------------------------------------------


def test_extract_propositions_batch_assigns_chunks_by_span(monkeypatch):
    monkeypatch.setattr(ki, "load_settings", lambda session: None)
    chunks = [
        _chunk("El gato duerme.", cid=1),
        _chunk("El perro corre.", cid=2),
    ]

    def fake_call(
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
                    "divisions": [
                        {
                            "section_path": "# Cap 1",
                            "propositions": [
                                {
                                    "core_idea_id": "CI-1",
                                    "argument_id": "ARG-1",
                                    "statement": "El gato duerme.",
                                    "text_span": "El gato duerme",
                                    "char_start": 0,
                                    "char_end": 14,
                                    "line_start": 1,
                                    "line_end": 1,
                                    "citations_references": [],
                                }
                            ],
                        },
                        {
                            "section_path": "# Cap 2",
                            "propositions": [
                                {
                                    "core_idea_id": "CI-2",
                                    "argument_id": "ARG-2",
                                    "statement": "El perro corre.",
                                    "text_span": "El perro corre",
                                    "char_start": 0,
                                    "char_end": 14,
                                    "line_start": 1,
                                    "line_end": 1,
                                    "citations_references": [],
                                }
                            ],
                        },
                    ]
                }
            ),
            "small",
            False,
        )

    monkeypatch.setattr(ki, "call_with_retries", fake_call)
    props = ki._extract_propositions_batch(
        _EmptySession(), doc_id=1, chunks=chunks, doc_path="doc.md", verbose=False
    )
    assert len(props) == 2
    assert props[0]["chunk_id"] == 1
    assert props[1]["chunk_id"] == 2
    assert props[0]["section_path"] == "# Cap 1"
    assert props[1]["section_path"] == "# Cap 2"
    assert props[0]["statement"] == "El gato duerme."
    assert props[1]["statement"] == "El perro corre."


def test_extract_propositions_batch_span_not_found_chunk_none(monkeypatch):
    """Degradación: span no localizable → chunk_id None (se descarta al guardar)."""
    monkeypatch.setattr(ki, "load_settings", lambda session: None)
    chunks = [_chunk("El gato duerme.", cid=1)]

    def fake_call(
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
                    "divisions": [
                        {
                            "section_path": "",
                            "propositions": [
                                {
                                    "core_idea_id": "CI-1",
                                    "argument_id": "ARG-1",
                                    "statement": "El gato duerme.",
                                    "text_span": "fragmento que no está",
                                    "char_start": 0,
                                    "char_end": 5,
                                    "line_start": 1,
                                    "line_end": 1,
                                    "citations_references": [],
                                }
                            ],
                        }
                    ]
                }
            ),
            "small",
            False,
        )

    monkeypatch.setattr(ki, "call_with_retries", fake_call)
    props = ki._extract_propositions_batch(
        _EmptySession(), doc_id=1, chunks=chunks, doc_path="doc.md", verbose=False
    )
    assert len(props) == 1
    assert props[0]["chunk_id"] is None
    assert props[0]["char_start"] is None


def test_extract_propositions_batch_llm_fails_returns_empty(monkeypatch):
    monkeypatch.setattr(ki, "load_settings", lambda session: None)

    def boom(
        session,
        *,
        prompt,
        system,
        model_size,
        response_format,
        retries,
        fallback_model=None,
    ):
        raise RuntimeError("LLM no disponible")

    monkeypatch.setattr(ki, "call_with_retries", boom)
    props = ki._extract_propositions_batch(
        _EmptySession(),
        doc_id=1,
        chunks=[_chunk("texto", cid=1)],
        doc_path="doc.md",
        verbose=False,
    )
    assert props == []


def test_extract_propositions_batch_uses_large_model_by_default(monkeypatch):
    """El batching usa el LLM grande por defecto (respuesta mucho mayor en
    extensión que la de un chunk suelto) — KAG_PROPOSITION_MODEL default
    "large". El prompt pair se resuelve con large_model."""
    monkeypatch.setattr(ki, "load_settings", lambda session: None)
    captured = {}

    def fake_call(
        session,
        *,
        prompt,
        system,
        model_size,
        response_format,
        retries,
        fallback_model=None,
    ):
        captured["model_size"] = model_size
        return (json.dumps({"propositions": []}), model_size, False)

    monkeypatch.setattr(ki, "call_with_retries", fake_call)
    ki._extract_propositions_batch(
        _EmptySession(),
        doc_id=1,
        chunks=[_chunk("texto", cid=1)],
        doc_path="doc.md",
        verbose=False,
    )
    assert captured["model_size"] == "large"


def test_extract_propositions_batch_model_size_small(monkeypatch):
    """model_size="small" explícito → se pasa a call_with_retries."""
    monkeypatch.setattr(ki, "load_settings", lambda session: None)
    captured = {}

    def fake_call(
        session,
        *,
        prompt,
        system,
        model_size,
        response_format,
        retries,
        fallback_model=None,
    ):
        captured["model_size"] = model_size
        return (json.dumps({"propositions": []}), model_size, False)

    monkeypatch.setattr(ki, "call_with_retries", fake_call)
    ki._extract_propositions_batch(
        _EmptySession(),
        doc_id=1,
        chunks=[_chunk("texto", cid=1)],
        doc_path="doc.md",
        verbose=False,
        model_size="small",
    )
    assert captured["model_size"] == "small"


def test_extract_propositions_batch_parses_divisions(monkeypatch):
    """El output con divisiones se parsea y asigna por section_path."""
    monkeypatch.setattr(ki, "load_settings", lambda session: None)
    chunks = [
        _chunk("El gato duerme.", cid=1),
        _chunk("El perro corre.", cid=2),
    ]

    def fake_call(
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
                    "divisions": [
                        {
                            "section_path": "# Cap 1",
                            "propositions": [
                                {
                                    "core_idea_id": "CI-1",
                                    "argument_id": "ARG-1",
                                    "statement": "El gato duerme.",
                                    "text_span": "El gato duerme",
                                    "char_start": 0,
                                    "char_end": 14,
                                    "line_start": 1,
                                    "line_end": 1,
                                    "citations_references": [],
                                }
                            ],
                        },
                        {
                            "section_path": "# Cap 2",
                            "propositions": [
                                {
                                    "core_idea_id": "CI-2",
                                    "argument_id": "ARG-2",
                                    "statement": "El perro corre.",
                                    "text_span": "El perro corre",
                                    "char_start": 0,
                                    "char_end": 14,
                                    "line_start": 1,
                                    "line_end": 1,
                                    "citations_references": [],
                                }
                            ],
                        },
                    ]
                }
            ),
            "large",
            False,
        )

    monkeypatch.setattr(ki, "call_with_retries", fake_call)
    props = ki._extract_propositions_batch(
        _EmptySession(),
        doc_id=1,
        chunks=chunks,
        doc_path="doc.md",
        verbose=False,
        section_path="# Cap 1",
    )
    assert len(props) == 2
    assert props[0]["chunk_id"] == 1
    assert props[0]["section_path"] == "# Cap 1"
    assert props[1]["chunk_id"] == 2
    assert props[1]["section_path"] == "# Cap 2"


def test_extract_propositions_batch_legacy_flat_output(monkeypatch):
    """Output legacy plano (artefacto compilado anterior) → una división con
    el section_path del lote (transición sin romper)."""
    monkeypatch.setattr(ki, "load_settings", lambda session: None)
    chunks = [_chunk("El gato duerme.", cid=1)]

    def fake_call(
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
                    "propositions": [
                        {
                            "core_idea_id": "CI-1",
                            "argument_id": "ARG-1",
                            "statement": "El gato duerme.",
                            "text_span": "El gato duerme",
                            "char_start": 0,
                            "char_end": 14,
                            "line_start": 1,
                            "line_end": 1,
                            "citations_references": [],
                        }
                    ]
                }
            ),
            "large",
            False,
        )

    monkeypatch.setattr(ki, "call_with_retries", fake_call)
    props = ki._extract_propositions_batch(
        _EmptySession(),
        doc_id=1,
        chunks=chunks,
        doc_path="doc.md",
        verbose=False,
        section_path="# Cap 1",
    )
    assert len(props) == 1
    assert props[0]["chunk_id"] == 1
    assert props[0]["section_path"] == "# Cap 1"


# ---------------------------------------------------------------------
# _run_proposition_batches_parallel (semáforo + orden determinista)
# ---------------------------------------------------------------------


def test_run_proposition_batches_parallel_semaphore_limits_concurrency(monkeypatch):
    """El semáforo limita la concurrencia: nunca excede N llamadas simultáneas."""
    import threading
    import time

    state = {"active": 0, "max_active": 0}
    lock = threading.Lock()

    def fake_batch(session, doc_id, chunks, doc_path, verbose, **kwargs):
        with lock:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
        time.sleep(0.02)
        with lock:
            state["active"] -= 1
        return []

    monkeypatch.setattr(ki, "_extract_propositions_batch", fake_batch)
    units = [("", [_chunk("a", cid=i)]) for i in range(6)]
    results = ki._run_proposition_batches_parallel(
        _EmptySession(),
        doc_id=1,
        doc_path="doc.md",
        units=units,
        verbose=False,
        max_parallel=2,
    )
    assert state["max_active"] <= 2
    assert len(results) == 6
    assert all(r == [] for r in results)


def test_run_proposition_batches_parallel_preserves_unit_order(monkeypatch):
    """Los resultados se devuelven en el orden de entrada (persistencia
    determinista) aunque los lotes terminen fuera de orden."""
    import time

    def fake_batch(session, doc_id, chunks, doc_path, verbose, **kwargs):
        # El lote con chunk_index mayor termina antes (completación fuera de
        # orden).
        time.sleep(0.02 * (5 - chunks[0]["id"]))
        return [{"chunk_id": chunks[0]["id"], "ok": True}]

    monkeypatch.setattr(ki, "_extract_propositions_batch", fake_batch)
    units = [("", [_chunk("a", cid=i)]) for i in range(1, 5)]
    results = ki._run_proposition_batches_parallel(
        _EmptySession(),
        doc_id=1,
        doc_path="doc.md",
        units=units,
        verbose=False,
        max_parallel=2,
    )
    assert [r[0]["chunk_id"] for r in results] == [1, 2, 3, 4]


# ---------------------------------------------------------------------
# _extract_propositions_for_doc (cache por content_hash + batching)
# ---------------------------------------------------------------------


class _Result:
    def __init__(self, first=None, scalar=None, fetchall=None):
        self._first = first
        self._scalar = scalar
        self._fetchall = fetchall if fetchall is not None else []

    def first(self):
        return self._first

    def scalar(self):
        return self._scalar

    def fetchall(self):
        return self._fetchall


class _PropsSession:
    """Sesión fake para _extract_propositions_for_doc."""

    def __init__(self, chunk_rows, cached_chunk_ids):
        self._chunk_rows = chunk_rows
        self._cached = cached_chunk_ids
        self.updates = []

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


def _chunk_row(cid, content, content_hash):
    return SimpleNamespace(
        id=cid,
        content=content,
        chunk_index=cid,
        section_path="s1",
        content_hash=content_hash,
    )


def test_extract_propositions_for_doc_cache_skips_extracted(monkeypatch):
    """Cache por content_hash: chunks ya extraídos con hash idéntico se saltan."""
    h1 = ki._chunk_content_hash("aaa")
    h3 = ki._chunk_content_hash("ccc")
    rows = [
        _chunk_row(1, "aaa", h1),  # cache hit (proposiciones + hash idéntico)
        _chunk_row(
            2, "bbb", "old-hash"
        ),  # proposiciones pero hash cambiado → re-extraer
        _chunk_row(3, "ccc", h3),  # sin proposiciones → extraer
    ]
    session = _PropsSession(rows, cached_chunk_ids=[1, 2])
    batches_seen = []
    stored = []

    def fake_batch(session, doc_id, chunks, doc_path, verbose, **kwargs):
        batches_seen.append([c["id"] for c in chunks])
        return [
            {
                "doc_id": doc_id,
                "chunk_id": c["id"],
                "core_idea_id": "CI",
                "argument_id": "ARG",
                "statement": "Prop.",
                "text_span": "",
                "char_start": None,
                "char_end": None,
                "line_start": None,
                "line_end": None,
                "citations": [],
            }
            for c in chunks
        ]

    def fake_store(session, doc_id, propositions, embed_fn=None):
        stored.append([p["chunk_id"] for p in propositions])

    monkeypatch.setattr(ki, "_extract_propositions_batch", fake_batch)
    monkeypatch.setattr(ki, "_store_propositions", fake_store)
    ki._extract_propositions_for_doc(
        session, doc_id=1, doc_path="doc.md", verbose=False, batch_size_tokens=250000
    )
    # Solo los chunks 2 y 3 se extraen (el 1 está en cache).
    assert batches_seen == [[2, 3]]
    assert stored == [[2, 3]]
    # El content_hash se registra solo para los chunks extraídos.
    updated = {u["id"] for u in session.updates}
    assert updated == {2, 3}
    assert 1 not in updated


def test_extract_propositions_for_doc_all_cached_skips_llm(monkeypatch):
    """Todos los chunks en cache → no se llama al LLM."""
    h1 = ki._chunk_content_hash("aaa")
    rows = [_chunk_row(1, "aaa", h1)]
    session = _PropsSession(rows, cached_chunk_ids=[1])
    called = []
    monkeypatch.setattr(
        ki, "_extract_propositions_batch", lambda *a, **k: called.append(1)
    )
    ki._extract_propositions_for_doc(
        session, doc_id=1, doc_path="doc.md", verbose=False, batch_size_tokens=250000
    )
    assert called == []


def test_extract_propositions_for_doc_drops_unassigned_props(monkeypatch):
    """Proposiciones sin chunk asignable se descartan (chunk_id NOT NULL)."""
    h1 = ki._chunk_content_hash("aaa")
    rows = [_chunk_row(1, "aaa", h1)]
    session = _PropsSession(rows, cached_chunk_ids=[])
    stored = []

    def fake_batch(session, doc_id, chunks, doc_path, verbose, **kwargs):
        return [
            {
                "doc_id": doc_id,
                "chunk_id": None,  # span no localizable
                "core_idea_id": "CI",
                "argument_id": "ARG",
                "statement": "Prop.",
                "text_span": "",
                "char_start": None,
                "char_end": None,
                "line_start": None,
                "line_end": None,
                "citations": [],
            }
        ]

    monkeypatch.setattr(ki, "_extract_propositions_batch", fake_batch)
    monkeypatch.setattr(
        ki,
        "_store_propositions",
        lambda s, d, props, embed_fn=None: stored.append(props),
    )
    ki._extract_propositions_for_doc(
        session, doc_id=1, doc_path="doc.md", verbose=False, batch_size_tokens=250000
    )
    assert stored == [[]]


def test_extract_propositions_for_doc_persists_in_chunk_index_order(monkeypatch):
    """El orden de persistencia es determinista (chunk_index) aunque los
    lotes se procesen en paralelo y terminen fuera de orden."""
    import time

    rows = [
        _chunk_row(1, "a" * 40000, ki._chunk_content_hash("a" * 40000)),  # 10k
        _chunk_row(2, "b" * 40000, ki._chunk_content_hash("b" * 40000)),  # 10k
        _chunk_row(3, "c" * 40000, ki._chunk_content_hash("c" * 40000)),  # 10k
        _chunk_row(4, "d" * 40000, ki._chunk_content_hash("d" * 40000)),  # 10k
    ]
    session = _PropsSession(rows, cached_chunk_ids=[])
    stored_order = []

    def fake_batch(session, doc_id, chunks, doc_path, verbose, **kwargs):
        # Completación fuera de orden: el lote con chunk_index mayor termina
        # antes.
        time.sleep(0.02 * (5 - chunks[0]["id"]))
        return [
            {
                "doc_id": doc_id,
                "chunk_id": c["id"],
                "section_path": kwargs.get("section_path", ""),
                "core_idea_id": "CI",
                "argument_id": "ARG",
                "statement": "Prop.",
                "text_span": "",
                "char_start": None,
                "char_end": None,
                "line_start": None,
                "line_end": None,
                "citations": [],
            }
            for c in chunks
        ]

    def fake_store(session, doc_id, propositions, embed_fn=None):
        stored_order.extend(p["chunk_id"] for p in propositions)

    monkeypatch.setattr(ki, "_extract_propositions_batch", fake_batch)
    monkeypatch.setattr(ki, "_store_propositions", fake_store)
    ki._extract_propositions_for_doc(
        session,
        doc_id=1,
        doc_path="doc.md",
        verbose=False,
        batch_size_tokens=20000,
        max_parallel=2,
    )
    # Los lotes se persisten en orden de chunk_index (1,2) antes que (3,4)
    # aunque el lote (3,4) termine primero.
    assert stored_order == [1, 2, 3, 4]


# ---------------------------------------------------------------------
# _proposition_flags (config + compatibilidad)
# ---------------------------------------------------------------------


def test_proposition_flags_param_false_disables(monkeypatch):
    monkeypatch.delenv("KAG_EXTRACT_PROPOSITIONS", raising=False)
    monkeypatch.delenv("KAG_PROPOSITION_BATCH_SIZE", raising=False)
    monkeypatch.delenv("KAG_PROPOSITION_MODEL", raising=False)
    monkeypatch.delenv("KAG_PROPOSITION_PARALLEL", raising=False)
    enabled, batch_size, model_size, max_parallel = ki._proposition_flags(
        None, extract_propositions=False
    )
    assert enabled is False
    assert batch_size == 400000
    assert model_size == "large"
    assert max_parallel == 3


def test_proposition_flags_defaults(monkeypatch):
    monkeypatch.delenv("KAG_EXTRACT_PROPOSITIONS", raising=False)
    monkeypatch.delenv("KAG_PROPOSITION_BATCH_SIZE", raising=False)
    monkeypatch.delenv("KAG_PROPOSITION_MODEL", raising=False)
    monkeypatch.delenv("KAG_PROPOSITION_PARALLEL", raising=False)
    enabled, batch_size, model_size, max_parallel = ki._proposition_flags(
        None, extract_propositions=True
    )
    assert enabled is True
    assert batch_size == 400000
    assert model_size == "large"
    assert max_parallel == 3


def test_proposition_flags_env_overrides(monkeypatch):
    monkeypatch.setenv("KAG_EXTRACT_PROPOSITIONS", "0")
    monkeypatch.setenv("KAG_PROPOSITION_BATCH_SIZE", "1000")
    monkeypatch.delenv("KAG_PROPOSITION_MODEL", raising=False)
    monkeypatch.delenv("KAG_PROPOSITION_PARALLEL", raising=False)
    enabled, batch_size, model_size, max_parallel = ki._proposition_flags(
        None, extract_propositions=True
    )
    assert enabled is False
    assert batch_size == 1000
    assert model_size == "large"
    assert max_parallel == 3


def test_proposition_flags_batch_size_env(monkeypatch):
    monkeypatch.delenv("KAG_EXTRACT_PROPOSITIONS", raising=False)
    monkeypatch.setenv("KAG_PROPOSITION_BATCH_SIZE", "50000")
    monkeypatch.delenv("KAG_PROPOSITION_MODEL", raising=False)
    monkeypatch.delenv("KAG_PROPOSITION_PARALLEL", raising=False)
    enabled, batch_size, model_size, max_parallel = ki._proposition_flags(
        None, extract_propositions=True
    )
    assert enabled is True
    assert batch_size == 50000
    assert model_size == "large"
    assert max_parallel == 3


def test_proposition_flags_model_env(monkeypatch):
    """KAG_PROPOSITION_MODEL=small cambia el modelo de extracción."""
    monkeypatch.delenv("KAG_EXTRACT_PROPOSITIONS", raising=False)
    monkeypatch.delenv("KAG_PROPOSITION_BATCH_SIZE", raising=False)
    monkeypatch.setenv("KAG_PROPOSITION_MODEL", "small")
    monkeypatch.delenv("KAG_PROPOSITION_PARALLEL", raising=False)
    enabled, _batch_size, model_size, max_parallel = ki._proposition_flags(
        None, extract_propositions=True
    )
    assert enabled is True
    assert model_size == "small"
    assert max_parallel == 3


def test_proposition_flags_model_invalid_falls_back_large(monkeypatch):
    """Valor inválido de KAG_PROPOSITION_MODEL → default "large" (nunca romper)."""
    monkeypatch.delenv("KAG_EXTRACT_PROPOSITIONS", raising=False)
    monkeypatch.delenv("KAG_PROPOSITION_BATCH_SIZE", raising=False)
    monkeypatch.setenv("KAG_PROPOSITION_MODEL", "gigante")
    monkeypatch.delenv("KAG_PROPOSITION_PARALLEL", raising=False)
    enabled, _batch_size, model_size, max_parallel = ki._proposition_flags(
        None, extract_propositions=True
    )
    assert enabled is True
    assert model_size == "large"
    assert max_parallel == 3


def test_proposition_flags_parallel_env(monkeypatch):
    """KAG_PROPOSITION_PARALLEL configura el semáforo de concurrencia."""
    monkeypatch.delenv("KAG_EXTRACT_PROPOSITIONS", raising=False)
    monkeypatch.delenv("KAG_PROPOSITION_BATCH_SIZE", raising=False)
    monkeypatch.delenv("KAG_PROPOSITION_MODEL", raising=False)
    monkeypatch.setenv("KAG_PROPOSITION_PARALLEL", "5")
    enabled, batch_size, _model_size, max_parallel = ki._proposition_flags(
        None, extract_propositions=True
    )
    assert enabled is True
    assert batch_size == 400000
    assert max_parallel == 5


# ---------------------------------------------------------------------
# index_document: compatibilidad extract_propositions=False → no LLM
# ---------------------------------------------------------------------


def _install_fake_modules(monkeypatch):
    """Inyecta módulos fake para los imports perezosos de index_document
    (segmentador/embeddings/entidades) — sin torch/spacy."""
    import sys
    import types

    fake_seg = types.ModuleType("src.kag.segmentador")
    fake_seg.build_segmenter = lambda *a, **k: SimpleNamespace(nlp=None)
    monkeypatch.setitem(sys.modules, "src.kag.segmentador", fake_seg)

    fake_emb = types.ModuleType("src.embeddings")
    fake_emb.embed_texts = lambda texts, **k: [[0.1] * 768 for _ in texts]
    monkeypatch.setitem(sys.modules, "src.embeddings", fake_emb)

    fake_ent = types.ModuleType("src.kag.entities")
    fake_ent.extract_entities_deterministic = lambda *a, **k: {
        "entities": [],
        "relations": [],
    }
    fake_ent.extract_entities_deterministic_batch = lambda *a, **k: [
        {"entities": [], "relations": []}
    ]
    monkeypatch.setitem(sys.modules, "src.kag.entities", fake_ent)


class _IndexSession:
    """Sesión fake para index_document (doc nuevo, 1 chunk)."""

    def __init__(self):
        self.executed = []

    def execute(self, stmt, params=None):
        sql = str(stmt)
        self.executed.append(sql)
        if "INSERT INTO kag_documents" in sql:
            return _Result(scalar=1)
        if "FROM kag_documents" in sql and "WHERE doc_path" in sql:
            return _Result(first=None)
        if "INSERT INTO kag_chunks" in sql:
            return _Result()
        if "FROM kag_chunks" in sql and "ORDER BY chunk_index" in sql:
            return _Result(
                fetchall=[
                    SimpleNamespace(
                        id=1,
                        content="Contenido del documento.",
                        chunk_index=0,
                        section_path="",
                    )
                ]
            )
        if "UPDATE kag_chunks" in sql:
            return _Result()
        if "COUNT(*) FROM kag_entities" in sql:
            return _Result(scalar=0)
        if "COUNT(*) FROM kag_relations" in sql:
            return _Result(scalar=0)
        if "UPDATE kag_documents" in sql:
            return _Result()
        return _Result()

    def commit(self):
        pass

    def rollback(self):
        pass


def _index_document_setup(monkeypatch, tmp_path):
    """Mocks comunes para correr index_document sin DB ni modelos pesados."""
    monkeypatch.setattr(ki, "DOCS_DIR", tmp_path)
    md = tmp_path / "doc_test.md"
    md.write_text("# Título\n\nContenido del documento.", encoding="utf-8")
    _install_fake_modules(monkeypatch)
    monkeypatch.setattr(
        ki,
        "chunk_markdown",
        lambda *a, **k: [
            {
                "section_path": "",
                "content": "Contenido del documento.",
                "token_estimate": 10,
            }
        ],
    )
    monkeypatch.setattr(ki, "_maintain_word_freq", lambda *a, **k: None)
    monkeypatch.setattr(ki, "set_stage", lambda *a, **k: None)
    monkeypatch.setattr(ki, "_store_entities_relations", lambda *a, **k: None)
    monkeypatch.setattr(ki, "_index_figures", lambda *a, **k: 0)
    monkeypatch.setattr(ki, "summarize_document", lambda *a, **k: "")
    return md


def test_index_document_extract_propositions_false_skips_llm(monkeypatch, tmp_path):
    """Compatibilidad: extract_propositions=False → no se llama al LLM."""
    md = _index_document_setup(monkeypatch, tmp_path)
    calls = {"props": 0}
    monkeypatch.setattr(
        ki,
        "_extract_propositions_for_doc",
        lambda *a, **k: calls.__setitem__("props", calls["props"] + 1),
    )
    result = ki.index_document(
        _IndexSession(), md, extract_propositions=False, no_summary=True
    )
    assert result["status"] == "indexed"
    assert calls["props"] == 0


def test_index_document_extract_propositions_true_calls_step(monkeypatch, tmp_path):
    """extract_propositions=True (default) → el paso de proposiciones se ejecuta."""
    md = _index_document_setup(monkeypatch, tmp_path)
    calls = {"props": 0}
    monkeypatch.setattr(
        ki,
        "_extract_propositions_for_doc",
        lambda *a, **k: calls.__setitem__("props", calls["props"] + 1),
    )
    result = ki.index_document(_IndexSession(), md, no_summary=True)
    assert result["status"] == "indexed"
    assert calls["props"] == 1
