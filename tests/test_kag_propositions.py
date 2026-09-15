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
    assert ":s0" in sql and ":s1" in sql
    assert "executemany" not in sql.lower()
    assert params["d0"] == 1
    assert params["c0"] == 2
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
