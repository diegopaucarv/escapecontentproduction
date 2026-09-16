"""Tests del índice de resúmenes jerárquicos (summary_index, migración 0032).

Patrón de tests/test_kag.py (summarize_document, L2456-2723): sesión falsa
+ monkeypatch de load_settings/_get_prompt_pair/complete_local. Aquí además
se parchea src.embeddings.embed_texts (import perezoso dentro de
_persist_summary_index) para verificar el batch de embeddings y su
degradación. Sin DB real.
"""

import uuid
from types import SimpleNamespace

import src.embeddings


class _SummarySession:
    """Sesión falsa: registra execute() y devuelve un UUID para RETURNING id."""

    def __init__(self):
        self.calls = []
        self.doc_row_id = str(uuid.uuid4())

    def execute(self, stmt, params=None):
        self.calls.append((str(stmt), params))

        class _R:
            def __init__(self, scalar_value):
                self._scalar = scalar_value

            def scalars(self):
                return self

            def first(self):
                return None

            def scalar(self):
                return self._scalar

        if "RETURNING id" in str(stmt):
            return _R(self.doc_row_id)
        return _R(None)


class _BrokenSession(_SummarySession):
    """Sesión que lanza en execute (persistencia caída)."""

    def execute(self, stmt, params=None):
        raise RuntimeError("db caída")


def _patch_summary_deps(monkeypatch, fake_complete, fake_embed=None):
    """Parchea load_settings/_get_prompt_pair/complete_local/embed_texts."""
    import src.kag_ingest as ki

    monkeypatch.setattr(
        ki,
        "load_settings",
        lambda session: SimpleNamespace(small_model="qwen2.5-3b"),
    )
    monkeypatch.setattr(
        ki,
        "_get_prompt_pair",
        lambda *a, **k: ("SYS", "<text>\n{text}\n</text>\n\nSummary:"),
    )
    monkeypatch.setattr(ki, "complete_local", fake_complete)
    monkeypatch.setattr(
        ki, "resolve_config", lambda session: {"KAG_SUMMARY_PARALLEL": 3}
    )
    if fake_embed is not None:
        monkeypatch.setattr(src.embeddings, "embed_texts", fake_embed)
    return ki


LONG_MD = (
    "# Cap 1\n\nContenido del capítulo uno.\n\n"
    "## Sección 1.1\n\nDetalle de la sección 1.1.\n\n"
    "# Cap 2\n\nContenido del capítulo dos.\n\n"
    "## Sección 2.1\n\nDetalle de la sección 2.1."
)


def _fake_complete(session, prompt, system=None, max_tokens=None, **kw):
    if max_tokens == 200:
        return "RESUMEN FINAL"
    inner = prompt.split("<text>\n", 1)[1].split("\n</text>", 1)[0]
    return f"map:{inner[:30]}"


def _fake_embed(texts, input_type="document"):
    return [[float(i)] for i in range(len(texts))]


def _insert_calls(session):
    """Devuelve las llamadas INSERT a summary_index (documento y secciones)."""
    return [c for c in session.calls if "INSERT INTO summary_index" in c[0]]


def test_persist_document_and_section_rows(monkeypatch):
    """doc_id=42 → persiste fila documento + 4 filas sección con parent_id."""
    import src.kag_ingest as ki

    session = _SummarySession()
    ki = _patch_summary_deps(monkeypatch, _fake_complete, _fake_embed)

    result = ki.summarize_document(session, LONG_MD, "long", doc_id=42)

    assert result == "RESUMEN FINAL"
    # DELETE previo (idempotencia para re-ingesta con --force).
    deletes = [c for c in session.calls if "DELETE FROM summary_index" in c[0]]
    assert len(deletes) == 1
    assert deletes[0][1] == {"doc_id": 42}
    inserts = _insert_calls(session)
    assert len(inserts) == 2  # 1 documento + 1 multi-VALUES de secciones
    # Fila documento: level='document', parent_id NULL, text=final.
    doc_stmt, doc_params = inserts[0]
    assert "RETURNING id" in doc_stmt
    assert doc_params["doc_id"] == 42
    assert doc_params["text"] == "RESUMEN FINAL"
    assert doc_params["emb"] == "[0.0]"
    # Fila secciones: 4 placeholders, parent_id = doc_row_id.
    sec_stmt, sec_params = inserts[1]
    assert sec_stmt.count("'section'") == 4
    for i in range(4):
        assert sec_params[f"d{i}"] == 42
        assert sec_params[f"p{i}"] == session.doc_row_id
        assert sec_params[f"emb{i}"] == f"[{i + 1}.0]"
    assert sec_params["t0"].startswith("map:# Cap 1")


def test_doc_id_none_does_not_persist(monkeypatch):
    """doc_id=None → comportamiento viejo: no persiste nada."""
    import src.kag_ingest as ki

    session = _SummarySession()
    ki = _patch_summary_deps(monkeypatch, _fake_complete, _fake_embed)

    result = ki.summarize_document(session, LONG_MD, "long")

    assert result == "RESUMEN FINAL"
    assert session.calls == []


def test_empty_final_summary_does_not_persist(monkeypatch):
    """Resumen final '' (reduce degradado) → no persiste nada."""
    import src.kag_ingest as ki

    def fake_complete(session, prompt, system=None, max_tokens=None, **kw):
        if max_tokens == 200:
            return ""  # reduce falla → final_summary ''
        inner = prompt.split("<text>\n", 1)[1].split("\n</text>", 1)[0]
        return f"map:{inner[:30]}"

    session = _SummarySession()
    ki = _patch_summary_deps(monkeypatch, fake_complete, _fake_embed)

    result = ki.summarize_document(session, LONG_MD, "long", doc_id=42)

    assert result == ""
    assert session.calls == []


def test_embedding_batch_called_with_final_plus_sections(monkeypatch):
    """El batch de embeddings recibe [final] + secciones (una sola llamada)."""
    import src.kag_ingest as ki

    calls = []

    def fake_embed(texts, input_type="document"):
        calls.append((list(texts), input_type))
        return [[float(i)] for i in range(len(texts))]

    session = _SummarySession()
    ki = _patch_summary_deps(monkeypatch, _fake_complete, fake_embed)

    result = ki.summarize_document(session, LONG_MD, "long", doc_id=42)

    assert result == "RESUMEN FINAL"
    assert len(calls) == 1
    texts, input_type = calls[0]
    assert input_type == "document"
    assert texts[0] == "RESUMEN FINAL"
    assert len(texts) == 5  # 1 final + 4 secciones
    assert all(t.startswith("map:") for t in texts[1:])


def test_embedding_failure_degrades_to_null(monkeypatch):
    """Si embed_texts lanza → embedding NULL (no rompe; el FTS sigue)."""
    import src.kag_ingest as ki

    def fake_embed(texts, input_type="document"):
        raise RuntimeError("modelo no disponible")

    session = _SummarySession()
    ki = _patch_summary_deps(monkeypatch, _fake_complete, fake_embed)

    result = ki.summarize_document(session, LONG_MD, "long", doc_id=42)

    assert result == "RESUMEN FINAL"
    inserts = _insert_calls(session)
    assert len(inserts) == 2
    assert inserts[0][1]["emb"] is None  # documento sin embedding
    for i in range(4):
        assert inserts[1][1][f"emb{i}"] is None  # secciones sin embedding


def test_persistence_failure_does_not_break_summary(monkeypatch):
    """La persistencia lanza → summarize_document sigue devolviendo el resumen."""
    import src.kag_ingest as ki

    session = _BrokenSession()
    ki = _patch_summary_deps(monkeypatch, _fake_complete, _fake_embed)

    result = ki.summarize_document(session, LONG_MD, "long", doc_id=42)

    assert result == "RESUMEN FINAL"
