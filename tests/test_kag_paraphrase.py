"""Tests de la Fase 3 (segmentación dentro de capítulos) y Fase 4
(paráfrasis de chunks) — sin DB real, sin torch/spacy/transformers.

Patrón de tests/test_kag.py: sesiones falsas + monkeypatch de
call_with_retries/_get_prompt_pair. NO se importa src.kag.segmentador a
nivel de módulo (carga torch/spacy, ~30s).
"""

import json
from types import SimpleNamespace

from src.kag.stages import KAG_INGEST_STAGES
from src.kag_ingest import (
    _paraphrase_chunks,
    chunk_markdown,
)

# ---------------------------------------------------------------------
# chunk_markdown con chapters (Fase 3)
# ---------------------------------------------------------------------


class _FakeSegmenter:
    """Segmenter duck-typed: devuelve listas fijas y registra las llamadas."""

    def __init__(self, segments_per_call=2):
        self.segments_per_call = segments_per_call
        self.calls = []

    def segment_text(self, text, max_tokens=800):
        self.calls.append((text, max_tokens))
        return [f"seg{i}:{text[:20]}" for i in range(self.segments_per_call)]


def test_chunk_markdown_with_chapters_assigns_chapter_id():
    md = (
        "# Cap 1\n\nContenido del capítulo uno.\n\n"
        "# Cap 2\n\nContenido del capítulo dos."
    )
    seg = _FakeSegmenter()
    chapters = [
        {"chapter_id": "uuid-1", "line_start": 1, "line_end": 3},
        {"chapter_id": "uuid-2", "line_start": 5, "line_end": 6},
    ]
    chunks = chunk_markdown(md, "short", seg, max_tokens=800, chapters=chapters)

    # Cada capítulo produce sus propios chunks con su chapter_id.
    assert len(chunks) == 2
    assert chunks[0]["chapter_id"] == "uuid-1"
    assert chunks[1]["chapter_id"] == "uuid-2"
    # El texto segmentado es el del capítulo (no el documento completo).
    assert "Cap 1" in chunks[0]["content"]
    assert "Cap 2" in chunks[1]["content"]
    assert "Cap 2" not in chunks[0]["content"]


def test_chunk_markdown_without_chapters_keeps_classic_behavior():
    md = "# Título\n\nContenido del documento."
    seg = _FakeSegmenter()
    chunks = chunk_markdown(md, "short", seg, max_tokens=800)
    assert len(chunks) == 1
    assert chunks[0]["chapter_id"] is None

    # chapters=[] explícito → mismo comportamiento clásico.
    chunks2 = chunk_markdown(md, "short", seg, max_tokens=800, chapters=[])
    assert len(chunks2) == 1
    assert chunks2[0]["chapter_id"] is None


def test_chunk_markdown_chapters_respects_max_tokens_merge():
    class _BigSegments:
        def segment_text(self, text, max_tokens=800):
            return ["a" * 2000, "b" * 2000]  # ~500 tokens cada uno

    md = "\n".join(["# Cap 1", "x" * 2000, "y" * 2000])
    chapters = [{"chapter_id": "uuid-1", "line_start": 1, "line_end": 3}]
    chunks = chunk_markdown(
        md, "short", _BigSegments(), max_tokens=800, chapters=chapters
    )
    assert len(chunks) == 2  # no caben juntos en 800 tokens
    assert all(c["chapter_id"] == "uuid-1" for c in chunks)


def test_chunk_markdown_chapters_use_coref_false_disables_coref():
    class _SegWithCoref:
        def __init__(self):
            self.coref_called = False

        def segment_text(self, text, max_tokens=800):
            segs = ["seg uno", "seg dos"]
            return self.resolve_coreferences(segs)

        def resolve_coreferences(self, segments):
            self.coref_called = True
            return segments

    md = "# Cap 1\n\nContenido."
    chapters = [{"chapter_id": "uuid-1", "line_start": 1, "line_end": 2}]
    seg = _SegWithCoref()
    chunk_markdown(md, "short", seg, use_coref=False, chapters=chapters)
    assert seg.coref_called is False  # no-op temporal

    seg2 = _SegWithCoref()
    chunk_markdown(md, "short", seg2, use_coref=True, chapters=chapters)
    assert seg2.coref_called is True  # comportamiento original preservado


# ---------------------------------------------------------------------
# _paraphrase_chunks (Fase 4)
# ---------------------------------------------------------------------


class _FakeSession:
    """Sesión falsa: registra los UPDATE de paraphrase por chunk."""

    def __init__(self, rows):
        self.rows = rows
        self.updates = []  # (id, paraphrase)
        self.commits = 0

    def execute(self, stmt, params=None):
        sql = str(stmt)
        if "FROM kag_chunks" in sql and "WHERE doc_id" in sql:
            return SimpleNamespace(fetchall=lambda: self.rows)
        if "UPDATE kag_chunks SET paraphrase" in sql:
            self.updates.append((params["id"], params["p"]))
            return SimpleNamespace()
        raise AssertionError(f"SQL inesperado: {sql}")

    def commit(self):
        self.commits += 1


def _row(id_, chunk_index, content, chapter_id):
    return SimpleNamespace(
        id=id_, chunk_index=chunk_index, content=content, chapter_id=chapter_id
    )


def test_paraphrase_chunks_updates_by_chunk(monkeypatch):
    rows = [
        _row(1, 0, "Contenido A.", "uuid-1"),
        _row(2, 1, "Contenido B.", "uuid-1"),
        _row(3, 2, "Contenido C.", None),
    ]
    session = _FakeSession(rows)

    def fake_call(session, prompt, system=None, model_size=None, **kwargs):
        # Devuelve paráfrasis para los 3 chunks (2 del capítulo + 1 sin capítulo).
        return (
            json.dumps(
                {
                    "paraphrases": [
                        {"chunk_index": 0, "paraphrase": "Paráfrasis A."},
                        {"chunk_index": 1, "paraphrase": "Paráfrasis B."},
                        {"chunk_index": 2, "paraphrase": "Paráfrasis C."},
                    ]
                }
            ),
            "model",
            False,
        )

    monkeypatch.setattr("src.kag_ingest.call_with_retries", fake_call)
    _paraphrase_chunks(session, 42, "docs/archivo.md", verbose=False)

    assert session.updates == [
        (1, "Paráfrasis A."),
        (2, "Paráfrasis B."),
        (3, "Paráfrasis C."),
    ]
    assert session.commits == 1


def test_paraphrase_chunks_group_without_chapter(monkeypatch):
    # Grupo sin capítulo (chapter_id NULL) → chapter_id "" en el prompt.
    rows = [
        _row(1, 0, "Contenido A.", None),
        _row(2, 1, "Contenido B.", None),
    ]
    session = _FakeSession(rows)
    seen_chapter_ids = []

    def fake_call(session, prompt, system=None, model_size=None, **kwargs):
        seen_chapter_ids.append(prompt)
        return (
            json.dumps(
                {
                    "paraphrases": [
                        {"chunk_index": 0, "paraphrase": "Paráfrasis A."},
                        {"chunk_index": 1, "paraphrase": "Paráfrasis B."},
                    ]
                }
            ),
            "model",
            False,
        )

    monkeypatch.setattr("src.kag_ingest.call_with_retries", fake_call)
    _paraphrase_chunks(session, 42, "docs/archivo.md", verbose=False)

    assert session.updates == [(1, "Paráfrasis A."), (2, "Paráfrasis B.")]
    # El prompt del grupo sin capítulo usa "(sin capítulo)".
    assert "(sin capítulo)" in seen_chapter_ids[0]


def test_paraphrase_chunks_llm_failure_degrades(monkeypatch):
    rows = [
        _row(1, 0, "Contenido A.", "uuid-1"),
        _row(2, 1, "Contenido B.", "uuid-1"),
    ]
    session = _FakeSession(rows)

    def fake_call(session, prompt, system=None, model_size=None, **kwargs):
        raise RuntimeError("LLM no disponible")

    monkeypatch.setattr("src.kag_ingest.call_with_retries", fake_call)
    _paraphrase_chunks(session, 42, "docs/archivo.md", verbose=False)

    # Degradación: sin UPDATE, sin excepción, commit igual.
    assert session.updates == []
    assert session.commits == 1


def test_paraphrase_chunks_invalid_json_degrades(monkeypatch):
    rows = [_row(1, 0, "Contenido A.", "uuid-1")]
    session = _FakeSession(rows)

    def fake_call(session, prompt, system=None, model_size=None, **kwargs):
        return "no es json", "model", False

    monkeypatch.setattr("src.kag_ingest.call_with_retries", fake_call)
    _paraphrase_chunks(session, 42, "docs/archivo.md", verbose=False)

    assert session.updates == []
    assert session.commits == 1


def test_paraphrase_chunks_no_chunks_noop(monkeypatch):
    session = _FakeSession([])
    called = []

    def fake_call(session, prompt, system=None, model_size=None, **kwargs):
        called.append(prompt)
        return "{}", "model", False

    monkeypatch.setattr("src.kag_ingest.call_with_retries", fake_call)
    _paraphrase_chunks(session, 42, "docs/archivo.md", verbose=False)

    assert called == []  # sin chunks → sin llamadas LLM
    assert session.updates == []
    assert session.commits == 0


# ---------------------------------------------------------------------
# Máquina de estados: etapa paraphrased
# ---------------------------------------------------------------------


def test_kag_ingest_stages_includes_paraphrased():
    assert "paraphrased" in KAG_INGEST_STAGES
    assert (
        KAG_INGEST_STAGES.index("paraphrased")
        == KAG_INGEST_STAGES.index("segmented") + 1
    )
    assert (
        KAG_INGEST_STAGES.index("paraphrased") == KAG_INGEST_STAGES.index("chunked") - 1
    )
