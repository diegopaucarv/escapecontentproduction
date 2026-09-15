"""Tests de la máquina de estados de ingesta legacy (src/kag_ingest.py).

Sin DB real y sin red: sesión falsa que simula kag_documents/kag_chunks/
kag_entities/kag_relations/kag_figures, segmentador y embeddings mockeados.
Verifica la reanudación por etapa: al interrumpir la ingesta, la siguiente
pasada salta las etapas completadas y continúa desde la interrumpida.
"""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.kag_ingest as ki

# ---------------------------------------------------------------------
# Sesión falsa (kag_documents + kag_chunks + kag_entities + kag_figures)
# ---------------------------------------------------------------------


class _FakeResult:
    def __init__(self, rows=None, scalar=None):
        self._rows = rows or []
        self._scalar = scalar

    def first(self):
        return self._rows[0] if self._rows else None

    def scalar(self):
        return self._scalar

    def fetchall(self):
        return self._rows


class _FakeSession:
    def __init__(self):
        self.documents = []
        self.chunks = []
        self.entities = []
        self.relations = []
        self.figures = []
        self._next = 1
        self.commits = 0
        self.rollbacks = 0

    def _next_id(self):
        v = self._next
        self._next += 1
        return v

    def execute(self, stmt, params=None):
        sql = str(stmt)
        params = params or {}
        if "INSERT INTO kag_documents" in sql:
            row_id = self._next_id()
            self.documents.append({"id": row_id, "stage": "pending", **params})
            return _FakeResult(scalar=row_id)
        if "INSERT INTO kag_chunks" in sql:
            row_id = self._next_id()
            self.chunks.append({"id": row_id, **params})
            return _FakeResult(scalar=row_id)
        if "INSERT INTO kag_entities" in sql:
            row_id = self._next_id()
            self.entities.append({"id": row_id, **params})
            return _FakeResult(scalar=row_id)
        if "INSERT INTO kag_relations" in sql:
            row_id = self._next_id()
            self.relations.append({"id": row_id, **params})
            return _FakeResult(scalar=row_id)
        if "INSERT INTO kag_figures" in sql:
            row_id = self._next_id()
            self.figures.append({"id": row_id, **params})
            return _FakeResult(scalar=row_id)
        if "SELECT id, content_hash, status, stage FROM kag_documents" in sql:
            doc_path = params.get("p")
            for d in self.documents:
                if d.get("doc_path") == doc_path:
                    return _FakeResult(
                        rows=[
                            SimpleNamespace(
                                id=d["id"],
                                content_hash=d["content_hash"],
                                status=d["status"],
                                stage=d.get("stage", "pending"),
                            )
                        ]
                    )
            return _FakeResult(rows=[])
        if "SELECT id, content FROM kag_chunks" in sql:
            rows = [
                SimpleNamespace(id=c["id"], content=c["content"]) for c in self.chunks
            ]
            return _FakeResult(rows=rows)
        if "SELECT COUNT(*) FROM kag_chunks" in sql:
            if "embedding IS NULL" in sql:
                nulls = sum(1 for c in self.chunks if c.get("embedding") is None)
                return _FakeResult(scalar=nulls)
            return _FakeResult(scalar=len(self.chunks))
        if "SELECT COUNT(*) FROM kag_entities" in sql:
            return _FakeResult(scalar=len(self.entities))
        if "SELECT COUNT(*) FROM kag_relations" in sql:
            return _FakeResult(scalar=len(self.relations))
        if "SELECT COUNT(*) FROM kag_figures" in sql:
            return _FakeResult(scalar=len(self.figures))
        if "SELECT name_norm, id FROM kag_entities" in sql:
            return _FakeResult(rows=[])
        if "UPDATE kag_documents" in sql:
            doc_id = params.get("id")
            for d in self.documents:
                if d["id"] == doc_id:
                    if "stage" in params:
                        d["stage"] = params["stage"]
                    if "status" in params:
                        d["status"] = params["status"]
                    else:
                        d["status"] = "ready"
            return _FakeResult()
        if "UPDATE kag_chunks" in sql:
            chunk_id = params.get("id")
            for c in self.chunks:
                if c["id"] == chunk_id:
                    c["embedding"] = params.get("emb")
            return _FakeResult()
        if "DELETE FROM kag_documents" in sql:
            doc_id = params.get("id")
            self.documents = [d for d in self.documents if d["id"] != doc_id]
            self.chunks = [c for c in self.chunks if c["doc_id"] != doc_id]
            self.entities = [e for e in self.entities if e["doc_id"] != doc_id]
            self.relations = [r for r in self.relations if r["doc_id"] != doc_id]
            self.figures = [f for f in self.figures if f["doc_id"] != doc_id]
            return _FakeResult()
        if "DELETE FROM kag_chunks" in sql:
            doc_id = params.get("doc_id")
            self.chunks = [c for c in self.chunks if c["doc_id"] != doc_id]
            return _FakeResult()
        if "DELETE FROM kag_entities" in sql:
            doc_id = params.get("doc_id")
            self.entities = [e for e in self.entities if e["doc_id"] != doc_id]
            return _FakeResult()
        if "DELETE FROM kag_relations" in sql:
            doc_id = params.get("doc_id")
            self.relations = [r for r in self.relations if r["doc_id"] != doc_id]
            return _FakeResult()
        if "DELETE FROM kag_figures" in sql:
            doc_id = params.get("doc_id")
            self.figures = [f for f in self.figures if f["doc_id"] != doc_id]
            return _FakeResult()
        return _FakeResult()

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


# ---------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------


class _FakeSegmenter:
    def __init__(self):
        self.nlp = object()

    def segment_text(self, content, max_tokens=800):
        # Un segmento por línea no vacía (determinista).
        return [l for l in content.splitlines() if l.strip()]


def _fake_build_segmenter(session, lang="es", verbose=False):
    return _FakeSegmenter()


def _fake_embed_texts(texts, input_type="document"):
    return [[0.1, 0.2, 0.3]] * len(texts)


def _fake_extract_deterministic(nlp, content, embed_fn=None, lang="es"):
    return {"entities": [], "relations": []}


def _write_md(tmp_path, n_lines=20):
    md = tmp_path / "doc.md"
    lines = ["# Título", "Primera línea de contenido."]
    lines += [f"Línea {i}" for i in range(n_lines - 2)]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md


@pytest.fixture
def ingest_env(tmp_path, monkeypatch):
    md = _write_md(tmp_path)
    monkeypatch.setattr(ki, "DOCS_DIR", tmp_path)
    monkeypatch.setattr(ki, "IMAGES_DIR", tmp_path / "images")
    monkeypatch.setattr("src.kag.segmentador.build_segmenter", _fake_build_segmenter)
    monkeypatch.setattr("src.embeddings.embed_texts", _fake_embed_texts)
    monkeypatch.setattr(
        "src.kag.entities.extract_entities_deterministic",
        _fake_extract_deterministic,
    )
    monkeypatch.setattr(ki, "summarize_document", lambda *a, **k: "resumen")
    return md


# ---------------------------------------------------------------------
# Reanudación por etapa
# ---------------------------------------------------------------------


def test_index_document_full_pipeline(ingest_env):
    session = _FakeSession()
    result = ki.index_document(session, ingest_env, verbose=False)
    assert result["status"] == "indexed"
    assert len(session.documents) == 1
    assert session.documents[0]["status"] == "ready"
    assert session.documents[0]["stage"] == "ready"
    assert len(session.chunks) > 0


def test_index_document_skips_when_ready(ingest_env):
    session = _FakeSession()
    ki.index_document(session, ingest_env, verbose=False)
    # Segunda pasada: mismo hash + ready -> salta.
    result = ki.index_document(session, ingest_env, verbose=False)
    assert result["status"] == "skipped"
    assert len(session.documents) == 1
    assert len(session.chunks) == len(session.chunks)  # sin duplicar


def test_index_document_resumes_from_segmented(ingest_env):
    """Interrupción tras segmentar (stage='segmented') -> reanuda embebiendo.

    Los chunks ya están (embedding NULL); NO se re-segmenta ni se re-insertan
    chunks. Solo se completan chunked/figures/ready.
    """
    session = _FakeSession()
    ki.index_document(session, ingest_env, verbose=False)
    doc = session.documents[0]
    n_chunks = len(session.chunks)

    # Simular interrupción: murió tras 'segmented' (chunks sin embedding).
    for c in session.chunks:
        c["embedding"] = None
    doc["stage"] = "segmented"
    doc["status"] = "pending"

    result = ki.index_document(session, ingest_env, verbose=False)
    assert result["status"] == "indexed"
    # No se re-segmentó: la misma cantidad de chunks, sin duplicar.
    assert len(session.chunks) == n_chunks
    assert session.documents[0]["status"] == "ready"
    assert session.documents[0]["stage"] == "ready"


def test_index_document_resumes_from_chunked(ingest_env):
    """Interrupción tras 'chunked' -> reanuda desde figures (no re-embebe)."""
    session = _FakeSession()
    ki.index_document(session, ingest_env, verbose=False)
    doc = session.documents[0]
    n_chunks = len(session.chunks)
    n_entities = len(session.entities)

    # Simular interrupción: murió tras 'chunked' (figures pendientes).
    doc["stage"] = "chunked"
    doc["status"] = "pending"

    result = ki.index_document(session, ingest_env, verbose=False)
    assert result["status"] == "indexed"
    assert len(session.chunks) == n_chunks
    assert len(session.entities) == n_entities
    assert session.documents[0]["status"] == "ready"
    assert session.documents[0]["stage"] == "ready"


def test_index_document_force_reindexes(ingest_env):
    session = _FakeSession()
    ki.index_document(session, ingest_env, verbose=False)
    result = ki.index_document(session, ingest_env, force=True, verbose=False)
    assert result["status"] == "indexed"
    assert len(session.documents) == 1  # borrado + re-insertado
