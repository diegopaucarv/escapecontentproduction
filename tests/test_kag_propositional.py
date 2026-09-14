"""Tests de la ingesta proposicional KAG (src/kag_propositional.py).

Sin DB real y sin red: sesiones falsas (_FakeSession/_FakeResult), LLM
mockeado (monkeypatch de call_with_retries / complete_vision), embeddings
mockeados y MultibookFinderTool/LibraryOfCongressAPITool mockeados. Patrón
de tests/test_kag.py.
"""

import json
from types import SimpleNamespace

import pytest

import src.kag_propositional as kp

# ---------------------------------------------------------------------
# Sesión falsa
# ---------------------------------------------------------------------


class _FakeResult:
    """Resultado mínimo de session.execute: first/scalar/fetchall/scalars."""

    def __init__(self, rows=None, scalar=None):
        self._rows = rows or []
        self._scalar = scalar

    def first(self):
        return self._rows[0] if self._rows else None

    def scalar(self):
        return self._scalar

    def fetchall(self):
        return self._rows

    def scalars(self):
        return self

    def all(self):
        return self._rows


class _FakeSession:
    """Sesión falsa que registra inserts por tabla y simula RETURNING id.

    - SELECT de idempotencia sobre documents devuelve la fila almacenada.
    - SELECT de chunks (árbol temático) devuelve los chunks insertados.
    - UPDATE documents SET status='ready' actualiza la fila almacenada.
    - Cualquier query de session_settings (load_settings) devuelve vacío.
    """

    def __init__(self):
        self.documents = []
        self.chapters = []
        self.chunks = []
        self.topic_nodes = []
        self.images = []
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
        if "session_settings" in sql:
            return _FakeResult(rows=[])
        if "INSERT INTO documents" in sql:
            row_id = self._next_id()
            self.documents.append({"id": row_id, **params})
            return _FakeResult(scalar=row_id)
        if "INSERT INTO document_chapters" in sql:
            row_id = self._next_id()
            self.chapters.append({"id": row_id, **params})
            return _FakeResult(scalar=row_id)
        if "INSERT INTO propositional_chunks" in sql:
            row_id = self._next_id()
            self.chunks.append({"id": row_id, **params})
            return _FakeResult(scalar=row_id)
        if "INSERT INTO topic_tree_nodes" in sql:
            row_id = self._next_id()
            self.topic_nodes.append({"id": row_id, **params})
            return _FakeResult(scalar=row_id)
        if "INSERT INTO document_images" in sql:
            row_id = self._next_id()
            self.images.append({"id": row_id, **params})
            return _FakeResult(scalar=row_id)
        if "SELECT id, content_hash, status FROM documents" in sql:
            did = params.get("did") or params.get("document_id")
            for d in self.documents:
                if d.get("document_id") == did:
                    return _FakeResult(
                        rows=[
                            SimpleNamespace(
                                id=d["id"],
                                content_hash=d["content_hash"],
                                status=d["status"],
                            )
                        ]
                    )
            return _FakeResult(rows=[])
        if "SELECT id, statement, embedding FROM propositional_chunks" in sql:
            rows = [
                SimpleNamespace(
                    id=c["id"], statement=c["statement"], embedding=c["emb"]
                )
                for c in self.chunks
            ]
            return _FakeResult(rows=rows)
        if "UPDATE documents" in sql:
            doc_id = params.get("id")
            for d in self.documents:
                if d["id"] == doc_id:
                    d["status"] = "ready"
            return _FakeResult()
        if "DELETE FROM documents" in sql:
            doc_id = params.get("id")
            self.documents = [d for d in self.documents if d["id"] != doc_id]
            # Simula ON DELETE CASCADE de la DB real: al borrar el documento
            # se borran sus capítulos, chunks, nodos temáticos e imágenes.
            self.chapters = [c for c in self.chapters if c["document_id"] != doc_id]
            self.chunks = [c for c in self.chunks if c["document_id"] != doc_id]
            self.topic_nodes = [
                t for t in self.topic_nodes if t["document_id"] != doc_id
            ]
            self.images = [i for i in self.images if i["document_id"] != doc_id]
            return _FakeResult()
        return _FakeResult()

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


# ---------------------------------------------------------------------
# Mocks compartidos
# ---------------------------------------------------------------------


class _FakeFinder:
    """MultibookFinderTool mockeado: un único documento en el .md."""

    def __init__(self, md_path, document_id="stacked_doc_001", line_end=60):
        self.md_path = md_path
        self.document_id = document_id
        self.line_end = line_end

    def execute(self, file_path):
        return [
            {
                "source_file": str(self.md_path),
                "document_id": self.document_id,
                "title": "Documento de prueba",
                "line_start": 1,
                "line_end": self.line_end,
                "total_lines": self.line_end,
                "structural_keyword_hits": 0,
            }
        ]


class _FakeLoC:
    """LibraryOfCongressAPITool mockeado: sin red, devuelve None (best-effort)."""

    def query_subject_heading(self, term):
        return None

    def query_classification_code(self, label):
        return None


def _fake_embed_texts(texts, input_type="document"):
    return [[0.1, 0.2, 0.3]] * len(texts)


def _write_stacked_md(tmp_path, n_lines=60):
    md = tmp_path / "stacked.md"
    lines = [
        "# Documento de prueba",
        "La cultura influye en la cognición humana.",
        "Las normas sociales determinan el comportamiento.",
    ]
    lines += [f"Línea {i}" for i in range(n_lines - 3)]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md


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
    """call_with_retries mockeado: devuelve JSON válido según el system prompt."""
    if system == kp.ANALYSIS_SYSTEM:
        # Llamada unificada (0019): ficha + capítulos + resumen + entidades.
        return (
            json.dumps(
                {
                    "source_file": "stacked.md",
                    "document_id": "stacked_doc_001",
                    "title": "Documento de prueba",
                    "technical_level": "intermediate",
                    "bibtex": "@book{prueba, title={Documento de prueba}}",
                    "thematic_areas_iso25964": [
                        {
                            "preferred_term": "Psicología Cultural",
                            "non_preferred_terms": ["Psicología de la cultura"],
                            "scope_note_disambiguation": "Estudio de la cultura en psicología",
                            "broader_term": "Psicología",
                            "narrower_term": "",
                            "related_terms": ["Antropología"],
                        }
                    ],
                    "library_of_congress": {
                        "lcsh_terms": [{"term": "Ethnopsychology", "uri": ""}],
                        "lcc_classification": {
                            "class_code": "GN",
                            "class_title": "",
                            "uri": "",
                        },
                    },
                    "key_entities": ["cultura", "cognición"],
                    "summary": "El texto fundamenta la relación entre cultura y cognición.",
                    "chapters": [
                        {
                            "chapter_index": 1,
                            "title": "Capítulo Uno",
                            "line_start": 1,
                            "line_end": 60,
                            "main_theme": "Introducción",
                            "summary": "Introducción a la psicología cultural.",
                            "subsections": [],
                        }
                    ],
                }
            ),
            "small",
            False,
        )
    if system == kp.METADATA_SYSTEM:
        return (
            json.dumps(
                {
                    "source_file": "stacked.md",
                    "document_id": "stacked_doc_001",
                    "title": "Documento de prueba",
                    "technical_level": "intermediate",
                    "bibtex": "@book{prueba, title={Documento de prueba}}",
                    "thematic_areas_iso25964": [
                        {
                            "preferred_term": "Psicología Cultural",
                            "non_preferred_terms": ["Psicología de la cultura"],
                            "scope_note_disambiguation": "Estudio de la cultura en psicología",
                            "broader_term": "Psicología",
                            "narrower_term": "",
                            "related_terms": ["Antropología"],
                        }
                    ],
                    "library_of_congress": {
                        "lcsh_terms": [{"term": "Ethnopsychology", "uri": ""}],
                        "lcc_classification": {
                            "class_code": "GN",
                            "class_title": "",
                            "uri": "",
                        },
                    },
                    "key_entities": ["cultura", "cognición"],
                }
            ),
            "small",
            False,
        )
    if system == kp.CHAPTERS_SYSTEM:
        return (
            json.dumps(
                {
                    "source_file": "stacked.md",
                    "document_id": "stacked_doc_001",
                    "total_chapters": 1,
                    "chapters": [
                        {
                            "chapter_index": 1,
                            "title": "Capítulo Uno",
                            "line_start": 1,
                            "line_end": 60,
                            "main_theme": "Introducción",
                            "subsections": [],
                        }
                    ],
                }
            ),
            "small",
            False,
        )
    if system == kp.PROPOSITIONAL_SYSTEM:
        return (
            json.dumps(
                {
                    "source_file": "stacked.md",
                    "document_id": "stacked_doc_001",
                    "chapter_index": 1,
                    "chapter_title": "Capítulo Uno",
                    "core_ideas": [
                        {
                            "core_idea_id": "CI_01",
                            "central_claim": "La cultura moldea la cognición",
                            "supporting_arguments": [
                                {
                                    "argument_id": "ARG_01_A",
                                    "argument_type": "theoretical_deduction",
                                    "argument_statement": "La cultura influye en la cognición",
                                    "propositional_chunks": [
                                        {
                                            "chunk_id": "CH_01",
                                            "proposition": "La cultura influye en la cognición humana.",
                                            "verbatim_span": "La cultura influye en la cognición humana.",
                                            "line_start": 2,
                                            "line_end": 2,
                                            "char_start": 0,
                                            "char_end": 40,
                                            "citations_references": [
                                                "Bourdieu, 1984, p. 52"
                                            ],
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ),
            "small",
            False,
        )
    if system == kp.TOPIC_LABEL_SYSTEM:
        return (
            json.dumps(
                {
                    "source_file": "stacked.md",
                    "document_id": "stacked_doc_001",
                    "sequential_order": 1,
                    "macro_phase_label": "Fundamentación Ontológica",
                    "representative_keywords": ["cultura", "cognición"],
                    "start_chunk_id": "1",
                    "end_chunk_id": "1",
                    "epistemic_summary": "El texto fundamenta la relación entre cultura y cognición.",
                }
            ),
            "small",
            False,
        )
    return ("{}", "small", False)


# ---------------------------------------------------------------------
# Pipeline completo (pasos 0-6)
# ---------------------------------------------------------------------


def test_index_stacked_file_full_pipeline(tmp_path, monkeypatch):
    md = _write_stacked_md(tmp_path)
    monkeypatch.setattr("src.kag.tools.MultibookFinderTool", lambda: _FakeFinder(md))
    monkeypatch.setattr("src.kag.tools.LibraryOfCongressAPITool", _FakeLoC)
    monkeypatch.setattr(kp, "call_with_retries", _fake_call_valid)
    monkeypatch.setattr("src.embeddings.embed_texts", _fake_embed_texts)
    monkeypatch.setattr(kp, "MANIFEST_PATH", tmp_path / "stacked_manifest.json")

    session = _FakeSession()
    results = kp.index_stacked_file(session, str(md), verbose=False)

    assert len(results) == 1
    assert results[0]["status"] == "indexed"
    assert results[0]["document_id"] == "stacked_doc_001"

    # documents (paso 1 + cierre)
    assert len(session.documents) == 1
    doc = session.documents[0]
    assert doc["status"] == "ready"
    assert doc["title"] == "Documento de prueba"
    assert doc["technical_level"] == "intermediate"
    assert "Psicología Cultural" in doc["scope_thematic"]
    assert json.loads(doc["thematic"])[0]["preferred_term"] == "Psicología Cultural"
    # Resumen ejecutivo global (llamada unificada 0019).
    assert doc["summary"] == (
        "El texto fundamenta la relación entre cultura y cognición."
    )

    # document_chapters (paso 2) — micro-resumen del capítulo
    assert len(session.chapters) == 1
    assert session.chapters[0]["title"] == "Capítulo Uno"
    assert session.chapters[0]["line_start"] == 1
    assert session.chapters[0]["line_end"] == 60
    assert session.chapters[0]["summary"] == "Introducción a la psicología cultural."

    # propositional_chunks (paso 3) — verbatim_span localizado en el texto
    assert len(session.chunks) == 1
    chunk = session.chunks[0]
    assert chunk["statement"] == "La cultura influye en la cognición humana."
    assert chunk["text_span"] == "La cultura influye en la cognición humana."
    # Línea 1 = "# Documento de prueba" (21 chars) + newline -> offset 22.
    # El span "La cultura influye en la cognición humana." mide 42 chars.
    assert chunk["char_start"] == 22
    assert chunk["char_end"] == 22 + 42
    assert json.loads(chunk["citations"]) == ["Bourdieu, 1984, p. 52"]
    assert chunk["emb"] == "[0.1,0.2,0.3]"

    # topic_tree_nodes (paso 4)
    assert len(session.topic_nodes) == 1
    node = session.topic_nodes[0]
    assert node["macro_phase_label"] == "Fundamentación Ontológica"
    assert node["sequential_order"] == 1
    assert node["start_chunk_id"] == 1
    assert node["end_chunk_id"] == 1
    assert json.loads(node["keywords"]) == ["cultura", "cognición"]


def test_index_stacked_file_skips_when_hash_unchanged(tmp_path, monkeypatch):
    md = _write_stacked_md(tmp_path)
    monkeypatch.setattr("src.kag.tools.MultibookFinderTool", lambda: _FakeFinder(md))
    monkeypatch.setattr("src.kag.tools.LibraryOfCongressAPITool", _FakeLoC)
    monkeypatch.setattr(kp, "call_with_retries", _fake_call_valid)
    monkeypatch.setattr("src.embeddings.embed_texts", _fake_embed_texts)
    monkeypatch.setattr(kp, "MANIFEST_PATH", tmp_path / "stacked_manifest.json")

    session = _FakeSession()
    first = kp.index_stacked_file(session, str(md), verbose=False)
    assert first[0]["status"] == "indexed"

    # Segunda pasada: mismo hash + status='ready' -> salta sin re-insertar.
    second = kp.index_stacked_file(session, str(md), verbose=False)
    assert second[0]["status"] == "skipped"
    assert len(session.documents) == 1
    assert len(session.chunks) == 1
    assert len(session.topic_nodes) == 1


def test_index_stacked_file_force_reindexes(tmp_path, monkeypatch):
    md = _write_stacked_md(tmp_path)
    monkeypatch.setattr("src.kag.tools.MultibookFinderTool", lambda: _FakeFinder(md))
    monkeypatch.setattr("src.kag.tools.LibraryOfCongressAPITool", _FakeLoC)
    monkeypatch.setattr(kp, "call_with_retries", _fake_call_valid)
    monkeypatch.setattr("src.embeddings.embed_texts", _fake_embed_texts)
    monkeypatch.setattr(kp, "MANIFEST_PATH", tmp_path / "stacked_manifest.json")

    session = _FakeSession()
    kp.index_stacked_file(session, str(md), verbose=False)
    # --force: re-indexa (borra y re-inserta) aunque el hash no cambió.
    results = kp.index_stacked_file(session, str(md), force=True, verbose=False)
    assert results[0]["status"] == "indexed"
    assert len(session.documents) == 1  # borrado + re-insertado
    assert len(session.chunks) == 1


# ---------------------------------------------------------------------
# Degradación: LLM caído -> pipeline sigue con datos crudos
# ---------------------------------------------------------------------


def test_index_stacked_file_degrades_when_llm_fails(tmp_path, monkeypatch):
    md = _write_stacked_md(tmp_path)
    monkeypatch.setattr("src.kag.tools.MultibookFinderTool", lambda: _FakeFinder(md))
    monkeypatch.setattr("src.kag.tools.LibraryOfCongressAPITool", _FakeLoC)
    monkeypatch.setattr("src.embeddings.embed_texts", _fake_embed_texts)
    monkeypatch.setattr(kp, "MANIFEST_PATH", tmp_path / "stacked_manifest.json")

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
        raise RuntimeError("LLM caído")

    monkeypatch.setattr(kp, "call_with_retries", boom)

    session = _FakeSession()
    results = kp.index_stacked_file(session, str(md), verbose=False)

    # El pipeline completa y marca ready con datos crudos.
    assert len(results) == 1
    assert results[0]["status"] == "indexed"
    assert len(session.documents) == 1
    assert session.documents[0]["status"] == "ready"
    assert (
        session.documents[0]["title"] == "Documento de prueba"
    )  # fallback del doc_info

    # Capítulo único degradado con todo el rango.
    assert len(session.chapters) == 1
    assert session.chapters[0]["line_start"] == 1
    assert session.chapters[0]["line_end"] == 60

    # Chunk crudo: statement == text_span == texto del bloque.
    assert len(session.chunks) == 1
    assert session.chunks[0]["statement"] == session.chunks[0]["text_span"]
    assert session.chunks[0]["statement"] != ""

    # Nodo temático degradado: etiqueta "Fase N" + keywords c-TF-IDF.
    assert len(session.topic_nodes) == 1
    assert session.topic_nodes[0]["macro_phase_label"] == "Fase 1"


# ---------------------------------------------------------------------
# Clustering secuencial (función pura)
# ---------------------------------------------------------------------


def test_sequential_clusters_only_merges_adjacent():
    # [0,1] similares, [2,3] similares, 1 y 2 disimilares.
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.1, 0.9],
    ]
    labels = kp._sequential_clusters(embeddings, distance_threshold=0.3)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[1] != labels[2]


def test_sequential_clusters_does_not_merge_non_adjacent():
    # 0 y 2 son muy similares pero NO adyacentes (1 está entre medio y es
    # disimilar): la conectividad bandeada impide la fusión.
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.99, 0.01, 0.0],
    ]
    labels = kp._sequential_clusters(embeddings, distance_threshold=0.2)
    assert labels[0] != labels[2]


def test_sequential_clusters_edge_cases():
    assert kp._sequential_clusters([], 0.5) == []
    assert kp._sequential_clusters([[1.0, 0.0]], 0.5) == [0]
    # Embeddings None (falló el modelo) -> un cluster por chunk.
    labels = kp._sequential_clusters([None, None, None], 0.5)
    assert labels == [0, 1, 2]


# ---------------------------------------------------------------------
# c-TF-IDF (función pura)
# ---------------------------------------------------------------------


def test_ctfidf_keywords_returns_distinctive_terms():
    statements = [
        "La teoría de la cultura estudia las normas sociales",
        "La cultura influye en la cognición y la emoción",
        "Las normas culturales determinan el comportamiento social",
    ]
    keywords = kp._ctfidf_keywords(statements, top_n=5)
    assert len(keywords) <= 5
    assert all(isinstance(k, str) and k for k in keywords)
    assert "cultura" in keywords
    assert "la" not in keywords  # stopword filtrada


def test_ctfidf_keywords_empty_and_single():
    assert kp._ctfidf_keywords([], top_n=3) == []
    # Un solo statement: TfidfTransformer(smooth_idf=True) da IDF=1 para
    # todos los términos (no 0), así que devuelve los términos de contenido
    # con su peso TF — las stopwords se filtran igual.
    keywords = kp._ctfidf_keywords(["La cultura influye en la cognición"], top_n=3)
    assert len(keywords) <= 3
    assert all(isinstance(k, str) and k for k in keywords)
    assert "la" not in keywords  # stopword filtrada
    assert "en" not in keywords  # stopword filtrada


# ---------------------------------------------------------------------
# Imágenes (paso 5)
# ---------------------------------------------------------------------


class _FakeImageExtractor:
    def __init__(self, images):
        self._images = images

    def extract_images_from_document(
        self, source_file, document_id, line_start, line_end
    ):
        return self._images


def _image_entry(tmp_path, exists=True):
    img_path = tmp_path / "img.png"
    if exists:
        img_path.write_bytes(b"dummy")
    return {
        "document_id": "doc_001",
        "anchor_line": 3,
        "markdown_tag": "![fig](img.png)",
        "caption": "fig",
        "file_path": str(img_path),
        "exists_on_disk": exists,
    }


def test_index_images_vlm_success(tmp_path, monkeypatch):
    md = tmp_path / "doc.md"
    md.write_text("# Doc\n\n![fig](img.png)\n", encoding="utf-8")
    monkeypatch.setattr(
        "src.kag.tools.MarkdownImageExtractorTool",
        lambda: _FakeImageExtractor([_image_entry(tmp_path, exists=True)]),
    )

    def fake_vision(session, prompt, image_url, system=None, response_format=None):
        return json.dumps(
            {
                "image_id": "doc_001_img_001",
                "document_id": "doc_001",
                "file_path": str(tmp_path / "img.png"),
                "anchor_line": 3,
                "caption": "fig",
                "image_type": "diagram",
                "dense_visual_description": "Un diagrama de cajas y flechas.",
                "epistemic_contribution": "Formaliza el flujo metodológico.",
                "faq_indexing": ["¿Qué variables relaciona el diagrama?"],
                "associated_entities": ["cultura"],
            }
        )

    monkeypatch.setattr("src.llm.together.complete_vision", fake_vision)

    session = _FakeSession()
    doc_info = {"document_id": "doc_001", "line_start": 1, "line_end": 3}
    count = kp._index_images(session, md, doc_info, md.read_text(encoding="utf-8"), 1)

    assert count == 1
    assert len(session.images) == 1
    img = session.images[0]
    assert img["image_id"] == "doc_001_img_001"
    assert img["dense"] == "Un diagrama de cajas y flechas."
    assert json.loads(img["faq"]) == ["¿Qué variables relaciona el diagrama?"]


def test_index_images_vlm_failure_degrades(tmp_path, monkeypatch):
    md = tmp_path / "doc.md"
    md.write_text("# Doc\n\n![fig](img.png)\n", encoding="utf-8")
    monkeypatch.setattr(
        "src.kag.tools.MarkdownImageExtractorTool",
        lambda: _FakeImageExtractor([_image_entry(tmp_path, exists=True)]),
    )

    def boom(session, prompt, image_url, system=None, response_format=None):
        raise RuntimeError("VLM caído")

    monkeypatch.setattr("src.llm.together.complete_vision", boom)

    session = _FakeSession()
    doc_info = {"document_id": "doc_001", "line_start": 1, "line_end": 3}
    count = kp._index_images(session, md, doc_info, md.read_text(encoding="utf-8"), 1)

    # La imagen queda registrada con descripción vacía (degradación).
    assert count == 1
    assert len(session.images) == 1
    assert session.images[0]["dense"] == ""
    assert session.images[0]["caption"] == "fig"
