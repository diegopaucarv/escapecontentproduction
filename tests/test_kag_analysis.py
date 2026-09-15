"""Tests de la Fase 2 (análisis documental) y Fase 2b (visión de figuras) — sin DB.

Cubre `_index_document_analysis`, `_scope_thematic_from_thematic`,
`_enrich_library_of_congress` e `_index_figures` de src/kag_ingest.py con
sesiones falsas (que capturan el SQL) y monkeypatch de `call_with_retries`,
`_get_prompt_pair`, `complete_vision` y `LibraryOfCongressAPITool`. NO se
importa src.kag.segmentador (carga torch/spacy, ~30s).
"""

import json
from types import SimpleNamespace

from src.kag_ingest import (
    _enrich_library_of_congress,
    _index_document_analysis,
    _index_figures,
    _scope_thematic_from_thematic,
)


class _FakeSession:
    """Sesión falsa que captura el SQL ejecutado y devuelve ids fake.

    `execute` registra (sql, params) en `calls` y devuelve un objeto con
    `.scalar()` (para el RETURNING id de kag_chapters) o `.fetchall()` /
    `.first()` según el SQL. `commit` es no-op.
    """

    def __init__(self):
        self.calls = []
        self._next_id = 0

    def execute(self, stmt, params=None):
        sql = str(stmt)
        self.calls.append((sql, params or {}))
        if "RETURNING id" in sql:
            self._next_id += 1
            return SimpleNamespace(scalar=lambda: f"chap-uuid-{self._next_id}")
        if sql.strip().startswith("SELECT") and "kag_chapters" in sql:
            return SimpleNamespace(fetchall=list)
        if "kag_chunks" in sql and "LIKE" in sql:
            return SimpleNamespace(first=lambda: None)
        return SimpleNamespace(fetchall=list, first=lambda: None, scalar=lambda: 0)

    def commit(self):
        pass

    def rollback(self):
        pass


def _md_text() -> str:
    return (
        "# Libro Uno\n"
        "\n"
        "Introducción del primer libro.\n"
        "\n"
        "## Capítulo 1\n"
        "\n"
        "Contenido del capítulo uno.\n"
        "\n"
        "## Capítulo 2\n"
        "\n"
        "Contenido del capítulo dos.\n"
    )


# ---------------------------------------------------------------------
# _scope_thematic_from_thematic
# ---------------------------------------------------------------------


def test_scope_thematic_preferred_y_no_preferred_con_pipe():
    ficha = {
        "thematic_areas_iso25964": [
            {
                "preferred_term": "Redes neuronales",
                "non_preferred_terms": ["RNA", "Redes artificiales"],
            },
            {
                "preferred_term": "Aprendizaje supervisado",
                "non_preferred_terms": [],
            },
        ]
    }
    scope = _scope_thematic_from_thematic(ficha)
    assert "Redes neuronales" in scope
    assert "RNA" in scope
    assert "Redes artificiales" in scope
    assert "Aprendizaje supervisado" in scope
    assert " | " in scope


def test_scope_thematic_sin_tematicas_vacio():
    assert _scope_thematic_from_thematic({}) == ""
    assert _scope_thematic_from_thematic({"thematic_areas_iso25964": []}) == ""


# ---------------------------------------------------------------------
# _enrich_library_of_congress
# ---------------------------------------------------------------------


def test_enrich_library_of_congress_anade_uris(monkeypatch):
    class _FakeLoCTool:
        def query_subject_heading(self, term):
            return {"preferred_label": term, "uri": f"https://id.loc.gov/{term}"}

        def query_classification_code(self, label):
            return {"lcc_call_number": "QA76.9", "lcc_uri": "https://id.loc.gov/QA76.9"}

    monkeypatch.setattr(
        "src.kag.tools.LibraryOfCongressAPITool", lambda: _FakeLoCTool()
    )
    ficha = {
        "library_of_congress": {
            "lcsh_terms": ["Redes neuronales"],
            "lcc_classification": {"label": "Neural networks", "call_number": ""},
        }
    }
    out = _enrich_library_of_congress(ficha, verbose=False)
    loc = out["library_of_congress"]
    assert loc["lcsh_terms"][0]["term"] == "Redes neuronales"
    assert loc["lcsh_terms"][0]["uri"] == "https://id.loc.gov/Redes neuronales"
    assert loc["lcc_classification"]["lcc_uri"] == "https://id.loc.gov/QA76.9"
    assert loc["lcc_classification"]["call_number"] == "QA76.9"


def test_enrich_library_of_congress_fallo_conserva_original(monkeypatch):
    class _FakeLoCTool:
        def query_subject_heading(self, term):
            raise RuntimeError("API caída")

        def query_classification_code(self, label):
            return None

    monkeypatch.setattr(
        "src.kag.tools.LibraryOfCongressAPITool", lambda: _FakeLoCTool()
    )
    ficha = {
        "library_of_congress": {
            "lcsh_terms": ["Redes neuronales"],
            "lcc_classification": {"label": "Neural networks", "call_number": "QA76"},
        }
    }
    out = _enrich_library_of_congress(ficha, verbose=False)
    loc = out["library_of_congress"]
    # El término se conserva con uri vacía (no rompe).
    assert loc["lcsh_terms"][0]["term"] == "Redes neuronales"
    assert loc["lcsh_terms"][0]["uri"] == ""
    # LCC conserva el call_number original.
    assert loc["lcc_classification"]["call_number"] == "QA76"


# ---------------------------------------------------------------------
# _index_document_analysis — LLM fake
# ---------------------------------------------------------------------


def test_analysis_llm_persiste_ficha_y_capitulos(monkeypatch):
    def fake_call_with_retries(session, **kwargs):
        return (
            json.dumps(
                {
                    "ficha": {
                        "title": "Libro Uno",
                        "technical_level": "intermediate",
                        "thematic_areas_iso25964": [
                            {
                                "preferred_term": "Redes neuronales",
                                "non_preferred_terms": ["RNA"],
                                "scope_note_disambiguation": "Modelos conexionistas",
                                "broader_term": "Inteligencia artificial",
                                "narrower_terms": ["Perceptrón"],
                                "related_terms": ["Backpropagation"],
                            }
                        ],
                        "library_of_congress": {
                            "lcsh_terms": ["Neural networks"],
                            "lcc_classification": {"label": "QA76.9"},
                        },
                        "bibtex": "@book{uno2026}",
                        "key_entities": [{"name": "Perceptrón", "type": "concepto"}],
                    },
                    "chapters": [
                        {
                            "chapter_id": "cap1",
                            "title": "Capítulo 1",
                            "line_start": 1,
                            "line_end": 5,
                            "has_images": True,
                        },
                        {
                            "chapter_id": "cap2",
                            "title": "Capítulo 2",
                            "line_start": 6,
                            "line_end": 10,
                            "has_images": False,
                        },
                    ],
                }
            ),
            "fake-large",
            False,
        )

    monkeypatch.setattr("src.kag_ingest.call_with_retries", fake_call_with_retries)
    session = _FakeSession()
    chapter_map = _index_document_analysis(
        session, "doc-1", "libro_uno.md", _md_text(), verbose=False
    )

    # Mapeo chapter_id_str -> uuid.
    assert chapter_map == {"cap1": "chap-uuid-1", "cap2": "chap-uuid-2"}

    # UPDATE de ficha_jsonb + sections_json.
    updates = [c for c in session.calls if "UPDATE kag_documents" in c[0]]
    assert len(updates) == 1
    ficha = json.loads(updates[0][1]["ficha"])
    assert ficha["title"] == "Libro Uno"
    assert ficha["thematic_areas_iso25964"][0]["preferred_term"] == "Redes neuronales"
    # scope_thematic derivado dentro de la ficha.
    assert "Redes neuronales" in ficha["scope_thematic"]
    sections = json.loads(updates[0][1]["sections"])
    assert len(sections) == 2
    assert sections[0]["chapter_id"] == "cap1"
    assert sections[0]["has_images"] is True
    # SIN summary de capítulo.
    assert "summary" not in sections[0]

    # INSERTs en kag_chapters.
    inserts = [c for c in session.calls if "INSERT INTO kag_chapters" in c[0]]
    assert len(inserts) == 2
    assert inserts[0][1]["chapter_id"] == "cap1"
    assert inserts[0][1]["has_images"] is True
    assert inserts[1][1]["chapter_id"] == "cap2"


def test_analysis_llm_falla_degrada_ficha_defaults_y_capitulo_unico(monkeypatch):
    def fake_call_with_retries(session, **kwargs):
        raise RuntimeError("LLM caído")

    monkeypatch.setattr("src.kag_ingest.call_with_retries", fake_call_with_retries)
    session = _FakeSession()
    chapter_map = _index_document_analysis(
        session, "doc-1", "libro_uno.md", _md_text(), verbose=False
    )

    # Un capítulo único con todo el rango del documento.
    assert len(chapter_map) == 1
    assert "" in chapter_map

    updates = [c for c in session.calls if "UPDATE kag_documents" in c[0]]
    assert len(updates) == 1
    ficha = json.loads(updates[0][1]["ficha"])
    assert ficha["title"] == "Libro Uno"
    assert ficha["technical_level"] == "intermediate"
    assert ficha["thematic_areas_iso25964"] == []
    assert ficha["library_of_congress"] == {}
    assert ficha["bibtex"] == ""
    assert ficha["key_entities"] == []

    sections = json.loads(updates[0][1]["sections"])
    assert len(sections) == 1
    assert sections[0]["line_start"] == 1
    assert sections[0]["line_end"] == 11  # total de líneas del slice
    assert sections[0]["has_images"] is False

    # El INSERT de capítulos ocurre SIEMPRE (incluso en degradación).
    inserts = [c for c in session.calls if "INSERT INTO kag_chapters" in c[0]]
    assert len(inserts) == 1


def test_analysis_llm_json_invalido_degrada(monkeypatch):
    def fake_call_with_retries(session, **kwargs):
        return ("no es json", "fake-large", False)

    monkeypatch.setattr("src.kag_ingest.call_with_retries", fake_call_with_retries)
    session = _FakeSession()
    chapter_map = _index_document_analysis(
        session, "doc-1", "libro_uno.md", _md_text(), verbose=False
    )
    assert len(chapter_map) == 1
    updates = [c for c in session.calls if "UPDATE kag_documents" in c[0]]
    assert len(updates) == 1


def test_analysis_index_malformado_usa_chapters_plano(monkeypatch):
    """FIX M1: index no vacío pero con divisiones malformadas (sin chapters
    list) → backward compat con el schema plano v1.0 (data["chapters"])."""

    def fake_call_with_retries(session, **kwargs):
        return (
            json.dumps(
                {
                    "ficha": {"title": "Libro Uno"},
                    "index": [
                        {"division": "Parte I", "chapters": "no-es-lista"},
                        {"division": "Parte II"},
                    ],
                    "chapters": [
                        {
                            "chapter_id": "cap1",
                            "title": "Capítulo 1",
                            "line_start": 1,
                            "line_end": 5,
                            "has_images": False,
                        }
                    ],
                }
            ),
            "fake-large",
            False,
        )

    monkeypatch.setattr("src.kag_ingest.call_with_retries", fake_call_with_retries)
    session = _FakeSession()
    chapter_map = _index_document_analysis(
        session, "doc-1", "libro_uno.md", _md_text(), verbose=False
    )
    assert chapter_map == {"cap1": "chap-uuid-1"}
    updates = [c for c in session.calls if "UPDATE kag_documents" in c[0]]
    sections = json.loads(updates[0][1]["sections"])
    assert len(sections) == 1
    assert sections[0]["chapter_id"] == "cap1"


def test_analysis_index_jerarquico_aplana_con_division(monkeypatch):
    """FIX: index jerárquico [{division, chapters: [...]}] se aplana y cada
    capítulo hereda su division en sections_json."""

    def fake_call_with_retries(session, **kwargs):
        return (
            json.dumps(
                {
                    "ficha": {"title": "Libro Uno"},
                    "index": [
                        {
                            "division": "PART I",
                            "chapters": [
                                {
                                    "chapter_id": "cap1",
                                    "title": "Capítulo 1",
                                    "line_start": 1,
                                    "line_end": 5,
                                    "has_images": False,
                                },
                                {
                                    "chapter_id": "cap2",
                                    "title": "Capítulo 2",
                                    "line_start": 6,
                                    "line_end": 10,
                                    "has_images": True,
                                },
                            ],
                        },
                        {
                            "division": "PART II",
                            "chapters": [
                                {
                                    "chapter_id": "cap3",
                                    "title": "Capítulo 3",
                                    "line_start": 11,
                                    "line_end": 15,
                                    "has_images": False,
                                }
                            ],
                        },
                    ],
                }
            ),
            "fake-large",
            False,
        )

    monkeypatch.setattr("src.kag_ingest.call_with_retries", fake_call_with_retries)
    session = _FakeSession()
    chapter_map = _index_document_analysis(
        session, "doc-1", "libro_uno.md", _md_text(), verbose=False
    )

    # Los 3 capítulos se aplanan en orden y cada uno hereda su division.
    assert chapter_map == {
        "cap1": "chap-uuid-1",
        "cap2": "chap-uuid-2",
        "cap3": "chap-uuid-3",
    }
    updates = [c for c in session.calls if "UPDATE kag_documents" in c[0]]
    sections = json.loads(updates[0][1]["sections"])
    assert len(sections) == 3
    assert sections[0]["chapter_id"] == "cap1"
    assert sections[0]["division"] == "PART I"
    assert sections[1]["chapter_id"] == "cap2"
    assert sections[1]["division"] == "PART I"
    assert sections[2]["chapter_id"] == "cap3"
    assert sections[2]["division"] == "PART II"


def test_analysis_index_vacio_usa_chapters_plano(monkeypatch):
    """FIX: index: [] (lista vacía) cae al fallback chapters plano v1.0."""

    def fake_call_with_retries(session, **kwargs):
        return (
            json.dumps(
                {
                    "ficha": {"title": "Libro Uno"},
                    "index": [],
                    "chapters": [
                        {
                            "chapter_id": "cap1",
                            "title": "Capítulo 1",
                            "line_start": 1,
                            "line_end": 5,
                            "has_images": False,
                        }
                    ],
                }
            ),
            "fake-large",
            False,
        )

    monkeypatch.setattr("src.kag_ingest.call_with_retries", fake_call_with_retries)
    session = _FakeSession()
    chapter_map = _index_document_analysis(
        session, "doc-1", "libro_uno.md", _md_text(), verbose=False
    )
    assert chapter_map == {"cap1": "chap-uuid-1"}
    updates = [c for c in session.calls if "UPDATE kag_documents" in c[0]]
    sections = json.loads(updates[0][1]["sections"])
    assert len(sections) == 1
    assert sections[0]["chapter_id"] == "cap1"
    assert sections[0]["division"] == ""


def test_analysis_has_images_string_normalizado(monkeypatch):
    """FIX M3: has_images "false"/"true" (str) no debe evaluar a True."""

    def fake_call_with_retries(session, **kwargs):
        return (
            json.dumps(
                {
                    "ficha": {"title": "Libro Uno"},
                    "chapters": [
                        {
                            "chapter_id": "cap1",
                            "title": "Capítulo 1",
                            "line_start": 1,
                            "line_end": 5,
                            "has_images": "false",
                        },
                        {
                            "chapter_id": "cap2",
                            "title": "Capítulo 2",
                            "line_start": 6,
                            "line_end": 10,
                            "has_images": "true",
                        },
                    ],
                }
            ),
            "fake-large",
            False,
        )

    monkeypatch.setattr("src.kag_ingest.call_with_retries", fake_call_with_retries)
    session = _FakeSession()
    _index_document_analysis(
        session, "doc-1", "libro_uno.md", _md_text(), verbose=False
    )
    updates = [c for c in session.calls if "UPDATE kag_documents" in c[0]]
    sections = json.loads(updates[0][1]["sections"])
    assert sections[0]["has_images"] is False
    assert sections[1]["has_images"] is True
    inserts = [c for c in session.calls if "INSERT INTO kag_chapters" in c[0]]
    assert inserts[0][1]["has_images"] is False
    assert inserts[1][1]["has_images"] is True


# ---------------------------------------------------------------------
# _index_figures — visión condicional
# ---------------------------------------------------------------------


def _fake_chapters_session(has_images_rows):
    """Sesión cuyo SELECT de kag_chapters devuelve filas con has_images."""

    class _S:
        def __init__(self):
            self.calls = []

        def execute(self, stmt, params=None):
            sql = str(stmt)
            self.calls.append((sql, params or {}))
            if "FROM kag_chapters" in sql:
                return SimpleNamespace(
                    fetchall=lambda: [
                        SimpleNamespace(
                            id=f"ch-{i}",
                            chapter_id=f"cap{i}",
                            line_start=1,
                            line_end=10,
                            has_images=has_images_rows[i],
                        )
                        for i in range(len(has_images_rows))
                    ]
                )
            if "kag_chunks" in sql and "LIKE" in sql:
                return SimpleNamespace(first=lambda: None)
            return SimpleNamespace(fetchall=list, first=lambda: None)

        def commit(self):
            pass

    return _S()


def test_index_figures_sin_capitulos_con_imagenes_skip(tmp_path, monkeypatch):
    md = tmp_path / "doc.md"
    md.write_text("# Título\n\n![fig](img.png)\n", encoding="utf-8")
    session = _fake_chapters_session([False, False])

    def fake_vision(session, figure, slice_text, doc_line_start, verbose):
        raise AssertionError("la visión no debe llamarse sin capítulos con imágenes")

    monkeypatch.setattr("src.kag_ingest._describe_figure_vision", fake_vision)
    count = _index_figures(
        session, "doc-1", md, "# Título\n\n![fig](img.png)\n", 1, verbose=False
    )
    assert count == 0
    # No se insertó ninguna figura.
    assert not [c for c in session.calls if "INSERT INTO kag_figures" in c[0]]


def test_index_figures_con_has_images_y_vlm_fake_inserta_faq(tmp_path, monkeypatch):
    md = tmp_path / "doc.md"
    img = tmp_path / "fig1.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
    md.write_text(
        "# Título\n\nTexto que referencia la figura.\n\n![Figura 1](fig1.png)\n\n"
        "Más texto.\n",
        encoding="utf-8",
    )
    session = _fake_chapters_session([True])

    def fake_vision(session, figure, slice_text, doc_line_start, verbose):
        return {
            "image_type": "chart_or_plot",
            "dense_visual_description": "Gráfico de barras con dos ejes.",
            "epistemic_contribution": "Muestra la tendencia temporal.",
            "faq_indexing": ["¿Cuál es la tendencia?", "¿Qué valores muestra?"],
            "associated_entities": ["tendencia", "barras"],
        }

    monkeypatch.setattr("src.kag_ingest._describe_figure_vision", fake_vision)
    count = _index_figures(
        session, "doc-1", md, md.read_text(encoding="utf-8"), 1, verbose=False
    )
    assert count == 1

    inserts = [c for c in session.calls if "INSERT INTO kag_figures" in c[0]]
    assert len(inserts) == 1
    params = inserts[0][1]
    assert params["image_type"] == "chart_or_plot"
    assert params["dense_visual_description"] == "Gráfico de barras con dos ejes."
    assert params["epistemic_contribution"] == "Muestra la tendencia temporal."
    faq = json.loads(params["faq_indexing"])
    assert faq == ["¿Cuál es la tendencia?", "¿Qué valores muestra?"]
    entities = json.loads(params["associated_entities"])
    assert entities == ["tendencia", "barras"]
    assert params["anchor_line"] > 0


def test_index_figures_vlm_falla_descripcion_vacia(tmp_path, monkeypatch):
    md = tmp_path / "doc.md"
    img = tmp_path / "fig1.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
    md.write_text(
        "# Título\n\n![Figura 1](fig1.png)\n\n",
        encoding="utf-8",
    )
    session = _fake_chapters_session([True])

    def fake_vision(session, figure, slice_text, doc_line_start, verbose):
        return {}

    monkeypatch.setattr("src.kag_ingest._describe_figure_vision", fake_vision)
    count = _index_figures(
        session, "doc-1", md, md.read_text(encoding="utf-8"), 1, verbose=False
    )
    assert count == 1

    inserts = [c for c in session.calls if "INSERT INTO kag_figures" in c[0]]
    assert len(inserts) == 1
    params = inserts[0][1]
    assert params["description"] == ""
    assert params["dense_visual_description"] == ""
    assert json.loads(params["faq_indexing"]) == []
    assert json.loads(params["associated_entities"]) == []
