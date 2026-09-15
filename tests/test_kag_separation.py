"""Tests de la Fase 1 (separación de documentos apilados) — sin DB, sin LLM real.

Cubre `_index_document_separation` de src/kag_ingest.py con una sesión falsa
(cuyo execute lanza: load_settings y _get_prompt_pair degradan a constantes)
y `call_with_retries` monkeypatcheado. NO se importa src.kag.segmentador
(carga torch/spacy, ~30s).
"""

import json

from src.kag_ingest import _build_skeleton, _index_document_separation


class _FakeSession:
    """Sesión falsa: cualquier execute lanza (sin DB en tests).

    load_settings y _get_prompt_pair degradan con gracia (try/except) y
    devuelven los fallbacks — comportamiento EXACTO de producción sin
    artefacto compilado.
    """

    def execute(self, *args, **kwargs):
        raise RuntimeError("sin DB en tests")


def _md_text() -> str:
    """Archivo pequeño (16 líneas): MultibookFinderTool ignora fragmentos <50."""
    return (
        "# Libro Uno\n"
        "\n"
        "Introducción del primer libro.\n"
        "\n"
        "## Capítulo 1\n"
        "\n"
        "Contenido del capítulo uno.\n"
        "\n"
        "---\n"
        "\n"
        "# Libro Dos\n"
        "\n"
        "Introducción del segundo libro.\n"
        "\n"
        "## Capítulo 1\n"
        "\n"
        "Más contenido del capítulo dos.\n"
    )


def _big_md() -> str:
    """Archivo con 2 libros de ~61 líneas separados por '---' (≥50 líneas)."""
    lines = ["# Libro Uno", ""]
    lines += [f"Contenido del libro uno, línea {i}." for i in range(58)]
    lines += ["", "---", "", "# Libro Dos", ""]
    lines += [f"Contenido del libro dos, línea {i}." for i in range(58)]
    return "\n".join(lines)


# ---------------------------------------------------------------------
# _build_skeleton
# ---------------------------------------------------------------------


def test_build_skeleton_incluye_headers_con_linea_real():
    md = "# Título\n\n## Sección\n\n### Sub\n\nTexto."
    sk = _build_skeleton(md)
    assert md[:2000] in sk
    assert "L1: # Título" in sk
    assert "L3: ## Sección" in sk
    assert "L5: ### Sub" in sk


def test_build_skeleton_sin_headers_solo_cabeza():
    md = "Solo texto plano, sin encabezados."
    sk = _build_skeleton(md)
    assert md in sk
    assert "Encabezados" not in sk


# ---------------------------------------------------------------------
# _index_document_separation — LLM fake
# ---------------------------------------------------------------------


def test_separation_llm_devuelve_n_docs(tmp_path, monkeypatch):
    md = tmp_path / "stack.md"
    md.write_text(_md_text(), encoding="utf-8")

    def fake_call_with_retries(session, **kwargs):
        return (
            json.dumps(
                {
                    "documents": [
                        {
                            "document_id": "doc_1",
                            "title": "Libro Uno",
                            "line_start": 1,
                            "line_end": 8,
                            "language": "es",
                        },
                        {
                            "document_id": "doc_2",
                            "title": "Libro Dos",
                            "line_start": 10,
                            "line_end": 16,
                            "language": "es",
                        },
                    ]
                }
            ),
            "fake-large",
            False,
        )

    monkeypatch.setattr("src.kag_ingest.call_with_retries", fake_call_with_retries)
    docs = _index_document_separation(_FakeSession(), md, _md_text(), verbose=False)

    assert len(docs) == 2
    assert docs[0]["document_id"] == "doc_1"
    assert docs[0]["title"] == "Libro Uno"
    assert docs[0]["line_start"] == 1
    assert docs[0]["line_end"] == 8
    assert docs[1]["document_id"] == "doc_2"
    assert docs[1]["line_start"] == 10
    assert docs[1]["line_end"] == 16


def test_separation_llm_sin_document_id_usa_vacio(tmp_path, monkeypatch):
    md = tmp_path / "single.md"
    md.write_text(_md_text(), encoding="utf-8")

    def fake_call_with_retries(session, **kwargs):
        return (
            json.dumps(
                {
                    "documents": [
                        {
                            "title": "Libro Uno",
                            "line_start": 1,
                            "line_end": 16,
                            "language": "es",
                        }
                    ]
                }
            ),
            "fake-large",
            False,
        )

    monkeypatch.setattr("src.kag_ingest.call_with_retries", fake_call_with_retries)
    docs = _index_document_separation(_FakeSession(), md, _md_text(), verbose=False)

    assert len(docs) == 1
    assert docs[0]["document_id"] == ""
    assert docs[0]["line_start"] == 1
    assert docs[0]["line_end"] == 16


# ---------------------------------------------------------------------
# _index_document_separation — sanitización de rangos
# ---------------------------------------------------------------------


def test_separation_sanitiza_rangos(tmp_path, monkeypatch):
    md = tmp_path / "stack.md"
    md.write_text(_md_text(), encoding="utf-8")

    def fake_call_with_retries(session, **kwargs):
        return (
            json.dumps(
                {
                    "documents": [
                        # line_start > line_end → swap
                        {
                            "document_id": "a",
                            "title": "A",
                            "line_start": 10,
                            "line_end": 2,
                            "language": "es",
                        },
                        # fuera de rango → clamp al archivo (17 líneas)
                        {
                            "document_id": "b",
                            "title": "B",
                            "line_start": -5,
                            "line_end": 999,
                            "language": "es",
                        },
                    ]
                }
            ),
            "fake-large",
            False,
        )

    monkeypatch.setattr("src.kag_ingest.call_with_retries", fake_call_with_retries)
    docs = _index_document_separation(_FakeSession(), md, _md_text(), verbose=False)

    assert docs[0]["line_start"] == 2
    assert docs[0]["line_end"] == 10
    assert docs[1]["line_start"] == 1
    assert docs[1]["line_end"] == 17


def test_separation_document_ids_duplicados_se_deduplican(tmp_path, monkeypatch):
    md = tmp_path / "stack.md"
    md.write_text(_md_text(), encoding="utf-8")

    def fake_call_with_retries(session, **kwargs):
        return (
            json.dumps(
                {
                    "documents": [
                        {
                            "document_id": "",
                            "title": "A",
                            "line_start": 1,
                            "line_end": 8,
                            "language": "es",
                        },
                        {
                            "document_id": "",
                            "title": "B",
                            "line_start": 10,
                            "line_end": 16,
                            "language": "es",
                        },
                    ]
                }
            ),
            "fake-large",
            False,
        )

    monkeypatch.setattr("src.kag_ingest.call_with_retries", fake_call_with_retries)
    docs = _index_document_separation(_FakeSession(), md, _md_text(), verbose=False)

    assert len(docs) == 2
    assert docs[0]["document_id"] == ""
    assert docs[1]["document_id"] == "doc_2"


def test_separation_llm_omite_doc_determinista_se_conserva(tmp_path, monkeypatch):
    """FIX M2: si el LLM omite un doc determinista, el hint se conserva SIEMPRE.

    El LLM devuelve solo el Libro Uno (1-61); el Libro Dos (62-123) detectado
    por MultibookFinderTool no debe perderse. El doc del LLM que colisiona con
    un hint se descarta (no se duplica).
    """
    md = tmp_path / "stack.md"
    md.write_text(_big_md(), encoding="utf-8")

    def fake_call_with_retries(session, **kwargs):
        return (
            json.dumps(
                {
                    "documents": [
                        {
                            "document_id": "llm_1",
                            "title": "Libro Uno",
                            "line_start": 1,
                            "line_end": 61,
                            "language": "es",
                        }
                    ]
                }
            ),
            "fake-large",
            False,
        )

    monkeypatch.setattr("src.kag_ingest.call_with_retries", fake_call_with_retries)
    docs = _index_document_separation(_FakeSession(), md, _big_md(), verbose=False)

    # 2 hints (siempre incluidos) + 0 del LLM (colisiona con el hint 1-61).
    assert len(docs) == 2
    assert docs[0]["line_start"] == 1
    assert docs[0]["line_end"] == 61
    assert docs[1]["line_start"] == 62
    assert docs[1]["line_end"] == 123


# ---------------------------------------------------------------------
# _index_document_separation — degradación
# ---------------------------------------------------------------------


def test_separation_degradacion_llm_falla_usa_multibook(tmp_path, monkeypatch):
    md = tmp_path / "stack.md"
    md.write_text(_big_md(), encoding="utf-8")

    def fake_call_with_retries(session, **kwargs):
        raise RuntimeError("LLM caído")

    monkeypatch.setattr("src.kag_ingest.call_with_retries", fake_call_with_retries)
    docs = _index_document_separation(_FakeSession(), md, _big_md(), verbose=False)

    # MultibookFinderTool detecta los 2 libros físicos (separador '---').
    assert len(docs) == 2
    assert docs[0]["line_start"] == 1
    assert docs[0]["line_end"] == 61
    assert docs[1]["line_start"] == 62
    assert docs[1]["line_end"] == 123


def test_separation_degradacion_total_un_documento(tmp_path, monkeypatch):
    md = tmp_path / "single.md"
    md.write_text(_md_text(), encoding="utf-8")

    def fake_call_with_retries(session, **kwargs):
        raise RuntimeError("LLM caído")

    monkeypatch.setattr("src.kag_ingest.call_with_retries", fake_call_with_retries)
    docs = _index_document_separation(_FakeSession(), md, _md_text(), verbose=False)

    # Sin LLM y sin pistas físicas (fragmentos <50 líneas) → 1 doc completo.
    assert len(docs) == 1
    assert docs[0]["document_id"] == ""
    assert docs[0]["line_start"] == 1
    assert docs[0]["line_end"] == 17
    assert docs[0]["title"] == "Libro Uno"
    assert docs[0]["language"] == "es"


def test_separation_llm_json_invalido_degrada(tmp_path, monkeypatch):
    md = tmp_path / "single.md"
    md.write_text(_md_text(), encoding="utf-8")

    def fake_call_with_retries(session, **kwargs):
        return ("esto no es json", "fake-large", False)

    monkeypatch.setattr("src.kag_ingest.call_with_retries", fake_call_with_retries)
    docs = _index_document_separation(_FakeSession(), md, _md_text(), verbose=False)

    assert len(docs) == 1
    assert docs[0]["document_id"] == ""
    assert docs[0]["line_start"] == 1
    assert docs[0]["line_end"] == 17
