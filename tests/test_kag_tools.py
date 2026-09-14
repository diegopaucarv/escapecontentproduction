"""Tests de las herramientas de ingesta documental (src/kag/tools.py).

Sin red y sin DB: httpx.get se mockea con monkeypatch y los .md se crean
en tmp_path. Patrón de tests/test_entities.py.
"""

import httpx
import pytest

from src.kag.tools import (
    LibraryOfCongressAPITool,
    MarkdownImageExtractorTool,
    MultibookFinderTool,
)

# ---------------------------------------------------------------------
# MultibookFinderTool
# ---------------------------------------------------------------------


def test_multibook_finder_detects_two_stacked_books(tmp_path):
    # Dos libros apilados: `---` + ISBN marcan la frontera entre ambos.
    # Cada libro tiene >= 50 líneas (el umbral de fragmentos residuales).
    md = tmp_path / "libros_apilados.md"
    lines = ["# Libro Uno"]
    lines += [f"Contenido del libro uno, párrafo {i}." for i in range(54)]
    lines += ["---", "ISBN 978-3-16-148410-0", "# Libro Dos"]
    lines += [f"Contenido del libro dos, párrafo {i}." for i in range(54)]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    docs = MultibookFinderTool().execute(str(md))

    assert len(docs) == 2
    assert docs[0]["document_id"] == "libros_apilados_doc_001"
    assert docs[1]["document_id"] == "libros_apilados_doc_002"
    assert docs[0]["title"] == "Libro Uno"
    assert docs[1]["title"] == "Libro Dos"
    # Libro 1: líneas 1-55 (incluye el separador `---`); libro 2: 56-112.
    assert docs[0]["line_start"] == 1
    assert docs[0]["line_end"] == 55
    assert docs[0]["total_lines"] == 55
    assert docs[1]["line_start"] == 56
    assert docs[1]["line_end"] == 112
    assert docs[1]["total_lines"] == 57


def test_multibook_finder_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        MultibookFinderTool().execute(str(tmp_path / "no_existe.md"))


def test_multibook_finder_empty_file_returns_empty(tmp_path):
    md = tmp_path / "vacio.md"
    md.write_text("", encoding="utf-8")
    assert MultibookFinderTool().execute(str(md)) == []


# ---------------------------------------------------------------------
# LibraryOfCongressAPITool (httpx mockeado)
# ---------------------------------------------------------------------


class _FakeResponse:
    """Respuesta mínima de httpx.Response: status_code + json()."""

    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def test_lc_query_subject_heading_parses_suggest_json(monkeypatch):
    # La API de id.loc.gov devuelve [query, [labels], [uris], [ids]].
    captured = {}

    def fake_get(url, params=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        captured["timeout"] = timeout
        return _FakeResponse(payload=["q", ["label"], ["uri"], ["id"]])

    monkeypatch.setattr(httpx, "get", fake_get)
    tool = LibraryOfCongressAPITool(timeout=7.5)
    result = tool.query_subject_heading("machine learning")

    # data[1][0] -> preferred_label, data[3][0] -> uri (posición ids).
    assert result == {"preferred_label": "label", "uri": "id"}
    assert captured["url"] == LibraryOfCongressAPITool.SUGGEST_URL
    assert captured["params"] == {"q": "machine learning", "count": 1}
    assert captured["timeout"] == 7.5


def test_lc_query_classification_parses_suggest_json(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        return _FakeResponse(payload=["q", ["QA76.73.P98"], ["uri"], ["id"]])

    monkeypatch.setattr(httpx, "get", fake_get)
    tool = LibraryOfCongressAPITool()
    result = tool.query_classification_code("Python")

    # data[1][0] -> lcc_call_number, data[3][0] -> lcc_uri (posición ids).
    assert result == {"lcc_call_number": "QA76.73.P98", "lcc_uri": "id"}


def test_lc_query_subject_heading_http_error_returns_none(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        raise httpx.HTTPError("boom")

    monkeypatch.setattr(httpx, "get", fake_get)
    tool = LibraryOfCongressAPITool()
    assert tool.query_subject_heading("term") is None


def test_lc_query_classification_timeout_returns_none(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        raise httpx.TimeoutException("timeout")

    monkeypatch.setattr(httpx, "get", fake_get)
    tool = LibraryOfCongressAPITool()
    assert tool.query_classification_code("Python") is None


def test_lc_query_subject_heading_non_200_returns_none(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        return _FakeResponse(status_code=500, payload=None)

    monkeypatch.setattr(httpx, "get", fake_get)
    tool = LibraryOfCongressAPITool()
    assert tool.query_subject_heading("term") is None


def test_lc_query_subject_heading_malformed_json_returns_none(monkeypatch):
    # JSON sin candidatos (listas vacías) → None, no revienta.
    def fake_get(url, params=None, timeout=None):
        return _FakeResponse(payload=["q", [], [], []])

    monkeypatch.setattr(httpx, "get", fake_get)
    tool = LibraryOfCongressAPITool()
    assert tool.query_subject_heading("term") is None


# ---------------------------------------------------------------------
# MarkdownImageExtractorTool
# ---------------------------------------------------------------------


def test_markdown_image_extractor_finds_md_and_html_images(tmp_path):
    # Imágenes dummy en disco para que exists_on_disk sea True.
    (tmp_path / "img.png").write_bytes(b"dummy")
    (tmp_path / "x.png").write_bytes(b"dummy")

    md = tmp_path / "doc.md"
    md.write_text(
        "# Doc\n"
        "![alt](img.png)\n"
        '<img src="x.png" alt="y">\n'
        "![sin archivo](missing.png)\n",
        encoding="utf-8",
    )

    images = MarkdownImageExtractorTool().extract_images_from_document(
        str(md), "doc_001", 1, 4
    )

    assert len(images) == 3

    md_img = images[0]
    assert md_img["document_id"] == "doc_001"
    assert md_img["anchor_line"] == 2
    assert md_img["caption"] == "alt"
    assert md_img["markdown_tag"] == "![alt](img.png)"
    assert md_img["file_path"] == str((tmp_path / "img.png").resolve())
    assert md_img["exists_on_disk"] is True

    html_img = images[1]
    assert html_img["anchor_line"] == 3
    assert html_img["caption"] == "y"
    assert html_img["markdown_tag"] == '<img src="x.png" alt="y">'
    assert html_img["file_path"] == str((tmp_path / "x.png").resolve())
    assert html_img["exists_on_disk"] is True

    missing = images[2]
    assert missing["anchor_line"] == 4
    assert missing["caption"] == "sin archivo"
    assert missing["exists_on_disk"] is False


def test_markdown_image_extractor_respects_line_range(tmp_path):
    md = tmp_path / "doc.md"
    md.write_text(
        "![fuera](a.png)\n![dentro](b.png)\n",
        encoding="utf-8",
    )

    images = MarkdownImageExtractorTool().extract_images_from_document(
        str(md), "doc_001", 2, 2
    )

    assert len(images) == 1
    assert images[0]["anchor_line"] == 2
    assert images[0]["caption"] == "dentro"
