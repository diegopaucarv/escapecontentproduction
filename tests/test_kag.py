"""Tests de lógica pura del sistema KAG — sin DB, sin torch/spacy/transformers.

NO se importa src.kag.segmentador a nivel de módulo (carga torch/spacy/
sentence-transformers, ~30s). Se testea la lógica pura de src/kag_ingest.py
y src/kag_query.py con sesiones falsas y monkeypatch (patrón de
tests/test_embeddings.py).
"""

from types import SimpleNamespace

import httpx
import pytest

from src.kag_ingest import (
    chunk_markdown,
    complete_local,
    detect_language,
    estimate_tokens,
    normalize_entity_name,
)
from src.kag_query import (
    assemble_context,
    build_adjacency,
    classify_query,
    personalized_pagerank,
)

# ---------------------------------------------------------------------
# estimate_tokens / normalize_entity_name
# ---------------------------------------------------------------------


def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("hola mundo") == 2  # 10 // 4
    assert estimate_tokens("a" * 100) == 25


def test_normalize_entity_name():
    assert normalize_entity_name("  Propagación   Hacia Atrás ") == (
        "propagación hacia atrás"
    )
    assert normalize_entity_name("Red Neuronal") == "red neuronal"
    assert normalize_entity_name("") == ""


# ---------------------------------------------------------------------
# detect_language
# ---------------------------------------------------------------------


def test_detect_language_spanish():
    assert detect_language("El gato y el perro están en la casa de la abuela") == "es"


def test_detect_language_english():
    assert (
        detect_language("The cat and the dog are in the house of the grandmother")
        == "en"
    )


def test_detect_language_portuguese():
    assert detect_language("O gato e o cachorro estão na casa da avó") == "pt"


def test_detect_language_german():
    assert detect_language("Der Hund und die Katze sind im Haus der Großmutter") == "de"


def test_detect_language_french():
    assert (
        detect_language("Le chat et le chien sont dans la maison de la grand-mère")
        == "fr"
    )


def test_detect_language_default_es():
    assert detect_language("xyz qwerty 12345") == "es"
    assert detect_language("") == "es"


# ---------------------------------------------------------------------
# chunk_markdown (segmenter fake)
# ---------------------------------------------------------------------


class _FakeSegmenter:
    """Segmenter duck-typed: devuelve listas fijas y registra las llamadas."""

    def __init__(self, segments_per_call=2):
        self.segments_per_call = segments_per_call
        self.calls = []

    def segment_text(self, text, max_tokens=800):
        self.calls.append((text, max_tokens))
        return [f"seg{i}:{text[:20]}" for i in range(self.segments_per_call)]


def test_chunk_markdown_short_splits_by_headers():
    md = (
        "# Introducción\n\nTexto intro.\n\n"
        "## Métodos\n\nDetalle métodos.\n\n"
        "### Sub\n\nDetalle sub."
    )
    seg = _FakeSegmenter()
    chunks = chunk_markdown(md, "short", seg, max_tokens=800)

    assert len(chunks) == 6  # 3 secciones × 2 segmentos
    assert chunks[0]["section_path"] == "# Introducción"
    assert chunks[1]["section_path"] == "# Introducción"
    assert chunks[2]["section_path"] == "# Introducción > ## Métodos"
    assert chunks[4]["section_path"] == "# Introducción > ## Métodos > ### Sub"
    assert chunks[0]["token_estimate"] == estimate_tokens(chunks[0]["content"])
    assert "seg0:" in chunks[0]["content"]
    # max_tokens se propaga al segmenter
    assert all(call[1] == 800 for call in seg.calls)


def test_chunk_markdown_long_presegments_h1_h2():
    md = (
        "# Cap 1\n\nContenido cap 1.\n\n"
        "### Sub detalle\n\nmás.\n\n"
        "## Cap 2\n\nContenido cap 2."
    )
    seg = _FakeSegmenter()
    chunks = chunk_markdown(md, "long", seg, max_tokens=800)

    # long: solo H1/H2 parten; ### queda dentro del contenido de la sección.
    assert len(chunks) == 4  # 2 secciones × 2 segmentos
    assert chunks[0]["section_path"] == "# Cap 1"
    # El texto pasado al segmenter para la sección 1 incluye el ### (no parte).
    assert "Sub detalle" in seg.calls[0][0]
    assert chunks[2]["section_path"] == "# Cap 1 > ## Cap 2"


def test_chunk_markdown_without_headers_uses_whole_text():
    seg = _FakeSegmenter()
    chunks = chunk_markdown("Solo texto sin encabezados.", "short", seg)
    assert len(chunks) == 2
    assert chunks[0]["section_path"] == ""


# ---------------------------------------------------------------------
# classify_query
# ---------------------------------------------------------------------


def test_classify_query_global():
    assert classify_query("¿De qué trata el libro?") == "global"
    assert classify_query("Resumen del documento") == "global"
    assert classify_query("¿Cuáles son los temas principales?") == "global"
    assert classify_query("main topics of the book") == "global"


def test_classify_query_local():
    assert classify_query("¿Qué fórmula usa la propagación hacia atrás?") == "local"
    assert classify_query("¿Cómo se calcula el gradiente?") == "local"


# ---------------------------------------------------------------------
# personalized_pagerank
# ---------------------------------------------------------------------


def test_personalized_pagerank_converges_and_seed_dominates():
    # Triángulo 1-2-3 + nodo 4 conectado a 1.
    adj = {
        1: {2: 1, 4: 1},
        2: {1: 1, 3: 1},
        3: {2: 1, 1: 1},
        4: {1: 1},
    }
    scores = personalized_pagerank(adj, seed=[1], alpha=0.15, max_iter=50, tol=1e-6)

    assert set(scores.keys()) == {1, 2, 3, 4}
    assert abs(sum(scores.values()) - 1.0) < 1e-6
    # La semilla domina.
    assert scores[1] > scores[2]
    assert scores[1] > scores[3]
    assert scores[1] > scores[4]


def test_personalized_pagerank_empty_seed_returns_empty():
    assert personalized_pagerank({1: {2: 1}, 2: {1: 1}}, seed=[]) == {}


# ---------------------------------------------------------------------
# build_adjacency
# ---------------------------------------------------------------------


def test_build_adjacency_symmetric_with_weights():
    class _Result:
        def __init__(self, rows):
            self._rows = rows

        def fetchall(self):
            return self._rows

    class _FakeSession:
        def __init__(self, rows):
            self._rows = rows

        def execute(self, stmt, params=None):
            return _Result(self._rows)

    rows = [
        SimpleNamespace(source_entity_id=1, target_entity_id=2),
        SimpleNamespace(source_entity_id=2, target_entity_id=3),
        SimpleNamespace(source_entity_id=1, target_entity_id=2),  # repetida
    ]
    adj = build_adjacency(_FakeSession(rows))

    assert adj[1][2] == 2  # frecuencia
    assert adj[2][1] == 2  # simétrico
    assert adj[2][3] == 1
    assert adj[3][2] == 1


# ---------------------------------------------------------------------
# assemble_context
# ---------------------------------------------------------------------


def test_assemble_context_contains_sections():
    chunks = [
        {
            "doc_path": "a.md",
            "section_path": "## Intro",
            "chunk_index": 0,
            "content": "contenido a",
        }
    ]
    triples = [{"source": "A", "type": "UTILIZA", "target": "B", "doc": "a.md"}]
    figures = [{"image_path": "images/a/fig1.png", "description": "una figura"}]
    summaries = [{"doc_path": "a.md", "summary": "resumen de a"}]

    ctx = assemble_context(chunks, triples, figures, summaries, "pregunta")

    assert "FRAGMENTOS RECUPERADOS" in ctx
    assert "SUBGRAFO DE ENTIDADES" in ctx
    assert "FIGURAS" in ctx
    assert "RESUMENES DE DOCUMENTO" in ctx
    assert "contenido a" in ctx
    assert "(A) -[UTILIZA]-> (B)" in ctx
    assert "una figura" in ctx
    assert "resumen de a" in ctx


def test_assemble_context_empty_graceful():
    ctx = assemble_context([], [], [], [], "pregunta")
    assert "sin fragmentos recuperados" in ctx
    assert "sin tripletas" in ctx
    assert "sin figuras" in ctx
    assert "sin resúmenes" in ctx


# ---------------------------------------------------------------------
# complete_local
# ---------------------------------------------------------------------


def _local_model_row():
    return SimpleNamespace(
        model_name="qwen2.5-3b-instruct-q4_k_m",
        syntax_profile={
            "base_url": "http://localhost:8080/v1",
            "sampling": {
                "temperature": 0.1,
                "top_p": 0.9,
                "repetition_penalty": 1.05,
                "max_tokens": 60,
                "stop": ["\n", ""],
            },
        },
    )


class _LocalSession:
    def __init__(self, row):
        self._row = row

    def execute(self, stmt, params=None):
        class _Result:
            def __init__(self, row):
                self._row = row

            def first(self):
                return self._row

        return _Result(self._row)


def test_complete_local_builds_body(monkeypatch):
    import src.kag_ingest as ingest

    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout

        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": [{"message": {"content": "resumen"}}]}

        return _Resp()

    monkeypatch.setattr(ingest.httpx, "post", fake_post)

    text = complete_local(
        _LocalSession(_local_model_row()),
        "prompt",
        system="sys",
        max_tokens=60,
        temperature=0.1,
    )

    assert text == "resumen"
    assert captured["url"] == "http://localhost:8080/v1/chat/completions"
    assert captured["json"]["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "prompt"},
    ]
    assert captured["json"]["temperature"] == 0.1
    assert captured["json"]["max_tokens"] == 60
    assert captured["json"]["top_p"] == 0.9
    assert captured["json"]["repetition_penalty"] == 1.05
    assert captured["json"]["stop"] == ["\n", ""]


def test_complete_local_raises_without_local_model():
    with pytest.raises(RuntimeError, match="local"):
        complete_local(_LocalSession(None), "prompt")


def test_complete_local_retries_then_raises(monkeypatch):
    import src.kag_ingest as ingest

    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(url)
        raise httpx.ConnectError("no server")

    monkeypatch.setattr(ingest.httpx, "post", fake_post)

    with pytest.raises(httpx.ConnectError):
        complete_local(_LocalSession(_local_model_row()), "prompt")
    assert len(calls) == 2  # retry simple de 2 intentos
