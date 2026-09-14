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
    apply_relevance_threshold,
    assemble_context,
    build_adjacency,
    classify_query,
    disambiguate_by_cooccurrence,
    grounded_entity_linking,
    hybrid_search,
    match_entities_candidates,
    personalized_pagerank,
    ppr_entity_selection,
    rrf_merge,
)


@pytest.fixture(autouse=True)
def _reset_adjacency_cache():
    """Resetea el caché module-level de build_adjacency entre tests.

    El caché es global al módulo src.kag_query y contamina entre tests: un
    test que cachea con una sesión falsa dejaría el dict/versión para el
    siguiente. Se resetea antes y después de cada test.
    """
    import src.kag_query as kq

    kq._adjacency_cache = None
    kq._adjacency_version = -1
    yield
    kq._adjacency_cache = None
    kq._adjacency_version = -1


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

    # 3 secciones × 2 segmentos, pero los segmentos son diminutos y se
    # fusionan hasta max_tokens → 1 chunk por sección.
    assert len(chunks) == 3
    assert chunks[0]["section_path"] == "# Introducción"
    assert chunks[1]["section_path"] == "# Introducción > ## Métodos"
    assert chunks[2]["section_path"] == "# Introducción > ## Métodos > ### Sub"
    assert chunks[0]["token_estimate"] == estimate_tokens(chunks[0]["content"])
    assert "seg0:" in chunks[0]["content"]
    assert "seg1:" in chunks[0]["content"]  # fusionado
    # max_tokens se propaga al segmenter
    assert all(call[1] == 800 for call in seg.calls)


def test_chunk_markdown_short_respects_max_tokens_merge():
    # Segmentos grandes no se fusionan si exceden max_tokens.
    class _BigSegments:
        def segment_text(self, text, max_tokens=800):
            return ["a" * 2000, "b" * 2000]  # ~500 tokens cada uno

    chunks = chunk_markdown("# T\n\ncontenido", "short", _BigSegments(), max_tokens=800)
    assert len(chunks) == 2  # no caben juntos en 800 tokens


def test_chunk_markdown_long_presegments_h1_h2():
    md = (
        "# Cap 1\n\nContenido cap 1.\n\n"
        "### Sub detalle\n\nmás.\n\n"
        "## Cap 2\n\nContenido cap 2."
    )
    seg = _FakeSegmenter()
    chunks = chunk_markdown(md, "long", seg, max_tokens=800)

    # long: solo H1/H2 parten; ### queda dentro del contenido de la sección.
    # 2 secciones × 2 segmentos diminutos → 1 chunk por sección (fusionados).
    assert len(chunks) == 2
    assert chunks[0]["section_path"] == "# Cap 1"
    # El texto pasado al segmenter para la sección 1 incluye el ### (no parte).
    assert "Sub detalle" in seg.calls[0][0]
    assert chunks[1]["section_path"] == "# Cap 1 > ## Cap 2"


def test_chunk_markdown_without_headers_uses_whole_text():
    seg = _FakeSegmenter()
    chunks = chunk_markdown("Solo texto sin encabezados.", "short", seg)
    assert len(chunks) == 1  # 2 segmentos diminutos fusionados
    assert chunks[0]["section_path"] == ""


def test_chunk_markdown_use_coref_false_disables_coref():
    class _SegWithCoref:
        def __init__(self):
            self.coref_called = False

        def segment_text(self, text, max_tokens=800):
            # Simula el segmentador real: segment_text llama a coref internamente.
            segs = ["seg uno", "seg dos"]
            return self.resolve_coreferences(segs)

        def resolve_coreferences(self, segments):
            self.coref_called = True
            return segments

    seg = _SegWithCoref()
    chunk_markdown("texto", "short", seg, use_coref=False)
    assert seg.coref_called is False  # no-op temporal

    seg2 = _SegWithCoref()
    chunk_markdown("texto", "short", seg2, use_coref=True)
    assert seg2.coref_called is True  # comportamiento original preservado


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


def test_build_adjacency_cached_by_version():
    """Con la tabla de versión (migración 0016), la segunda llamada con la
    misma versión devuelve el MISMO objeto sin releer kag_relations."""
    import src.kag_query as kq

    class _Result:
        def __init__(self, rows=None, scalar=None):
            self._rows = rows or []
            self._scalar = scalar

        def fetchall(self):
            return self._rows

        def scalar(self):
            return self._scalar

    class _FakeSession:
        def __init__(self, version, rows):
            self.version = version
            self.rows = rows
            self.relation_queries = 0

        def execute(self, stmt, params=None):
            sql = str(stmt)
            if "kag_graph_state" in sql:
                return _Result(scalar=self.version)
            self.relation_queries += 1
            return _Result(rows=self.rows)

    rows = [
        SimpleNamespace(source_entity_id=1, target_entity_id=2),
        SimpleNamespace(source_entity_id=2, target_entity_id=3),
    ]
    session = _FakeSession(version=1, rows=rows)

    adj1 = build_adjacency(session)
    assert session.relation_queries == 1
    assert adj1[1][2] == 1
    assert adj1[2][3] == 1

    adj2 = build_adjacency(session)
    assert session.relation_queries == 1  # no relee
    assert adj2 is adj1  # mismo objeto


def test_build_adjacency_rebuilds_on_version_change():
    """Si la versión en DB cambió (ingesta), la siguiente llamada relee
    kag_relations y reconstruye el dict."""
    import src.kag_query as kq

    class _Result:
        def __init__(self, rows=None, scalar=None):
            self._rows = rows or []
            self._scalar = scalar

        def fetchall(self):
            return self._rows

        def scalar(self):
            return self._scalar

    class _FakeSession:
        def __init__(self, version, rows):
            self.version = version
            self.rows = rows
            self.relation_queries = 0

        def execute(self, stmt, params=None):
            sql = str(stmt)
            if "kag_graph_state" in sql:
                return _Result(scalar=self.version)
            self.relation_queries += 1
            return _Result(rows=self.rows)

    rows_v1 = [
        SimpleNamespace(source_entity_id=1, target_entity_id=2),
        SimpleNamespace(source_entity_id=2, target_entity_id=3),
    ]
    session = _FakeSession(version=1, rows=rows_v1)

    adj1 = build_adjacency(session)
    assert session.relation_queries == 1
    assert 3 in adj1[2]

    # Ingesta: versión sube y las relaciones cambian.
    session.version = 2
    session.rows = [
        SimpleNamespace(source_entity_id=1, target_entity_id=4),
        SimpleNamespace(source_entity_id=4, target_entity_id=5),
    ]

    adj2 = build_adjacency(session)
    assert session.relation_queries == 2
    assert 2 not in adj2  # entidad 2 ya no está en el grafo
    assert adj2[1][4] == 1
    assert adj2[4][5] == 1


def test_build_adjacency_degrades_without_version_table():
    """Sin la migración 0016 (o sesión sin .scalar()), build_adjacency
    reconstruye en cada llamada (comportamiento viejo) y no cachea."""
    import src.kag_query as kq

    class _Result:
        def __init__(self, rows):
            self._rows = rows

        def fetchall(self):
            return self._rows

    class _FakeSession:
        def __init__(self, rows):
            self.rows = rows
            self.relation_queries = 0

        def execute(self, stmt, params=None):
            sql = str(stmt)
            if "kag_graph_state" in sql:
                raise AttributeError("no .scalar() en sesión falsa")
            self.relation_queries += 1
            return _Result(rows=self.rows)

    rows = [
        SimpleNamespace(source_entity_id=1, target_entity_id=2),
        SimpleNamespace(source_entity_id=2, target_entity_id=3),
    ]
    session = _FakeSession(rows=rows)

    adj1 = build_adjacency(session)
    assert session.relation_queries == 1
    adj2 = build_adjacency(session)
    assert session.relation_queries == 2  # relee cada vez
    assert adj2 is not adj1  # no cachea
    assert adj2[1][2] == 1


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


def test_assemble_context_with_history():
    chunks = [
        {
            "doc_path": "a.md",
            "section_path": "## Intro",
            "chunk_index": 0,
            "content": "contenido a",
            "is_anchor": True,
            "score": 0.5,
        }
    ]
    history = [
        {"role": "user", "content": "¿qué es X?"},
        {"role": "assistant", "content": "X es Y."},
    ]
    ctx = assemble_context(chunks, [], [], [], "pregunta", history=history)
    assert "HISTORIAL DE CONVERSACIÓN" in ctx
    assert "[USER] ¿qué es X?" in ctx
    assert "[ASSISTANT] X es Y." in ctx
    # El historial no rompe las secciones normales.
    assert "FRAGMENTOS RECUPERADOS" in ctx
    assert "contenido a" in ctx


def test_assemble_context_without_history_omits_section():
    ctx = assemble_context([], [], [], [], "pregunta")
    assert "HISTORIAL DE CONVERSACIÓN" not in ctx


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


# ---------------------------------------------------------------------
# rrf_merge (opt 4)
# ---------------------------------------------------------------------


def test_rrf_merge_combines_ranks_positionally():
    dense = [(1, 0.9), (2, 0.8), (3, 0.7)]
    sparse = [(3, 5.0), (1, 4.0)]
    merged = rrf_merge(dense, sparse, k=60, top_k=10)
    scores = dict(merged)
    # Ambos en rank 1 → el que aparece en ambas listas arriba gana.
    assert scores[1] == pytest.approx(1 / 61 + 1 / 62)
    assert scores[3] == pytest.approx(1 / 63 + 1 / 61)
    assert scores[2] == pytest.approx(1 / 62)
    # Ordenado desc por score.
    assert [cid for cid, _ in merged] == sorted(scores, key=lambda c: -scores[c])


def test_rrf_merge_top_k_limits():
    dense = [(i, 1.0) for i in range(1, 11)]
    sparse = []
    merged = rrf_merge(dense, sparse, k=60, top_k=3)
    assert len(merged) == 3


def test_rrf_merge_empty():
    assert rrf_merge([], [], k=60, top_k=5) == []


def test_rrf_merge_three_layers():
    # Tres capas (vector, regex, PPR): cada una aporta 1/(k+rank).
    vec = [(1, 0.9), (2, 0.8)]
    regex = [(2, 5.0), (3, 4.0)]
    ppr = [(3, 0.0), (4, 0.0)]
    merged = rrf_merge(vec, regex, ppr, k=60, top_k=10)
    scores = dict(merged)
    assert scores[2] == pytest.approx(1 / 62 + 1 / 61)  # rank2 vec + rank1 regex
    assert scores[3] == pytest.approx(1 / 62 + 1 / 61)  # rank2 regex + rank1 ppr
    assert scores[1] == pytest.approx(1 / 61)
    assert scores[4] == pytest.approx(1 / 62)
    # Sin duplicación: cada id aparece una sola vez.
    assert len(merged) == 4


def test_rrf_merge_single_layer_preserves_order():
    # RRF de una sola lista preserva el orden (monótono en rank).
    hits = [(3, 0.9), (1, 0.8), (2, 0.7)]
    merged = rrf_merge(hits, k=60, top_k=10)
    assert [cid for cid, _ in merged] == [3, 1, 2]


# ---------------------------------------------------------------------
# ppr_entity_selection — umbral de cercanía en el grafo
# ---------------------------------------------------------------------


def test_ppr_entity_selection_auto_knee():
    # Semilla domina (0.33); primer salto 0.05; segundo salto 0.007; cola 1e-4.
    # Con auto=True el codo (Kneedle) corta en 0.05 → {1..5} (el 6, 0.007,
    # queda en la cola). El piso relativo (0.02×0.33=0.0066) habría dejado
    # pasar al 6 — el codo es más estricto y data-driven.
    scores = {1: 0.33, 2: 0.33, 3: 0.33, 4: 0.05, 5: 0.05, 6: 0.007, 7: 1e-4, 8: 1e-4}
    selected = ppr_entity_selection(scores, min_ratio=0.02)
    assert set(selected) == {1, 2, 3, 4, 5}
    # Ordenado por score desc.
    assert selected == sorted(selected, key=lambda e: -scores[e])


def test_ppr_entity_selection_fixed_floor():
    # Con auto=False se vuelve al comportamiento fijo: el piso relativo
    # (0.02×0.33=0.0066) deja pasar al 6 (0.007).
    scores = {1: 0.33, 2: 0.33, 3: 0.33, 4: 0.05, 5: 0.05, 6: 0.007, 7: 1e-4, 8: 1e-4}
    selected = ppr_entity_selection(scores, min_ratio=0.02, auto=False)
    assert set(selected) == {1, 2, 3, 4, 5, 6}


def test_ppr_entity_selection_empty():
    assert ppr_entity_selection({}) == []


def test_ppr_entity_selection_large_graph_statistical_guard():
    # 200 nodos: 3 semilla (0.3), 5 primer salto (0.05), resto ruido (1e-4).
    # El piso estadístico (mean + z*std) separa la señal de la cola.
    scores = {}
    for i in range(3):
        scores[i] = 0.3
    for i in range(3, 8):
        scores[i] = 0.05
    for i in range(8, 200):
        scores[i] = 1e-4
    selected = ppr_entity_selection(scores, min_ratio=0.02, z=0.5)
    assert set(range(8)) <= set(selected)
    assert not any(e >= 8 for e in selected)


def test_ppr_entity_selection_small_graph_no_statistical_guard():
    # Grafo pequeño (n < 100): el piso es solo relativo al máximo — el
    # término estadístico sobre-filtraría (la media es significativa).
    scores = {1: 0.33, 2: 0.33, 3: 0.33, 4: 0.05, 5: 0.05, 6: 0.007}
    selected = ppr_entity_selection(scores, min_ratio=0.02, z=0.5)
    assert set(selected) == {1, 2, 3, 4, 5, 6}


# ---------------------------------------------------------------------
# apply_relevance_threshold — codo automático (Kneedle)
# ---------------------------------------------------------------------


def test_apply_relevance_threshold_auto_knee():
    # Curva con codo claro: 0.9, 0.5, 0.3, 0.1, 0.05, 0.01. El codo corta en
    # 0.1 → quedan los 4 primeros. El piso relativo (0.15×0.9=0.135) habría
    # dejado solo 3 — el codo es data-driven.
    hits = [(1, 0.9), (2, 0.5), (3, 0.3), (4, 0.1), (5, 0.05), (6, 0.01)]
    result = apply_relevance_threshold(hits)
    assert [cid for cid, _ in result] == [1, 2, 3, 4]


def test_apply_relevance_threshold_fixed_floor():
    # Con auto=False: piso relativo 0.15×0.9=0.135 → solo los 3 primeros.
    hits = [(1, 0.9), (2, 0.5), (3, 0.3), (4, 0.1), (5, 0.05), (6, 0.01)]
    result = apply_relevance_threshold(hits, auto=False)
    assert [cid for cid, _ in result] == [1, 2, 3]


def test_apply_relevance_threshold_empty():
    assert apply_relevance_threshold([]) == []


def test_apply_relevance_threshold_no_signal():
    # max_score <= 0 → sin señal → nada pasa.
    assert apply_relevance_threshold([(1, 0.0), (2, -0.1)]) == []


def test_apply_relevance_threshold_no_knee_falls_back_to_floor():
    # Curva casi lineal (sin codo claro) → fallback al piso relativo.
    hits = [(1, 0.9), (2, 0.8), (3, 0.7), (4, 0.6), (5, 0.5)]
    result = apply_relevance_threshold(hits)
    # Piso = 0.15×0.9 = 0.135 → todos pasan.
    assert [cid for cid, _ in result] == [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------
# disambiguate_by_cooccurrence (opt 2)
# ---------------------------------------------------------------------


def test_disambiguate_picks_candidate_sharing_neighbors():
    # Entidad 1 confirmada (mención única). Mención ambigua: 2 o 3.
    # 2 comparte vecino (4) con 1; 3 está aislada.
    adj = {
        1: {4: 1},
        2: {4: 1},
        3: {},
        4: {1: 1, 2: 1},
    }
    groups = [[1], [2, 3]]
    result = disambiguate_by_cooccurrence(None, groups, adjacency=adj)
    assert result == [1, 2]


def test_disambiguate_no_confirmed_keeps_first():
    adj = {1: {2: 1}, 2: {1: 1}, 3: {}}
    groups = [[1, 2], [3, 1]]
    result = disambiguate_by_cooccurrence(None, groups, adjacency=adj)
    assert result == [1, 3]


def test_disambiguate_empty_groups():
    assert disambiguate_by_cooccurrence(None, [], adjacency={}) == []


def test_disambiguate_overlap_normalizes_by_degree():
    # Confirmada: 1 con vecinos {2,3,4,5}. Candidato A (hub, grado 100)
    # comparte 3; candidato B (grado 3) comparte 3. El overlap coefficient
    # prefiere a B (3/3=1.0) sobre A (3/4=0.75); el conteo crudo empataría
    # y el código viejo elegiría al primero (el hub).
    adj = {1: {2: 1, 3: 1, 4: 1, 5: 1}}
    adj[10] = {2: 1, 3: 1, 4: 1}
    for i in range(97):
        adj[10][f"h{i}"] = 1  # grado 100
    adj[11] = {2: 1, 3: 1, 4: 1}  # grado 3
    groups = [[1], [10, 11]]
    result = disambiguate_by_cooccurrence(
        None, groups, adjacency=adj, min_overlap=0.1, margin=0.05
    )
    assert result == [1, 11]


def test_disambiguate_ambiguous_falls_back_to_first():
    # Dos candidatos con el mismo overlap (0.5) → margen 0 → primer candidato.
    adj = {1: {2: 1, 3: 1, 4: 1}, 2: {1: 1, 3: 1}, 3: {1: 1, 2: 1}, 4: {1: 1}}
    groups = [[1], [2, 3]]
    result = disambiguate_by_cooccurrence(
        None, groups, adjacency=adj, min_overlap=0.1, margin=0.05
    )
    assert result == [1, 2]


def test_disambiguate_below_min_overlap_falls_back():
    # El mejor candidato no comparte vecinos (overlap 0 < min_overlap) →
    # ambiguo → primer candidato.
    adj = {1: {2: 1, 3: 1}, 2: {1: 1}, 3: {1: 1}, 4: {}}
    groups = [[1], [2, 3]]
    result = disambiguate_by_cooccurrence(
        None, groups, adjacency=adj, min_overlap=0.1, margin=0.05
    )
    assert result == [1, 2]


# ---------------------------------------------------------------------
# match_entities_candidates (opt 2)
# ---------------------------------------------------------------------


class _CandidatesSession:
    """Sesión falsa: devuelve ids según name_norm exacto o patrón LIKE.

    exact: {name_norm: [ids]}; like: {substring: [(id, name)]}.
    Devuelve filas con .id y .name (lo que esperan los callers).
    """

    def __init__(self, exact=None, like=None):
        self.exact = exact or {}  # name_norm -> [ids]
        self.like = like or {}  # substring -> [(id, name)]

    def execute(self, stmt, params=None):
        sql = str(stmt)
        if "name_norm = :nn" in sql:
            ids = self.exact.get(params["nn"], [])
            rows = [SimpleNamespace(id=i, name=f"entidad-{i}") for i in ids]
        elif "name_norm LIKE :pat" in sql:
            pat = params["pat"].strip("%")
            rows = []
            for key, vals in self.like.items():
                if key in pat or pat in key:
                    rows.extend(SimpleNamespace(id=i, name=n) for i, n in vals)
        else:
            rows = []

        class _Result:
            def __init__(self, rows):
                self._rows = rows

            def fetchall(self):
                return self._rows

        return _Result(rows)


def test_match_entities_candidates_exact_then_like():
    session = _CandidatesSession(
        exact={"red neuronal": [1]},
        like={"red": [(2, "Red de Petri"), (3, "Red Social")]},
    )
    groups = match_entities_candidates(session, ["Red Neuronal", "red"])
    assert groups == [[1], [2, 3]]


def test_match_entities_candidates_no_match_skips():
    session = _CandidatesSession(exact={}, like={})
    assert match_entities_candidates(session, ["nada que ver"]) == []


# ---------------------------------------------------------------------
# grounded_entity_linking (opt 1)
# ---------------------------------------------------------------------


def test_grounded_entity_linking_llm_selects_from_pool(monkeypatch):
    import src.kag_query as kq

    # Pool determinista: "red neuronal" y "red de petri" (vía LIKE).
    session = _CandidatesSession(
        exact={},
        like={
            "red neuronal": [(1, "Red Neuronal")],
            "red de petri": [(2, "Red de Petri")],
        },
    )

    # spaCy no disponible → _noun_chunk_fallback degrada a determinista.
    def fake_get_spacy(session, lang):
        raise RuntimeError("no spacy")

    monkeypatch.setattr(kq, "_get_spacy_nlp", fake_get_spacy)

    monkeypatch.setattr(
        kq,
        "load_settings",
        lambda session: SimpleNamespace(llm_retries=3, fallback_model=None),
    )

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
        return '{"entities": ["Red Neuronal"]}', "small", False

    monkeypatch.setattr(kq, "call_with_retries", fake_call)

    names = grounded_entity_linking(session, "¿Qué usa la red neuronal?")
    assert names == ["Red Neuronal"]
    # El prompt incluye el pool de candidatos.
    assert "Red Neuronal" in captured["prompt"]
    assert "Red de Petri" in captured["prompt"]


def test_grounded_entity_linking_llm_fails_returns_pool(monkeypatch):
    import src.kag_query as kq

    session = _CandidatesSession(
        exact={},
        like={"red neuronal": [(1, "Red Neuronal")]},
    )

    def fake_get_spacy(session, lang):
        raise RuntimeError("no spacy")

    monkeypatch.setattr(kq, "_get_spacy_nlp", fake_get_spacy)

    monkeypatch.setattr(
        kq,
        "load_settings",
        lambda session: SimpleNamespace(llm_retries=3, fallback_model=None),
    )

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
        raise RuntimeError("LLM caído")

    monkeypatch.setattr(kq, "call_with_retries", fake_call)

    names = grounded_entity_linking(session, "¿Qué usa la red neuronal?")
    assert names == ["Red Neuronal"]  # pool determinista como degradación


def test_grounded_entity_linking_empty_pool(monkeypatch):
    import src.kag_query as kq

    session = _CandidatesSession(exact={}, like={})

    def fake_get_spacy(session, lang):
        raise RuntimeError("no spacy")

    monkeypatch.setattr(kq, "_get_spacy_nlp", fake_get_spacy)

    assert grounded_entity_linking(session, "sin entidades aquí") == []


def test_grounded_entity_linking_filters_out_of_pool(monkeypatch):
    import src.kag_query as kq

    # Pool: solo "Red Neuronal". El LLM alucina "Red de Petri" (fuera del pool).
    session = _CandidatesSession(
        exact={},
        like={"red neuronal": [(1, "Red Neuronal")]},
    )

    def fake_get_spacy(session, lang):
        raise RuntimeError("no spacy")

    monkeypatch.setattr(kq, "_get_spacy_nlp", fake_get_spacy)
    monkeypatch.setattr(
        kq,
        "load_settings",
        lambda session: SimpleNamespace(llm_retries=3, fallback_model=None),
    )

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
        return '{"entities": ["Red de Petri"]}', "small", False

    monkeypatch.setattr(kq, "call_with_retries", fake_call)

    # El nombre fuera del pool se descarta → cae al pool determinista.
    names = grounded_entity_linking(session, "¿Qué usa la red neuronal?")
    assert names == ["Red Neuronal"]


def test_disambiguate_empty_adjacency_does_not_crash():
    # C2: adjacency vacío + mención ambigua con confirmada → no ValueError.
    groups = [[1], [2, 3]]
    result = disambiguate_by_cooccurrence(None, groups, adjacency={})
    assert result == [1, 2]  # default=g[0] para la ambigua


def test_disambiguate_auto_margin_uses_median():
    # Confirmada: 1 con vecinos {2..8}. Cuatro menciones ambiguas:
    #   [10,11]: 10 overlap 0.8, 11 overlap 1.0 → gap 0.2
    #   [21,20]: 20 overlap 0.8, 21 overlap 0.667 → gap 0.133
    #   [30,31]: 30 overlap 0.8, 31 overlap 1.0 → gap 0.2
    # Gaps = [0.2, 0.133, 0.2] → mediana 0.2. Con auto, el margen es 0.2:
    #   [10,11] gap 0.2 >= 0.2 → 11; [21,20] gap 0.133 < 0.2 → ambiguo (21);
    #   [30,31] gap 0.2 >= 0.2 → 31.
    adj = {1: {2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1}}
    adj[10] = {2: 1, 3: 1, 4: 1, 5: 1, 100: 1}  # overlap 0.8
    adj[11] = {2: 1, 3: 1}  # overlap 1.0
    adj[20] = {2: 1, 3: 1, 4: 1, 5: 1, 200: 1}  # overlap 0.8
    adj[21] = {2: 1, 3: 1, 201: 1}  # overlap 0.667
    adj[30] = {2: 1, 3: 1, 4: 1, 5: 1, 300: 1}  # overlap 0.8
    adj[31] = {2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 301: 1}  # overlap 1.0
    groups = [[1], [10, 11], [21, 20], [30, 31]]
    result = disambiguate_by_cooccurrence(
        None, groups, adjacency=adj, min_overlap=0.1, margin=0.05
    )
    assert result == [1, 11, 21, 31]


def test_disambiguate_fixed_margin_resolves_all():
    # El mismo grafo con auto=False y margen fijo 0.05: todos los gaps
    # (0.2, 0.133, 0.2) superan 0.05 → todas las menciones se resuelven al
    # mejor candidato.
    adj = {1: {2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1}}
    adj[10] = {2: 1, 3: 1, 4: 1, 5: 1, 100: 1}
    adj[11] = {2: 1, 3: 1}
    adj[20] = {2: 1, 3: 1, 4: 1, 5: 1, 200: 1}
    adj[21] = {2: 1, 3: 1, 201: 1}
    adj[30] = {2: 1, 3: 1, 4: 1, 5: 1, 300: 1}
    adj[31] = {2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 301: 1}
    groups = [[1], [10, 11], [21, 20], [30, 31]]
    result = disambiguate_by_cooccurrence(
        None, groups, adjacency=adj, min_overlap=0.1, margin=0.05, auto=False
    )
    assert result == [1, 11, 20, 31]


def test_noun_chunk_fallback_no_phrases_degrades(monkeypatch):
    import src.kag_query as kq

    class _FakeNlp:
        def __call__(self, text):
            return _FakeDoc()

    class _FakeDoc:
        noun_chunks = []

        def __iter__(self):
            return iter([])

    session = _CandidatesSession(
        exact={},
        like={"calcula": [(5, "Cálculo")]},
    )

    monkeypatch.setattr(kq, "_get_spacy_nlp", lambda session, lang: _FakeNlp())

    # Sin noun chunks ni PROPN → degrada al fallback determinista por tokens.
    found = kq._noun_chunk_fallback(session, "¿Cómo se calcula?")
    assert found == ["Cálculo"]


# ---------------------------------------------------------------------
# critic_regex_search — LLM crítico + heurística determinista
# ---------------------------------------------------------------------


class _RegexSession:
    """Sesión falsa: responde a la query FTS por término."""

    def __init__(self, hits_by_term):
        self.hits_by_term = hits_by_term  # {term: [(id, score)]}

    def execute(self, stmt, params=None):
        term = params["q"]
        rows = [
            SimpleNamespace(id=cid, score=s)
            for cid, s in self.hits_by_term.get(term, [])
        ]

        class _Result:
            def __init__(self, rows):
                self._rows = rows

            def fetchall(self):
                return self._rows

        return _Result(rows)


def test_deterministic_regex_terms():
    import src.kag_query as kq

    assert kq._deterministic_regex_terms("¿Qué es CVE-2024-3094?") == ["CVE-2024-3094"]
    assert kq._deterministic_regex_terms("¿Quién es Pierre Bourdieu?") == [
        "Pierre Bourdieu"
    ]
    assert kq._deterministic_regex_terms("¿qué es el capital cultural?") == []


def test_critic_regex_search_returns_terms(monkeypatch):
    import src.kag_query as kq

    session = _RegexSession({"CVE-2024-3094": [(7, 3.0), (8, 2.0)]})
    monkeypatch.setattr(
        kq,
        "load_settings",
        lambda session: SimpleNamespace(llm_retries=3, fallback_model=None),
    )

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
        return '{"needs_regex": true, "terms": ["CVE-2024-3094"]}', "small", False

    monkeypatch.setattr(kq, "call_with_retries", fake_call)

    hits, terms = kq.critic_regex_search(session, "¿Qué es CVE-2024-3094?", top_k=5)
    assert terms == ["CVE-2024-3094"]
    assert hits == [(7, 3.0), (8, 2.0)]


def test_critic_regex_search_no_terms_returns_empty(monkeypatch):
    import src.kag_query as kq

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
        return '{"needs_regex": false, "terms": []}', "small", False

    monkeypatch.setattr(kq, "call_with_retries", fake_call)
    monkeypatch.setattr(
        kq,
        "load_settings",
        lambda session: SimpleNamespace(llm_retries=3, fallback_model=None),
    )

    hits, terms = kq.critic_regex_search(None, "¿Qué es el capital cultural?", top_k=5)
    assert hits == []
    assert terms == []


def test_critic_regex_search_llm_fails_uses_heuristic(monkeypatch):
    import src.kag_query as kq

    session = _RegexSession({"CVE-2024-3094": [(7, 3.0)]})

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
        raise RuntimeError("LLM caído")

    monkeypatch.setattr(kq, "call_with_retries", fake_call)
    monkeypatch.setattr(
        kq,
        "load_settings",
        lambda session: SimpleNamespace(llm_retries=3, fallback_model=None),
    )

    hits, terms = kq.critic_regex_search(session, "¿Qué es CVE-2024-3094?", top_k=5)
    assert terms == ["CVE-2024-3094"]  # heurística determinista
    assert hits == [(7, 3.0)]


# ---------------------------------------------------------------------
# hybrid_search (opt 4) — degradación sin FTS
# ---------------------------------------------------------------------


def test_hybrid_search_degrades_to_dense_when_fts_missing(monkeypatch):
    from sqlalchemy.exc import ProgrammingError

    import src.kag_query as kq

    class _FakeSession:
        def rollback(self):
            pass

    session = _FakeSession()

    def fake_vector(session, q_emb, top_k):
        return [(1, 0.9), (2, 0.8)]

    def fake_fts(session, query_text, top_k):
        raise ProgrammingError(
            "stmt", {}, Exception("column content_tsv does not exist")
        )

    monkeypatch.setattr(kq, "vector_search", fake_vector)
    monkeypatch.setattr(kq, "fts_search", fake_fts)

    hits = hybrid_search(session, "pregunta", [0.1, 0.2], top_k=5)
    assert hits == [(1, 0.9), (2, 0.8)]


def test_hybrid_search_propagates_non_fts_errors(monkeypatch):
    import src.kag_query as kq

    class _FakeSession:
        def rollback(self):
            pass

    session = _FakeSession()

    def fake_vector(session, q_emb, top_k):
        return [(1, 0.9)]

    def fake_fts(session, query_text, top_k):
        raise RuntimeError("otro error real")

    monkeypatch.setattr(kq, "vector_search", fake_vector)
    monkeypatch.setattr(kq, "fts_search", fake_fts)

    with pytest.raises(RuntimeError, match="otro error real"):
        hybrid_search(session, "pregunta", [0.1, 0.2], top_k=5)


def test_fts_search_empty_query_returns_empty(monkeypatch):
    import src.kag_query as kq

    def fake_execute(stmt, params=None):
        raise AssertionError("no debería ejecutar SQL con query vacía")

    class _FakeSession:
        def execute(self, stmt, params=None):
            return fake_execute(stmt, params)

    assert kq.fts_search(_FakeSession(), "", 5) == []
    assert kq.fts_search(_FakeSession(), "   ", 5) == []


def test_hybrid_search_merges_dense_and_sparse(monkeypatch):
    import src.kag_query as kq

    class _FakeSession:
        def rollback(self):
            pass

    session = _FakeSession()

    def fake_vector(session, q_emb, top_k):
        return [(1, 0.9), (2, 0.8)]

    def fake_fts(session, query_text, top_k):
        return [(2, 5.0), (3, 4.0)]

    monkeypatch.setattr(kq, "vector_search", fake_vector)
    monkeypatch.setattr(kq, "fts_search", fake_fts)

    hits = hybrid_search(session, "pregunta", [0.1, 0.2], top_k=5)
    ids = [cid for cid, _ in hits]
    assert 1 in ids and 2 in ids and 3 in ids
    # El chunk 2 (rank 2 denso + rank 1 sparse) debe liderar.
    assert hits[0][0] == 2


def test_hybrid_search_none_embedding_degrades_to_fts_only(monkeypatch):
    """Sin query_embedding (embeddings no disponibles), no se llama a
    vector_search y se devuelven solo los hits de FTS."""
    import src.kag_query as kq

    class _FakeSession:
        def rollback(self):
            pass

    session = _FakeSession()

    def fake_vector(session, q_emb, top_k):
        raise AssertionError("vector_search no debe llamarse con embedding None")

    def fake_fts(session, query_text, top_k):
        return [(2, 5.0), (3, 4.0)]

    monkeypatch.setattr(kq, "vector_search", fake_vector)
    monkeypatch.setattr(kq, "fts_search", fake_fts)

    hits = hybrid_search(session, "pregunta", None, top_k=5)
    assert hits == [(2, 5.0), (3, 4.0)]


def test_hybrid_search_none_embedding_and_no_fts_returns_empty(monkeypatch):
    """Sin embedding y sin FTS (migración 0014 sin aplicar) → [] sin error."""
    from sqlalchemy.exc import ProgrammingError

    import src.kag_query as kq

    class _FakeSession:
        def rollback(self):
            pass

    session = _FakeSession()

    def fake_vector(session, q_emb, top_k):
        raise AssertionError("vector_search no debe llamarse con embedding None")

    def fake_fts(session, query_text, top_k):
        raise ProgrammingError(
            "stmt", {}, Exception("column content_tsv does not exist")
        )

    monkeypatch.setattr(kq, "vector_search", fake_vector)
    monkeypatch.setattr(kq, "fts_search", fake_fts)

    hits = hybrid_search(session, "pregunta", None, top_k=5)
    assert hits == []
