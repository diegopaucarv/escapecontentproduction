"""
Consulta del sistema KAG (Fase 2 del diseño).

Responde preguntas sobre el knowledge_repository combinando búsqueda
vectorial (pgvector), entity linking + Personalized PageRank estilo HippoRAG
(multi-hop) y resúmenes de documento (Qwen 2.5 local) como referencia
secundaria.

Módulo LIGERO a propósito: no importa torch/spacy/transformers. El embedding
de la query usa src.embeddings (carga perezosa del modelo Jina).

Uso:
    python -m src.kag_query "¿Qué fórmula usa la propagación hacia atrás?"
    python -m src.kag_query --top-k 12 "pregunta"
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import threading
from pathlib import Path

from sqlalchemy import bindparam, text
from sqlalchemy.exc import ProgrammingError

from src.kag_ingest import (
    detect_language,
    embedding_to_sql,
    normalize_entity_name,
)
from src.llm.base import call_with_retries, load_settings, parse_llm_output

# ---------------------------------------------------------------------
# Parámetros configurables del querying (más adelante se moverán a globals
# / settings de la DB; por ahora viven aquí para ajuste rápido).
# ---------------------------------------------------------------------

# Ventana de contexto: chunks consecutivos del MISMO documento alrededor de
# cada resultado ancla (p. ej. 5 = ±5 chunks). El LLM recibe el grupo
# completo (ancla + vecinos) para entender el texto contiguo.
CONTEXT_WINDOW = 5
# Cap total de chunks en el contexto final (anclas + ventanas). Evita
# reventar el presupuesto de tokens del LLM (8 anclas × 11 = 88 chunks
# ≈ 70k tokens sin cap).
MAX_CONTEXT_CHUNKS = 24
# Threshold de relevancia RELATIVO: se descartan resultados con score <
# ratio × max_score. Escala-agnóstico (RRF, coseno o ts_rank según la
# degradación del día).
MIN_SCORE_RATIO = 0.15
# Umbral mínimo absoluto de score para considerar un resultado (seguro
# contra queries sin señal: si el mejor score es ~0, nada pasa).
MIN_ABS_SCORE = 0.001

# ── Umbrales de cercanía en el GRAFO ────────────────────────────────────
# El score PPR depende del tamaño del grafo, de la distribución de grados
# y de alpha: un umbral absoluto fijo no funciona (en un grafo de 10k nodos
# la semilla puntúa ~0.3 y el segundo salto ~0.007; en uno de 50 nodos todo
# es más plano). Se usa un piso RELATIVO al máximo (la cola power-law del
# PPR cae rápido; el ratio captura el codo natural). Con alpha=0.15 cada
# salto decae ~×0.15: min_ratio=0.02 ≈ semilla + 2 saltos.
PPR_MIN_RATIO = 0.02
# Guarda estadística SOLO para grafos grandes (n >= 100): score >= mean +
# z*std. En grafos pequeños la media es significativa y este término
# sobre-filtra.
PPR_Z = 0.5
# Desambiguación por copresencia: overlap coefficient (vecinos compartidos
# normalizados por grado) + margen sobre el segundo candidato. Si el mejor
# no supera el piso o el margen, se conserva el primero (ambiguo).
DISAMBIG_MIN_OVERLAP = 0.1
DISAMBIG_MARGIN = 0.05

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _fix_db_host() -> None:
    """Reemplaza '@db:' por '@localhost:' en DATABASE_URL/DATABASE_URL_ASYNC.

    El host 'db' es la red Docker y no resuelve desde el host; las credenciales
    son las mismas. Debe llamarse ANTES de importar src.db.session. Si la
    variable no está en el entorno, la lee del .env (pydantic-settings da
    prioridad a las env vars reales sobre el archivo .env).
    """
    env_path = Path(__file__).resolve().parent.parent / ".env"
    for var in ("DATABASE_URL", "DATABASE_URL_ASYNC"):
        val = os.environ.get(var, "")
        if not val and env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith(f"{var}="):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
        if "@db:" in val:
            os.environ[var] = val.replace("@db:", "@localhost:")


# ---------------------------------------------------------------------
# Clasificación de la consulta
# ---------------------------------------------------------------------

GLOBAL_KEYWORDS = [
    "resumen",
    "conclusiones",
    "principales",
    "temas",
    "overview",
    "summary",
    "main topics",
    "libro",
    "book",
    "de que trata",
    "de qué trata",
    "que trata",
    "panorama",
    "ideas clave",
    "key ideas",
    "estructura",
]


def classify_query(query: str) -> str:
    """Heurística determinista de keywords: 'global' | 'local'."""
    q = query.lower()
    for kw in GLOBAL_KEYWORDS:
        if kw in q:
            return "global"
    return "local"


# ---------------------------------------------------------------------
# Entity linking
# ---------------------------------------------------------------------

# Stopwords planas para el fallback determinista de entity linking.
_QUERY_STOPWORDS = {
    "el",
    "la",
    "los",
    "las",
    "de",
    "del",
    "y",
    "en",
    "un",
    "una",
    "que",
    "es",
    "por",
    "para",
    "con",
    "se",
    "su",
    "al",
    "lo",
    "the",
    "and",
    "of",
    "to",
    "in",
    "is",
    "are",
    "for",
    "with",
    "on",
    "that",
    "this",
    "it",
    "como",
    "cual",
    "cuál",
    "qué",
    "quién",
    "quien",
    "dónde",
    "donde",
    "cuándo",
    "cuando",
    "cómo",
    "cuáles",
    "cuales",
    "usa",
    "usan",
    "utiliza",
    "hace",
    "hacen",
    "tiene",
    "tienen",
    "son",
    "fue",
    "era",
    "a",
    "o",
    "e",
}


def _deterministic_entity_fallback(session, query: str) -> list:
    """Tokeniza la query, quita stopwords y busca por substring contra name_norm."""
    words = [
        w
        for w in re.findall(r"[a-záéíóúüñçàèìòùâêîôû0-9]+", query.lower())
        if w not in _QUERY_STOPWORDS and len(w) > 2
    ]
    found = []
    for w in words:
        rows = session.execute(
            text(
                "SELECT DISTINCT e.name FROM kag_entities e "
                "JOIN kag_documents d ON d.id = e.doc_id "
                "WHERE e.name_norm LIKE :pat AND d.status = 'ready' LIMIT 5"
            ),
            {"pat": f"%{w}%"},
        ).fetchall()
        for r in rows:
            if r.name not in found:
                found.append(r.name)
    return found[:10]


# ---------------------------------------------------------------------
# Fallback con noun chunks de spaCy (opt 3)
# ---------------------------------------------------------------------

# Modelos spaCy por idioma (mismos defaults que src/kag/segmentador.py).
# Se duplican aquí a propósito: importar el segmentador cargaría torch/spacy
# a nivel de módulo, y kag_query.py es LIGERO a propósito.
_SPACY_MODELS = {
    "es": "es_core_news_md",
    "en": "en_core_web_md",
    "pt": "pt_core_news_md",
    "de": "de_core_news_md",
    "fr": "fr_core_news_md",
}

_spacy_nlp_cache: dict = {}
_spacy_nlp_lock = threading.Lock()


def _spacy_model_for(session, lang: str) -> str:
    """Modelo spaCy del idioma desde kag_segmenter_settings (fallback local)."""
    try:
        row = session.execute(
            text(
                "SELECT spacy_models FROM kag_segmenter_settings "
                "WHERE is_active = TRUE ORDER BY id DESC LIMIT 1"
            )
        ).first()
        if row and row.spacy_models and lang in row.spacy_models:
            return row.spacy_models[lang]
    except Exception:  # noqa: BLE001 — sin config: defaults locales
        pass
    return _SPACY_MODELS.get(lang, "es_core_news_md")


def _get_spacy_nlp(session, lang: str):
    """Carga spaCy perezosamente (el módulo es ligero) y cachea por idioma."""
    if lang in _spacy_nlp_cache:
        return _spacy_nlp_cache[lang]
    with _spacy_nlp_lock:
        if lang in _spacy_nlp_cache:
            return _spacy_nlp_cache[lang]
        import spacy

        model = _spacy_model_for(session, lang)
        nlp = spacy.load(model)
        _spacy_nlp_cache[lang] = nlp
        return nlp


def _noun_chunk_fallback(session, query: str) -> list:
    """Fallback con noun chunks de spaCy: frases nominales completas.

    Mejor granularidad que tokens sueltos para entidades multi-palabra
    ("red neuronal" no se parte). Si spaCy no está disponible o no
    encuentra frases, degrada al fallback determinista por tokens.
    """
    lang = detect_language(query)
    try:
        nlp = _get_spacy_nlp(session, lang)
    except Exception:  # noqa: BLE001 — spaCy no disponible: degradación
        return _deterministic_entity_fallback(session, query)
    doc = nlp(query)
    phrases = [chunk.text for chunk in doc.noun_chunks]
    phrases += [tok.text for tok in doc if tok.pos_ == "PROPN"]
    if not phrases:
        return _deterministic_entity_fallback(session, query)
    found = []
    for phrase in phrases:
        nn = normalize_entity_name(phrase)
        if len(nn) < 3:
            continue
        rows = session.execute(
            text(
                "SELECT DISTINCT e.name FROM kag_entities e "
                "JOIN kag_documents d ON d.id = e.doc_id "
                "WHERE e.name_norm LIKE :pat AND d.status = 'ready' LIMIT 5"
            ),
            {"pat": f"%{nn}%"},
        ).fetchall()
        for r in rows:
            if r.name not in found:
                found.append(r.name)
    return found[:10]


# ---------------------------------------------------------------------
# Entity linking anclado (opt 1)
# ---------------------------------------------------------------------

GROUNDED_ENTITIES_PROMPT = """Entidades candidatas del grafo de conocimiento:
{candidates}

Pregunta: {query}

Devuelve SOLO JSON:
{{"entities": ["Entidad 1", "Entidad 2"]}}

Elige SOLO de la lista de candidatas. Si ninguna se menciona en la
pregunta, devuelve {{"entities": []}}.
"""


def grounded_entity_linking(session, query: str) -> list:
    """Entity linking anclado: el LLM selecciona entidades canónicas SOLO
    entre los candidatos reales del grafo (pre-filtro determinista).

    Elimina el LIKE de la vía principal: el LLM devuelve nombres que ya
    existen en kag_entities → el match exacto basta. Los nombres que el
    LLM devuelva FUERA del pool se descartan (anclaje real, no cosmético).
    Si el LLM falla, devuelve el pool de candidatos (degradación natural).
    """
    candidates = _noun_chunk_fallback(session, query)
    if not candidates:
        return []
    settings = load_settings(session)
    retries = int(getattr(settings, "llm_retries", 3) or 3) if settings else 3
    fallback = getattr(settings, "fallback_model", None) if settings else None
    try:
        text_out, _model, _used_fallback = call_with_retries(
            session,
            prompt=GROUNDED_ENTITIES_PROMPT.format(
                candidates="\n".join(f"- {c}" for c in candidates),
                query=query,
            ),
            system="Eres un selector de entidades. Devuelve JSON válido.",
            model_size="small",
            response_format={"type": "json_object"},
            retries=retries,
            fallback_model=fallback,
        )
        data = parse_llm_output(text_out)
        names = [str(e).strip() for e in (data.get("entities") or []) if str(e).strip()]
        # Anclaje real: solo nombres que están en el pool (normalizados).
        pool_norm = {normalize_entity_name(c) for c in candidates}
        anchored = [n for n in names if normalize_entity_name(n) in pool_norm]
        if anchored:
            return anchored
    except Exception:  # noqa: BLE001 — LLM no disponible: pool determinista
        pass
    return candidates


def match_entities_candidates(session, names: list) -> list:
    """Candidatos por mención: lista de listas de ids.

    Exacto por name_norm primero; si no hay, LIKE (hasta 5). Cada mención
    puede tener 0..N candidatos — la desambiguación por copresencia decide.
    """
    groups = []
    for name in names:
        nn = normalize_entity_name(name)
        group = []
        rows = session.execute(
            text(
                "SELECT e.id FROM kag_entities e "
                "JOIN kag_documents d ON d.id = e.doc_id "
                "WHERE e.name_norm = :nn AND d.status = 'ready'"
            ),
            {"nn": nn},
        ).fetchall()
        group.extend(r.id for r in rows)
        if not group:
            rows = session.execute(
                text(
                    "SELECT e.id FROM kag_entities e "
                    "JOIN kag_documents d ON d.id = e.doc_id "
                    "WHERE e.name_norm LIKE :pat AND d.status = 'ready' LIMIT 5"
                ),
                {"pat": f"%{nn}%"},
            ).fetchall()
            group.extend(r.id for r in rows)
        # Dedup dentro del grupo (el LIKE puede solaparse con el exacto).
        group = list(dict.fromkeys(group))
        if group:
            groups.append(group)
    return groups


def disambiguate_by_cooccurrence(
    session,
    groups: list,
    adjacency=None,
    min_overlap: float = DISAMBIG_MIN_OVERLAP,
    margin: float = DISAMBIG_MARGIN,
) -> list:
    """Desambiguación por copresencia en el grafo.

    Para cada mención con varios candidatos, elige el que comparte más
    vecinos (a 1 salto) con las entidades confirmadas (menciones con un
    solo candidato). La cercanía se mide con el OVERLAP COEFFICIENT
    (vecinos compartidos / min(grado del candidato, vecinos confirmados)):
    normaliza por grado — un candidato de grado 100 con 5 vecinos
    compartidos NO es más cercano que uno de grado 3 con 3 compartidos.

    Confianza: el mejor candidato debe superar `min_overlap` Y tener un
    margen >= `margin` sobre el segundo. Si no, se conserva el primer
    candidato (ambiguo). Sin entidades confirmadas, conserva el primer
    candidato de cada mención (no hay señal de copresencia).
    """
    if not groups:
        return []
    confirmed = {g[0] for g in groups if len(g) == 1}
    if not confirmed:
        return list(dict.fromkeys(g[0] for g in groups))
    if adjacency is None:
        adjacency = build_adjacency(session)
    confirmed_neighbors = set()
    for eid in confirmed:
        confirmed_neighbors.update(adjacency.get(eid, {}).keys())
    result = []
    for g in groups:
        if len(g) == 1:
            result.append(g[0])
            continue
        scored = []
        for c in g:
            shared = len(confirmed_neighbors & set(adjacency.get(c, {}).keys()))
            denom = min(len(adjacency.get(c, {})), len(confirmed_neighbors)) or 1
            scored.append((c, shared / denom))
        scored.sort(key=lambda x: -x[1])
        best_c, best_s = scored[0]
        second_s = scored[1][1] if len(scored) > 1 else 0.0
        if best_s >= min_overlap and (best_s - second_s) >= margin:
            result.append(best_c)
        else:
            result.append(g[0])  # ambiguo: primer candidato
    return list(dict.fromkeys(result))


# ---------------------------------------------------------------------
# Grafo + Personalized PageRank (HippoRAG)
# ---------------------------------------------------------------------


def build_adjacency(session):
    """Grafo desde kag_relations: {entity_id: {neighbor_id: weight}}.

    Simétrico: cada relación aporta arista en ambos sentidos, ponderada por
    frecuencia.
    """
    rows = session.execute(
        text("SELECT source_entity_id, target_entity_id FROM kag_relations")
    ).fetchall()
    adj = {}
    for r in rows:
        s, t = r.source_entity_id, r.target_entity_id
        adj.setdefault(s, {})
        adj.setdefault(t, {})
        adj[s][t] = adj[s].get(t, 0) + 1
        adj[t][s] = adj[t].get(s, 0) + 1
    return adj


def personalized_pagerank(adjacency, seed, alpha=0.15, max_iter=50, tol=1e-6):
    """Personalized PageRank (HippoRAG): v = (1-α)·seed + α·Mᵀ·v.

    `seed` es una lista de entity_ids (semilla uniforme). Devuelve dict
    {entity_id: score}. Con semilla vacía devuelve {} (degradación natural:
    solo vector search).
    """
    if not seed:
        return {}
    nodes = set(adjacency.keys())
    nodes.update(seed)
    nodes = sorted(nodes)
    if not nodes:
        return {}
    n = len(nodes)
    idx = {node: i for i, node in enumerate(nodes)}

    # Matriz de transición M (column-stochastic): M[i][j] = P(j -> i).
    M = [[0.0] * n for _ in range(n)]
    for node in nodes:
        neighbors = adjacency.get(node, {})
        total = sum(neighbors.values())
        if total == 0:
            continue
        for nb, w in neighbors.items():
            M[idx[nb]][idx[node]] = w / total

    # Semilla: uniforme sobre las entidades matcheadas.
    seed_vec = [0.0] * n
    for s in seed:
        if s in idx:
            seed_vec[idx[s]] += 1.0
    ssum = sum(seed_vec)
    if ssum > 0:
        seed_vec = [v / ssum for v in seed_vec]
    else:
        seed_vec = [1.0 / n] * n

    v = [1.0 / n] * n
    for _ in range(max_iter):
        v_new = [
            (1 - alpha) * seed_vec[i] + alpha * sum(M[i][j] * v[j] for j in range(n))
            for i in range(n)
        ]
        diff = sum(abs(a - b) for a, b in zip(v_new, v))
        v = v_new
        if diff < tol:
            break
    return {nodes[i]: v[i] for i in range(n)}


def ppr_entity_selection(
    scores: dict, min_ratio: float = PPR_MIN_RATIO, z: float = PPR_Z
) -> list:
    """Entidades PPR 'cerca' de la semilla — umbral de cercanía en el grafo.

    El score PPR depende del tamaño del grafo, de la distribución de grados
    y de alpha: un umbral absoluto fijo no funciona. Piso doble
    escala-agnóstico:
      - relativo al máximo: score >= min_ratio × max_score. La cola
        power-law del PPR cae rápido; el ratio captura el codo natural
        (min_ratio=0.02 ≈ semilla + 2 saltos con alpha=0.15).
      - estadístico (solo grafos grandes, n >= 100): score >= mean + z*std.
        En grafos pequeños la media es significativa y este término
        sobre-filtra; en grafos grandes separa la señal de la cola.

    Devuelve lista de entity_ids ordenados por score desc.
    """
    if not scores:
        return []
    vals = list(scores.values())
    max_s = max(vals)
    floor = min_ratio * max_s
    if len(vals) >= 100:
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        floor = max(floor, mean + z * std)
    return [eid for eid, s in sorted(scores.items(), key=lambda x: -x[1]) if s >= floor]


# ---------------------------------------------------------------------
# Recuperación
# ---------------------------------------------------------------------


def vector_search(session, query_embedding, top_k):
    """pgvector <=> (coseno). Devuelve lista de (chunk_id, score).

    Solo chunks de documentos 'ready': los docs pending/failed (proceso
    interrumpido) no deben contaminar los resultados (degradación elegante).
    """
    q = embedding_to_sql(query_embedding)
    rows = session.execute(
        text(
            "SELECT c.id, 1 - (c.embedding <=> CAST(:q AS vector)) AS score "
            "FROM kag_chunks c "
            "JOIN kag_documents d ON d.id = c.doc_id "
            "WHERE c.embedding IS NOT NULL AND d.status = 'ready' "
            "ORDER BY c.embedding <=> CAST(:q AS vector) LIMIT :top_k"
        ),
        {"q": q, "top_k": top_k},
    ).fetchall()
    return [(r.id, float(r.score)) for r in rows]


# ---------------------------------------------------------------------
# Búsqueda híbrida: densa + léxica (FTS) + RRF (opt 4)
# ---------------------------------------------------------------------


def rrf_merge(*ranked_lists, k=60, top_k=20):
    """Fusión de rankings por Reciprocal Rank Fusion (RRF).

    Cada lista es una secuencia de (id, score) ordenada por relevancia
    (la posición 1-based es el rank). Acepta N listas (densa, léxica,
    regex, PPR...): cada capa aporta 1/(k+rank) por documento. Devuelve
    lista de (id, rrf_score) ordenada desc, limitada a top_k.
    """
    scores = {}
    for ranked in ranked_lists:
        for rank, (cid, _score) in enumerate(ranked, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])[:top_k]


def fts_search(session, query_text, top_k):
    """Búsqueda léxica con FTS de Postgres (ts_rank_cd sobre content_tsv).

    Requiere la migración 0014 (columna generada content_tsv + índice GIN).
    Config 'simple' a propósito: agnóstica de idioma (el corpus es
    multilingüe) y sin stemming (ideal para términos exactos: acrónimos,
    códigos, nombres propios). Query vacía o sin tokens → [] (sin error).
    """
    if not query_text or not query_text.strip():
        return []
    rows = session.execute(
        text(
            "SELECT c.id, ts_rank_cd(c.content_tsv, plainto_tsquery('simple', :q)) "
            "AS score FROM kag_chunks c "
            "JOIN kag_documents d ON d.id = c.doc_id "
            "WHERE c.content_tsv @@ plainto_tsquery('simple', :q) "
            "AND d.status = 'ready' "
            "ORDER BY score DESC LIMIT :top_k"
        ),
        {"q": query_text, "top_k": top_k},
    ).fetchall()
    return [(r.id, float(r.score)) for r in rows]


def hybrid_search(session, query_text, query_embedding, top_k, rrf_k=60, verbose=False):
    """Búsqueda híbrida: densa (pgvector) + léxica (FTS) + RRF.

    Si query_embedding es None (embeddings no disponibles), degrada a solo
    FTS. Si la migración 0014 no está aplicada (columna content_tsv ausente,
    ProgrammingError), degrada a solo búsqueda densa. Otros errores se
    propagan al caller (ask() los degrada a vec_hits=[]).
    """
    dense_hits = []
    if query_embedding is not None:
        dense_hits = vector_search(session, query_embedding, top_k)
    try:
        sparse_hits = fts_search(session, query_text, top_k)
    except ProgrammingError as exc:  # migración 0014 sin aplicar
        session.rollback()
        if verbose:
            print(f"[KAG] ⚠ FTS no disponible ({exc}); solo búsqueda densa.")
        return dense_hits
    if not dense_hits:
        return sparse_hits
    return rrf_merge(dense_hits, sparse_hits, k=rrf_k, top_k=top_k)


def chunks_for_entities(session, entity_ids, top_n):
    """Chunks de las entidades dadas, ordenados por frecuencia de mención."""
    if not entity_ids:
        return []
    rows = session.execute(
        text(
            "SELECT c.id, c.doc_id, c.section_path, c.content, d.doc_path, "
            "COUNT(*) AS mentions "
            "FROM kag_chunks c "
            "JOIN kag_entities e ON e.chunk_id = c.id "
            "JOIN kag_documents d ON d.id = c.doc_id "
            "WHERE e.id IN :ids AND d.status = 'ready' "
            "GROUP BY c.id, d.doc_path "
            "ORDER BY mentions DESC LIMIT :top_n"
        ).bindparams(bindparam("ids", expanding=True)),
        {"ids": list(entity_ids), "top_n": top_n},
    ).fetchall()
    return [
        {
            "chunk_id": r.id,
            "doc_id": r.doc_id,
            "section_path": r.section_path,
            "content": r.content,
            "doc_path": r.doc_path,
        }
        for r in rows
    ]


def subgraph_triples(session, entity_ids, limit=25):
    """Tripletas (source_name, relation_type, target_name, doc_path)."""
    if not entity_ids:
        return []
    rows = session.execute(
        text(
            "SELECT se.name AS source_name, r.relation_type, te.name AS target_name, "
            "d.doc_path "
            "FROM kag_relations r "
            "JOIN kag_entities se ON se.id = r.source_entity_id "
            "JOIN kag_entities te ON te.id = r.target_entity_id "
            "JOIN kag_documents d ON d.id = r.doc_id "
            "WHERE (r.source_entity_id IN :ids OR r.target_entity_id IN :ids) "
            "AND d.status = 'ready' "
            "LIMIT :limit"
        ).bindparams(bindparam("ids", expanding=True)),
        {"ids": list(entity_ids), "limit": limit},
    ).fetchall()
    return [
        {
            "source": r.source_name,
            "type": r.relation_type,
            "target": r.target_name,
            "doc": r.doc_path,
        }
        for r in rows
    ]


def figures_for_chunks(session, chunk_ids):
    """Figuras de los chunks dados."""
    if not chunk_ids:
        return []
    rows = session.execute(
        text(
            "SELECT image_path, caption, description FROM kag_figures "
            "WHERE chunk_id IN :ids"
        ).bindparams(bindparam("ids", expanding=True)),
        {"ids": list(chunk_ids)},
    ).fetchall()
    return [
        {"image_path": r.image_path, "caption": r.caption, "description": r.description}
        for r in rows
    ]


def doc_summaries(session, doc_ids):
    """Resúmenes de los documentos dados (fuente secundaria)."""
    if not doc_ids:
        return []
    rows = session.execute(
        text(
            "SELECT doc_path, summary FROM kag_documents "
            "WHERE id IN :ids AND summary != ''"
        ).bindparams(bindparam("ids", expanding=True)),
        {"ids": list(doc_ids)},
    ).fetchall()
    return [{"doc_path": r.doc_path, "summary": r.summary} for r in rows]


# ---------------------------------------------------------------------
# Threshold de relevancia + ventana de contexto
# ---------------------------------------------------------------------


def apply_relevance_threshold(hits, min_ratio=MIN_SCORE_RATIO, min_abs=MIN_ABS_SCORE):
    """Filtra hits (id, score) por relevancia relativa y absoluta.

    Escala-agnóstico: funciona con RRF (~0.01-0.03), coseno ([0,1]) o
    ts_rank (sin cota). Se descartan los resultados con score <
    min_ratio × max_score (cola larga irrelevante) o < min_abs (sin señal).
    """
    if not hits:
        return []
    max_score = max(s for _, s in hits)
    if max_score <= 0:
        return []
    floor = max(min_abs, min_ratio * max_score)
    return [(cid, s) for cid, s in hits if s >= floor]


def expand_chunk_window(session, chunk, window=CONTEXT_WINDOW):
    """Expande un chunk ancla con sus ±window vecinos del mismo documento.

    Devuelve lista de dicts (ancla primero, luego vecinos por chunk_index)
    con la misma forma que los chunks de ask(): chunk_id, doc_id,
    section_path, content, chunk_index, doc_path, score, is_anchor.
    """
    rows = session.execute(
        text(
            "SELECT c.id, c.doc_id, c.section_path, c.content, c.chunk_index, "
            "d.doc_path FROM kag_chunks c "
            "JOIN kag_documents d ON d.id = c.doc_id "
            "WHERE c.doc_id = :doc AND c.chunk_index BETWEEN :lo AND :hi "
            "AND d.status = 'ready' ORDER BY c.chunk_index"
        ),
        {
            "doc": chunk["doc_id"],
            "lo": chunk["chunk_index"] - window,
            "hi": chunk["chunk_index"] + window,
        },
    ).fetchall()
    out = []
    for r in rows:
        out.append(
            {
                "chunk_id": r.id,
                "doc_id": r.doc_id,
                "section_path": r.section_path,
                "content": r.content,
                "chunk_index": r.chunk_index,
                "doc_path": r.doc_path,
                "score": chunk.get("score", 0.0),
                "is_anchor": r.id == chunk["chunk_id"],
            }
        )
    return out


def _group_chunks_with_window(
    session, chunks, window=CONTEXT_WINDOW, max_chunks=MAX_CONTEXT_CHUNKS
):
    """Agrupa los chunks recuperados con su ventana ±window.

    Cada grupo = ancla + vecinos (orden por chunk_index). Los grupos se
    ordenan por el score del ancla (importancia). Dedup global por
    chunk_id: un chunk que ya está en un grupo de mayor importancia no se
    repite. Cap total en max_chunks (los grupos de menor importancia se
    truncan).
    """
    groups = []
    seen = set()
    for c in chunks:
        group = expand_chunk_window(session, c, window=window)
        # Marcar el ancla: el primero de la ventana con is_anchor=True.
        group = [g for g in group if g["chunk_id"] not in seen or g["is_anchor"]]
        if not group:
            continue
        # Dedup dentro del grupo (un vecino puede ser ancla de otro grupo).
        deduped = []
        for g in group:
            if g["chunk_id"] in seen and not g["is_anchor"]:
                continue
            seen.add(g["chunk_id"])
            deduped.append(g)
        if deduped:
            groups.append(deduped)
    # Ordenar grupos por score del ancla desc.
    groups.sort(key=lambda g: g[0].get("score", 0.0), reverse=True)
    # Cap total de chunks.
    flat = []
    for g in groups:
        for c in g:
            if len(flat) >= max_chunks:
                break
            flat.append(c)
        if len(flat) >= max_chunks:
            break
    return flat


# ---------------------------------------------------------------------
# Búsqueda textual dirigida por el LLM crítico (regex / términos exactos)
# ---------------------------------------------------------------------

CRITIC_PROMPT = """Eres un crítico de búsqueda. Dada una pregunta, decide si
contiene términos EXACTOS que requieren búsqueda textual (regex/FTS) en vez
de búsqueda semántica: nombres propios, países, ciudades, organizaciones,
códigos alfanuméricos (CVE-2024-3094, SKU-123), acrónimos, fechas, cifras,
identificadores o términos técnicos raros.

Devuelve SOLO JSON:
{{"needs_regex": true/false, "terms": ["término1", "término2"]}}

- needs_regex: true si hay al menos un término exacto que buscar.
- terms: los términos exactos (máx 5), tal como aparecen en la pregunta.
- Si no hay términos exactos, devuelve {{"needs_regex": false, "terms": []}}.

Pregunta: {query}
"""

# Heurística determinista de respaldo: tokens con mayúscula inicial o
# códigos alfanuméricos (el LLM crítico puede fallar o no estar disponible).
_REGEX_TERM_RE = re.compile(
    r"(?:[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]+(?:\s+[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ]+)*)"
    r"|(?:[A-Z0-9]+[-_][A-Za-z0-9-]+)"
)


def _deterministic_regex_terms(query: str) -> list:
    """Extrae términos exactos por heurística (fallback del LLM crítico)."""
    terms = []
    for m in _REGEX_TERM_RE.finditer(query):
        t = m.group(0).strip()
        if t and t.lower() not in _QUERY_STOPWORDS and len(t) > 2:
            terms.append(t)
    return terms[:5]


def critic_regex_search(session, query, top_k=10, verbose=False):
    """El LLM crítico decide si hace falta búsqueda textual y la aplica.

    Devuelve (hits, terms):
      - hits: lista de (chunk_id, score) de la búsqueda regex/FTS sobre los
        términos exactos (pocos chunks, alta precisión).
      - terms: los términos exactos usados — también alimentan el entity
        linking + PPR (el grafo se aplica sobre las keywords de la pregunta
        Y sobre las del crítico).

    Si el LLM falla o no hay términos, devuelve ([], []) (el flujo normal
    sigue). Degradación natural: la heurística determinista cubre el caso
    sin LLM.
    """
    terms = None
    settings = load_settings(session)
    retries = int(getattr(settings, "llm_retries", 3) or 3) if settings else 3
    fallback = getattr(settings, "fallback_model", None) if settings else None
    try:
        text_out, _model, _used_fallback = call_with_retries(
            session,
            prompt=CRITIC_PROMPT.format(query=query),
            system="Eres un crítico de búsqueda. Devuelve JSON válido.",
            model_size="small",
            response_format={"type": "json_object"},
            retries=retries,
            fallback_model=fallback,
        )
        data = parse_llm_output(text_out)
        if data.get("needs_regex"):
            terms = [
                str(t).strip() for t in (data.get("terms") or []) if str(t).strip()
            ]
    except Exception:  # noqa: BLE001 — LLM no disponible: heurística
        pass
    if not terms:
        terms = _deterministic_regex_terms(query)
    if not terms:
        return [], []
    if verbose:
        print(f"[KAG] 🔍 Crítico: búsqueda textual con términos {terms}")

    # Búsqueda FTS por cada término (config 'simple': agnóstica de idioma,
    # sin stemming — ideal para códigos, acrónimos y nombres propios).
    hits = []
    seen = set()
    for t in terms:
        rows = session.execute(
            text(
                "SELECT c.id, ts_rank_cd(c.content_tsv, plainto_tsquery('simple', :q)) "
                "AS score FROM kag_chunks c "
                "JOIN kag_documents d ON d.id = c.doc_id "
                "WHERE c.content_tsv @@ plainto_tsquery('simple', :q) "
                "AND d.status = 'ready' ORDER BY score DESC LIMIT :top_k"
            ),
            {"q": t, "top_k": top_k},
        ).fetchall()
        for r in rows:
            if r.id not in seen:
                seen.add(r.id)
                hits.append((r.id, float(r.score)))
    return hits[:top_k], terms


# ---------------------------------------------------------------------
# Ensamblado + respuesta
# ---------------------------------------------------------------------


def assemble_context(chunks, triples, figures, summaries, query, history=None):
    """Ensambla el bloque de contexto para la respuesta final.

    Los chunks llegan ya agrupados con su ventana (ancla + vecinos) y
    ordenados por importancia. Cada chunk muestra su procedencia exacta
    (doc, sección, chunk_index) y si es resultado directo o contexto
    adyacente. `history` (opcional) es una lista de dicts {"role",
    "content"} del historial de conversación — se incluye como sección
    informativa (preparado, aún sin probar con Docker).
    """
    parts = []
    if history:
        parts.append("--- HISTORIAL DE CONVERSACIÓN (referencia) ---")
        for turn in history[-6:]:
            role = turn.get("role", "?").upper()
            content = turn.get("content", "")
            parts.append(f"[{role}] {content}")
        parts.append("")
    parts.append("--- FRAGMENTOS RECUPERADOS (orden de importancia) ---")
    if chunks:
        for i, c in enumerate(chunks, start=1):
            doc = c.get("doc_path", "?")
            section = c.get("section_path", "") or "(sin sección)"
            idx = c.get("chunk_index", "?")
            score = c.get("score", 0.0)
            marker = "RESULTADO" if c.get("is_anchor", True) else "contexto"
            parts.append(
                f"[{i}] {marker} | doc: {doc} | sección: {section} | "
                f"chunk {idx} | score: {score:.4f}"
            )
            parts.append(c.get("content", ""))
            parts.append("")
    else:
        parts.append("(sin fragmentos recuperados)")
        parts.append("")
    parts.append("--- SUBGRAFO DE ENTIDADES ---")
    if triples:
        for t in triples:
            parts.append(
                f"({t['source']}) -[{t['type']}]-> ({t['target']}) [doc: {t['doc']}]"
            )
    else:
        parts.append("(sin tripletas)")
    parts.append("")
    parts.append("--- FIGURAS ---")
    if figures:
        for f in figures:
            parts.append(f"[{f['image_path']}] {f['description']}")
    else:
        parts.append("(sin figuras)")
    parts.append("")
    parts.append("--- RESUMENES DE DOCUMENTO (referencia secundaria) ---")
    if summaries:
        for s in summaries:
            parts.append(f"[{s['doc_path']}] {s['summary']}")
    else:
        parts.append("(sin resúmenes)")
    return "\n".join(parts)


ANSWER_SYSTEM = (
    "Eres un asistente de conocimiento. Responde la pregunta del usuario "
    "usando SOLO el contexto proporcionado. Si el contexto no contiene la "
    "respuesta, dilo claramente. Cita los documentos cuando sea posible. "
    "Responde en el idioma de la pregunta."
)

ANSWER_PROMPT = """Contexto:
{context}

Pregunta: {query}

Responde con precisión basándote en el contexto."""


def generate_answer(session, context, query):
    """Respuesta final con el LLM grande. Si falla, devuelve el contexto crudo."""
    settings = load_settings(session)
    retries = int(getattr(settings, "llm_retries", 3) or 3) if settings else 3
    fallback = getattr(settings, "fallback_model", None) if settings else None
    try:
        text_out, _model, _used_fallback = call_with_retries(
            session,
            prompt=ANSWER_PROMPT.format(context=context, query=query),
            system=ANSWER_SYSTEM,
            model_size="large",
            response_format=None,
            retries=retries,
            fallback_model=fallback,
        )
        return text_out
    except Exception as exc:  # noqa: BLE001 — degradación: contexto crudo
        return (
            f"[LLM no disponible — contexto crudo]\n\n{context}\n\n"
            f"(Nota: el LLM de respuesta falló: {exc})"
        )


# ---------------------------------------------------------------------
# Flujo completo
# ---------------------------------------------------------------------


def ask(session, query, top_k=8, global_top_k=20, verbose=True, history=None):
    """Flujo completo de consulta KAG (§3.1 del diseño). Devuelve la respuesta.

    `history` (opcional) es una lista de dicts {"role", "content"} del
    historial de conversación. PREPARADO pero aún sin probar con Docker:
    se incluye como sección informativa en el contexto, no modifica la
    búsqueda.
    """
    if verbose:
        print(f"\n🔎 Pregunta: {query}")

    # 1. Clasificar
    qtype = classify_query(query)
    k = global_top_k if qtype == "global" else top_k
    if verbose:
        print(f"[KAG] Clasificación: {qtype} (top_k={k})")

    # 2. Búsqueda híbrida (densa + FTS + RRF)
    q_emb = None
    try:
        # Import perezoso: src.embeddings importa src.db.session (que lee .env
        # al importar) — debe ocurrir DESPUÉS de _fix_db_host().
        from src.embeddings import embed_text

        q_emb = embed_text(query, input_type="query")
    except Exception as exc:  # noqa: BLE001 — degradación natural
        session.rollback()  # la transacción queda abortada tras el error
        if verbose:
            print(f"[KAG] ⚠ Embeddings no disponibles ({exc}); solo FTS.")
    try:
        vec_hits = hybrid_search(session, query, q_emb, k, verbose=verbose)
    except Exception as exc:  # noqa: BLE001 — degradación natural
        session.rollback()  # la transacción queda abortada tras el error
        if verbose:
            print(f"[KAG] ⚠ Búsqueda híbrida falló: {exc}")
        vec_hits = []
    # Threshold de relevancia: descarta la cola larga irrelevante.
    vec_hits = apply_relevance_threshold(vec_hits)
    if verbose:
        print(f"[KAG] Búsqueda híbrida: {len(vec_hits)} chunks (tras threshold)")
        for cid, score in vec_hits[:5]:
            print(f"    - chunk {cid}: score {score:.4f}")

    # 2.5. LLM crítico: términos exactos → búsqueda textual (regex/FTS).
    #      Devuelve (hits, terms): los términos también alimentan el entity
    #      linking + PPR — el grafo se aplica sobre las keywords de la
    #      pregunta Y sobre las del crítico.
    regex_hits, regex_terms = [], []
    try:
        regex_hits, regex_terms = critic_regex_search(
            session, query, top_k=k, verbose=verbose
        )
    except Exception as exc:  # noqa: BLE001 — degradación natural
        session.rollback()
        if verbose:
            print(f"[KAG] ⚠ Búsqueda textual del crítico falló: {exc}")
    if verbose and regex_hits:
        print(
            f"[KAG] Crítico: {len(regex_hits)} chunks textuales "
            f"(términos: {regex_terms})"
        )

    # 3. Entity linking (anclado + copresencia) sobre la pregunta Y los
    #    términos del crítico (una sola llamada con el texto combinado).
    link_text = query
    if regex_terms:
        link_text = f"{query} {' '.join(regex_terms)}"
    names = grounded_entity_linking(session, link_text)
    groups = match_entities_candidates(session, names)
    adj = build_adjacency(session) if groups else {}
    entity_ids = disambiguate_by_cooccurrence(
        session,
        groups,
        adjacency=adj,
        min_overlap=DISAMBIG_MIN_OVERLAP,
        margin=DISAMBIG_MARGIN,
    )
    if verbose:
        print(
            f"[KAG] Entity linking: {len(names)} nombres → "
            f"{len(entity_ids)} entidades (tras copresencia)"
        )

    # 4. PPR (HippoRAG)
    ppr_scores = {}
    if entity_ids:
        ppr_scores = personalized_pagerank(adj, entity_ids)
        if verbose:
            top_ppr = sorted(ppr_scores.items(), key=lambda x: -x[1])[:5]
            print(f"[KAG] PPR: {len(ppr_scores)} entidades rankeadas")
            for eid, score in top_ppr:
                print(f"    - entidad {eid}: {score:.4f}")

    # 4.5. Umbral de cercanía en el grafo: entidades PPR 'cerca' de la
    #      semilla (relativo al máximo + baseline estadístico). Reemplaza
    #      el top-10 fijo: si solo 3 entidades están cerca, no arrastra 7
    #      irrelevantes; si 20 están cerca, no descarta la mitad.
    ppr_entities = ppr_entity_selection(ppr_scores)
    if verbose:
        print(f"[KAG] PPR cercanas: {len(ppr_entities)} entidades (umbral relativo)")

    # 5. Merge + dedup: RRF sobre las tres capas (vector, regex, PPR) →
    #    una sola lista de chunks, sin duplicación. Cada capa aporta su
    #    rank; el RRF es escala-agnóstico (ts_rank, coseno y menciones no
    #    comparten escala).
    ppr_chunks = (
        chunks_for_entities(session, ppr_entities, top_n=10) if ppr_entities else []
    )
    merged_hits = rrf_merge(
        vec_hits,
        regex_hits,
        [(pc["chunk_id"], 0.0) for pc in ppr_chunks],
        k=60,
        top_k=len(vec_hits) + len(regex_hits) + len(ppr_chunks),
    )
    chunks = []
    for cid, score in merged_hits:
        row = session.execute(
            text(
                "SELECT c.doc_id, c.section_path, c.content, c.chunk_index, "
                "d.doc_path FROM kag_chunks c "
                "JOIN kag_documents d ON d.id = c.doc_id WHERE c.id = :id"
            ),
            {"id": cid},
        ).first()
        if row:
            chunks.append(
                {
                    "chunk_id": cid,
                    "doc_id": row.doc_id,
                    "section_path": row.section_path,
                    "content": row.content,
                    "chunk_index": row.chunk_index,
                    "doc_path": row.doc_path,
                    "score": score,
                }
            )
    if verbose:
        print(f"[KAG] Merge: {len(chunks)} chunks ancla (vector + regex + PPR, dedup)")

    # 5.5. Ventana de contexto: cada ancla se expande con sus ±CONTEXT_WINDOW
    #      vecinos del mismo doc (agrupador de chunks consecutivos). El
    #      resultado se ordena por importancia del ancla y se capa en
    #      MAX_CONTEXT_CHUNKS.
    chunks = _group_chunks_with_window(session, chunks)
    if verbose:
        n_anchors = sum(1 for c in chunks if c.get("is_anchor", True))
        print(
            f"[KAG] Ventana ±{CONTEXT_WINDOW}: {len(chunks)} chunks totales "
            f"({n_anchors} anclas) — cap {MAX_CONTEXT_CHUNKS}"
        )

    # 6. Subgrafo de tripletas (entidades PPR cercanas, no top-10 fijo)
    triples = subgraph_triples(session, ppr_entities or entity_ids, limit=25)
    if verbose:
        print(f"[KAG] Subgrafo: {len(triples)} tripletas")

    # 7. Figuras
    chunk_ids = [c["chunk_id"] for c in chunks]
    figures = figures_for_chunks(session, chunk_ids)
    if verbose:
        print(f"[KAG] Figuras: {len(figures)}")

    # 8. Resúmenes (fuente secundaria, siempre)
    doc_ids = list({c["doc_id"] for c in chunks})
    summaries = doc_summaries(session, doc_ids)
    if verbose:
        print(f"[KAG] Resúmenes: {len(summaries)} documentos")

    # 9. Ensamblar contexto
    context = assemble_context(
        chunks, triples, figures, summaries, query, history=history
    )
    if verbose:
        print("\n[KAG] Contexto ensamblado:")
        print(context[:2000])
        if len(context) > 2000:
            print("...")

    # 10. Respuesta
    answer = generate_answer(session, context, query)
    if verbose:
        print("\n[KAG] Respuesta final:")
    return answer


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def main() -> None:
    # Windows: la consola usa cp1252 y no imprime emojis — forzar UTF-8.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Consulta KAG del knowledge_repository."
    )
    parser.add_argument("query", help="Pregunta a responder.")
    parser.add_argument(
        "--top-k", type=int, default=8, help="Chunks vectoriales (consulta local)."
    )
    parser.add_argument(
        "--global-top-k", type=int, default=20, help="Chunks vectoriales (global)."
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True, help="Prints descriptivos."
    )
    args = parser.parse_args()

    _fix_db_host()
    from src.db.session import SessionLocal

    session = SessionLocal()
    try:
        answer = ask(
            session,
            args.query,
            top_k=args.top_k,
            global_top_k=args.global_top_k,
            verbose=args.verbose,
        )
        print(answer)
    finally:
        session.close()


if __name__ == "__main__":
    main()
