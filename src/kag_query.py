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
                "SELECT DISTINCT name FROM kag_entities "
                "WHERE name_norm LIKE :pat LIMIT 5"
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
                "SELECT DISTINCT name FROM kag_entities "
                "WHERE name_norm LIKE :pat LIMIT 5"
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
            text("SELECT id FROM kag_entities WHERE name_norm = :nn"),
            {"nn": nn},
        ).fetchall()
        group.extend(r.id for r in rows)
        if not group:
            rows = session.execute(
                text("SELECT id FROM kag_entities WHERE name_norm LIKE :pat LIMIT 5"),
                {"pat": f"%{nn}%"},
            ).fetchall()
            group.extend(r.id for r in rows)
        # Dedup dentro del grupo (el LIKE puede solaparse con el exacto).
        group = list(dict.fromkeys(group))
        if group:
            groups.append(group)
    return groups


def disambiguate_by_cooccurrence(session, groups: list, adjacency=None) -> list:
    """Desambiguación por copresencia en el grafo.

    Para cada mención con varios candidatos, elige el que comparte más
    vecinos (a 1 salto) con las entidades confirmadas (menciones con un
    solo candidato). Sin entidades confirmadas, conserva el primer
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
        else:
            best = max(
                g,
                key=lambda c: len(
                    confirmed_neighbors & set(adjacency.get(c, {}).keys())
                ),
                default=g[0],  # sin vecinos compartidos: primer candidato
            )
            result.append(best)
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


# ---------------------------------------------------------------------
# Recuperación
# ---------------------------------------------------------------------


def vector_search(session, query_embedding, top_k):
    """pgvector <=> (coseno). Devuelve lista de (chunk_id, score)."""
    q = embedding_to_sql(query_embedding)
    rows = session.execute(
        text(
            "SELECT id, 1 - (embedding <=> CAST(:q AS vector)) AS score "
            "FROM kag_chunks WHERE embedding IS NOT NULL "
            "ORDER BY embedding <=> CAST(:q AS vector) LIMIT :top_k"
        ),
        {"q": q, "top_k": top_k},
    ).fetchall()
    return [(r.id, float(r.score)) for r in rows]


# ---------------------------------------------------------------------
# Búsqueda híbrida: densa + léxica (FTS) + RRF (opt 4)
# ---------------------------------------------------------------------


def rrf_merge(dense_hits, sparse_hits, k=60, top_k=20):
    """Fusión de rankings por Reciprocal Rank Fusion (RRF).

    dense_hits/sparse_hits: listas de (id, score) ordenadas por relevancia
    (la posición 1-based es el rank). Devuelve lista de (id, rrf_score)
    ordenada desc, limitada a top_k.
    """
    scores = {}
    for rank, (cid, _score) in enumerate(dense_hits, start=1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    for rank, (cid, _score) in enumerate(sparse_hits, start=1):
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
            "SELECT id, ts_rank_cd(content_tsv, plainto_tsquery('simple', :q)) "
            "AS score FROM kag_chunks "
            "WHERE content_tsv @@ plainto_tsquery('simple', :q) "
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
            "WHERE e.id IN :ids "
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
            "WHERE r.source_entity_id IN :ids OR r.target_entity_id IN :ids "
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
# Ensamblado + respuesta
# ---------------------------------------------------------------------


def assemble_context(chunks, triples, figures, summaries, query):
    """Ensambla el bloque de contexto para la respuesta final."""
    parts = []
    parts.append("--- FRAGMENTOS RECUPERADOS (keywords + PPR) ---")
    if chunks:
        for c in chunks:
            doc = c.get("doc_path", "?")
            section = c.get("section_path", "")
            idx = c.get("chunk_index", "?")
            parts.append(f"[doc: {doc} | sección: {section} | chunk {idx}]")
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


def ask(session, query, top_k=8, global_top_k=20, verbose=True):
    """Flujo completo de consulta KAG (§3.1 del diseño). Devuelve la respuesta."""
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
    if verbose:
        print(f"[KAG] Búsqueda híbrida: {len(vec_hits)} chunks")
        for cid, score in vec_hits[:5]:
            print(f"    - chunk {cid}: score {score:.4f}")

    # 3. Entity linking (anclado + copresencia)
    names = grounded_entity_linking(session, query)
    groups = match_entities_candidates(session, names)
    adj = build_adjacency(session) if groups else {}
    entity_ids = disambiguate_by_cooccurrence(session, groups, adjacency=adj)
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

    # 5. Merge + dedup: vectoriales primero, luego PPR no incluidos
    chunks = []
    seen_chunk_ids = set()
    for cid, score in vec_hits:
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
            seen_chunk_ids.add(cid)
    if ppr_scores:
        top_ppr_ids = [
            eid for eid, _ in sorted(ppr_scores.items(), key=lambda x: -x[1])[:10]
        ]
        for pc in chunks_for_entities(session, top_ppr_ids, top_n=10):
            if pc["chunk_id"] not in seen_chunk_ids:
                chunks.append(pc)
                seen_chunk_ids.add(pc["chunk_id"])
    if verbose:
        print(f"[KAG] Merge: {len(chunks)} chunks finales")

    # 6. Subgrafo de tripletas
    top_entity_ids = (
        [eid for eid, _ in sorted(ppr_scores.items(), key=lambda x: -x[1])[:10]]
        if ppr_scores
        else entity_ids
    )
    triples = subgraph_triples(session, top_entity_ids, limit=25)
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
    context = assemble_context(chunks, triples, figures, summaries, query)
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
