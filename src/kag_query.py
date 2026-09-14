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

from src.kag.thresholds import auto_cutoff
from src.kag.thresholds import auto_margin as _auto_margin
from src.kag_ingest import (
    detect_language,
    embedding_to_sql,
    normalize_entity_name,
)
from src.llm.base import call_with_retries, load_settings, parse_llm_output

# ---------------------------------------------------------------------
# Prompt-as-code: task keys (specs en src/db/seed_kag_prompts.py)
# ---------------------------------------------------------------------

TASK_GROUNDED_ENTITIES = "kag_grounded_entities"
TASK_CRITIC_REGEX = "kag_critic_regex"
TASK_CRITIC_LINKING = "kag_critic_linking"
TASK_QUERY_ANSWER = "kag_query_answer"

# System prompts cortos actuales — fallback EXACTO de hoy cuando no hay
# artefacto compilado (tests sin DB: get_active_prompt devuelve None).
GROUNDED_ENTITIES_SYSTEM_SHORT = "Eres un selector de entidades. Devuelve JSON válido."
CRITIC_SYSTEM_SHORT = "Eres un crítico de búsqueda. Devuelve JSON válido."
COMBINED_SYSTEM_SHORT = (
    "Eres un crítico de búsqueda y selector de entidades. Devuelve JSON válido."
)
ANSWER_SYSTEM_SHORT = (
    "Eres un asistente de conocimiento. Responde la pregunta del usuario "
    "usando SOLO el contexto proporcionado. Si el contexto no contiene la "
    "respuesta, dilo claramente. Cita los documentos cuando sea posible. "
    "Responde en el idioma de la pregunta."
)


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

# Umbrales automáticos (Tier 1): Kneedle para codos (PPR, relevancia) y
# mediana de gaps para márgenes (desambiguación, canonicalización). Con
# AUTO_THRESHOLDS = True los umbrales fijos de arriba son solo el FALLBACK
# cuando la distribución no tiene codo claro o hay <2 gaps. Con False se
# vuelve al comportamiento fijo de siempre.
AUTO_THRESHOLDS = True

# ── Reranker cross-encoder (OPCIONAL, default OFF) ─────────────────────
# El RRF fusiona por posición pero ignora la afinidad semántica exacta
# entre la query y el texto. Un cross-encoder (p. ej. jina-reranker-v2)
# reordena el top-N antes de expandir la ventana ±CONTEXT_WINDOW, filtrando
# falsos positivos del FTS/PPR. Es una dependencia pesada (sentence-
# transformers + ~1GB de modelo), por eso está DESACTIVADO por defecto:
# RERANK_ENABLED = True lo activa con lazy-load y degradación natural
# (si el modelo no carga, el flujo sigue sin rerankear).
RERANK_ENABLED = False
RERANK_MODEL = "jinaai/jina-reranker-v2-base-multilingual"
RERANK_TOP_N = 30

_reranker = None
_reranker_lock = threading.Lock()

# Caché del grafo de entidades (build_adjacency) por versión en DB. La
# versión la mantiene la migración 0016_kag_graph_version: un trigger sobre
# kag_relations la incrementa en cada ingesta (INSERT/UPDATE/DELETE). En
# consulta se lee la versión (O(1), una fila) y se reutiliza el dict si no
# cambió — espejo del índice HNSW, que Postgres mantiene incrementalmente.
# Si la migración no está aplicada, version = None y se degrada al
# comportamiento viejo (reconstruir en cada llamada, sin cachear).
_adjacency_cache: dict | None = None
_adjacency_version: int = -1

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


def _get_system_prompt(session, model_name: str, task_key: str, fallback: str) -> str:
    """System prompt desde el artefacto compilado, con fallback a la constante.

    Import lazy dentro de try/except: si no hay DB, no hay artefacto o el
    compilador no existe, se devuelve la constante actual (comportamiento
    exacto de hoy — los tests usan sesiones falsas sin DB).
    """
    try:
        from src.llm.compiler import get_active_prompt

        artifact = get_active_prompt(session, model_name, task_key)
        if artifact is not None and artifact.prompt_text:
            return artifact.prompt_text
    except Exception:  # noqa: BLE001 — degradación natural
        pass
    return fallback


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
    small_model = getattr(settings, "small_model", None) if settings else None
    try:
        text_out, _model, _used_fallback = call_with_retries(
            session,
            prompt=GROUNDED_ENTITIES_PROMPT.format(
                candidates="\n".join(f"- {c}" for c in candidates),
                query=query,
            ),
            system=_get_system_prompt(
                session,
                small_model,
                TASK_GROUNDED_ENTITIES,
                GROUNDED_ENTITIES_SYSTEM_SHORT,
            ),
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
    auto: bool = AUTO_THRESHOLDS,
    verbose: bool = False,
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

    Con `auto=True` el margen se deriva de la distribución observada:
    se puntúan TODAS las menciones ambiguas, se recolectan los gaps
    (best - second) de las que tienen señal (best >= min_overlap) y el
    margen es la MEDIANA de esos gaps (robusta: la mitad de las menciones
    con señal se resuelven, la otra mitad queda ambigua). Con <2 gaps se
    usa el margen fijo (fallback).
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

    def _score_group(g):
        scored = []
        for c in g:
            shared = len(confirmed_neighbors & set(adjacency.get(c, {}).keys()))
            denom = min(len(adjacency.get(c, {})), len(confirmed_neighbors)) or 1
            scored.append((c, shared / denom))
        scored.sort(key=lambda x: -x[1])
        return scored

    # Pasada 1: puntuar todas las menciones ambiguas y recolectar los gaps
    # de las que tienen señal (best >= min_overlap). El margen automático es
    # la mediana de esos gaps — la distribución manda, no una constante.
    gaps = []
    scored_groups = []
    for g in groups:
        if len(g) == 1:
            scored_groups.append(None)
            continue
        scored = _score_group(g)
        scored_groups.append(scored)
        best_s = scored[0][1]
        second_s = scored[1][1] if len(scored) > 1 else 0.0
        if best_s >= min_overlap:
            gaps.append(best_s - second_s)
    if auto:
        margin = _auto_margin(gaps, 0.5, floor=margin)
        if verbose:
            print(
                f"[KAG] Desambiguación: margen automático = {margin:.4f} "
                f"(mediana de {len(gaps)} gaps)"
            )

    # Pasada 2: decidir con el margen (automático o fijo).
    result = []
    for g, scored in zip(groups, scored_groups):
        if scored is None:
            result.append(g[0])
            continue
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
    frecuencia. Cacheado por versión en DB (migración 0016): si la versión
    no cambió, devuelve el dict en memoria sin releer kag_relations. Si la
    migración no está aplicada (o la sesión no soporta el SELECT de versión),
    degrada al comportamiento viejo: reconstruir en cada llamada, sin
    cachear.
    """
    global _adjacency_cache, _adjacency_version

    try:
        version = session.execute(
            text("SELECT version FROM kag_graph_state WHERE id = 1")
        ).scalar()
    except Exception:
        # Migración 0016 sin aplicar, o sesión falsa de tests sin .scalar().
        version = None

    if (
        version is not None
        and version == _adjacency_version
        and _adjacency_cache is not None
    ):
        return _adjacency_cache

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

    if version is not None:
        _adjacency_cache = adj
        _adjacency_version = version
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


def ego_network(adjacency, seed, hops=2):
    """Subgrafo inducido de la vecindad de `hops` saltos de la semilla.

    Correr PPR sobre el grafo completo es O(N²) en el peor caso (matriz de
    transición densa) y crece con el corpus. El PPR de HippoRAG solo necesita
    la vecindad local de la semilla: con alpha=0.15, la masa de probabilidad
    decae ~×0.15 por salto, así que 2 saltos capturan la señal relevante.

    Devuelve un dict de adyacencia (misma forma que build_adjacency) con
    SOLO los nodos a <=hops saltos de la semilla y las aristas entre ellos.
    Con semilla vacía devuelve {}.
    """
    if not seed:
        return {}
    frontier = set(seed)
    seen = set(seed)
    for _ in range(hops):
        nxt = set()
        for node in frontier:
            nxt.update(adjacency.get(node, {}).keys())
        nxt -= seen
        seen |= nxt
        frontier = nxt
        if not frontier:
            break
    sub = {}
    for node in seen:
        neighbors = {nb: w for nb, w in adjacency.get(node, {}).items() if nb in seen}
        if neighbors:
            sub[node] = neighbors
    return sub


def ppr_entity_selection(
    scores: dict,
    min_ratio: float = PPR_MIN_RATIO,
    z: float = PPR_Z,
    auto: bool = AUTO_THRESHOLDS,
    verbose: bool = False,
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

    Con `auto=True` (default) el corte es el CODO de la curva (Kneedle,
    Satopää et al. 2011): normaliza rank→[0,1] y score→[0,1], encuentra el
    punto de máxima curvatura y corta ahí. El codo manda — el piso relativo
    es solo el fallback cuando la curva no tiene codo claro (plana, corta o
    casi lineal). Devuelve lista de entity_ids ordenados por score desc.
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
    sorted_items = sorted(scores.items(), key=lambda x: -x[1])
    if auto:
        cutoff = auto_cutoff(vals)
        if cutoff is not None:
            if verbose:
                print(f"[KAG] PPR: codo automático en score {cutoff:.4f}")
            return [eid for eid, s in sorted_items if s >= cutoff]
        if verbose:
            print(f"[KAG] PPR: sin codo claro → piso relativo {floor:.4f}")
    return [eid for eid, s in sorted_items if s >= floor]


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


def chunks_by_ids(session, chunk_ids):
    """Chunks por ids en UNA consulta, preservando el orden de entrada.

    Reemplaza el loop N+1 de `SELECT chunk WHERE id=:id` por merged hit:
    `WHERE id = ANY(:ids)` + `array_position(:ids, id)` para ordenar por la
    posición en la lista de entrada (el orden de importancia del RRF).
    Devuelve lista de dicts con la misma forma que chunks_for_entities.
    """
    if not chunk_ids:
        return []
    ids = list(dict.fromkeys(chunk_ids))
    rows = session.execute(
        text(
            "SELECT c.id, c.doc_id, c.section_path, c.content, c.chunk_index, "
            "d.doc_path FROM kag_chunks c "
            "JOIN kag_documents d ON d.id = c.doc_id "
            "WHERE c.id = ANY(:ids) AND d.status = 'ready' "
            "ORDER BY array_position(:ids, c.id)"
        ).bindparams(bindparam("ids", expanding=True)),
        {"ids": ids},
    ).fetchall()
    by_id = {r.id: r for r in rows}
    out = []
    for cid in ids:
        r = by_id.get(cid)
        if r is None:
            continue
        out.append(
            {
                "chunk_id": r.id,
                "doc_id": r.doc_id,
                "section_path": r.section_path,
                "content": r.content,
                "chunk_index": r.chunk_index,
                "doc_path": r.doc_path,
            }
        )
    return out


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


def apply_relevance_threshold(
    hits,
    min_ratio=MIN_SCORE_RATIO,
    min_abs=MIN_ABS_SCORE,
    auto: bool = AUTO_THRESHOLDS,
    verbose: bool = False,
):
    """Filtra hits (id, score) por relevancia relativa y absoluta.

    Escala-agnóstico: funciona con RRF (~0.01-0.03), coseno ([0,1]) o
    ts_rank (sin cota). Se descartan los resultados con score <
    min_ratio × max_score (cola larga irrelevante) o < min_abs (sin señal).

    Con `auto=True` (default) el corte es el CODO de la curva de scores
    (Kneedle): el punto donde la relevancia cae abruptamente. El codo manda
    — el piso relativo es solo el fallback cuando no hay codo claro.
    """
    if not hits:
        return []
    max_score = max(s for _, s in hits)
    if max_score <= 0:
        return []
    floor = max(min_abs, min_ratio * max_score)
    if auto:
        cutoff = auto_cutoff([s for _, s in hits])
        if cutoff is not None:
            if verbose:
                print(f"[KAG] Relevancia: codo automático en score {cutoff:.4f}")
            return [(cid, s) for cid, s in hits if s >= cutoff]
        if verbose:
            print(f"[KAG] Relevancia: sin codo claro → piso relativo {floor:.4f}")
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


def _get_reranker():
    """Carga el cross-encoder perezosamente (una vez, cacheado)."""
    global _reranker
    if _reranker is not None:
        return _reranker
    with _reranker_lock:
        if _reranker is not None:
            return _reranker
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(RERANK_MODEL)
        return _reranker


def rerank_chunks(query, chunks, top_n=RERANK_TOP_N):
    """Reordena los chunks ancla por relevancia semántica (cross-encoder).

    Solo actúa si RERANK_ENABLED = True. Reordena el top-N por la afinidad
    exacta (query, chunk) y actualiza el score del ancla con la puntuación
    del reranker — así `_group_chunks_with_window` ordena los grupos por la
    nueva relevancia. Devuelve la misma lista (mismo orden) si el modelo no
    está disponible o falla — degradación natural.
    """
    if not RERANK_ENABLED or not chunks:
        return chunks
    try:
        model = _get_reranker()
        pairs = [(query, c.get("content", "")) for c in chunks[:top_n]]
        scores = model.predict(pairs)
        ranked = sorted(zip(chunks[:top_n], scores), key=lambda x: -x[1])
        out = []
        for c, s in ranked:
            c["score"] = float(s)
            out.append(c)
        return out + chunks[top_n:]
    except Exception:  # noqa: BLE001 — degradación: sin rerank
        return chunks


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
    small_model = getattr(settings, "small_model", None) if settings else None
    try:
        text_out, _model, _used_fallback = call_with_retries(
            session,
            prompt=CRITIC_PROMPT.format(query=query),
            system=_get_system_prompt(
                session, small_model, TASK_CRITIC_REGEX, CRITIC_SYSTEM_SHORT
            ),
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

    # Búsqueda FTS de TODOS los términos en UNA consulta (config 'simple':
    # agnóstica de idioma, sin stemming — ideal para códigos, acrónimos y
    # nombres propios). Se construye un tsquery booleano OR: cada término se
    # escapa como frase literal entre comillas dobles ("término") y se une
    # con |. Las comillas dobles evitan que caracteres como '-' (operador
    # NOT de tsquery) o espacios rompan el parseo. El ranking usa ts_rank_cd
    # sobre el tsquery combinado; los chunks que matchean varios términos
    # puntúan más alto (suma de relevancia por término).
    tsq = " | ".join(f'"{t.replace(chr(34), chr(34) * 2)}"' for t in terms)
    rows = session.execute(
        text(
            "SELECT c.id, ts_rank_cd(c.content_tsv, to_tsquery('simple', :tsq)) "
            "AS score FROM kag_chunks c "
            "JOIN kag_documents d ON d.id = c.doc_id "
            "WHERE c.content_tsv @@ to_tsquery('simple', :tsq) "
            "AND d.status = 'ready' ORDER BY score DESC LIMIT :top_k"
        ),
        {"tsq": tsq, "top_k": top_k},
    ).fetchall()
    hits = [(r.id, float(r.score)) for r in rows]
    return hits[:top_k], terms


# ---------------------------------------------------------------------
# CRIT + EL fusionados (una sola llamada LLM)
# ---------------------------------------------------------------------

COMBINED_PROMPT = """Eres un crítico de búsqueda y selector de entidades.
Dada una pregunta:

1. Decide si contiene términos EXACTOS que requieren búsqueda textual
   (regex/FTS): nombres propios, países, ciudades, organizaciones, códigos
   alfanuméricos (CVE-2024-3094, SKU-123), acrónimos, fechas, cifras,
   identificadores o términos técnicos raros.
2. Selecciona las entidades canónicas SOLO entre los candidatos del grafo
   que se mencionan en la pregunta.

Candidatos del grafo:
{candidates}

Devuelve SOLO JSON:
{{"needs_regex": true/false, "terms": ["término1"], "entities": ["Entidad 1"]}}

- needs_regex: true si hay al menos un término exacto que buscar.
- terms: los términos exactos (máx 5), tal como aparecen en la pregunta.
- entities: las entidades de la lista de candidatos que se mencionan en la
  pregunta. Si ninguna, [].
- Si no hay términos exactos, devuelve {{"needs_regex": false, "terms": []}}.

Pregunta: {query}
"""


def critic_and_linking(session, query, top_k=10, verbose=False):
    """Fusiona el LLM crítico y el entity linking anclado en UNA llamada.

    Devuelve (regex_hits, regex_terms, names):
      - regex_hits: chunks por FTS sobre los términos exactos (si los hay).
      - regex_terms: términos exactos (alimentan el PPR).
      - names: entidades canónicas del pool (ancladas al grafo).

    Ahorra un round-trip LLM por consulta (CRIT + EL → 1 llamada). Si el
    LLM falla, degrada por separado: heurística determinista para los
    términos y pool determinista para las entidades (comportamiento viejo).
    """
    candidates = _noun_chunk_fallback(session, query)
    settings = load_settings(session)
    retries = int(getattr(settings, "llm_retries", 3) or 3) if settings else 3
    fallback = getattr(settings, "fallback_model", None) if settings else None
    small_model = getattr(settings, "small_model", None) if settings else None
    terms = None
    names = None
    try:
        text_out, _model, _used_fallback = call_with_retries(
            session,
            prompt=COMBINED_PROMPT.format(
                candidates="\n".join(f"- {c}" for c in candidates)
                or "(sin candidatos)",
                query=query,
            ),
            system=_get_system_prompt(
                session, small_model, TASK_CRITIC_LINKING, COMBINED_SYSTEM_SHORT
            ),
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
        raw_names = [
            str(e).strip() for e in (data.get("entities") or []) if str(e).strip()
        ]
        # Anclaje real: solo nombres que están en el pool (normalizados).
        pool_norm = {normalize_entity_name(c) for c in candidates}
        anchored = [n for n in raw_names if normalize_entity_name(n) in pool_norm]
        if anchored:
            names = anchored
    except Exception:  # noqa: BLE001 — LLM no disponible: degradación
        pass
    if not terms:
        terms = _deterministic_regex_terms(query)
    if names is None:
        names = candidates

    # Búsqueda FTS combinada (una consulta, OR de términos).
    regex_hits = []
    if terms:
        if verbose:
            print(f"[KAG] 🔍 Crítico: búsqueda textual con términos {terms}")
        tsq = " | ".join(f'"{t.replace(chr(34), chr(34) * 2)}"' for t in terms)
        rows = session.execute(
            text(
                "SELECT c.id, ts_rank_cd(c.content_tsv, to_tsquery('simple', :tsq)) "
                "AS score FROM kag_chunks c "
                "JOIN kag_documents d ON d.id = c.doc_id "
                "WHERE c.content_tsv @@ to_tsquery('simple', :tsq) "
                "AND d.status = 'ready' ORDER BY score DESC LIMIT :top_k"
            ),
            {"tsq": tsq, "top_k": top_k},
        ).fetchall()
        regex_hits = [(r.id, float(r.score)) for r in rows][:top_k]
    return regex_hits, terms, names


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

    ORDEN PARA PROMPT CACHING: el subgrafo y los resúmenes (bloques
    semi-estáticos, ordenados de forma determinista por doc) van ANTES de
    los fragmentos y figuras (dinámicos). Así el prefijo del prompt
    (system + subgrafo + resúmenes) es cacheable entre queries que
    comparten documentos; la cola dinámica (chunks + figuras) y la query
    del usuario quedan al final.
    """
    parts = []
    if history:
        parts.append("--- HISTORIAL DE CONVERSACIÓN (referencia) ---")
        for turn in history[-6:]:
            role = turn.get("role", "?").upper()
            content = turn.get("content", "")
            parts.append(f"[{role}] {content}")
        parts.append("")
    parts.append("--- SUBGRAFO DE ENTIDADES ---")
    if triples:
        for t in sorted(
            triples,
            key=lambda t: (t.get("doc", ""), t.get("source", ""), t.get("target", "")),
        ):
            parts.append(
                f"({t['source']}) -[{t['type']}]-> ({t['target']}) [doc: {t['doc']}]"
            )
    else:
        parts.append("(sin tripletas)")
    parts.append("")
    parts.append("--- RESUMENES DE DOCUMENTO (referencia secundaria) ---")
    if summaries:
        for s in sorted(summaries, key=lambda s: s.get("doc_path", "")):
            parts.append(f"[{s['doc_path']}] {s['summary']}")
    else:
        parts.append("(sin resúmenes)")
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
    parts.append("--- FIGURAS ---")
    if figures:
        for f in figures:
            parts.append(f"[{f['image_path']}] {f['description']}")
    else:
        parts.append("(sin figuras)")
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
    large_model = getattr(settings, "large_model", None) if settings else None
    try:
        text_out, _model, _used_fallback = call_with_retries(
            session,
            prompt=ANSWER_PROMPT.format(context=context, query=query),
            system=_get_system_prompt(
                session, large_model, TASK_QUERY_ANSWER, ANSWER_SYSTEM_SHORT
            ),
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
    vec_hits = apply_relevance_threshold(vec_hits, verbose=verbose)
    if verbose:
        print(f"[KAG] Búsqueda híbrida: {len(vec_hits)} chunks (tras threshold)")
        for cid, score in vec_hits[:5]:
            print(f"    - chunk {cid}: score {score:.4f}")

    # 2.5 + 3. CRIT + EL fusionados en UNA llamada LLM (ahorra un round-trip):
    #      el crítico decide los términos exactos (regex/FTS) y el entity
    #      linking anclado selecciona las entidades canónicas del pool del
    #      grafo. Devuelve (regex_hits, regex_terms, names). Si el LLM falla,
    #      degrada por separado (heurística determinista + pool determinista).
    regex_hits, regex_terms, names = [], [], []
    try:
        regex_hits, regex_terms, names = critic_and_linking(
            session, query, top_k=k, verbose=verbose
        )
    except Exception as exc:  # noqa: BLE001 — degradación natural
        session.rollback()
        if verbose:
            print(f"[KAG] ⚠ CRIT+EL fusionado falló: {exc}")
    if verbose and regex_hits:
        print(
            f"[KAG] Crítico: {len(regex_hits)} chunks textuales "
            f"(términos: {regex_terms})"
        )

    # 3. Entity linking: candidatos del grafo + desambiguación por
    #    copresencia. `names` ya viene del LLM fusionado (o del pool
    #    determinista si falló).
    groups = match_entities_candidates(session, names)
    adj = build_adjacency(session) if groups else {}
    entity_ids = disambiguate_by_cooccurrence(
        session,
        groups,
        adjacency=adj,
        min_overlap=DISAMBIG_MIN_OVERLAP,
        margin=DISAMBIG_MARGIN,
        verbose=verbose,
    )
    if verbose:
        print(
            f"[KAG] Entity linking: {len(names)} nombres → "
            f"{len(entity_ids)} entidades (tras copresencia)"
        )

    # 4. PPR (HippoRAG) sobre la vecindad de 2 saltos de la semilla: correr
    #    PPR sobre el grafo completo es O(N²) y crece con el corpus; con
    #    alpha=0.15 la masa decae ~×0.15 por salto, así que 2 saltos capturan
    #    la señal relevante (ego_network extrae el subgrafo inducido).
    ppr_scores = {}
    if entity_ids:
        sub = ego_network(adj, entity_ids, hops=2)
        ppr_scores = personalized_pagerank(sub, entity_ids)
        if verbose:
            top_ppr = sorted(ppr_scores.items(), key=lambda x: -x[1])[:5]
            print(f"[KAG] PPR: {len(ppr_scores)} entidades rankeadas (ego 2-hop)")
            for eid, score in top_ppr:
                print(f"    - entidad {eid}: {score:.4f}")

    # 4.5. Umbral de cercanía en el grafo: entidades PPR 'cerca' de la
    #      semilla (relativo al máximo + baseline estadístico). Reemplaza
    #      el top-10 fijo: si solo 3 entidades están cerca, no arrastra 7
    #      irrelevantes; si 20 están cerca, no descarta la mitad.
    ppr_entities = ppr_entity_selection(ppr_scores, verbose=verbose)
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
    # Una sola consulta por todos los ids (en vez del loop N+1 por hit),
    # preservando el orden de importancia del RRF (array_position).
    merged_ids = [cid for cid, _score in merged_hits]
    fetched = chunks_by_ids(session, merged_ids)
    score_by_id = dict(merged_hits)
    chunks = []
    for c in fetched:
        c["score"] = score_by_id.get(c["chunk_id"], 0.0)
        chunks.append(c)
    if verbose:
        print(f"[KAG] Merge: {len(chunks)} chunks ancla (vector + regex + PPR, dedup)")

    # 5.5. Reranker opcional (cross-encoder): reordena los anclas por
    #      afinidad semántica exacta (query, chunk) antes de expandir la
    #      ventana. Default OFF (RERANK_ENABLED); si el modelo no carga,
    #      degrada sin rerank (mismo orden).
    if RERANK_ENABLED:
        chunks = rerank_chunks(query, chunks)
        if verbose:
            print(f"[KAG] Reranker: {len(chunks)} anclas reordenadas (cross-encoder)")

    # 5.6. Ventana de contexto: cada ancla se expande con sus ±CONTEXT_WINDOW
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
