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
    python -m src.kag_query --mode audited --rerank "pregunta"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

from sqlalchemy import bindparam, text
from sqlalchemy.exc import ProgrammingError

from src.kag.prompts import (
    ANSWER_SYSTEM_SHORT,
    AUDIT_FUSED_SYSTEM_SHORT,
    AUDITED_ANSWER_SYSTEM_SHORT,
    COMBINED_SYSTEM_SHORT,
    CONTRADICTION_SYSTEM_SHORT,
    CRITIC_SYSTEM_SHORT,
    GROUNDED_ENTITIES_SYSTEM_SHORT,
    QUERY_METADATA_SYSTEM_SHORT,
    QUERY_STRATEGY_SYSTEM_SHORT,
    QUERY_SUBQUERIES_SYSTEM_SHORT,
    SUFFICIENCY_SYSTEM_SHORT,
    SYNTHESIS_SYSTEM_SHORT,
    TASK_ANSWER,
    TASK_AUDIT_FUSED,
    TASK_CONTRADICTIONS,
    TASK_CRITIC_LINKING,
    TASK_CRITIC_REGEX,
    TASK_GROUNDED_ENTITIES,
    TASK_QUERY_ANSWER,
    TASK_QUERY_METADATA,
    TASK_QUERY_STRATEGY,
    TASK_QUERY_SUBQUERIES,
    TASK_SUFFICIENCY,
    TASK_SYNTHESIS,
    _get_prompt_pair,
)
from src.kag.thresholds import auto_cutoff
from src.kag.thresholds import auto_margin as _auto_margin
from src.kag_ingest import (
    detect_language,
    embedding_to_sql,
    normalize_entity_name,
)
from src.llm.base import call_with_retries, load_settings, parse_llm_output


def _with_own_session(fn, *args, **kwargs):
    """Ejecuta fn con una sesión propia del hilo (ver src.db.session).

    SQLAlchemy Session NO es thread-safe: compartir la sesión del caller
    entre hilos lanza InvalidSessionError ("session is provisioning a new
    connection") cuando dos hilos pisan la adquisición de conexión. Cada
    tarea paralela abre su propia sesión (SessionLocal) y la cierra al
    terminar. Las tareas son lecturas read-only: no hay transacción que
    propagar de vuelta al caller.
    """
    from src.db.session import run_in_own_session

    return run_in_own_session(fn, *args, **kwargs)


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


# Estrategias de consulta que el SLM puede devolver (spec kag_query_strategy).
QUERY_STRATEGIES = {"subqueries", "metadata", "hierarchical", "graph", "multidoc"}

# Canales de recuperación que el SLM puede seleccionar (spec kag_query_strategy).
CHANNEL_SET = {"dense", "fts", "paraphrase", "propositions", "graph", "summaries"}

# Canales por defecto por estrategia (validación post-parseo y fallback).
DEFAULT_CHANNELS = {
    "hierarchical": ["dense", "fts", "summaries"],
    "graph": ["dense", "fts", "paraphrase", "propositions", "graph"],
    "metadata": ["dense", "fts"],
    "subqueries": ["dense", "fts", "paraphrase", "propositions", "graph"],
    "multidoc": ["dense", "fts", "summaries"],
}


def _validated_channels(strategy, raw):
    """Valida channels del SLM: subconjunto no vacío de CHANNEL_SET.

    Si raw no es una lista no vacía o contiene canales fuera del set →
    default por estrategia (DEFAULT_CHANNELS).
    """
    if isinstance(raw, list) and raw:
        channels = [str(c).strip() for c in raw]
        if all(c in CHANNEL_SET for c in channels):
            seen = set()
            out = []
            for c in channels:
                if c not in seen:
                    seen.add(c)
                    out.append(c)
            return out
    return list(DEFAULT_CHANNELS[strategy])


def _top_k_for(strategy, top_k_label, top_k, global_top_k) -> int:
    """Resuelve el k efectivo según la etiqueta de amplitud del SLM.

    narrow → acotado a [3, 6]; standard → top_k del caller; wide → el
    global_top_k (marco temático amplio). `strategy` se conserva en la
    firma por si una estrategia necesita afinar el mapeo.
    """
    if top_k_label == "narrow":
        return max(3, min(top_k, 6))
    if top_k_label == "wide":
        return global_top_k
    return top_k


def classify_query_strategy(session, query) -> tuple[str, list[str], str, str, bool]:
    """Clasifica la estrategia de consulta con UNA llamada al SLM.

    El SLM (model_size="small") NO corre tools: solo clasifica la consulta
    en una de las cinco estrategias (spec kag_query_strategy) y selecciona
    los canales de recuperación + la amplitud de top_k. El LLM grande
    planifica según la clase (agentes B/C/D). Devuelve
    (strategy, channels, top_k_label, reason, used_fallback).

    Fallback: si la llamada lanza, el JSON es inválido o la estrategia no
    está en el set → degrada a la heurística actual (classify_query):
    "global"→"hierarchical", "local"→"graph", reason="fallback: <motivo>",
    used_fallback=True.
    """
    if not _kag_config_value(session, "KAG_QUERY_STRATEGY_SLM", True):
        return _query_strategy_fallback(query, "KAG_QUERY_STRATEGY_SLM=False")
    settings = load_settings(session)
    retries = int(getattr(settings, "llm_retries", 3) or 3) if settings else 3
    fallback = getattr(settings, "fallback_model", None) if settings else None
    small_model = getattr(settings, "small_model", None) if settings else None
    system, user_template = _get_prompt_pair(
        session,
        small_model,
        TASK_QUERY_STRATEGY,
        QUERY_STRATEGY_SYSTEM_SHORT,
    )
    try:
        text_out, _model, _used_fallback = call_with_retries(
            session,
            prompt=user_template.replace("{query}", query),
            system=system,
            model_size="small",
            response_format={"type": "json_object"},
            retries=retries,
            fallback_model=fallback,
        )
        data = parse_llm_output(text_out)
        strategy = str(data.get("strategy", "")).strip()
        if strategy not in QUERY_STRATEGIES:
            return _query_strategy_fallback(query, f"estrategia inválida: {strategy!r}")
        channels = _validated_channels(strategy, data.get("channels"))
        top_k_label = str(data.get("top_k") or "").strip()
        if top_k_label not in {"narrow", "standard", "wide"}:
            top_k_label = "standard"
        reason = str(data.get("reason") or "").strip()
        return strategy, channels, top_k_label, reason, False
    except Exception as exc:  # noqa: BLE001 — degradación natural
        return _query_strategy_fallback(query, str(exc))


def _query_strategy_fallback(
    query: str, motivo: str
) -> tuple[str, list[str], str, str, bool]:
    """Degrada a la heurística actual: 'global'→'hierarchical', 'local'→'graph'."""
    qtype = classify_query(query)
    if qtype == "global":
        return (
            "hierarchical",
            ["dense", "fts", "summaries"],
            "wide",
            f"fallback: {motivo}",
            True,
        )
    return (
        "graph",
        ["dense", "fts", "paraphrase", "propositions", "graph"],
        "standard",
        f"fallback: {motivo}",
        True,
    )


# ---------------------------------------------------------------------
# Estrategia metadata: filtrado guiado por metadatos
# ---------------------------------------------------------------------


def _corpus_metadata_brief(session) -> str:
    """JSON breve de títulos/doc_types/idiomas del corpus (para el prompt del
    extractor de filtros). Degradación: error SQL → "[]" sin romper."""
    rows = []
    try:
        rows = session.execute(
            text(
                "SELECT title, doc_type, language FROM kag_documents "
                "WHERE status = 'ready' LIMIT 50"
            )
        ).fetchall()
    except Exception:  # noqa: BLE001 — degradación natural
        _safe_rollback(session)
        rows = []
    return _json_dumps(
        [
            {
                "title": r.title,
                "doc_type": r.doc_type,
                "language": r.language,
            }
            for r in rows
        ]
    )


def extract_metadata_filters(session, query) -> dict:
    """Extrae filtros de metadatos de la consulta con el LLM grande.

    La EXTRACCIÓN de filtros es PLANIFICACIÓN (el SLM decide la estrategia,
    el LLM planifica): model_size="large". Devuelve
    {"filters": {"authors": [], "years": [], "fields": [], "works": [],
    "languages": []}, "reason": str}. Degradación: flag apagada, llamada
    falla o JSON inválido → {"filters": {}, "reason": ""} (nunca rompe).
    """
    if not _kag_config_value(session, "KAG_METADATA_FILTER", True):
        return {"filters": {}, "reason": ""}
    settings = load_settings(session)
    retries = int(getattr(settings, "llm_retries", 3) or 3) if settings else 3
    fallback = getattr(settings, "fallback_model", None) if settings else None
    large_model = getattr(settings, "large_model", None) if settings else None
    system, user_template = _get_prompt_pair(
        session,
        large_model,
        TASK_QUERY_METADATA,
        QUERY_METADATA_SYSTEM_SHORT,
    )
    try:
        text_out, _model, _used_fallback = call_with_retries(
            session,
            prompt=user_template.replace("{query}", query).replace(
                "{corpus_metadata}", _corpus_metadata_brief(session)
            ),
            system=system,
            model_size="large",
            response_format={"type": "json_object"},
            retries=retries,
            fallback_model=fallback,
        )
        data = parse_llm_output(text_out)
        filters = data.get("filters") or {}
        if not isinstance(filters, dict):
            filters = {}
        reason = str(data.get("reason") or "").strip()
        return {"filters": filters, "reason": reason}
    except Exception:  # noqa: BLE001 — degradación natural
        _safe_rollback(session)
        return {"filters": {}, "reason": ""}


def generate_subqueries(session, query) -> list:
    """Descompone la consulta en subconsultas atómicas con el LLM grande.

    La GENERACIÓN de subconsultas es PLANIFICACIÓN (el SLM decide la
    estrategia, el LLM planifica): model_size="large". Devuelve lista de
    dicts [{"query": str, "intent": str}] (máx 4, strings saneados).
    Degradación: flag apagada, llamada falla, JSON inválido o más de 4
    subconsultas → [{"query": query, "intent": "consulta original"}]
    (nunca rompe).
    """
    if not _kag_config_value(session, "KAG_SUBQUERIES", True):
        return [{"query": query, "intent": "consulta original"}]
    settings = load_settings(session)
    retries = int(getattr(settings, "llm_retries", 3) or 3) if settings else 3
    fallback = getattr(settings, "fallback_model", None) if settings else None
    large_model = getattr(settings, "large_model", None) if settings else None
    system, user_template = _get_prompt_pair(
        session,
        large_model,
        TASK_QUERY_SUBQUERIES,
        QUERY_SUBQUERIES_SYSTEM_SHORT,
    )
    try:
        text_out, _model, _used_fallback = call_with_retries(
            session,
            prompt=user_template.replace("{query}", query).replace(
                "{corpus_metadata}", _corpus_metadata_brief(session)
            ),
            system=system,
            model_size="large",
            response_format={"type": "json_object"},
            retries=retries,
            fallback_model=fallback,
        )
        data = parse_llm_output(text_out)
        raw = data.get("subqueries") or []
        if not isinstance(raw, list):
            raw = []
        subs = []
        for item in raw[:4]:
            if not isinstance(item, dict):
                continue
            sub_q = str(item.get("query") or "").strip()
            if not sub_q:
                continue
            subs.append(
                {
                    "query": sub_q,
                    "intent": str(item.get("intent") or "").strip(),
                }
            )
        if subs:
            return subs
    except Exception:  # noqa: BLE001 — degradación natural
        _safe_rollback(session)
    return [{"query": query, "intent": "consulta original"}]


def _doc_ids_for_filters(session, filters) -> list:
    """Ids de kag_documents que cumplen los filtros de metadatos.

    UN SELECT con ORs sobre kag_documents:
      - works → title ILIKE '%' || :w || '%' (por cada obra)
      - authors/years → ficha_jsonb->>'bibtex' ILIKE '%' || :a || '%'
        (el bibtex contiene autor/año)
      - fields → ficha_jsonb->'library_of_congress'->'lcsh_terms' @> :f
        (array) O EXISTS sobre thematic_areas_iso25964
      - languages → language = ANY(:langs)
    Si filters vacío → [] (sin filtro). Degradación: error SQL → rollback
    + [] sin romper.
    """
    if not filters:
        return []
    works = [str(w).strip() for w in (filters.get("works") or []) if str(w).strip()]
    authors = [str(a).strip() for a in (filters.get("authors") or []) if str(a).strip()]
    years = [str(y).strip() for y in (filters.get("years") or []) if str(y).strip()]
    fields = [str(f).strip() for f in (filters.get("fields") or []) if str(f).strip()]
    langs = [str(l).strip() for l in (filters.get("languages") or []) if str(l).strip()]
    if not (works or authors or years or fields or langs):
        return []
    conds = []
    params = {}
    for i, w in enumerate(works):
        conds.append(f"title ILIKE '%' || :w{i} || '%'")
        params[f"w{i}"] = w
    for i, a in enumerate(authors + years):
        conds.append(f"ficha_jsonb->>'bibtex' ILIKE '%' || :a{i} || '%'")
        params[f"a{i}"] = a
    for i, f in enumerate(fields):
        conds.append(
            f"(ficha_jsonb->'library_of_congress'->'lcsh_terms' @> :f{i} "
            f"OR EXISTS (SELECT 1 FROM jsonb_array_elements("
            f"ficha_jsonb->'thematic_areas_iso25964') t "
            f"WHERE t->>'preferred_term' ILIKE '%' || :fl{i} || '%'))"
        )
        params[f"f{i}"] = [f]
        params[f"fl{i}"] = f
    if langs:
        conds.append("language = ANY(:langs)")
        params["langs"] = langs
    sql = (
        "SELECT id FROM kag_documents WHERE status = 'ready' AND ("
        + " OR ".join(conds)
        + ")"
    )
    try:
        rows = session.execute(text(sql), params).fetchall()
        return [r.id for r in rows]
    except Exception:  # noqa: BLE001 — degradación natural
        _safe_rollback(session)
        return []


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
    # Verbos copulativos / existenciales comunes (el crítico SLM a veces los
    # propone como términos exactos — son ruido; el resto de verbos los
    # decide el SLM con el contexto de frecuencia del corpus).
    "existe",
    "existen",
    "hay",
    "ser",
    "estar",
    "está",
    "están",
    "was",
    "were",
    "be",
    "been",
    "being",
    "has",
    "have",
    "had",
    "does",
    "do",
    "did",
    "will",
    "would",
    "can",
    "could",
    "should",
    "may",
    "might",
    "must",
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
    if found:
        return found[:10]
    # Sin matches por LIKE (p. ej. query en otro idioma que el grafo):
    # devolver los noun chunks crudos (sin artículos) como candidatos. El
    # entity linking los resolverá por similitud coseno (name_embedding).
    raw = []
    for phrase in phrases:
        nn = normalize_entity_name(phrase)
        # Quitar artículos/determinantes iniciales: "la cultura" -> "cultura".
        for art in (
            "el ",
            "la ",
            "los ",
            "las ",
            "un ",
            "una ",
            "unos ",
            "unas ",
            "the ",
            "a ",
            "an ",
        ):
            if nn.startswith(art):
                nn = nn[len(art) :].strip()
                break
        if len(nn) >= 3 and nn not in raw:
            raw.append(nn)
    return raw[:10]


# ---------------------------------------------------------------------
# Entity linking anclado (opt 1)
# ---------------------------------------------------------------------


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
    system, user_template = _get_prompt_pair(
        session,
        small_model,
        TASK_GROUNDED_ENTITIES,
        GROUNDED_ENTITIES_SYSTEM_SHORT,
    )
    try:
        text_out, _model, _used_fallback = call_with_retries(
            session,
            prompt=user_template.format(
                candidates="\n".join(f"- {c}" for c in candidates),
                query=query,
            ),
            system=system,
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


def match_entities_candidates(session, names: list, embed_fn=None) -> list:
    """Candidatos por mención: lista de listas de ids.

    Exacto por name_norm primero; si no hay, LIKE (hasta 5); si aún no hay,
    similitud coseno sobre name_embedding (embed_fn, p. ej. embed_texts) —
    permite cruzar idiomas (query en español vs entidades en inglés). Cada
    mención puede tener 0..N candidatos — la desambiguación por copresencia
    decide.
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
        if not group and embed_fn is not None:
            try:
                emb = embed_fn([name])[0]
                rows = session.execute(
                    text(
                        "SELECT e.id FROM kag_entities e "
                        "JOIN kag_documents d ON d.id = e.doc_id "
                        "WHERE d.status = 'ready' AND e.name_embedding IS NOT NULL "
                        "ORDER BY e.name_embedding <=> CAST(:emb AS vector) "
                        "LIMIT 5"
                    ),
                    {"emb": embedding_to_sql(emb)},
                ).fetchall()
                group.extend(r.id for r in rows)
            except Exception:  # noqa: BLE001 — sin embedding: sin candidatos
                pass
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


def _read_graph_version(session):
    """Versión actual del grafo (kag_graph_state, migración 0016).

    Devuelve None si la migración no está aplicada o la sesión no soporta
    el SELECT (misma degradación que build_adjacency).
    """
    try:
        return session.execute(
            text("SELECT version FROM kag_graph_state WHERE id = 1")
        ).scalar()
    except Exception:
        # Migración 0016 sin aplicar, o sesión falsa de tests sin .scalar().
        return None


def _fetch_adjacency_rows(session, sql):
    """SELECT de una capa del grafo con degradación: tabla ausente → [].

    Si la tabla no existe (migración sin aplicar) o la sesión no soporta la
    consulta, hace rollback (la transacción queda aborted tras un error en
    Postgres) y devuelve [] para que build_adjacency continúe con las capas
    disponibles.
    """
    try:
        return session.execute(text(sql)).fetchall()
    except Exception:  # noqa: BLE001 — degradación natural
        try:
            session.rollback()
        except Exception:  # noqa: BLE001 — sesión falsa de tests sin rollback
            pass
        return []


def build_adjacency(session):
    """Grafo desde kag_relations: {entity_id: {neighbor_id: weight}}.

    Simétrico: cada relación aporta arista en ambos sentidos, ponderada por
    frecuencia. Cacheado por versión en DB (migración 0016): si la versión
    no cambió, devuelve el dict en memoria sin releer kag_relations. Si la
    migración no está aplicada (o la sesión no soporta el SELECT de versión),
    degrada al comportamiento viejo: reconstruir en cada llamada, sin
    cachear.

    Capa proposicional (0034, flag KAG_GRAPH_PROPOSITION_LAYER): además de
    kag_relations, lee kag_proposition_links y agrega nodos de proposición
    con namespace de STRING "p:{proposition_id}" conectados a sus entidades
    (arista bidireccional, weight 1). El dict mezcla keys INT (entidades) y
    STRING "p:..." (proposiciones). Si la tabla no existe, degrada al grafo
    de entidades sin romper.

    Capa tesauro (0035, flag KAG_GRAPH_THESAURUS_LAYER): además, lee
    kag_entities (doc→entidad), kag_document_thesaurus (doc→término) y
    kag_thesaurus_terms (término→término por broader/narrower/related,
    resueltos por term_norm dentro del mismo source; si el término
    relacionado no existe como fila, se omite) y agrega nodos de documento
    "d:{doc_id}" y de tesauro "t:{term_id}". Si alguna tabla no existe,
    degrada a las capas disponibles sin romper.
    """
    global _adjacency_cache, _adjacency_version

    version = _read_graph_version(session)

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

    if _kag_config_value(session, "KAG_GRAPH_PROPOSITION_LAYER", True):
        try:
            link_rows = session.execute(
                text("SELECT proposition_id, entity_id FROM kag_proposition_links")
            ).fetchall()
        except Exception:  # noqa: BLE001 — tabla ausente: grafo de entidades
            link_rows = []
        for r in link_rows:
            # Defensivo: sesiones falsas de tests pueden devolver filas con
            # otra forma (p. ej. las de kag_relations) — se ignoran.
            pid = getattr(r, "proposition_id", None)
            eid = getattr(r, "entity_id", None)
            if pid is None or eid is None:
                continue
            pnode = f"p:{pid}"
            adj.setdefault(pnode, {})
            adj.setdefault(eid, {})
            adj[pnode][eid] = 1
            adj[eid][pnode] = 1

    if _kag_config_value(session, "KAG_GRAPH_THESAURUS_LAYER", True):
        # Capa documento: doc → entidades (kag_entities) y doc → términos
        # (kag_document_thesaurus). Cada consulta degrada por separado.
        doc_ent_rows = _fetch_adjacency_rows(
            session, "SELECT DISTINCT doc_id, id AS entity_id FROM kag_entities"
        )
        doc_term_rows = _fetch_adjacency_rows(
            session, "SELECT doc_id, term_id FROM kag_document_thesaurus"
        )
        for r in doc_ent_rows:
            did = getattr(r, "doc_id", None)
            eid = getattr(r, "entity_id", None)
            if did is None or eid is None:
                continue
            dnode = f"d:{did}"
            adj.setdefault(dnode, {})
            adj.setdefault(eid, {})
            adj[dnode][eid] = 1
            adj[eid][dnode] = 1
        for r in doc_term_rows:
            did = getattr(r, "doc_id", None)
            tid = getattr(r, "term_id", None)
            if did is None or tid is None:
                continue
            dnode = f"d:{did}"
            tnode = f"t:{tid}"
            adj.setdefault(dnode, {})
            adj.setdefault(tnode, {})
            adj[dnode][tnode] = 1
            adj[tnode][dnode] = 1
        # Capa tesauro: término → término por broader/narrower/related,
        # resueltos por term_norm dentro del mismo source.
        term_rows = _fetch_adjacency_rows(
            session,
            "SELECT id, term_norm, source, broader, narrower, related "
            "FROM kag_thesaurus_terms",
        )
        terms_by_key = {}
        for r in term_rows:
            tid = getattr(r, "id", None)
            norm = getattr(r, "term_norm", None)
            src = getattr(r, "source", None)
            if tid is None or not norm or not src:
                continue
            terms_by_key[(norm, src)] = tid
        for r in term_rows:
            tid = getattr(r, "id", None)
            norm = getattr(r, "term_norm", None)
            src = getattr(r, "source", None)
            if tid is None or not norm or not src:
                continue
            tnode = f"t:{tid}"
            for field in ("broader", "narrower", "related"):
                rels = getattr(r, field, None)
                if isinstance(rels, str):
                    try:
                        rels = json.loads(rels)
                    except Exception:  # noqa: BLE001 — JSONB malformado
                        rels = []
                if not isinstance(rels, list):
                    continue
                for other in rels:
                    other_norm = normalize_entity_name(str(other))
                    if not other_norm:
                        continue
                    other_id = terms_by_key.get((other_norm, src))
                    if other_id is None:
                        continue
                    onode = f"t:{other_id}"
                    adj.setdefault(tnode, {})
                    adj.setdefault(onode, {})
                    adj[tnode][onode] = 1
                    adj[onode][tnode] = 1

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
    # Clave mixta INT (entidades) + STRING "p:..." (proposiciones, 0034):
    # ordenar por tipo primero para que sorted() no compare int con str.
    nodes = sorted(nodes, key=lambda n: (isinstance(n, str), n))
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


class _GraphVersionNotCached(Exception):
    """La matriz de adyacencia de esta versión no está en memoria.

    Solo ocurre en una carrera: el grafo cambió entre la lectura de versión
    y la construcción de la matriz. El wrapper público degrada al cálculo
    directo (la excepción nunca se cachea en lru_cache).
    """


def _adjacency_for_version(graph_version):
    """Matriz de adyacencia cacheada para `graph_version`.

    build_adjacency mantiene el invariante: _adjacency_cache corresponde a
    _adjacency_version. Si la versión pedida no es la cacheada, devuelve
    None y el wrapper degrada al cálculo directo.
    """
    if _adjacency_version == graph_version and _adjacency_cache is not None:
        return _adjacency_cache
    return None


@lru_cache(maxsize=256)
def _ego_network_cached(entity_ids: frozenset, graph_version, hops=2):
    """Ego-network de `hops` saltos cacheado por (semilla, versión del grafo).

    La matriz de adyacencia se lee del caché module-level de build_adjacency
    (invariante garantizada por el wrapper público). Mismas entidades semilla
    en consultas del mismo dominio → el BFS de 2 saltos no se repite.
    """
    adj = _adjacency_for_version(graph_version)
    if adj is None:
        raise _GraphVersionNotCached(graph_version)
    return ego_network(adj, list(entity_ids), hops=hops)


@lru_cache(maxsize=256)
def _ppr_cached(entity_ids: frozenset, graph_version, alpha):
    """Power iteration de PPR cacheada por (semilla, versión, alpha).

    Reutiliza el caché de ego-network: la extracción del subgrafo y la
    power iteration solo se recalculan cuando cambia la semilla, la
    versión del grafo o alpha.
    """
    sub = _ego_network_cached(entity_ids, graph_version, 2)
    return personalized_pagerank(sub, list(entity_ids), alpha=alpha)


def ego_network_cached(session, entity_ids, hops=2):
    """ego_network con LRU cache por (frozenset(entity_ids), graph_version).

    Lee la versión del grafo (kag_graph_state) y delega en el core cacheado.
    Sin tabla de versión → cálculo directo (degradación, sin cachear).
    """
    graph_version = _read_graph_version(session)
    if graph_version is None:
        adj = build_adjacency(session)
        return ego_network(adj, entity_ids, hops=hops)
    build_adjacency(session)  # garantiza _adjacency_cache para esta versión
    try:
        return _ego_network_cached(frozenset(entity_ids), graph_version, hops)
    except _GraphVersionNotCached:
        adj = build_adjacency(session)
        return ego_network(adj, entity_ids, hops=hops)


def personalized_pagerank_cached(session, entity_ids, alpha=None):
    """personalized_pagerank con LRU cache por (semilla, versión, alpha).

    `alpha` viene de KAG_PPR_ALPHA (src/kag/config.py) si no se pasa
    explícito. Sin tabla de versión → cálculo directo (degradación).
    """
    if alpha is None:
        alpha = _kag_config_value(session, "KAG_PPR_ALPHA", 0.15)
    graph_version = _read_graph_version(session)
    if graph_version is None:
        adj = build_adjacency(session)
        sub = ego_network(adj, entity_ids, hops=2)
        return personalized_pagerank(sub, entity_ids, alpha=alpha)
    build_adjacency(session)  # garantiza _adjacency_cache para esta versión
    try:
        return _ppr_cached(frozenset(entity_ids), graph_version, float(alpha))
    except _GraphVersionNotCached:
        adj = build_adjacency(session)
        sub = ego_network(adj, entity_ids, hops=2)
        return personalized_pagerank(sub, entity_ids, alpha=alpha)


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
    # Solo nodos entidad (keys INT): los nodos "p:..." (capa proposicional,
    # 0034) y "d:..."/"t:..." (capa tesauro, 0035) propagan el PPR pero no
    # se seleccionan como semilla de chunks.
    entity_items = [(eid, s) for eid, s in scores.items() if isinstance(eid, int)]
    if not entity_items:
        return []
    vals = [s for _, s in entity_items]
    max_s = max(vals)
    floor = min_ratio * max_s
    if len(vals) >= 100:
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        floor = max(floor, mean + z * std)
    sorted_items = sorted(entity_items, key=lambda x: -x[1])
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


def vector_search(session, query_embedding, top_k, doc_ids=None):
    """pgvector <=> (coseno). Devuelve lista de (chunk_id, score).

    Solo chunks de documentos 'ready': los docs pending/failed (proceso
    interrumpido) no deben contaminar los resultados (degradación elegante).
    `doc_ids` (opcional) restringe a los documentos dados (estrategia
    metadata).
    """
    q = embedding_to_sql(query_embedding)
    doc_cond = ""
    params = {"q": q, "top_k": top_k}
    if doc_ids:
        doc_cond = " AND c.doc_id = ANY(:doc_ids)"
        params["doc_ids"] = list(doc_ids)
    rows = session.execute(
        text(
            "SELECT c.id, 1 - (c.embedding <=> CAST(:q AS vector)) AS score "
            "FROM kag_chunks c "
            "JOIN kag_documents d ON d.id = c.doc_id "
            "WHERE c.embedding IS NOT NULL AND d.status = 'ready' "
            + doc_cond
            + " ORDER BY c.embedding <=> CAST(:q AS vector) LIMIT :top_k"
        ),
        params,
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


def fts_search(session, query_text, top_k, doc_ids=None):
    """Búsqueda léxica con FTS de Postgres (ts_rank_cd sobre content_tsv).

    Requiere la migración 0014 (columna generada content_tsv + índice GIN).
    Config 'simple' a propósito: agnóstica de idioma (el corpus es
    multilingüe) y sin stemming (ideal para términos exactos: acrónimos,
    códigos, nombres propios). Query vacía o sin tokens → [] (sin error).
    `doc_ids` (opcional) restringe a los documentos dados (estrategia
    metadata).
    """
    if not query_text or not query_text.strip():
        return []
    doc_cond = ""
    params = {"q": query_text, "top_k": top_k}
    if doc_ids:
        doc_cond = " AND c.doc_id = ANY(:doc_ids)"
        params["doc_ids"] = list(doc_ids)
    rows = session.execute(
        text(
            "SELECT c.id, ts_rank_cd(c.content_tsv, plainto_tsquery('simple', :q)) "
            "AS score FROM kag_chunks c "
            "JOIN kag_documents d ON d.id = c.doc_id "
            "WHERE c.content_tsv @@ plainto_tsquery('simple', :q) "
            "AND d.status = 'ready' " + doc_cond + " ORDER BY score DESC LIMIT :top_k"
        ),
        params,
    ).fetchall()
    return [(r.id, float(r.score)) for r in rows]


def _paraphrase_fts_search(session, query_text, top_k, doc_ids=None):
    """Búsqueda léxica sobre paraphrase_tsv (migración 0033).

    Canal de paráfrasis: FTS sobre la columna generada paraphrase_tsv
    (config 'simple', corpus multilingüe). Devuelve [(chunk_id, score)].
    Degradación: query vacía → []; columna ausente (migración no aplicada)
    → ProgrammingError que hybrid_search convierte en [] sin romper.
    `doc_ids` (opcional) restringe a los documentos dados (estrategia
    metadata).
    """
    if not query_text or not query_text.strip():
        return []
    doc_cond = ""
    params = {"q": query_text, "top_k": top_k}
    if doc_ids:
        doc_cond = " AND doc_id = ANY(:doc_ids)"
        params["doc_ids"] = list(doc_ids)
    rows = session.execute(
        text(
            "SELECT id, ts_rank_cd(paraphrase_tsv, plainto_tsquery('simple', :q)) "
            "AS score FROM kag_chunks "
            "WHERE paraphrase_tsv @@ plainto_tsquery('simple', :q) "
            + doc_cond
            + " ORDER BY score DESC LIMIT :top_k"
        ),
        params,
    ).fetchall()
    return [(r.id, float(r.score)) for r in rows]


def hybrid_search(
    session,
    query_text,
    query_embedding,
    top_k,
    rrf_k=60,
    verbose=False,
    doc_ids=None,
    channels=None,
):
    """Búsqueda híbrida: densa (pgvector) + léxica (FTS) + RRF.

    La búsqueda densa, la FTS, el canal de paráfrasis (FTS sobre
    paraphrase_tsv, migración 0033) y el canal semántico de proposiciones
    (kag_propositions, migración 0024) son lecturas read-only independientes
    y corren en paralelo (ThreadPoolExecutor), cada una con su propia sesión
    (_with_own_session: Session no es thread-safe). El resultado es idéntico
    al secuencial: mismo RRF, mismo orden, mismo score.

    Los canales nuevos entran como listas adicionales en el rrf_merge final
    (junto a densa y FTS), gated por KAG_PARAPHRASE_CHANNEL y
    KAG_PROPOSITION_CHANNEL (default True vía _kag_config_value).

    `doc_ids` (opcional, lista de ints): restringe TODOS los canales a los
    documentos dados (estrategia metadata). None/[] = comportamiento actual
    (sin filtro).

    `channels` (opcional, lista de str): subconjunto de canales a correr
    ("dense"→vector_search, "fts"→fts_search, "paraphrase"→
    _paraphrase_fts_search, "propositions"→proposition_vector_search).
    "graph"/"summaries" se ignoran (no son canales de hybrid_search). None =
    los 4 canales actuales (gated por KAG_PARAPHRASE_CHANNEL /
    KAG_PROPOSITION_CHANNEL).

    Si query_embedding es None (embeddings no disponibles), degrada a solo
    FTS + paráfrasis (el canal de proposiciones devuelve []). Si la
    migración 0014 no está aplicada (columna content_tsv ausente,
    ProgrammingError), degrada a solo búsqueda densa. Si la migración 0033
    o 0024 no está aplicada, el canal correspondiente degrada a [] sin
    romper. Otros errores se propagan al caller (ask() los degrada a
    vec_hits=[]).
    """
    dense_hits = []
    sparse_hits = []
    para_hits = []
    prop_hits = []
    fts_error = None
    # Solo se pasa doc_ids a los canales cuando es no vacío: con None/[] la
    # firma es EXACTAMENTE la de antes (backward-compatible con los fakes de
    # tests que no aceptan el kwarg).
    channel_kwargs = {"doc_ids": doc_ids} if doc_ids else {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        dense_future = None
        if (channels is None or "dense" in channels) and query_embedding is not None:
            dense_future = pool.submit(
                _with_own_session,
                vector_search,
                query_embedding,
                top_k,
                **channel_kwargs,
            )
        sparse_future = None
        if channels is None or "fts" in channels:
            sparse_future = pool.submit(
                _with_own_session, fts_search, query_text, top_k, **channel_kwargs
            )
        para_future = None
        if (channels is None or "paraphrase" in channels) and _kag_config_value(
            session, "KAG_PARAPHRASE_CHANNEL", True
        ):
            para_future = pool.submit(
                _with_own_session,
                _paraphrase_fts_search,
                query_text,
                top_k,
                **channel_kwargs,
            )
        prop_future = None
        if (channels is None or "propositions" in channels) and _kag_config_value(
            session, "KAG_PROPOSITION_CHANNEL", True
        ):
            prop_future = pool.submit(
                _with_own_session,
                proposition_vector_search,
                query_embedding,
                top_k,
                **channel_kwargs,
            )
        if dense_future is not None:
            # Esperar primero la densa: si falla, su error se propaga (misma
            # precedencia que el flujo secuencial, donde la densa corre antes).
            dense_hits = dense_future.result()
        if sparse_future is not None:
            try:
                sparse_hits = sparse_future.result()
            except ProgrammingError as exc:  # migración 0014 sin aplicar
                fts_error = exc
        if para_future is not None:
            try:
                para_hits = para_future.result()
            except ProgrammingError as exc:  # migración 0033 sin aplicar
                session.rollback()
                if verbose:
                    print(f"[KAG] ⚠ Canal paráfrasis no disponible ({exc}); sin canal.")
                para_hits = []
        if prop_future is not None:
            try:
                prop_rows = prop_future.result()
                prop_hits = [
                    (p["chunk_id"], p["score"]) for p in prop_rows if p.get("chunk_id")
                ]
            except ProgrammingError as exc:  # migración 0024 sin aplicar
                session.rollback()
                if verbose:
                    print(
                        f"[KAG] ⚠ Canal proposiciones no disponible ({exc}); sin canal."
                    )
                prop_hits = []
    if fts_error is not None:
        session.rollback()
        if verbose:
            print(f"[KAG] ⚠ FTS no disponible ({fts_error}); solo búsqueda densa.")
        return dense_hits
    if not dense_hits:
        # Sin capa densa (embeddings no disponibles): fusionar las capas
        # léxicas restantes (FTS + paráfrasis + proposiciones).
        return rrf_merge(sparse_hits, para_hits, prop_hits, k=rrf_k, top_k=top_k)
    return rrf_merge(
        dense_hits, sparse_hits, para_hits, prop_hits, k=rrf_k, top_k=top_k
    )


# ---------------------------------------------------------------------
# Recuperación de resúmenes (summary_index, migración 0032) y
# proposiciones por embedding (kag_propositions, migración 0024)
# ---------------------------------------------------------------------


def search_summaries(
    session, query_text, query_embedding, top_k, level=None, rrf_k=60, verbose=False
):
    """Búsqueda híbrida sobre summary_index: densa + FTS + RRF (§3.1).

    Devuelve lista de DICTS con el texto del resumen (el caller necesita el
    texto, no solo ids): [{"id", "doc_id", "level", "chapter_id", "text",
    "score"}]. La densa usa pgvector sobre summary_index.embedding; la FTS
    usa la columna generada tsv (config 'simple', corpus multilingüe). Si una
    de las dos capas no produce hits, se devuelve la otra; si ambas vacías,
    []. Degradación: si la tabla no existe (migración 0032 no aplicada),
    ProgrammingError → [] sin romper.
    """
    dense_hits = []
    sparse_hits = []
    try:
        if query_embedding is not None:
            q = embedding_to_sql(query_embedding)
            rows = session.execute(
                text(
                    "SELECT id, doc_id, level, chapter_id, text, "
                    "1 - (embedding <=> CAST(:q AS vector)) AS score "
                    "FROM summary_index "
                    "WHERE embedding IS NOT NULL "
                    "AND (:level IS NULL OR level = :level) "
                    "ORDER BY embedding <=> CAST(:q AS vector) LIMIT :top_k"
                ),
                {"q": q, "top_k": top_k, "level": level},
            ).fetchall()
            dense_hits = [(r.id, float(r.score)) for r in rows]
        if query_text and query_text.strip():
            rows = session.execute(
                text(
                    "SELECT id, doc_id, level, chapter_id, text, "
                    "ts_rank_cd(tsv, plainto_tsquery('simple', :q)) AS score "
                    "FROM summary_index "
                    "WHERE tsv @@ plainto_tsquery('simple', :q) "
                    "AND (:level IS NULL OR level = :level) "
                    "ORDER BY score DESC LIMIT :top_k"
                ),
                {"q": query_text, "top_k": top_k, "level": level},
            ).fetchall()
            sparse_hits = [(r.id, float(r.score)) for r in rows]
    except ProgrammingError:  # migración 0032 no aplicada
        session.rollback()
        return []
    if not dense_hits:
        merged = sparse_hits
    elif not sparse_hits:
        merged = dense_hits
    else:
        merged = rrf_merge(dense_hits, sparse_hits, k=rrf_k, top_k=top_k)
    if not merged:
        return []
    # Segundo SELECT por los ids fusionados. id es UUID: se castea a text
    # para ANY(:ids) con strings; el orden del RRF se preserva en Python
    # (tabla chica — evita array_position con UUIDs).
    ids = [str(i) for i, _score in merged]
    try:
        rows = session.execute(
            text(
                "SELECT id, doc_id, level, chapter_id, text "
                "FROM summary_index WHERE id::text = ANY(:ids)"
            ),
            {"ids": ids},
        ).fetchall()
    except ProgrammingError:  # migración 0032 no aplicada
        session.rollback()
        return []
    by_id = {str(r.id): r for r in rows}
    out = []
    for i, score in merged:
        r = by_id.get(str(i))
        if r is None:
            continue
        out.append(
            {
                "id": r.id,
                "doc_id": r.doc_id,
                "level": r.level,
                "chapter_id": r.chapter_id,
                "text": r.text,
                "score": score,
            }
        )
    if verbose:
        print(f"[KAG] Resúmenes (índice): {len(out)} hits")
    return out


def proposition_vector_search(session, query_embedding, top_k, doc_ids=None):
    """Búsqueda semántica directa sobre kag_propositions.embedding (§3.2).

    El embedding de proposiciones ya existe (migración 0024) pero nunca se
    usa para búsqueda directa: esta función lo explota como capa micro.
    Devuelve lista de dicts con id, chunk_id, doc_id, statement y score.
    Degradación: query_embedding None o tabla ausente (migración no
    aplicada) → [] sin romper. `doc_ids` (opcional) restringe a los
    documentos dados (estrategia metadata).
    """
    if query_embedding is None:
        return []
    q = embedding_to_sql(query_embedding)
    doc_cond = ""
    params = {"q": q, "top_k": top_k}
    if doc_ids:
        doc_cond = " AND doc_id = ANY(:doc_ids)"
        params["doc_ids"] = list(doc_ids)
    try:
        rows = session.execute(
            text(
                "SELECT id, chunk_id, doc_id, statement, "
                "1 - (embedding <=> CAST(:q AS vector)) AS score "
                "FROM kag_propositions "
                "WHERE embedding IS NOT NULL "
                + doc_cond
                + " ORDER BY embedding <=> CAST(:q AS vector) LIMIT :top_k"
            ),
            params,
        ).fetchall()
    except ProgrammingError:  # migración 0024 no aplicada
        session.rollback()
        return []
    return [dict(r._mapping) for r in rows]


def chunks_for_entities(session, entity_ids, top_n):
    """Chunks de las entidades dadas, ordenados por frecuencia de mención."""
    if not entity_ids:
        return []
    rows = session.execute(
        text(
            "SELECT c.id, c.doc_id, c.chapter_id, c.content, d.doc_path, "
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
            "chapter_id": r.chapter_id,
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
            "SELECT c.id, c.doc_id, c.chapter_id, c.content, c.chunk_index, "
            "d.doc_path FROM kag_chunks c "
            "JOIN kag_documents d ON d.id = c.doc_id "
            "WHERE c.id = ANY(:ids) AND d.status = 'ready' "
            "ORDER BY array_position(:ids, c.id)"
        ),
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
                "chapter_id": r.chapter_id,
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


def propositions_for_chunks(session, chunk_ids, per_chunk=6, max_total=120):
    """Proposiciones atómicas de los chunks dados (capa micro, kag_propositions).

    Devuelve lista de dicts con la proposición, su span verbatim y el contexto
    de documento/sección. Degradación natural: si la tabla no existe (migración
    0024 no aplicada) o no hay proposiciones, devuelve [] sin romper.
    """
    if not chunk_ids:
        return []
    ids = list(dict.fromkeys(chunk_ids))
    try:
        rows = session.execute(
            text(
                "SELECT p.id AS prop_id, p.chunk_id, p.statement, p.text_span, "
                "p.char_start, p.char_end, p.citation_references, "
                "p.doc_id AS document_id, "
                "d.title AS doc_title, "
                "COALESCE(kc.title, '') AS chapter_title "
                "FROM kag_propositions p "
                "JOIN kag_documents d ON p.doc_id = d.id "
                "JOIN kag_chunks c ON p.chunk_id = c.id "
                "LEFT JOIN kag_chapters kc ON kc.id = c.chapter_id "
                "WHERE p.chunk_id = ANY(:ids) "
                "ORDER BY p.chunk_id, p.id"
            ),
            {"ids": ids},
        ).fetchall()
    except Exception:  # noqa: BLE001 — tabla ausente (migración no aplicada)
        return []
    out = []
    per_chunk_count: dict[int, int] = {}
    for r in rows:
        if per_chunk_count.get(r.chunk_id, 0) >= per_chunk:
            continue
        per_chunk_count[r.chunk_id] = per_chunk_count.get(r.chunk_id, 0) + 1
        out.append(
            {
                "chunk_id": r.prop_id,
                "document_id": r.document_id,
                "statement": r.statement,
                "text_span": r.text_span,
                "char_start": r.char_start,
                "char_end": r.char_end,
                "citation_references": r.citation_references or [],
                "doc_title": r.doc_title,
                "chapter_title": r.chapter_title,
            }
        )
        if len(out) >= max_total:
            break
    return out


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
    chapter_id, content, chunk_index, doc_path, score, is_anchor.
    """
    rows = session.execute(
        text(
            "SELECT c.id, c.doc_id, c.chapter_id, c.content, c.chunk_index, "
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
                "chapter_id": r.chapter_id,
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


def rerank_chunks(query, chunks, top_n=RERANK_TOP_N, enabled=None):
    """Reordena los chunks ancla por relevancia semántica (cross-encoder).

    Solo actúa si `enabled` es True (default: RERANK_ENABLED). Reordena el
    top-N por la afinidad exacta (query, chunk) y actualiza el score del
    ancla con la puntuación del reranker — así `_group_chunks_with_window`
    ordena los grupos por la nueva relevancia. Devuelve la misma lista
    (mismo orden) si el modelo no está disponible o falla — degradación
    natural.
    """
    if enabled is None:
        enabled = RERANK_ENABLED
    if not enabled or not chunks:
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


# Caché del índice de frecuencia de palabras + idiomas soportados (el corpus
# cambia al indexar; TTL corto para no servir datos muy viejos).
_CORPUS_CACHE: dict = {"ts": 0.0, "words": None, "langs": None}
_CORPUS_CACHE_TTL = 600.0  # 10 min


def _corpus_common_words(session, top_n: int = 200) -> list:
    """Palabras más frecuentes del corpus (índice kag_word_freq, migración 0023).

    Se inyecta al crítico SLM como contexto: palabras que matchearían
    demasiados chunks y no sirven como términos exactos. El SLM decide qué
    proponer con esta información (no hay filtro duro en código).
    """
    cache = _CORPUS_CACHE
    now = time.time()
    if cache["words"] is not None and now - cache["ts"] < _CORPUS_CACHE_TTL:
        return cache["words"]
    words = []
    try:
        rows = session.execute(
            text(
                "SELECT word FROM kag_word_freq "
                "GROUP BY word ORDER BY SUM(nentry) DESC, COUNT(*) DESC LIMIT :n"
            ),
            {"n": top_n},
        ).fetchall()
        words = [r.word for r in rows]
    except Exception:  # noqa: BLE001 — sin índice: el crítico decide sin contexto
        words = []
    cache["ts"] = now
    cache["words"] = words
    return words


def _corpus_languages(session) -> list:
    """Idiomas soportados por el sistema (kag_segmenter_settings.spacy_models).

    Los instalados inicialmente (es/en/pt/de/fr). El crítico SLM traduce cada
    término a TODOS estos idiomas para que el FTS matchee el corpus
    multilingüe.
    """
    cache = _CORPUS_CACHE
    now = time.time()
    if cache["langs"] is not None and now - cache["ts"] < _CORPUS_CACHE_TTL:
        return cache["langs"]
    langs = []
    try:
        row = session.execute(
            text(
                "SELECT spacy_models FROM kag_segmenter_settings "
                "WHERE is_active = TRUE ORDER BY id LIMIT 1"
            )
        ).first()
        if row and row.spacy_models:
            langs = [str(k) for k in row.spacy_models.keys()]
    except Exception:  # noqa: BLE001 — degradación natural
        langs = []
    if not langs:
        langs = ["es", "en", "pt", "de", "fr"]
    cache["ts"] = now
    cache["langs"] = langs
    return langs


def _clean_llm_terms(raw: list) -> list:
    """Safety net mínimo sobre los términos del crítico SLM.

    El SLM decide qué términos proponer (con el contexto de frecuencia del
    corpus y la traducción a todos los idiomas). Aquí solo se descartan
    palabras de función y copulas — el resto lo decide el SLM.
    """
    cleaned = []
    for t in raw:
        s = str(t).strip()
        if s and s.lower() not in _QUERY_STOPWORDS and len(s) > 2:
            cleaned.append(s)
    return cleaned[:15]


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
    system, user_template = _get_prompt_pair(
        session,
        small_model,
        TASK_CRITIC_REGEX,
        CRITIC_SYSTEM_SHORT,
    )
    try:
        common = _corpus_common_words(session)
        langs = _corpus_languages(session)
        text_out, _model, _used_fallback = call_with_retries(
            session,
            prompt=user_template.format(
                query=query,
                common_words=", ".join(common[:80]) or "(sin datos)",
                languages=", ".join(langs),
            ),
            system=system,
            model_size="small",
            response_format={"type": "json_object"},
            retries=retries,
            fallback_model=fallback,
        )
        data = parse_llm_output(text_out)
        if data.get("needs_regex"):
            terms = _clean_llm_terms(
                [str(t).strip() for t in (data.get("terms") or [])]
            )
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
    # nombres propios). Se construye un tsquery booleano OR con
    # websearch_to_tsquery: tolera frases con espacios y acentos (a
    # diferencia de to_tsquery, que exige sintaxis tsquery estricta y
    # revienta con '"discriminación negativa"'). Las comillas dobles internas
    # se reemplazan por espacio (en websearch las comillas delimitan frases).
    # El ranking usa ts_rank_cd sobre el tsquery combinado; los chunks que
    # matchean varios términos puntúan más alto (suma de relevancia por
    # término).
    tsq = " OR ".join(f'"{t.replace(chr(34), " ")}"' for t in terms)
    rows = session.execute(
        text(
            "SELECT c.id, ts_rank_cd(c.content_tsv, websearch_to_tsquery('simple', :tsq)) "
            "AS score FROM kag_chunks c "
            "JOIN kag_documents d ON d.id = c.doc_id "
            "WHERE c.content_tsv @@ websearch_to_tsquery('simple', :tsq) "
            "AND d.status = 'ready' ORDER BY score DESC LIMIT :top_k"
        ),
        {"tsq": tsq, "top_k": top_k},
    ).fetchall()
    hits = [(r.id, float(r.score)) for r in rows]
    return hits[:top_k], terms


# ---------------------------------------------------------------------
# CRIT + EL fusionados (una sola llamada LLM)
# ---------------------------------------------------------------------


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
    system, user_template = _get_prompt_pair(
        session,
        small_model,
        TASK_CRITIC_LINKING,
        COMBINED_SYSTEM_SHORT,
    )
    try:
        common = _corpus_common_words(session)
        langs = _corpus_languages(session)
        text_out, _model, _used_fallback = call_with_retries(
            session,
            prompt=user_template.format(
                candidates="\n".join(f"- {c}" for c in candidates)
                or "(sin candidatos)",
                query=query,
                common_words=", ".join(common[:80]) or "(sin datos)",
                languages=", ".join(langs),
            ),
            system=system,
            model_size="small",
            response_format={"type": "json_object"},
            retries=retries,
            fallback_model=fallback,
        )
        data = parse_llm_output(text_out)
        if data.get("needs_regex"):
            terms = _clean_llm_terms(
                [str(t).strip() for t in (data.get("terms") or [])]
            )
        raw_names = [
            str(e).strip() for e in (data.get("entities") or []) if str(e).strip()
        ]
        # Fusión de candidatos: anclados al pool + propuestas del LLM
        # (multilingüe, puede proponer nombres en el idioma del grafo) + pool
        # determinista. Más candidatos → más chances de resolver el cruce de
        # idiomas; la desambiguación por copresencia filtra el ruido.
        pool_norm = {normalize_entity_name(c) for c in candidates}
        anchored = [n for n in raw_names if normalize_entity_name(n) in pool_norm]
        names = list(dict.fromkeys(anchored + raw_names + candidates))[:12]
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
        try:
            tsq = " OR ".join(f'"{t.replace(chr(34), " ")}"' for t in terms)
            rows = session.execute(
                text(
                    "SELECT c.id, ts_rank_cd(c.content_tsv, websearch_to_tsquery('simple', :tsq)) "
                    "AS score FROM kag_chunks c "
                    "JOIN kag_documents d ON d.id = c.doc_id "
                    "WHERE c.content_tsv @@ websearch_to_tsquery('simple', :tsq) "
                    "AND d.status = 'ready' ORDER BY score DESC LIMIT :top_k"
                ),
                {"tsq": tsq, "top_k": top_k},
            ).fetchall()
            regex_hits = [(r.id, float(r.score)) for r in rows][:top_k]
        except Exception:  # noqa: BLE001 — FTS falla (p.ej. tsquery inválido):
            # no debe matar el entity linking; `names` ya se calculó arriba.
            if verbose:
                print("[KAG] ⚠ FTS textual falló; continúo solo con entity linking.")
            session.rollback()
    return regex_hits, terms, names


# ---------------------------------------------------------------------
# Ensamblado + respuesta
# ---------------------------------------------------------------------


def assemble_context(
    chunks,
    triples,
    figures,
    summaries,
    query,
    history=None,
    propositions=None,
    summary_hits=None,
):
    """Ensambla el bloque de contexto para la respuesta final.

    Las tres capas de evidencia se entregan como bloques EXPLÍCITAMENTE
    separados y etiquetados (diseño §3.3): el resumen da el MARCO, el chunk
    da la CITA verificable y la proposición da la CAPA ATÓMICA. No se
    mezclan en un solo ranking: cada bloque tiene su encabezado para que el
    LLM distinga la granularidad.

    Los chunks llegan ya agrupados con su ventana (ancla + vecinos) y
    ordenados por importancia. Cada chunk muestra su procedencia exacta
    (doc, doc_id, capítulo, chapter_id, chunk_index) y si es resultado
    directo o contexto adyacente. `history` (opcional) es una lista de
    dicts {"role", "content"} del historial de conversación — se incluye
    como sección informativa (preparado, aún sin probar con Docker).
    `propositions` (opcional) es la capa micro: proposiciones atómicas de
    los chunks ganadores (kag_propositions) con su chunk_id y span.
    `summary_hits` (opcional) son los resúmenes recuperados del índice
    temático (search_summaries, §3.3): dan el MARCO de la consulta global.
    """
    parts = []
    # 1. MARCO TEMÁTICO: resúmenes de documento/sección (índice temático).
    if summary_hits is not None:
        parts.append("--- MARCO TEMÁTICO (resúmenes de documento/sección) ---")
        if summary_hits:
            for s in summary_hits:
                doc_id = s.get("doc_id", "?")
                level = s.get("level", "document")
                chapter = s.get("chapter_id")
                if level == "section" and chapter:
                    parts.append(
                        f"[doc_id: {doc_id} | nivel: section | capítulo: {chapter}] "
                        f"{s.get('text', '')}"
                    )
                else:
                    parts.append(
                        f"[doc_id: {doc_id} | nivel: {level}] {s.get('text', '')}"
                    )
        else:
            parts.append("(sin resúmenes recuperados)")
        parts.append("")
    # 2. EVIDENCIA TEXTUAL: chunks con cita verificable (doc_id/chapter_id).
    parts.append("--- EVIDENCIA TEXTUAL (chunks con cita) ---")
    if chunks:
        for i, c in enumerate(chunks, start=1):
            doc = c.get("doc_path", "?")
            section = c.get("chapter_title", "") or "(sin capítulo)"
            idx = c.get("chunk_index", "?")
            score = c.get("score", 0.0)
            marker = "RESULTADO" if c.get("is_anchor", True) else "contexto"
            parts.append(
                f"[{i}] {marker} | doc: {doc} | doc_id: {c.get('doc_id', '?')} | "
                f"capítulo: {section} | chapter_id: {c.get('chapter_id', '?')} | "
                f"chunk {idx} | score: {score:.4f}"
            )
            parts.append(c.get("content", ""))
            parts.append("")
    else:
        parts.append("(sin fragmentos recuperados)")
        parts.append("")
    # 3. CAPA ATÓMICA: proposiciones con chunk_id y span.
    if propositions is not None:
        parts.append("--- CAPA ATÓMICA (proposiciones) ---")
        if propositions:
            for i, p in enumerate(propositions, start=1):
                doc = p.get("doc_title", "?")
                section = p.get("chapter_title", "") or "(sin sección)"
                line = (
                    f"[{i}] doc: {doc} | sección: {section} | "
                    f"chunk_id: {p.get('chunk_id', '?')} | {p.get('statement', '')}"
                )
                span = p.get("text_span") or ""
                if span:
                    line += f" (cita: {span})"
                parts.append(line)
                parts.append("")
        else:
            parts.append("(sin proposiciones)")
            parts.append("")
    # 4. Historial, subgrafo, resúmenes de documento y figuras (como hoy).
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
    parts.append("--- FIGURAS ---")
    if figures:
        for f in figures:
            parts.append(f"[{f['image_path']}] {f['description']}")
    else:
        parts.append("(sin figuras)")
    return "\n".join(parts)


def generate_answer(session, context, query):
    """Respuesta final con el LLM grande. Si falla, devuelve el contexto crudo."""
    settings = load_settings(session)
    retries = int(getattr(settings, "llm_retries", 3) or 3) if settings else 3
    fallback = getattr(settings, "fallback_model", None) if settings else None
    large_model = getattr(settings, "large_model", None) if settings else None
    system, user_template = _get_prompt_pair(
        session, large_model, TASK_QUERY_ANSWER, ANSWER_SYSTEM_SHORT
    )
    try:
        text_out, _model, _used_fallback = call_with_retries(
            session,
            prompt=user_template.format(context=context, query=query),
            system=system,
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
# Modo audited — auditoría epistémica (portado de src/kag_agents.py,
# adaptado a kag_propositions). Se eliminará src/kag_agents.py al final.
# ---------------------------------------------------------------------


def _json_dumps(obj) -> str:
    """json.dumps con ensure_ascii=False y fallback a repr si serializar falla."""
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return repr(obj)


def _safe_rollback(session):
    """Rollback defensivo: sesiones reales siempre lo tienen; fakes en tests no."""
    rollback = getattr(session, "rollback", None)
    if callable(rollback):
        rollback()


def _fts_tsquery(terms: list) -> str:
    """Construye un tsquery 'simple' con comillas DOBLES siempre.

    El `-` es el operador NOT de tsquery, así que cada término se escapa
    duplicando las comillas dobles y se envuelve en comillas dobles.
    """
    parts = []
    for term in terms:
        t = str(term).strip()
        if not t:
            continue
        t = t.replace('"', '""')
        parts.append(f'"{t}"')
    return " | ".join(parts)


def _settings_retries(session) -> tuple:
    """Lee llm_retries y fallback_model de session_settings (defaults seguros)."""
    try:
        settings = load_settings(session)
    except Exception:  # noqa: BLE001 — degradación natural
        settings = None
    if settings is None:
        return 3, None
    retries = int(getattr(settings, "llm_retries", 3) or 3)
    fallback = getattr(settings, "fallback_model", None)
    return retries, fallback


def _settings_models(session) -> tuple:
    """Lee small_model y large_model de session_settings (defaults None)."""
    try:
        settings = load_settings(session)
    except Exception:  # noqa: BLE001 — degradación natural
        settings = None
    if settings is None:
        return None, None
    return getattr(settings, "small_model", None), getattr(
        settings, "large_model", None
    )


def _synthesize_facts(session, query, propositions, verbose=False) -> list:
    """Sintetiza las proposiciones recuperadas en hechos atómicos (LLM pequeño).

    Degradación: cada proposición como supporting_evidence con
    atomic_summary=statement truncado y verbatim_evidence=text_span.
    """
    if not propositions:
        return []
    retries, fallback = _settings_retries(session)
    small_model, _large_model = _settings_models(session)
    system, user_template = _get_prompt_pair(
        session, small_model, TASK_SYNTHESIS, SYNTHESIS_SYSTEM_SHORT
    )
    prompt = user_template.format(
        query=query, candidate_chunks_json=_json_dumps(propositions)
    )
    try:
        text_out, _model, _used_fallback = call_with_retries(
            session,
            prompt=prompt,
            system=system,
            model_size="small",
            response_format={"type": "json_object"},
            retries=retries,
            fallback_model=fallback,
        )
        data = parse_llm_output(text_out)
        facts = data.get("synthesized_facts") or []
        if isinstance(facts, list) and facts:
            if verbose:
                print(
                    f"[KAG] Síntesis: {len(facts)} hechos (fallback={_used_fallback})"
                )
            return facts
    except Exception as exc:  # noqa: BLE001 — degradación natural
        _safe_rollback(session)
        if verbose:
            print(f"[KAG] ⚠ Síntesis LLM falló ({exc}); degradando.")
    degraded = []
    for p in propositions:
        statement = (p.get("statement") or "").strip()
        text_span = (p.get("text_span") or "").strip()
        degraded.append(
            {
                "chunk_id": str(p.get("chunk_id", "")),
                "document_id": str(p.get("document_id") or ""),
                "source_file": str(p.get("doc_title", "")),
                "relevance_level": "supporting_evidence",
                "atomic_summary": statement[:500],
                "verbatim_evidence": text_span[:1000],
                "academic_citations": p.get("citation_references") or [],
            }
        )
    if verbose:
        print(f"[KAG] Síntesis degradada: {len(degraded)} hechos.")
    return degraded


def _resolve_contradictions(session, query, facts, verbose=False) -> dict:
    """Tipifica contradicciones entre hechos sintetizados (LLM pequeño).

    Degradación: contradictions_detected=False, analysis_cases=[].
    """
    if not facts:
        return {"contradictions_detected": False, "analysis_cases": []}
    retries, fallback = _settings_retries(session)
    small_model, _large_model = _settings_models(session)
    system, user_template = _get_prompt_pair(
        session,
        small_model,
        TASK_CONTRADICTIONS,
        CONTRADICTION_SYSTEM_SHORT,
    )
    prompt = user_template.format(
        query=query, synthesized_facts_json=_json_dumps(facts)
    )
    try:
        text_out, _model, _used_fallback = call_with_retries(
            session,
            prompt=prompt,
            system=system,
            model_size="small",
            response_format={"type": "json_object"},
            retries=retries,
            fallback_model=fallback,
        )
        data = parse_llm_output(text_out)
        report = {
            "contradictions_detected": bool(data.get("contradictions_detected", False)),
            "analysis_cases": data.get("analysis_cases") or [],
        }
        if verbose:
            print(
                f"[KAG] Contradicciones: {report['contradictions_detected']} "
                f"({len(report['analysis_cases'])} casos)"
            )
        return report
    except Exception as exc:  # noqa: BLE001 — degradación natural
        _safe_rollback(session)
        if verbose:
            print(f"[KAG] ⚠ Análisis de contradicciones falló ({exc}).")
        return {"contradictions_detected": False, "analysis_cases": []}


def _evaluate_sufficiency(
    session, query, facts, corpus_metadata, verbose=False
) -> dict:
    """Dictamina si la recuperación basta para responder (LLM pequeño).

    Degradación: verdict='SUFFICIENT_FOR_SYNTHESIS' si hay ≥1 fact con
    relevance_level direct_answer/supporting_evidence; si no,
    'INSUFFICIENT_TRIGGER_BRANCH_B'.
    """
    retries, fallback = _settings_retries(session)
    small_model, _large_model = _settings_models(session)
    system, user_template = _get_prompt_pair(
        session,
        small_model,
        TASK_SUFFICIENCY,
        SUFFICIENCY_SYSTEM_SHORT,
    )
    prompt = user_template.format(
        query=query,
        active_corpus_metadata=_json_dumps(corpus_metadata),
        synthesized_propositions_json=_json_dumps(facts),
        parent_contexts_json="[]",
    )
    try:
        text_out, _model, _used_fallback = call_with_retries(
            session,
            prompt=prompt,
            system=system,
            model_size="small",
            response_format={"type": "json_object"},
            retries=retries,
            fallback_model=fallback,
        )
        data = parse_llm_output(text_out)
        verdict = data.get("verdict")
        if verdict in (
            "SUFFICIENT_FOR_SYNTHESIS",
            "INSUFFICIENT_TRIGGER_BRANCH_B",
            "NEGATIVE_REJECTION",
        ):
            if verbose:
                print(
                    f"[KAG] Suficiencia: {verdict} "
                    f"(confianza={data.get('confidence_score')})"
                )
            return data
    except Exception as exc:  # noqa: BLE001 — degradación natural
        _safe_rollback(session)
        if verbose:
            print(f"[KAG] ⚠ Auditoría de suficiencia falló ({exc}).")
    relevant = [
        f
        for f in facts
        if f.get("relevance_level") in ("direct_answer", "supporting_evidence")
    ]
    verdict = (
        "SUFFICIENT_FOR_SYNTHESIS" if relevant else "INSUFFICIENT_TRIGGER_BRANCH_B"
    )
    if verbose:
        print(f"[KAG] Suficiencia degradada: {verdict}.")
    return {"verdict": verdict, "confidence_score": 0.5}


def _audit_epistemic_fused(
    session,
    query,
    propositions,
    corpus_metadata,
    verbose=False,
    summary_hits=None,
    chunks=None,
) -> tuple:
    """Auditoría epistémica en UNA llamada al LLM pequeño (fusión de
    síntesis + contradicciones + suficiencia).

    Devuelve (facts, contradiction_report, evaluation) con los MISMOS shapes
    que _synthesize_facts / _resolve_contradictions / _evaluate_sufficiency.
    Si el JSON del LLM falla o viene malformado, degrada a las llamadas
    separadas actuales — nunca romper. `summary_hits` (opcional) son los
    resúmenes recuperados del índice temático (§3.3) y `chunks` (opcional)
    los fragmentos con cita: se entregan al prompt como TRES bloques de
    evidencia separados (marco → chunks → proposiciones) con los mismos
    encabezados que assemble_context, para que el LLM distinga la
    granularidad de cada capa.
    """
    if not propositions:
        return (
            [],
            {"contradictions_detected": False, "analysis_cases": []},
            {
                "verdict": "INSUFFICIENT_TRIGGER_BRANCH_B",
                "confidence_score": 0.5,
            },
        )
    retries, fallback = _settings_retries(session)
    small_model, _large_model = _settings_models(session)
    system, user_template = _get_prompt_pair(
        session,
        small_model,
        TASK_AUDIT_FUSED,
        AUDIT_FUSED_SYSTEM_SHORT,
    )
    prompt = (
        user_template.replace("{query}", query)
        .replace("{active_corpus_metadata}", _json_dumps(corpus_metadata))
        .replace("{candidate_chunks_json}", _json_dumps(propositions))
    )
    if chunks is not None or summary_hits is not None:
        prompt += "\n\nEvidencia por capas de granularidad:\n"
        prompt += (
            "--- MARCO TEMÁTICO (resúmenes de documento/sección) ---\n"
            + _json_dumps(summary_hits or [])
            + "\n--- EVIDENCIA TEXTUAL (chunks con cita) ---\n"
            + _json_dumps(chunks or [])
            + "\n--- CAPA ATÓMICA (proposiciones) ---\n"
            + _json_dumps(propositions)
        )
    try:
        text_out, _model, _used_fallback = call_with_retries(
            session,
            prompt=prompt,
            system=system,
            model_size="small",
            response_format={"type": "json_object"},
            retries=retries,
            fallback_model=fallback,
        )
        data = parse_llm_output(text_out)
        facts_raw = data.get("facts")
        if not isinstance(facts_raw, list):
            raise ValueError("fused audit: facts ausente o no-lista")
        facts_out = _fused_facts_to_shape(facts_raw, propositions)
        contradictions = data.get("contradictions") or []
        report = {
            "contradictions_detected": bool(contradictions),
            "analysis_cases": [
                {
                    "conflict_type": c.get(
                        "type", "paradigmatic_theoretical_divergence"
                    ),
                    "divergence_summary": c.get("resolution", ""),
                    "thesis_a": {},
                    "thesis_b": {},
                    "epistemic_reconciliation": c.get("resolution", ""),
                }
                for c in contradictions
                if isinstance(c, dict)
            ],
        }
        suff = data.get("sufficiency") or {}
        verdict_raw = str(suff.get("verdict", "")).strip().upper()
        verdict_map = {
            "SUFFICIENT": "SUFFICIENT_FOR_SYNTHESIS",
            "INSUFFICIENT": "INSUFFICIENT_TRIGGER_BRANCH_B",
            "NEGATIVE_REJECTION": "NEGATIVE_REJECTION",
        }
        verdict = verdict_map.get(verdict_raw)
        if verdict not in (
            "SUFFICIENT_FOR_SYNTHESIS",
            "INSUFFICIENT_TRIGGER_BRANCH_B",
            "NEGATIVE_REJECTION",
        ):
            raise ValueError(f"fused audit: veredicto inválido {verdict_raw!r}")
        try:
            confidence = float(suff.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        evaluation = {"verdict": verdict, "confidence_score": confidence}
        if verdict == "NEGATIVE_REJECTION":
            evaluation["negative_rejection_details"] = {
                "reason": "total_absence_in_knowledge_graph",
                "closest_available_topics": [],
                "formal_abstention_statement": (
                    "El corpus disponible no cubre el dominio conceptual requerido "
                    "por la consulta (abstención temprana)."
                ),
            }
        if verbose:
            print(
                f"[KAG] Auditoría fusionada: {len(facts_out)} hechos, "
                f"{len(report['analysis_cases'])} tensiones, "
                f"verdict={verdict} (fallback={_used_fallback})"
            )
        return facts_out, report, evaluation
    except Exception as exc:  # noqa: BLE001 — degradación natural
        _safe_rollback(session)
        if verbose:
            print(
                f"[KAG] ⚠ Auditoría fusionada falló ({exc}); degradando a llamadas separadas."
            )
    # Degradación: comportamiento actual (3 llamadas separadas).
    facts = _synthesize_facts(session, query, propositions, verbose=verbose)
    report = _resolve_contradictions(session, query, facts, verbose=verbose)
    evaluation = _evaluate_sufficiency(
        session, query, facts, corpus_metadata, verbose=verbose
    )
    return facts, report, evaluation


def _fused_facts_to_shape(facts_raw, propositions) -> list:
    """Convierte los hechos del JSON fusionado al shape de _synthesize_facts.

    Re-ancla cada hecho a su proposición original (chunk_id, document_id,
    source_file, academic_citations) buscando por statement/verbatim.
    """
    if not isinstance(facts_raw, list):
        return []
    by_statement: dict = {}
    by_span: dict = {}
    for p in propositions:
        stmt = (p.get("statement") or "").strip()
        span = (p.get("text_span") or "").strip()
        if stmt:
            by_statement.setdefault(stmt, p)
        if span:
            by_span.setdefault(span, p)
    out: list = []
    for f in facts_raw:
        if not isinstance(f, dict):
            continue
        statement = (f.get("statement") or "").strip()
        verbatim = (f.get("verbatim_evidence") or "").strip()
        relevance = str(f.get("relevance") or "supporting_evidence").strip()
        if relevance not in (
            "direct_answer",
            "supporting_evidence",
            "contextual_background",
            "irrelevant",
        ):
            relevance = "supporting_evidence"
        prop = by_statement.get(statement) or by_span.get(verbatim) or {}
        out.append(
            {
                "chunk_id": str(prop.get("chunk_id", "")),
                "document_id": str(prop.get("document_id") or ""),
                "source_file": str(prop.get("doc_title", "")),
                "relevance_level": relevance,
                "atomic_summary": statement[:500],
                "verbatim_evidence": verbatim[:1000],
                "academic_citations": prop.get("citation_references") or [],
            }
        )
    return out


def _kag_config_value(session, name, default=None):
    """Lee una flag KAG_* de la config resuelta (env > DB > default).

    Import perezoso: src.kag.config es puro (sin torch/spacy), pero se
    mantiene el patrón lazy del módulo. Nunca lanza.
    """
    try:
        from src.kag.config import get_config_value, resolve_config

        config = resolve_config(session)
        return get_config_value(config, name, default)
    except Exception:  # noqa: BLE001 — degradación natural
        return default


def _verify_grounding(session, facts, verbose=False) -> list:
    """Verifica el grounding verbatim de cada fact (rapidfuzz partial_ratio).

    PODA: la verificación difusa (partial_ratio >= KAG_GROUNDING_THRESHOLD,
    default 95.0) se aplica SOLO a hechos `direct_answer` o con citas
    bibliográficas explícitas. El material puramente contextual de fondo
    (supporting_evidence sin citas) NO requiere verificación carácter por
    carácter: pasa con el chequeo barato de substring.

    Descarta los hechos que no pasan. Enriquecen con metadatos del doc.

    Con KAG_GROUNDING_PARALLEL=True (default) y varios hechos, los cálculos
    por fact (lectura read-only + partial_ratio) corren en ThreadPoolExecutor
    (max_workers 2-4, CPU-bound en Python), cada uno con su propia sesión
    (_with_own_session: Session no es thread-safe). El resultado es idéntico
    al secuencial: mismo orden en `verified`, misma poda.
    """
    if not facts:
        return []
    try:
        from rapidfuzz import fuzz
    except Exception:  # noqa: BLE001 — sin rapidfuzz: grounding por substring
        fuzz = None
    threshold = float(_kag_config_value(session, "KAG_GROUNDING_THRESHOLD", 95.0))
    parallel = bool(_kag_config_value(session, "KAG_GROUNDING_PARALLEL", True))

    def _check_fact(session, f):
        if f.get("relevance_level") not in ("direct_answer", "supporting_evidence"):
            return None
        chunk_id = f.get("chunk_id")
        verbatim = f.get("verbatim_evidence") or ""
        if not chunk_id or not verbatim:
            return None
        citations = f.get("academic_citations") or []
        needs_fuzzy = f.get("relevance_level") == "direct_answer" or bool(citations)
        try:
            row = session.execute(
                text(
                    "SELECT p.text_span, p.citation_references, p.doc_id, "
                    "d.title AS doc_title, "
                    "COALESCE(kc.title, '') AS chapter_title "
                    "FROM kag_propositions p "
                    "JOIN kag_documents d ON p.doc_id = d.id "
                    "JOIN kag_chunks c ON p.chunk_id = c.id "
                    "LEFT JOIN kag_chapters kc ON kc.id = c.chapter_id "
                    "WHERE p.id = :cid"
                ),
                {"cid": chunk_id},
            ).fetchone()
        except Exception as exc:  # noqa: BLE001 — degradación natural
            _safe_rollback(session)
            if verbose:
                print(f"[KAG] ⚠ Grounding falló para chunk {chunk_id}: {exc}")
            return None
        if not row:
            return None
        db_span = row.text_span or ""
        if verbatim in db_span:
            match_score = 100.0
        elif needs_fuzzy:
            match_score = (
                float(fuzz.partial_ratio(verbatim, db_span))
                if fuzz is not None
                else 0.0
            )
        else:
            # Contextual de fondo: sin verificación difusa carácter por carácter.
            match_score = 100.0
        if needs_fuzzy and match_score < threshold:
            if verbose:
                print(
                    f"[KAG] Grounding rechazado: chunk {chunk_id} "
                    f"(score {match_score:.1f})."
                )
            return None
        return {
            "claim_id": str(chunk_id),
            "verified_fact": f.get("atomic_summary") or f.get("statement", ""),
            "verbatim_quote": verbatim,
            "document_id": row.doc_id,
            "document_title": row.doc_title,
            "chapter_title": row.chapter_title,
            "bibtex_citation_key": None,
            "academic_citations": row.citation_references or [],
        }

    if parallel and len(facts) > 1:
        with ThreadPoolExecutor(max_workers=min(4, len(facts))) as pool:
            futures = [pool.submit(_with_own_session, _check_fact, f) for f in facts]
            results = [future.result() for future in futures]
    else:
        results = [_check_fact(session, f) for f in facts]
    return [r for r in results if r is not None]


def _branch_b_expand(
    session, query, entity_ids, visited_chunks, top_k=8, verbose=False
) -> list:
    """Expansión Branch B: FTS sobre kag_propositions + vecinos del grafo.

    Devuelve proposiciones nuevas (misma forma que propositions_for_chunks)
    deduplicadas contra visited_chunks.
    """
    new_props: list = []
    seen: set = set(visited_chunks or [])
    terms: list = []
    # (a) Términos del crítico (regex/FTS) para subconsultas ortogonales.
    try:
        _regex_hits, regex_terms, _names = critic_and_linking(
            session, query, top_k=top_k, verbose=False
        )
        terms = list(regex_terms or [])
    except Exception as exc:  # noqa: BLE001 — degradación natural
        _safe_rollback(session)
        if verbose:
            print(f"[KAG] ⚠ Branch B crítico falló: {exc}")
    # (b) Vecinos del grafo (entidades semilla).
    if entity_ids:
        try:
            rows = session.execute(
                text(
                    "SELECT e.name FROM kag_relations r "
                    "JOIN kag_entities e ON r.target_entity_id = e.id "
                    "WHERE r.source_entity_id = ANY(:entity_ids) "
                    "ORDER BY r.id DESC LIMIT 10"
                ),
                {"entity_ids": list(entity_ids)},
            ).fetchall()
            terms = list(dict.fromkeys(terms + [r.name for r in rows]))
        except Exception as exc:  # noqa: BLE001 — degradación natural
            _safe_rollback(session)
            if verbose:
                print(f"[KAG] ⚠ Branch B grafo falló: {exc}")
    # (c) FTS sobre kag_propositions con los términos. Las subconsultas por
    #      término son lecturas read-only independientes: con
    #      KAG_QUERY_PARALLEL=True corren en ThreadPoolExecutor (max_workers
    #      2-4), cada una con su propia sesión (_with_own_session: Session no
    #      es thread-safe). El resultado es idéntico al secuencial: los
    #      futures se recogen en orden de envío y el dedup/append corre en el
    #      hilo principal en ese mismo orden.
    valid_terms = []
    for term in terms:
        if not term or not str(term).strip():
            continue
        tokens = [t for t in re.split(r"[\s,;]+", str(term).strip()) if t]
        if not tokens:
            continue
        valid_terms.append((term, tokens))

    def _fts_term(session, term, tokens):
        tsq = _fts_tsquery(tokens)
        try:
            return session.execute(
                text(
                    "SELECT p.id AS prop_id, p.chunk_id, p.statement, p.text_span, "
                    "p.char_start, p.char_end, p.citation_references, "
                    "d.title AS doc_title, "
                    "COALESCE(kc.title, '') AS chapter_title, "
                    "ts_rank_cd(to_tsvector('simple', p.statement), "
                    "to_tsquery('simple', :tsq)) AS score "
                    "FROM kag_propositions p "
                    "JOIN kag_documents d ON p.doc_id = d.id "
                    "JOIN kag_chunks c ON p.chunk_id = c.id "
                    "LEFT JOIN kag_chapters kc ON kc.id = c.chapter_id "
                    "WHERE to_tsvector('simple', p.statement) @@ "
                    "to_tsquery('simple', :tsq) "
                    "ORDER BY score DESC LIMIT 5"
                ),
                {"tsq": tsq},
            ).fetchall()
        except Exception as exc:  # noqa: BLE001 — degradación natural
            _safe_rollback(session)
            if verbose:
                print(f"[KAG] ⚠ Branch B FTS falló: {exc}")
            return []

    if (
        bool(_kag_config_value(session, "KAG_QUERY_PARALLEL", True))
        and len(valid_terms) > 1
    ):
        with ThreadPoolExecutor(max_workers=min(4, len(valid_terms))) as pool:
            futures = [
                pool.submit(_with_own_session, _fts_term, term, tokens)
                for term, tokens in valid_terms
            ]
            rowsets = [future.result() for future in futures]
    else:
        rowsets = [_fts_term(session, term, tokens) for term, tokens in valid_terms]

    for rows in rowsets:
        for r in rows:
            if r.prop_id in seen:
                continue
            seen.add(r.prop_id)
            new_props.append(
                {
                    "chunk_id": r.prop_id,
                    "document_id": None,
                    "statement": r.statement,
                    "text_span": r.text_span,
                    "char_start": r.char_start,
                    "char_end": r.char_end,
                    "citation_references": r.citation_references or [],
                    "doc_title": r.doc_title,
                    "chapter_title": r.chapter_title,
                }
            )
    return new_props


def _retrieval_phase(
    session, query, k, q_emb, doc_ids=None, verbose=False, channels=None
) -> dict:
    """Fase de recuperación compartida por ask() y _ask_audited().

    Encapsula: hybrid_search ∥ critic_and_linking (ThreadPoolExecutor +
    _with_own_session) → apply_relevance_threshold →
    match_entities_candidates → build_adjacency →
    disambiguate_by_cooccurrence → personalized_pagerank_cached →
    ppr_entity_selection → chunks_for_entities → rrf_merge.

    `channels` (opcional): subconjunto de canales para hybrid_search (None =
    los 4 actuales).

    Devuelve dict con {"merged_hits", "vec_hits", "regex_hits",
    "ppr_chunks", "names", "groups", "entity_ids", "ppr_scores"}.
    """
    vec_hits = []
    regex_hits, regex_terms, names = [], [], []
    if _kag_config_value(session, "KAG_QUERY_PARALLEL", True):
        with ThreadPoolExecutor(max_workers=2) as pool:
            hybrid_kwargs = {"doc_ids": doc_ids} if doc_ids else {}
            if channels is not None:
                hybrid_kwargs["channels"] = channels
            hybrid_future = pool.submit(
                _with_own_session,
                hybrid_search,
                query,
                q_emb,
                k,
                verbose=verbose,
                **hybrid_kwargs,
            )
            crit_future = pool.submit(
                _with_own_session, critic_and_linking, query, top_k=k, verbose=verbose
            )
            try:
                vec_hits = hybrid_future.result()
            except Exception as exc:  # noqa: BLE001 — degradación natural
                _safe_rollback(session)
                if verbose:
                    print(f"[KAG] ⚠ Búsqueda híbrida falló: {exc}")
                vec_hits = []
            try:
                regex_hits, regex_terms, names = crit_future.result()
            except Exception as exc:  # noqa: BLE001 — degradación natural
                _safe_rollback(session)
                if verbose:
                    print(f"[KAG] ⚠ CRIT+EL fusionado falló: {exc}")
    else:
        try:
            hybrid_kwargs = {"doc_ids": doc_ids} if doc_ids else {}
            if channels is not None:
                hybrid_kwargs["channels"] = channels
            vec_hits = hybrid_search(
                session, query, q_emb, k, verbose=verbose, **hybrid_kwargs
            )
        except Exception as exc:  # noqa: BLE001 — degradación natural
            _safe_rollback(session)
            if verbose:
                print(f"[KAG] ⚠ Búsqueda híbrida falló: {exc}")
            vec_hits = []
        try:
            regex_hits, regex_terms, names = critic_and_linking(
                session, query, top_k=k, verbose=verbose
            )
        except Exception as exc:  # noqa: BLE001 — degradación natural
            _safe_rollback(session)
            if verbose:
                print(f"[KAG] ⚠ CRIT+EL fusionado falló: {exc}")
    vec_hits = apply_relevance_threshold(vec_hits, verbose=verbose)
    if verbose:
        print(f"[KAG] Búsqueda híbrida: {len(vec_hits)} chunks (tras threshold)")
        for cid, score in vec_hits[:5]:
            print(f"    - chunk {cid}: score {score:.4f}")
    if verbose and regex_hits:
        print(
            f"[KAG] Crítico: {len(regex_hits)} chunks textuales "
            f"(términos: {regex_terms})"
        )
    embed_fn = None
    try:
        from src.embeddings import embed_texts as _embed_texts

        embed_fn = _embed_texts
    except Exception:  # noqa: BLE001 — sin modelo de embeddings: solo léxico
        pass
    groups = match_entities_candidates(session, names, embed_fn=embed_fn)
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
    ppr_scores = {}
    if entity_ids:
        ppr_scores = personalized_pagerank_cached(session, entity_ids)
        if verbose:
            top_ppr = sorted(ppr_scores.items(), key=lambda x: -x[1])[:5]
            print(f"[KAG] PPR: {len(ppr_scores)} entidades rankeadas (ego 2-hop)")
            for eid, score in top_ppr:
                print(f"    - entidad {eid}: {score:.4f}")
    ppr_entities = ppr_entity_selection(ppr_scores, verbose=verbose)
    if verbose:
        print(f"[KAG] PPR cercanas: {len(ppr_entities)} entidades (umbral relativo)")
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
    return {
        "merged_hits": merged_hits,
        "vec_hits": vec_hits,
        "regex_hits": regex_hits,
        "ppr_chunks": ppr_chunks,
        "names": names,
        "groups": groups,
        "entity_ids": entity_ids,
        "ppr_scores": ppr_scores,
    }


def _ask_audited(
    session, query, top_k=8, global_top_k=20, verbose=False, rerank=False
) -> dict:
    """Flujo audited completo: recuperación clásica → proposiciones → síntesis
    → contradicciones → suficiencia → Branch B → grounding → respuesta.

    Devuelve dict: {"answer", "verdict", "grounded_evidence",
    "epistemic_tensions", "used_fallback"}.
    """
    if verbose:
        print(f"\n🔎 Pregunta (audited): {query}")

    # 1. Recuperación clásica (misma lógica que ask fast). Con
    #    KAG_QUERY_ALL_CHANNELS=True (default) se saltea el routing SLM y se
    #    recupera de TODOS los canales a la vez.
    all_channels = _kag_config_value(session, "KAG_QUERY_ALL_CHANNELS", True)
    if all_channels:
        strategy = "graph"
        channels = ["dense", "fts", "paraphrase", "propositions", "graph", "summaries"]
        top_k_label = "standard"
        reason = "KAG_QUERY_ALL_CHANNELS: todos los canales a la vez"
        used_fallback = False
        k = top_k
    else:
        strategy, channels, top_k_label, reason, used_fallback = (
            classify_query_strategy(session, query)
        )
        k = _top_k_for(strategy, top_k_label, top_k, global_top_k)
    if verbose:
        print(f"[KAG] Clasificación: {strategy} (top_k={k})")
        print(f"[KAG] Canales: {channels} | top_k: {top_k_label}")
        if reason:
            print(f"[KAG]   razón: {reason}")
        if used_fallback:
            print(f"[KAG]   ⚠ fallback de estrategia: {reason}")
        if strategy == "multidoc":
            print(f"[KAG] Estrategia: {strategy} (planificación LLM pendiente)")
    # 1.0. Estrategia metadata: extraer filtros de metadatos (LLM grande) y
    #      resolver los doc_ids que los cumplen. Si la extracción falla
    #      (filters vacío) → comportamiento local normal (sin filtro).
    metadata_doc_ids = None
    if strategy == "metadata":
        try:
            filters = extract_metadata_filters(session, query)
            doc_ids = _doc_ids_for_filters(session, filters.get("filters") or {})
            if doc_ids:
                metadata_doc_ids = doc_ids
                if verbose:
                    print(
                        f"[KAG] Metadata: {len(doc_ids)} doc(s) filtrado(s) — "
                        f"{filters.get('reason') or ''}"
                    )
        except Exception as exc:  # noqa: BLE001 — degradación natural
            _safe_rollback(session)
            if verbose:
                print(f"[KAG] ⚠ Filtros de metadatos fallaron ({exc}); sin filtro.")
    q_emb = None
    try:
        from src.embeddings import embed_text

        q_emb = embed_text(query, input_type="query")
    except Exception as exc:  # noqa: BLE001 — degradación natural
        _safe_rollback(session)
        if verbose:
            print(f"[KAG] ⚠ Embeddings no disponibles ({exc}); solo FTS.")
    # 1.1. Canales nuevos (§3.2/§3.3): proposiciones (capa micro) y
    #      resúmenes temáticos (marco global). El canal de proposiciones ya
    #      vive DENTRO de hybrid_search (KAG_PROPOSITION_CHANNEL); aquí solo
    #      queda el marco temático. Ambos degradan a [] sin romper;
    #      KAG_SUMMARY_CHANNEL apaga el marco.
    summary_hits = []
    if "summaries" in channels and _kag_config_value(
        session, "KAG_SUMMARY_CHANNEL", True
    ):
        try:
            summary_hits = search_summaries(
                session, query, q_emb, top_k=global_top_k, verbose=verbose
            )
        except Exception as exc:  # noqa: BLE001 — degradación natural
            _safe_rollback(session)
            if verbose:
                print(f"[KAG] ⚠ Índice de resúmenes falló: {exc}")
    # 1.2. Estrategia subqueries: descomponer la consulta en subconsultas
    #      atómicas (LLM grande), recuperar cada una en paralelo y fusionar
    #      con RRF. Si la descomposición degrada a la original → path normal.
    if strategy == "subqueries":
        subs = generate_subqueries(session, query)
        if len(subs) == 1 and subs[0]["query"] == query:
            # Degradación: la descomposición devolvió la original → path normal.
            retrieval = _retrieval_phase(
                session,
                query,
                k,
                q_emb,
                doc_ids=metadata_doc_ids,
                verbose=verbose,
                channels=channels,
            )
        else:
            if verbose:
                print(f"[KAG] Subqueries: {len(subs)} — {[s['query'] for s in subs]}")
            with ThreadPoolExecutor(max_workers=min(4, len(subs))) as pool:
                futures = [
                    pool.submit(
                        _with_own_session,
                        _retrieval_phase,
                        s["query"],
                        k,
                        q_emb,
                        None,
                        verbose,
                        channels,
                    )
                    for s in subs
                ]
                results = [f.result() for f in futures]
            merged_hits = rrf_merge(
                *[r["merged_hits"] for r in results],
                k=60,
                top_k=sum(len(r["merged_hits"]) for r in results),
            )
            retrieval = {
                "merged_hits": merged_hits,
                "entity_ids": list(
                    dict.fromkeys(eid for r in results for eid in r["entity_ids"])
                ),
                "ppr_scores": {},
            }
    else:
        retrieval = _retrieval_phase(
            session,
            query,
            k,
            q_emb,
            doc_ids=metadata_doc_ids,
            verbose=verbose,
            channels=channels,
        )
    merged_hits = retrieval["merged_hits"]
    merged_ids = [cid for cid, _score in merged_hits]
    fetched = chunks_by_ids(session, merged_ids)
    score_by_id = dict(merged_hits)
    chunks = []
    for c in fetched:
        c["score"] = score_by_id.get(c["chunk_id"], 0.0)
        chunks.append(c)
    # Reranker opcional (cross-encoder): reordena los anclas por afinidad
    # semántica exacta antes de expandir la ventana. Se activa con la flag
    # --rerank o con RERANK_ENABLED; si el modelo no carga, degrada sin
    # rerank (mismo orden).
    if rerank or RERANK_ENABLED:
        chunks = rerank_chunks(query, chunks, enabled=rerank or RERANK_ENABLED)
        if verbose:
            print(f"[KAG] Reranker: {len(chunks)} anclas reordenadas (cross-encoder)")
    chunks = _group_chunks_with_window(session, chunks)
    chunk_ids = [c["chunk_id"] for c in chunks]
    entity_ids = retrieval["entity_ids"]

    # 2. Proposiciones de los chunks ganadores.
    propositions = propositions_for_chunks(session, chunk_ids)
    if verbose:
        print(f"[KAG] Proposiciones: {len(propositions)}")

    # 3-5. Auditoría epistémica FUSIONADA en UNA llamada al LLM pequeño
    #       (síntesis + contradicciones + suficiencia). Si el JSON falla o
    #       viene malformado, degrada a las 3 llamadas separadas.
    corpus_metadata = []
    try:
        rows = session.execute(
            text(
                "SELECT title, doc_type, summary FROM kag_documents "
                "WHERE status = 'ready' LIMIT 20"
            )
        ).fetchall()
        corpus_metadata = [
            {
                "title": r.title,
                "doc_type": r.doc_type,
                "summary": r.summary or "",
            }
            for r in rows
        ]
    except Exception as exc:  # noqa: BLE001 — degradación natural
        _safe_rollback(session)
        if verbose:
            print(f"[KAG] ⚠ Metadatos de corpus fallaron ({exc}).")
    facts, contradiction_report, evaluation = _audit_epistemic_fused(
        session,
        query,
        propositions,
        corpus_metadata,
        verbose=verbose,
        summary_hits=summary_hits,
        chunks=chunks,
    )
    verdict = evaluation.get("verdict", "SUFFICIENT_FOR_SYNTHESIS")

    if verdict == "NEGATIVE_REJECTION":
        details = evaluation.get("negative_rejection_details") or {}
        abstention = details.get("formal_abstention_statement") or (
            "El corpus disponible no cubre el dominio conceptual requerido "
            "por la consulta (abstención temprana)."
        )
        if verbose:
            print(f"[KAG] ⛔ Abstención formal: {abstention}")
        return {
            "answer": abstention,
            "verdict": verdict,
            "grounded_evidence": [],
            "epistemic_tensions": [],
            "used_fallback": False,
        }

    # 5b. Branch B (expansión iterativa) — solo si el veredicto lo pide y
    #      KAG_BRANCH_B_MAX_ITERS > 0. Aborta de inmediato si la expansión
    #      no recupera doc_ids/entidades nuevos o si la ganancia de
    #      suficiencia es marginal (Δ < 0.05).
    visited_chunks = [p.get("chunk_id") for p in propositions if p.get("chunk_id")]
    max_iterations = int(_kag_config_value(session, "KAG_BRANCH_B_MAX_ITERS", 2))
    iteration = 0
    prev_confidence = float(evaluation.get("confidence_score") or 0.0)
    while (
        verdict == "INSUFFICIENT_TRIGGER_BRANCH_B"
        and max_iterations > 0
        and iteration < max_iterations
    ):
        if verbose:
            print(f"[KAG] Branch B iteración {iteration} (max {max_iterations}).")
        extra = _branch_b_expand(
            session, query, entity_ids, visited_chunks, top_k=k, verbose=verbose
        )
        if not extra:
            if verbose:
                print("[KAG] Branch B abortado: sin proposiciones nuevas.")
            break
        new_doc_ids = {p.get("document_id") for p in extra if p.get("document_id")}
        new_entities = set()
        for p in extra:
            for ent in p.get("entities") or []:
                if ent:
                    new_entities.add(str(ent))
        if not new_doc_ids and not new_entities:
            if verbose:
                print(
                    "[KAG] Branch B abortado: sin doc_ids ni entidades nuevos "
                    "(short-circuit)."
                )
            break
        extra_facts = _synthesize_facts(session, query, extra, verbose=verbose)
        facts = list(facts) + extra_facts
        contradiction_report = _resolve_contradictions(
            session, query, facts, verbose=verbose
        )
        evaluation = _evaluate_sufficiency(
            session, query, facts, corpus_metadata, verbose=verbose
        )
        verdict = evaluation.get("verdict", "SUFFICIENT_FOR_SYNTHESIS")
        new_confidence = float(evaluation.get("confidence_score") or 0.0)
        gain = new_confidence - prev_confidence
        prev_confidence = new_confidence
        iteration += 1
        visited_chunks = list(
            dict.fromkeys(visited_chunks + [p.get("chunk_id") for p in extra])
        )
        if gain < 0.05:
            if verbose:
                print(
                    f"[KAG] Branch B abortado: ganancia marginal (Δ={gain:.3f} < 0.05)."
                )
            break

    # 6. Grounding verbatim.
    grounded_evidence = _verify_grounding(session, facts, verbose=verbose)

    # 7. Tensiones epistémicas.
    tensions = []
    for case in contradiction_report.get("analysis_cases") or []:
        tensions.append(
            {
                "tension_label": case.get(
                    "conflict_type", "paradigmatic_theoretical_divergence"
                ),
                "divergence_description": case.get("divergence_summary", ""),
                "framework_a": (case.get("thesis_a") or {}).get(
                    "author_or_framework", ""
                ),
                "framework_b": (case.get("thesis_b") or {}).get(
                    "author_or_framework", ""
                ),
            }
        )

    # 8. Respuesta final con el LLM grande.
    retries, fallback = _settings_retries(session)
    _small_model, large_model = _settings_models(session)
    used_fallback = False
    system, user_template = _get_prompt_pair(
        session,
        large_model,
        TASK_ANSWER,
        AUDITED_ANSWER_SYSTEM_SHORT,
    )
    prompt = user_template.format(
        query=query,
        grounded_evidence_json=_json_dumps(grounded_evidence),
        epistemic_tensions_json=_json_dumps(tensions),
    )
    try:
        text_out, _model, used_fallback = call_with_retries(
            session,
            prompt=prompt,
            system=system,
            model_size="large",
            response_format=None,
            retries=retries,
            fallback_model=fallback,
        )
        answer = text_out
    except Exception as exc:  # noqa: BLE001 — degradación: contexto crudo
        _safe_rollback(session)
        if verbose:
            print(f"[KAG] ⚠ Respuesta LLM falló ({exc}); contexto crudo.")
        answer = _json_dumps(
            {"grounded_evidence": grounded_evidence, "epistemic_tensions": tensions}
        )
        used_fallback = True

    if verbose:
        print(f"[KAG] Verdict: {verdict} | Evidencia: {len(grounded_evidence)}")
    return {
        "answer": answer,
        "verdict": verdict,
        "grounded_evidence": grounded_evidence,
        "epistemic_tensions": tensions,
        "used_fallback": used_fallback,
    }


# ---------------------------------------------------------------------
# Flujo completo
# ---------------------------------------------------------------------


def ask(
    session,
    query,
    top_k=8,
    global_top_k=20,
    verbose=True,
    history=None,
    mode="fast",
    rerank=False,
):
    """Flujo completo de consulta KAG (§3.1 del diseño).

    `history` (opcional) es una lista de dicts {"role", "content"} del
    historial de conversación. PREPARADO pero aún sin probar con Docker:
    se incluye como sección informativa en el contexto, no modifica la
    búsqueda.

    `mode` (opcional): "fast" (default) devuelve la respuesta como str con
    el contexto clásico + proposiciones atómicas de los chunks ganadores;
    "audited" devuelve un dict con la auditoría epistémica completa
    (síntesis, contradicciones, suficiencia, Branch B, grounding).

    `rerank` (opcional): True activa el reranker cross-encoder para esta
    consulta (independiente de RERANK_ENABLED).
    """
    if mode == "audited":
        return _ask_audited(
            session,
            query,
            top_k=top_k,
            global_top_k=global_top_k,
            verbose=verbose,
            rerank=rerank,
        )
    if verbose:
        print(f"\n🔎 Pregunta: {query}")

    # 1. Clasificar (SLM decide la estrategia; el LLM planifica — agentes B/C/D).
    #    Con KAG_QUERY_ALL_CHANNELS=True (default) se SALTEA el routing: se
    #    recupera de TODOS los canales a la vez (densa + FTS + paráfrasis +
    #    proposiciones + grafo + resúmenes) y se le muestran al LLM todos los
    #    resultados de los índices juntos — los resúmenes dan el marco, pero
    #    los detalles específicos (que a veces no están en los resúmenes)
    #    siempre llegan por los canales de chunks/proposiciones.
    all_channels = _kag_config_value(session, "KAG_QUERY_ALL_CHANNELS", True)
    if all_channels:
        strategy = "graph"
        channels = ["dense", "fts", "paraphrase", "propositions", "graph", "summaries"]
        top_k_label = "standard"
        reason = "KAG_QUERY_ALL_CHANNELS: todos los canales a la vez"
        used_fallback = False
        k = top_k
    else:
        strategy, channels, top_k_label, reason, used_fallback = (
            classify_query_strategy(session, query)
        )
        k = _top_k_for(strategy, top_k_label, top_k, global_top_k)
    if verbose:
        print(f"[KAG] Clasificación: {strategy} (top_k={k})")
        print(f"[KAG] Canales: {channels} | top_k: {top_k_label}")
        if reason:
            print(f"[KAG]   razón: {reason}")
        if used_fallback:
            print(f"[KAG]   ⚠ fallback de estrategia: {reason}")
        if strategy == "multidoc":
            print(f"[KAG] Estrategia: {strategy} (planificación LLM pendiente)")

    # 1.5. Estrategia metadata: extraer filtros de metadatos (LLM grande) y
    #      resolver los doc_ids que los cumplen. Si la extracción falla
    #      (filters vacío) → comportamiento local normal (sin filtro).
    metadata_doc_ids = None
    if strategy == "metadata":
        try:
            filters = extract_metadata_filters(session, query)
            doc_ids = _doc_ids_for_filters(session, filters.get("filters") or {})
            if doc_ids:
                metadata_doc_ids = doc_ids
                if verbose:
                    print(
                        f"[KAG] Metadata: {len(doc_ids)} doc(s) filtrado(s) — "
                        f"{filters.get('reason') or ''}"
                    )
        except Exception as exc:  # noqa: BLE001 — degradación natural
            session.rollback()  # la transacción queda abortada tras el error
            if verbose:
                print(f"[KAG] ⚠ Filtros de metadatos fallaron ({exc}); sin filtro.")

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
    # 2.1. Canales nuevos (§3.2/§3.3): proposiciones (capa micro) y
    #      resúmenes temáticos (marco global). El canal de proposiciones ya
    #      vive DENTRO de hybrid_search (KAG_PROPOSITION_CHANNEL); aquí solo
    #      queda el marco temático. Ambos degradan a [] sin romper;
    #      KAG_SUMMARY_CHANNEL apaga el marco.
    summary_hits = []
    if "summaries" in channels and _kag_config_value(
        session, "KAG_SUMMARY_CHANNEL", True
    ):
        try:
            summary_hits = search_summaries(
                session, query, q_emb, top_k=global_top_k, verbose=verbose
            )
        except Exception as exc:  # noqa: BLE001 — degradación natural
            session.rollback()  # la transacción queda abortada tras el error
            if verbose:
                print(f"[KAG] ⚠ Índice de resúmenes falló: {exc}")
    # 2.5. Estrategia subqueries: descomponer la consulta en subconsultas
    #      atómicas (LLM grande), recuperar cada una en paralelo y fusionar
    #      con RRF. Si la descomposición degrada a la original → path normal.
    if strategy == "subqueries":
        subs = generate_subqueries(session, query)
        if len(subs) == 1 and subs[0]["query"] == query:
            # Degradación: la descomposición devolvió la original → path normal.
            retrieval = _retrieval_phase(
                session,
                query,
                k,
                q_emb,
                doc_ids=metadata_doc_ids,
                verbose=verbose,
                channels=channels,
            )
        else:
            if verbose:
                print(f"[KAG] Subqueries: {len(subs)} — {[s['query'] for s in subs]}")
            with ThreadPoolExecutor(max_workers=min(4, len(subs))) as pool:
                futures = [
                    pool.submit(
                        _with_own_session,
                        _retrieval_phase,
                        s["query"],
                        k,
                        q_emb,
                        None,
                        verbose,
                        channels,
                    )
                    for s in subs
                ]
                results = [f.result() for f in futures]
            merged_hits = rrf_merge(
                *[r["merged_hits"] for r in results],
                k=60,
                top_k=sum(len(r["merged_hits"]) for r in results),
            )
            retrieval = {
                "merged_hits": merged_hits,
                "entity_ids": list(
                    dict.fromkeys(eid for r in results for eid in r["entity_ids"])
                ),
                "ppr_scores": {},
            }
    else:
        retrieval = _retrieval_phase(
            session,
            query,
            k,
            q_emb,
            doc_ids=metadata_doc_ids,
            verbose=verbose,
            channels=channels,
        )
    # 3. Merge + dedup: RRF sobre las capas (vector, regex, PPR) → una sola
    #    lista de chunks, sin duplicación. Cada capa aporta su rank; el RRF
    #    es escala-agnóstico (ts_rank, coseno y menciones no comparten
    #    escala). El canal de proposiciones ya vive DENTRO de hybrid_search.
    merged_hits = retrieval["merged_hits"]
    # Una sola consulta por todos los ids (en vez del loop N+1 por hit),
    # preservando el orden de importancia del RRF (array_position). El
    # subgrafo de tripletas es independiente (solo usa las entidades PPR),
    # así que ambas lecturas corren en paralelo.
    merged_ids = [cid for cid, _score in merged_hits]
    score_by_id = dict(merged_hits)
    triple_entities = (
        ppr_entity_selection(retrieval["ppr_scores"], verbose=False)
        or retrieval["entity_ids"]
    )
    if _kag_config_value(session, "KAG_QUERY_PARALLEL", True):
        with ThreadPoolExecutor(max_workers=2) as pool:
            chunks_future = pool.submit(_with_own_session, chunks_by_ids, merged_ids)
            triples_future = pool.submit(
                _with_own_session, subgraph_triples, triple_entities, 25
            )
            fetched = chunks_future.result()
            triples = triples_future.result()
    else:
        fetched = chunks_by_ids(session, merged_ids)
        triples = subgraph_triples(session, triple_entities, limit=25)
    chunks = []
    for c in fetched:
        c["score"] = score_by_id.get(c["chunk_id"], 0.0)
        chunks.append(c)
    if verbose:
        print(f"[KAG] Merge: {len(chunks)} chunks ancla (vector + regex + PPR, dedup)")

    # 5.5. Reranker opcional (cross-encoder): reordena los anclas por
    #      afinidad semántica exacta (query, chunk) antes de expandir la
    #      ventana. Se activa con la flag --rerank o con RERANK_ENABLED;
    #      si el modelo no carga, degrada sin rerank (mismo orden).
    if rerank or RERANK_ENABLED:
        chunks = rerank_chunks(query, chunks, enabled=rerank or RERANK_ENABLED)
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
    if verbose:
        print(f"[KAG] Subgrafo: {len(triples)} tripletas")

    # 7-8.5. Figuras, resúmenes y proposiciones: lecturas ortogonales sobre
    #      los chunks ganadores (ya con ventana) — corren en paralelo.
    chunk_ids = [c["chunk_id"] for c in chunks]
    doc_ids = list({c["doc_id"] for c in chunks})
    if _kag_config_value(session, "KAG_QUERY_PARALLEL", True):
        with ThreadPoolExecutor(max_workers=3) as pool:
            figures_future = pool.submit(
                _with_own_session, figures_for_chunks, chunk_ids
            )
            summaries_future = pool.submit(_with_own_session, doc_summaries, doc_ids)
            propositions_future = pool.submit(
                _with_own_session, propositions_for_chunks, chunk_ids
            )
            figures = figures_future.result()
            summaries = summaries_future.result()
            propositions = propositions_future.result()
    else:
        figures = figures_for_chunks(session, chunk_ids)
        summaries = doc_summaries(session, doc_ids)
        propositions = propositions_for_chunks(session, chunk_ids)
    if verbose:
        print(f"[KAG] Figuras: {len(figures)}")
        print(f"[KAG] Resúmenes: {len(summaries)} documentos")
        print(f"[KAG] Proposiciones: {len(propositions)}")

    # 9. Ensamblar contexto
    context = assemble_context(
        chunks,
        triples,
        figures,
        summaries,
        query,
        history=history,
        propositions=propositions,
        summary_hits=summary_hits,
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
    parser.add_argument(
        "--mode",
        choices=["fast", "audited"],
        default="fast",
        help="Modo de consulta: fast (respuesta directa) o audited (auditoría epistémica).",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        default=False,
        help=(
            "Activa el reranker cross-encoder (jina-reranker-v2, ~1GB) para "
            "reordenar los chunks por afinidad semántica exacta antes de armar "
            "el contexto."
        ),
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
            mode=args.mode,
            rerank=args.rerank,
        )
        print(answer)
    finally:
        session.close()


if __name__ == "__main__":
    main()
