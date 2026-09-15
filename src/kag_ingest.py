"""
Ingesta del sistema KAG (Fase 1 del diseño).

Indexa los `.md` de `data/knowledge_repository/docs/` (+ figuras de
`images/[docname]/` si existen) en las tablas KAG: segmentación con el
segmentador propio, embeddings locales Jina, extracción LLM de entidades y
relaciones, descripción VLM de figuras y resumen jerárquico con Qwen 2.5
local (fuente secundaria).

Módulo LIGERO a propósito: el segmentador (torch/spacy/sentence-transformers)
se importa SOLO dentro de las funciones que lo necesitan, nunca a nivel de
módulo — así los tests de lógica pura no cargan modelos pesados.

Uso:
    python -m src.kag_ingest                 # indexa todo lo pendiente/cambiado
    python -m src.kag_ingest --doc paper_nanotech.md
    python -m src.kag_ingest --force         # re-indexa todo
    python -m src.kag_ingest --no-summary
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from sqlalchemy import bindparam, text

from src.kag.config import get_config_value, resolve_config
from src.kag.stages import (
    KAG_INGEST_STAGES,
    cleanup_stage,
    get_stage,
    resume_from,
    set_stage,
)
from src.llm.base import call_with_retries, load_settings, parse_llm_output
from src.llm.together import complete_vision

# ---------------------------------------------------------------------
# Prompt-as-code: task keys (specs en src/db/seed_kag_prompts.py)
# ---------------------------------------------------------------------

TASK_EXTRACT_ENTITIES = "kag_extract_entities"
TASK_QWEN_SUMMARY = "kag_qwen_summary"
TASK_PROPOSITION_CHUNKING = "kag_proposition_chunking"
TASK_DOCUMENT_SEPARATION = "kag_document_separation"
TASK_DOCUMENT_ANALYSIS = "kag_document_analysis"
TASK_CHUNK_PARAPHRASE = "kag_chunk_paraphrase"
TASK_CHAPTER_PROPOSITIONS = "kag_chapter_propositions"

# System prompts cortos actuales — fallback EXACTO de hoy cuando no hay
# artefacto compilado (tests sin DB: get_active_prompt devuelve None).
EXTRACT_SYSTEM_SHORT = "Eres un extractor de conocimiento. Devuelve JSON válido."

# Fase 1 — separación de documentos apilados (spec kag_document_separation).
# El user fallback es el template EXACTO de la spec (src/db/seed_kag_prompts.py):
# se rellena con _fill_prompt (replace por clave, NUNCA .format() — el template
# contiene llaves JSON literales).
DOCUMENT_SEPARATION_SYSTEM_SHORT = (
    "Eres un bibliotecario digital. Identifica los documentos (libros, papers, "
    "artículos) apilados en un archivo Markdown y devuelve sus límites físicos "
    "de línea."
)
DOCUMENT_SEPARATION_USER_SHORT = """Archivo: {source_file}

Esqueleto del archivo (líneas del texto plano):
---
{skeleton}
---

Documentos detectados determinísticamente (MultibookFinderTool, límites físicos por ISBN/separadores):
---
{deterministic_documents}
---

Identifica los documentos contenidos en el archivo y devuelve el JSON:
{{"documents": [{{"document_id": str, "title": str, "line_start": int, "line_end": int, "language": str}}]}}

REGLAS:
- La lista determinista es la BASE: confirma cada documento detectado (puedes ajustar títulos/idioma).
- AÑADE divisiones adicionales SOLO si encuentras libros/papers/artículos SEPARADOS que la detección física no capturó (p. ej. un libro que empieza sin ISBN ni separador).
- NO dividas un libro en capítulos/secciones: los capítulos se detectan en la Fase 2 (análisis documental), no aquí.
- Cada documento debe tener un document_id unico y estable; line_start/line_end delimitan su rango en el archivo fuente."""

# Fase 2 — análisis documental (spec kag_document_analysis). El user fallback
# es el template EXACTO de la spec (src/db/seed_kag_prompts.py): se rellena con
# _fill_prompt (replace por clave, NUNCA .format() — el template contiene llaves
# JSON literales).
TASK_DOCUMENT_ANALYSIS = "kag_document_analysis"
DOCUMENT_ANALYSIS_SYSTEM_SHORT = (
    "Eres un analista documental y bibliotecario. Produce la ficha documental "
    "(tesauro ISO 25964, clasificación LCC/LCSH, cita BibTeX) y detecta los "
    "capítulos del documento con su rango de líneas."
)
DOCUMENT_ANALYSIS_USER_SHORT = """Archivo: {source_file}
Documento: {document_id}

Contexto del documento COMPLETO (texto plano, líneas 1..N):
---
{document_context}
---

Analiza el documento y devuelve el JSON:
{{"ficha": {{"title": str, "technical_level": str, "thematic_areas_iso25964": [{{"preferred_term": str, "non_preferred_terms": [str], "scope_note_disambiguation": str, "broader_term": str, "narrower_terms": [str], "related_terms": [str]}}], "library_of_congress": {{"lcsh_terms": [str], "lcc_classification": {{"label": str, "call_number": str}}}}, "bibtex": str, "key_entities": [{{"name": str, "type": str}}]}}, "index": [{{"division": str, "chapters": [{{"chapter_id": str, "title": str, "line_start": int, "line_end": int, "has_images": bool}}]}}]}}

REGLAS:
- El contexto es el documento COMPLETO: úsalo para detectar TODOS los capítulos reales (no solo los que tengan headers de markdown).
- Los divisores estructurales (p. ej. "PART I", "Parte 1", portadas, páginas de título, entradas del TOC) NO son capítulos por sí mismos: solo son capítulos los títulos (numerados o no) que van SEGUIDOS de texto real.
- Si una división "PART X" no tiene contenido propio, úsala solo como `division` del índice, nunca como capítulo. Un capítulo debe tener un rango de líneas con contenido sustancial (no 1-2 líneas de solo título).
- Reagrupa los capítulos en un índice JERÁRQUICO: cada división (p. ej. "PART I Foundations") agrupa sus capítulos. Si el documento no tiene divisiones, usa UNA división con el título del documento.
- line_start/line_end son RELATIVOS al documento (línea 1 = primera línea del contexto).
- has_images: true si el capítulo contiene imágenes o figuras."""


def _get_system_prompt(session, model_name, task_key, fallback: str) -> str:
    """System prompt desde el artefacto compilado, o el fallback actual.

    Import perezoso + try/except: si no hay DB/artefacto (tests con sesiones
    falsas), degrada a la constante de hoy sin romper nada.
    """
    try:
        from src.llm.compiler import get_active_prompt

        artifact = get_active_prompt(session, model_name, task_key)
        if artifact is not None and artifact.prompt_text:
            return artifact.prompt_text
    except Exception:  # noqa: BLE001 — degradación natural
        pass
    return fallback


def _get_prompt_pair(
    session, model_name, task_key, system_fallback: str, user_fallback: str
) -> tuple[str, str]:
    """(system, user) desde el artefacto compilado, o los fallbacks actuales.

    El artefacto (0021) congela el SYSTEM renderizado en `prompt_text` y el
    USER template parametrizable en `user_template`. Sin artefacto (tests sin
    DB) devuelve las constantes actuales — comportamiento EXACTO de hoy.
    """
    try:
        from src.llm.compiler import get_active_prompt

        artifact = get_active_prompt(session, model_name, task_key)
        if artifact is not None and artifact.prompt_text and artifact.user_template:
            return artifact.prompt_text, artifact.user_template
    except Exception:  # noqa: BLE001 — degradación natural
        pass
    return system_fallback, user_fallback


# Raíces del repositorio de conocimiento (relativas a la raíz del proyecto).
REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "data" / "knowledge_repository" / "docs"
IMAGES_DIR = REPO_ROOT / "data" / "knowledge_repository" / "images"

# Umbral de clasificación short/long (tokens estimados).
LONG_DOC_THRESHOLD = 20000
# Tamaño máximo de chunk (tokens) para el segmentador.
CHUNK_MAX_TOKENS = 800
# Truncado del texto para el resumen local (~16k tokens ≈ 64k chars).
SUMMARY_MAX_CHARS = 16000 * 4
# Límite de contexto del LLM por llamada (chars) para la extracción de
# proposiciones atómicas (mismo valor que el proposicional).
MAX_CONTEXT_CHARS = 12000
# Umbral de agrupación por capítulo (tokens): un capítulo (chapter_id ya
# detectado por LLM, Fase 2) con MÁS tokens que esto forma su propio
# grupo de proposiciones; los capítulos ≤ umbral se agrupan a nivel de
# archivo (una llamada LLM por capítulo grande, una por archivo para el resto).
CHAPTER_TOKEN_THRESHOLD = 30000

# ---------------------------------------------------------------------
# Helpers de host / texto
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


def estimate_tokens(text: str) -> int:
    """Estimación burda de tokens (≈4 chars por token)."""
    return len(text) // 4


def normalize_entity_name(name: str) -> str:
    """Minúsculas + strip + colapsar espacios (clave de dedup)."""
    return " ".join(name.lower().strip().split())


def sanitize_text(value: str) -> str:
    """Elimina caracteres que PostgreSQL rechaza en columnas text/varchar.

    El NUL (\x00) rompe la inserción con "A string literal cannot contain NUL
    (0x00) characters" (aparece en .md extraídos de PDFs). También se quitan
    otros C0 de control (excepto \n, \r, \t) que ensucian el texto.
    """
    if not value:
        return value
    return "".join(
        ch for ch in value if ch == "\n" or ch == "\r" or ch == "\t" or ord(ch) >= 32
    )


def embedding_to_sql(emb):
    """Convierte una lista de floats a la sintaxis literal de pgvector.

    psycopg2 adapta las listas Python a numeric[], que pgvector no acepta;
    el literal '[0.1,0.2,...]' con CAST AS vector sí funciona.
    """
    if emb is None:
        return None
    return "[" + ",".join(repr(float(x)) for x in emb) + "]"


def _fill_prompt(template: str, **kwargs) -> str:
    """Rellena placeholders {name} sin tocar las llaves JSON literales del prompt.

    Los prompts contienen schemas JSON con llaves que .format() interpretaría
    como placeholders (KeyError); por eso se usa replace() por clave.
    """
    out = template
    for key, value in kwargs.items():
        out = out.replace("{" + key + "}", str(value))
    return out


# Stopwords por idioma para detect_language (heurística determinista).
_STOPWORDS = {
    "es": {
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
    },
    "en": {
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
        "as",
        "at",
        "a",
        "an",
        "by",
        "from",
    },
    "pt": {
        "o",
        "a",
        "os",
        "as",
        "de",
        "do",
        "da",
        "e",
        "em",
        "um",
        "uma",
        "que",
        "é",
        "por",
        "para",
        "com",
        "no",
        "na",
        "dos",
        "das",
    },
    "de": {
        "der",
        "die",
        "das",
        "und",
        "von",
        "zu",
        "in",
        "ist",
        "sind",
        "für",
        "mit",
        "auf",
        "den",
        "dem",
        "ein",
        "eine",
        "im",
        "nicht",
    },
    "fr": {
        "le",
        "la",
        "les",
        "de",
        "du",
        "des",
        "et",
        "en",
        "un",
        "une",
        "que",
        "est",
        "pour",
        "avec",
        "sur",
        "dans",
        "au",
        "aux",
        "pas",
    },
}


def detect_language(text: str) -> str:
    """Heurística determinista por stopwords (es/en/pt/de/fr). Default 'es'."""
    words = set(re.findall(r"[a-záéíóúüñçàèìòùâêîôû]+", text.lower()))
    if not words:
        return "es"
    best_lang, best_score = "es", 0
    for lang, stops in _STOPWORDS.items():
        score = len(words & stops)
        if score > best_score:
            best_lang, best_score = lang, score
    return best_lang


# ---------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------


def chunk_markdown(
    md_text,
    doc_type,
    segmenter,
    max_tokens=CHUNK_MAX_TOKENS,
    use_coref=True,
    chapters=None,
):
    """Segmenta el texto directamente (sin jerarquía de headers determinista).

    El `section_path` (jerarquía de headers markdown calculada por esta
    función) se ELIMINA: la relación nueva es kag_chapters (detectados por
    LLM, Fase 2) → kag_chunks.chapter_id. Los chunks se segmentan DENTRO de
    cada capítulo (Fase 3): cada chunk resultante lleva el `chapter_id` del
    capítulo que lo contiene.

    `chapters` (opcional): lista de dicts {"chapter_id": uuid_str,
    "line_start": int, "line_end": int} con líneas RELATIVAS a `md_text`
    (1-based). Si es None o vacío, se segmenta el texto completo con
    `chapter_id = None` (comportamiento clásico EXACTO). Si hay capítulos,
    se extrae el texto de cada uno (`\n`.join de sus líneas), se segmenta
    con la misma lógica de fusión hasta `max_tokens`, y cada chunk lleva el
    `chapter_id` del capítulo. Los capítulos se procesan en orden; los
    chunks se numeran globalmente por chunk_index (el INSERT ya lo hace con
    enumerate).

    El segmentador produce cortes semánticos (a veces muy pequeños, ~50-100
    tokens). Para la ingesta KAG fusionamos segmentos adyacentes hasta
    `max_tokens` (respetando los cortes del segmentador como fronteras
    duras): reduce el número de chunks (y de llamadas LLM de extracción) sin
    perder los límites semánticos.

    `use_coref=False` omite la resolución de correferencias del segmentador
    (monkeypatch temporal en la instancia): el coref Stanza cuesta ~11s por
    segmento y es prohibitivo en book stacks; los docs cortos sí lo usan.

    Devuelve lista de dicts {"content", "chapter_id", "token_estimate"}.
    """
    # Coref opcional: no-op temporal en la instancia (no se modifica el
    # segmentador; el método original se restaura al salir).
    _orig_resolve = getattr(segmenter, "resolve_coreferences", None)
    if not use_coref and _orig_resolve is not None:
        segmenter.resolve_coreferences = lambda segments: segments  # noqa: E731
    try:
        chunks = []
        if not chapters:
            # Sin capítulos: comportamiento clásico EXACTO (texto completo,
            # chapter_id=None).
            return _merge_segments(
                segmenter.segment_text(md_text.strip(), max_tokens=max_tokens),
                chapter_id=None,
                max_tokens=max_tokens,
            )
        lines = md_text.splitlines()
        for chapter in chapters:
            ls = int(chapter.get("line_start") or 1)
            le = int(chapter.get("line_end") or len(lines))
            if ls > le:
                ls, le = le, ls
            ls = max(1, min(ls, len(lines)))
            le = max(1, min(le, len(lines)))
            chapter_text = "\n".join(lines[ls - 1 : le])
            chapter_id = chapter.get("chapter_id")
            chunks.extend(
                _merge_segments(
                    segmenter.segment_text(chapter_text.strip(), max_tokens=max_tokens),
                    chapter_id=chapter_id,
                    max_tokens=max_tokens,
                )
            )
        return chunks
    finally:
        if _orig_resolve is not None:
            segmenter.resolve_coreferences = _orig_resolve


def _merge_segments(segments, chapter_id, max_tokens):
    """Fusiona segmentos adyacentes hasta `max_tokens` (fronteras duras).

    El segmentador produce cortes semánticos (a veces muy pequeños, ~50-100
    tokens); fusionar reduce el número de chunks (y de llamadas LLM de
    extracción) sin perder los límites semánticos. Cada chunk resultante
    lleva `chapter_id` (None = nivel documento).
    """
    chunks = []
    buffer = ""
    buffer_tokens = 0
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        seg_tokens = estimate_tokens(seg)
        # Fusiona hasta max_tokens; un segmento que ya excede se
        # inserta solo (el segmentador ya lo cortó quirúrgicamente).
        if buffer and buffer_tokens + seg_tokens > max_tokens:
            chunks.append(
                {
                    "content": buffer,
                    "chapter_id": chapter_id,
                    "token_estimate": estimate_tokens(buffer),
                }
            )
            buffer = ""
            buffer_tokens = 0
        if buffer:
            buffer += "\n\n" + seg
            buffer_tokens += seg_tokens
        else:
            buffer = seg
            buffer_tokens = seg_tokens
    if buffer:
        chunks.append(
            {
                "content": buffer,
                "chapter_id": chapter_id,
                "token_estimate": estimate_tokens(buffer),
            }
        )
    return chunks


# ---------------------------------------------------------------------
# LLM local (Qwen 2.5 vía llama.cpp server, API OpenAI-compatible)
# ---------------------------------------------------------------------


def complete_local(
    session,
    prompt,
    system=None,
    max_tokens=None,
    temperature=None,
    timeout=120.0,
):
    """Llama al modelo local activo (provider='local') en {base_url}/chat/completions.

    Lee el syntax_profile del modelo (base_url, sampling) de la DB. Retry
    simple (2 intentos). Lanza excepción si no hay modelo local o el servidor
    no responde — el caller degrada (p. ej. summary='').
    """
    row = session.execute(
        text(
            "SELECT model_name, syntax_profile FROM llm_models "
            "WHERE provider = 'local' AND is_active = TRUE "
            "ORDER BY created_at DESC LIMIT 1"
        )
    ).first()
    if row is None:
        raise RuntimeError(
            "No hay modelo local activo (provider='local') en llm_models."
        )
    profile = row.syntax_profile or {}
    base_url = (profile.get("base_url") or "http://localhost:8080/v1").rstrip("/")
    sampling = profile.get("sampling") or {}

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    body = {
        "model": row.model_name,
        "messages": messages,
        "temperature": (
            temperature if temperature is not None else sampling.get("temperature", 0.1)
        ),
        "max_tokens": (
            max_tokens if max_tokens is not None else sampling.get("max_tokens", 60)
        ),
    }
    if sampling.get("top_p") is not None:
        body["top_p"] = sampling["top_p"]
    if sampling.get("repetition_penalty") is not None:
        body["repetition_penalty"] = sampling["repetition_penalty"]
    if sampling.get("stop"):
        body["stop"] = sampling["stop"]

    last_error = None
    for _attempt in range(2):
        try:
            resp = httpx.post(
                f"{base_url}/chat/completions",
                headers={
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001 — reintento simple
            last_error = exc
    raise last_error  # type: ignore[misc]


# ---------------------------------------------------------------------
# Extracción LLM de entidades y relaciones
# ---------------------------------------------------------------------

EXTRACT_PROMPT = """Extrae las entidades y relaciones del siguiente fragmento de texto.

Devuelve SOLO JSON con esta forma exacta:
{
  "entities": [
    {"name": "Nombre de la entidad", "type": "concept|method|law|person|org|figure", "description": "breve descripción"}
  ],
  "relations": [
    {"source": "Entidad origen", "target": "Entidad destino", "type": "RELACIÓN_EN_MAYÚSCULAS", "description": "breve descripción"}
  ]
}

Reglas:
- Entidades: conceptos, métodos, leyes, personas, organizaciones o figuras relevantes.
- Relaciones: solo entre entidades presentes en el fragmento.
- Si no hay entidades, devuelve {"entities": [], "relations": []}.

Texto:
<text>
{chunk}
</text>
"""


def extract_entities_relations(session, chunk_text):
    """Extrae entidades y relaciones de un chunk con el modelo pequeño (Together).

    Usa call_with_retries (retries y fallback_model de session_settings).
    Devuelve dict; si el LLM falla, devuelve {} (degradación de ingesta).
    """
    settings = load_settings(session)
    retries = int(getattr(settings, "llm_retries", 3) or 3) if settings else 3
    fallback = getattr(settings, "fallback_model", None) if settings else None
    small_model = getattr(settings, "small_model", None) if settings else None
    system, user_template = _get_prompt_pair(
        session,
        small_model,
        TASK_EXTRACT_ENTITIES,
        EXTRACT_SYSTEM_SHORT,
        EXTRACT_PROMPT,
    )
    # str.replace en vez de .format(): el prompt contiene llaves JSON literales
    # que .format() interpretaría como placeholders (KeyError).
    prompt = user_template.replace("{chunk}", chunk_text[:8000])
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
        return parse_llm_output(text_out)
    except Exception:  # noqa: BLE001 — LLM no disponible: degradación de ingesta
        return {}


def _maintain_word_freq(session, doc_id: int) -> None:
    """Reescribe el índice de frecuencia de palabras de un documento.

    Se llama al final de la etapa 'segmented' (los chunks ya están insertados
    y content_tsv es una columna generada, migración 0014). DELETE + INSERT
    por doc: idempotente y consistente con el patrón de re-indexado (misma
    característica que name_embedding: se recalcula al añadir/modificar un
    documento).
    """
    session.execute(
        text("DELETE FROM kag_word_freq WHERE doc_id = :doc_id"),
        {"doc_id": doc_id},
    )
    session.execute(
        text(
            "INSERT INTO kag_word_freq (doc_id, word, nentry) "
            "SELECT :doc_id, t.lexeme, COUNT(*) "
            "FROM kag_chunks c "
            "CROSS JOIN LATERAL unnest(c.content_tsv) AS t(lexeme, positions, weights) "
            "WHERE c.doc_id = :doc_id "
            "GROUP BY t.lexeme"
        ),
        {"doc_id": doc_id},
    )


def _store_entities_relations(session, doc_id, chunk_id, data, embed_fn=None):
    """Guarda entidades y relaciones de un chunk con dedup por name_norm (por doc).

    Batch: una sola consulta de existentes por doc + un solo INSERT multi-VALUES
    por lote de entidades y otro por lote de relaciones (en vez de N+1
    SELECT/INSERT por entidad). El grafo es incremental a nivel de documento:
    solo se tocan las filas de este doc_id (el trigger de versión 0016
    invalida la caché de adyacencia una vez por statement).

    `embed_fn` (opcional, p. ej. src.embeddings.embed_texts) embebe los nombres
    de las entidades nuevas para el entity linking por similitud (columna
    name_embedding, migración 0022). Sin embed_fn, name_embedding queda NULL.
    """
    entities = data.get("entities") or []
    relations = data.get("relations") or []
    if not entities and not relations:
        return

    # 1. Existentes del doc en UNA consulta (name_norm -> id).
    norms = [normalize_entity_name(str(e.get("name", "")).strip()) for e in entities]
    norms = [n for n in norms if n]
    existing = {}
    if norms:
        rows = session.execute(
            text(
                "SELECT name_norm, id FROM kag_entities "
                "WHERE doc_id = :doc_id AND name_norm = ANY(:norms)"
            ),
            {"doc_id": doc_id, "norms": list(dict.fromkeys(norms))},
        ).fetchall()
        existing = {r.name_norm: r.id for r in rows}

    # 2. Insertar solo las entidades nuevas (multi-VALUES, un statement).
    entity_ids = dict(existing)
    new_entities = []
    for ent in entities:
        name = sanitize_text(str(ent.get("name", "")).strip())[:300]
        if not name:
            continue
        name_norm = normalize_entity_name(name)[:300]
        if name_norm in entity_ids:
            continue
        entity_ids[name_norm] = None  # placeholder: se rellena con RETURNING
        new_entities.append(
            (
                doc_id,
                chunk_id,
                name,
                name_norm,
                sanitize_text(str(ent.get("type", "concept")))[:100],
                sanitize_text(str(ent.get("description", ""))),
            )
        )
    if new_entities:
        # Embeddings de los nombres nuevos (entity linking por similitud).
        name_embs: list = []
        if embed_fn is not None:
            try:
                name_embs = embed_fn([n for _, _, n, _, _, _ in new_entities])
            except Exception:  # noqa: BLE001 — sin embedding: name_embedding NULL
                name_embs = [None] * len(new_entities)
        else:
            name_embs = [None] * len(new_entities)
        # Un solo INSERT multi-VALUES con UN set de parámetros. Pasar una
        # lista de dicts haría executemany, y psycopg2 no devuelve filas con
        # executemany + RETURNING (ResourceClosedError "does not return rows").
        placeholders = ", ".join(
            f"(:d{i}, :c{i}, :n{i}, :nn{i}, :e{i}, :de{i}, CAST(:emb{i} AS vector))"
            for i in range(len(new_entities))
        )
        params: dict = {}
        for i, (d, c, n, nn, et, de) in enumerate(new_entities):
            params.update(
                {
                    f"d{i}": d,
                    f"c{i}": c,
                    f"n{i}": n,
                    f"nn{i}": nn,
                    f"e{i}": et,
                    f"de{i}": de,
                    f"emb{i}": embedding_to_sql(name_embs[i]),
                }
            )
        rows = session.execute(
            text(
                "INSERT INTO kag_entities "
                "(doc_id, chunk_id, name, name_norm, entity_type, description, "
                "name_embedding) "
                f"VALUES {placeholders} "
                "RETURNING id, name_norm"
            ),
            params,
        ).fetchall()
        for r in rows:
            entity_ids[r.name_norm] = r.id

    # 3. Relaciones (multi-VALUES, un statement). Solo las que referencian
    #    entidades ya insertadas/existentes de este chunk.
    new_relations = []
    for rel in relations:
        src = normalize_entity_name(sanitize_text(str(rel.get("source", ""))))
        tgt = normalize_entity_name(sanitize_text(str(rel.get("target", ""))))
        if src not in entity_ids or tgt not in entity_ids:
            continue
        if entity_ids[src] is None or entity_ids[tgt] is None:
            continue
        new_relations.append(
            (
                doc_id,
                chunk_id,
                entity_ids[src],
                entity_ids[tgt],
                sanitize_text(str(rel.get("type", "RELACIONA")))[:200],
                sanitize_text(str(rel.get("description", ""))),
            )
        )
    if new_relations:
        placeholders = ", ".join(
            f"(:d{i}, :c{i}, :s{i}, :t{i}, :rt{i}, :de{i})"
            for i in range(len(new_relations))
        )
        params = {}
        for i, (d, c, s, t, rt, de) in enumerate(new_relations):
            params.update(
                {
                    f"d{i}": d,
                    f"c{i}": c,
                    f"s{i}": s,
                    f"t{i}": t,
                    f"rt{i}": rt,
                    f"de{i}": de,
                }
            )
        session.execute(
            text(
                "INSERT INTO kag_relations "
                "(doc_id, chunk_id, source_entity_id, target_entity_id, "
                "relation_type, description) "
                f"VALUES {placeholders}"
            ),
            params,
        )


# ---------------------------------------------------------------------
# Extracción LLM de proposiciones atómicas (capa micro, migración 0024)
# ---------------------------------------------------------------------

# Intent del analista epistemológico (corto) — fallback EXACTO cuando no hay
# artefacto compilado (tests sin DB: get_active_prompt devuelve None).
PROPOSITIONAL_SYSTEM_SHORT = (
    "Eres un analista de epistemología y análisis del discurso. Tu objetivo "
    "es descomponer el texto en proposiciones atómicas autocontenidas: cada "
    "proposición debe ser gramaticalmente independiente (reemplaza anáforas "
    "como 'éste', 'lo anterior', 'dicho autor' por el sujeto explícito). "
    "'text_span' debe contener el fragmento de texto EXACTO del original. "
    "REGLA DE REFERENCIAS DUPLICADAS: si una frase contiene una referencia "
    "académica (ej. 'Bourdieu, 1984, p. 52'), dicha referencia DEBE "
    "preservarse y duplicarse en 'citations_references' de TODAS las "
    "proposiciones que deriven de ella. Devuelve JSON válido."
)

PROPOSITIONAL_USER_SHORT = """Archivo: {source_file}
Documento: {document_id}
Capítulo: {chapter_id}
Rango: Línea {line_start} a Línea {line_end}

Texto a procesar:
---
{chapter_text_content}
---

Genera el JSON con las proposiciones organizadas por divisiones (capítulos del texto):
{{"divisions": [{{"chapter_id": str, "propositions": [{{"core_idea_id": str, "argument_id": str, "statement": str, "text_span": str, "char_start": int, "char_end": int, "line_start": int, "line_end": int, "citations_references": [str]}}]}}]}}"""

# Paráfrasis de chunks (Fase 4, etapa 'paraphrased') — fallback EXACTO cuando
# no hay artefacto compilado (tests sin DB: get_active_prompt devuelve None).
# El template USER es el de la spec kag_chunk_paraphrase (seed_kag_prompts.py)
# con el schema de salida ajustado a {"paraphrases": [{"chunk_index": int,
# "paraphrase": str}]} (por chunk_index, no por chunk_id).
PARAPHRASE_SYSTEM_SHORT = (
    "Eres un parafraseador académico. Parafrasea cada chunk preservando el "
    "significado exacto, sin añadir ni omitir información."
)

PARAPHRASE_USER_SHORT = """Archivo: {source_file}
Documento: {document_id}
Capítulo: {chapter_id}

Chunks del capítulo:
---
{chunks_json}
---

Parafrasea cada chunk y devuelve el JSON:
{{"paraphrases": [{{"chunk_index": int, "paraphrase": str}}]}}"""

# Proposiciones + entidades + relaciones por documento (Fase 5, etapa
# 'chunked') — fallback EXACTO cuando no hay artefacto compilado (tests sin
# DB: get_active_prompt devuelve None). El template USER es el de la spec
# kag_chapter_propositions (seed_kag_prompts.py) con el schema de salida
# {"propositions": [{"chunk_index": int, ...}], "entities": [...],
# "relations": [...]} (asignación primaria por chunk_index).
CHAPTER_PROPOSITIONS_SYSTEM_SHORT = (
    "Eres un analista de epistemología y análisis del discurso. Tu objetivo "
    "es descomponer las paráfrasis del capítulo en proposiciones atómicas "
    "autocontenidas y extraer las entidades y relaciones del capítulo. "
    "'text_span' debe contener el fragmento de texto EXACTO de la paráfrasis. "
    "REGLA DE REFERENCIAS DUPLICADAS: si una frase contiene una referencia "
    "académica (ej. 'Bourdieu, 1984, p. 52'), dicha referencia DEBE "
    "preservarse y duplicarse en 'citations_references' de TODAS las "
    "proposiciones que deriven de ella. Devuelve JSON válido."
)

CHAPTER_PROPOSITIONS_USER_SHORT = """Archivo: {source_file}
Documento: {document_id}
Capítulo: {chapter_id}

Paráfrasis del capítulo:
---
{paraphrases_json}
---

Extrae las proposiciones atomicas y devuelve el JSON:
{{"propositions": [{{"chunk_index": int, "core_idea_id": str, "argument_id": str, "statement": str, "text_span": str, "citations_references": [str]}}], "entities": [{{"name": str, "type": str, "description": str}}], "relations": [{{"source": str, "target": str, "type": str, "description": str}}]}}"""


def _span_offsets(content: str, char_start: int, char_end: int):
    """(char_start, char_end, line_start, line_end) 1-based del chunk."""
    line_start = content.count("\n", 0, char_start) + 1
    line_end = content.count("\n", 0, max(char_start, char_end - 1)) + 1
    return (char_start, char_end, line_start, line_end)


def _map_norm_to_orig(content: str, norm_pos: int) -> int:
    """Mapea un índice del contenido normalizado (espacios colapsados) al
    índice original del chunk."""
    norm_idx = 0
    prev_ws = False
    for orig_idx, ch in enumerate(content):
        if ch.isspace():
            if prev_ws:
                continue
            prev_ws = True
        else:
            prev_ws = False
        if norm_idx == norm_pos:
            return orig_idx
        norm_idx += 1
    return len(content)


def _locate_span_in_chunk(content: str, text_span: str):
    """Localiza `text_span` dentro del chunk (substring, normalizando espacios).

    Devuelve (char_start, char_end, line_start, line_end) absolutos del chunk
    (líneas 1-based). Si el span no aparece (el LLM parafraseó), devuelve
    (None, None, None, None) — la proposición se guarda igual con spans NULL.
    """
    if not text_span:
        return (None, None, None, None)
    pos = content.find(text_span)
    if pos != -1:
        return _span_offsets(content, pos, pos + len(text_span))
    # Normalización de espacios: colapsar runs de whitespace a un espacio.
    norm_content = re.sub(r"\s+", " ", content)
    norm_span = re.sub(r"\s+", " ", text_span).strip()
    if not norm_span:
        return (None, None, None, None)
    npos = norm_content.find(norm_span)
    if npos == -1:
        return (None, None, None, None)
    char_start = _map_norm_to_orig(content, npos)
    char_end = _map_norm_to_orig(content, npos + len(norm_span))
    return _span_offsets(content, char_start, char_end)


def _propositions_from_llm(data, default_chapter_id=""):
    """Extrae las proposiciones del output del LLM (organizado por divisiones).

    Formato nuevo (exigido por el prompt): {"divisions": [{"chapter_id":
    str, "propositions": [...]}]}. Formato legacy (artefacto compilado
    anterior a la agrupación por capítulo): {"propositions": [...]} — se
    trata como una única división con `default_chapter_id`. Devuelve lista
    de (chapter_id, prop_dict) o [] si no hay proposiciones.
    """
    if not isinstance(data, dict):
        return []
    divisions = data.get("divisions")
    if isinstance(divisions, list) and divisions:
        out = []
        for div in divisions:
            if not isinstance(div, dict):
                continue
            div_path = str(div.get("chapter_id") or "").strip()
            div_path = div_path or default_chapter_id
            for p in div.get("propositions") or []:
                if isinstance(p, dict):
                    out.append((div_path, p))
        return out
    propositions = data.get("propositions")
    if isinstance(propositions, list):
        return [(default_chapter_id, p) for p in propositions if isinstance(p, dict)]
    return []


def _extract_propositions(
    session, doc_id, chunk_id, content, doc_path, chunk_index, chapter_id, verbose
):
    """Extrae proposiciones atómicas de un chunk con el modelo pequeño.

    Usa _get_prompt_pair (artefacto compilado o fallback a constantes) +
    call_with_retries (model_size="small"). Devuelve lista de dicts
    {doc_id, chunk_id, chapter_id, core_idea_id, argument_id, statement,
    text_span, char_start, char_end, line_start, line_end, citations}. Si el
    LLM falla o devuelve JSON inválido → [] (degradación: la ingesta sigue).
    """
    settings = load_settings(session)
    retries = int(getattr(settings, "llm_retries", 3) or 3) if settings else 3
    fallback = getattr(settings, "fallback_model", None) if settings else None
    small_model = getattr(settings, "small_model", None) if settings else None
    system, user_template = _get_prompt_pair(
        session,
        small_model,
        TASK_PROPOSITION_CHUNKING,
        PROPOSITIONAL_SYSTEM_SHORT,
        PROPOSITIONAL_USER_SHORT,
    )
    # Truncado del contexto (~12000 chars) si el chunk es muy grande.
    chunk_text = content[:MAX_CONTEXT_CHARS]
    prompt = _fill_prompt(
        user_template,
        source_file=doc_path,
        document_id=doc_id,
        chapter_id=chapter_id or "",
        chunk_index=chunk_index,
        chunk_title=chapter_id or "",
        line_start=1,
        line_end=chunk_text.count("\n") + 1,
        chapter_text_content=chunk_text,
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
    except Exception as exc:  # noqa: BLE001 — LLM no disponible: degradación
        if verbose:
            print(f"[KAG] ⚠ Proposiciones fallaron (chunk {chunk_id}): {exc}")
        return []
    out = []
    for div_path, p in _propositions_from_llm(
        data, default_chapter_id=chapter_id or ""
    ):
        statement = str(p.get("statement") or "").strip()
        if not statement:
            continue
        text_span = str(p.get("text_span") or "")
        char_start, char_end, line_start, line_end = _locate_span_in_chunk(
            chunk_text, text_span
        )
        citations = p.get("citations_references") or []
        if not isinstance(citations, list):
            citations = []
        out.append(
            {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "chapter_id": div_path,
                "core_idea_id": str(p.get("core_idea_id") or "")[:50],
                "argument_id": str(p.get("argument_id") or "")[:50],
                "statement": statement,
                "text_span": text_span,
                "char_start": char_start,
                "char_end": char_end,
                "line_start": line_start,
                "line_end": line_end,
                "citations": [str(c) for c in citations],
            }
        )
    return out


def _store_propositions(session, doc_id, propositions, embed_fn=None):
    """Guarda proposiciones atómicas de un chunk (multi-VALUES, un statement).

    Mismo patrón que _store_entities_relations: UN solo INSERT multi-VALUES
    con placeholders :d0,:c0,... (una lista de dicts haría executemany, y
    psycopg2 no devuelve filas con executemany + RETURNING). `embed_fn`
    embebe los statements en batch (degradación: [None]*n si falla).
    `citation_references` = json.dumps(citations).
    """
    if not propositions:
        return
    embs: list = []
    if embed_fn is not None:
        try:
            embs = embed_fn([p["statement"] for p in propositions])
        except Exception:  # noqa: BLE001 — sin embedding: embedding NULL
            embs = [None] * len(propositions)
    else:
        embs = [None] * len(propositions)
    placeholders = ", ".join(
        f"(:d{i}, :c{i}, :ch{i}, :ci{i}, :a{i}, :s{i}, :ts{i}, :cs{i}, :ce{i}, "
        f":ls{i}, :le{i}, CAST(:cr{i} AS jsonb), CAST(:emb{i} AS vector))"
        for i in range(len(propositions))
    )
    params: dict = {}
    for i, p in enumerate(propositions):
        params.update(
            {
                f"d{i}": p["doc_id"],
                f"c{i}": p["chunk_id"],
                f"ch{i}": p.get("chapter_id"),
                f"ci{i}": p["core_idea_id"],
                f"a{i}": p["argument_id"],
                f"s{i}": sanitize_text(p["statement"]),
                f"ts{i}": sanitize_text(p["text_span"]),
                f"cs{i}": p["char_start"],
                f"ce{i}": p["char_end"],
                f"ls{i}": p["line_start"],
                f"le{i}": p["line_end"],
                f"cr{i}": json.dumps(p["citations"], ensure_ascii=False),
                f"emb{i}": embedding_to_sql(embs[i]),
            }
        )
    session.execute(
        text(
            "INSERT INTO kag_propositions "
            "(doc_id, chunk_id, chapter_id, core_idea_id, argument_id, statement, "
            "text_span, char_start, char_end, line_start, line_end, "
            "citation_references, embedding) "
            f"VALUES {placeholders}"
        ),
        params,
    )


def _chunk_content_hash(content: str) -> str:
    """sha256 hex del contenido del chunk (cache de proposiciones, 0026)."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _proposition_flags(
    session, extract_propositions: bool
) -> tuple[bool, int, str, int]:
    """Deriva las flags de proposiciones desde config (compatibilidad).

    `extract_propositions` (param de index_document) se combina con la flag
    KAG_EXTRACT_PROPOSITIONS de config: ambas deben estar activas. Devuelve
    (enabled, batch_size_tokens, model_size, max_parallel) con
    KAG_PROPOSITION_BATCH_SIZE (default 400000 tokens), KAG_PROPOSITION_MODEL
    (default "large" — la respuesta de un lote de hasta 400k tokens es mucho
    mayor en extensión que la de un chunk suelto, así que la extracción usa
    el LLM grande) y KAG_PROPOSITION_PARALLEL (default 3 llamadas LLM
    concurrentes, semáforo).
    """
    config = resolve_config(session)
    enabled = bool(extract_propositions) and bool(
        get_config_value(config, "KAG_EXTRACT_PROPOSITIONS", True)
    )
    batch_size = int(
        get_config_value(config, "KAG_PROPOSITION_BATCH_SIZE", 400000) or 400000
    )
    model_size = get_config_value(config, "KAG_PROPOSITION_MODEL", "large")
    if model_size not in ("small", "large"):
        model_size = "large"
    max_parallel = int(get_config_value(config, "KAG_PROPOSITION_PARALLEL", 3) or 3)
    if max_parallel < 1:
        max_parallel = 3
    return enabled, batch_size, model_size, max_parallel


def _batch_chunks_by_tokens(
    chunks, batch_size_tokens: int, estimate_fn=estimate_tokens
):
    """Agrupa chunks consecutivos por presupuesto de tokens.

    Acumula `estimate_fn(content)` hasta alcanzar `batch_size_tokens`; cada
    lote = UNA llamada LLM. El corte es por el límite de tokens, NO por un
    número fijo de chunks. Los chunks consecutivos del mismo `chapter_id`
    quedan juntos (orden por chunk_index); un chunk que excede el presupuesto
    forma su propio lote (no se parte).
    """
    batches: list[list[dict]] = []
    current: list[dict] = []
    current_tokens = 0
    for chunk in chunks:
        tokens = estimate_fn(chunk["content"])
        if current and current_tokens + tokens > batch_size_tokens:
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(chunk)
        current_tokens += tokens
    if current:
        batches.append(current)
    return batches


def _group_chunks_by_chapter(
    chunks, chapter_token_threshold=CHAPTER_TOKEN_THRESHOLD, estimate_fn=estimate_tokens
):
    """Agrupa chunks por documento→capítulo (chapter_id ya detectado).

    Los chunks llegan ordenados por chunk_index. Cada capítulo (chapter_id)
    con > `chapter_token_threshold` tokens forma su propio grupo (una llamada
    LLM por capítulo); los capítulos ≤ umbral se fusionan en un único grupo a
    nivel de archivo (chapter_id ""). Si TODOS los chunks tienen chapter_id
    NULL (Fase 3 aún no segmenta dentro de capítulos), se produce un único
    grupo a nivel de documento. Devuelve lista de dicts
    {"chapter_id": str, "chunks": [...]} en orden de aparición.
    """
    chapters: list[dict] = []
    for ch in chunks:
        path = ch.get("chapter_id") or ""
        if chapters and chapters[-1]["chapter_id"] == path:
            chapters[-1]["chunks"].append(ch)
        else:
            chapters.append({"chapter_id": path, "chunks": [ch]})
    groups: list[dict] = []
    file_level: list = []
    for chapter in chapters:
        tokens = sum(estimate_fn(c["content"]) for c in chapter["chunks"])
        if tokens > chapter_token_threshold:
            groups.append(chapter)
        else:
            file_level.extend(chapter["chunks"])
    if file_level:
        groups.append({"chapter_id": "", "chunks": file_level})
    return groups


def _run_proposition_batches_parallel(
    session,
    doc_id,
    doc_path,
    units,
    verbose,
    model_size="large",
    max_parallel=3,
    batch_fn=None,
):
    """Dispara las llamadas LLM de los lotes en paralelo (ThreadPoolExecutor).

    `units`: lista de (chapter_id, chunks) — cada lote es UNA llamada LLM
    independiente. `max_parallel` (semáforo) limita las llamadas LLM
    concurrentes para respetar el rate limit del proveedor. `batch_fn`:
    función de extracción por lote (default `_extract_propositions_batch`;
    el flujo por documento usa `_extract_document_batch`, que devuelve
    (propositions, entities, relations) por lote). Devuelve los resultados
    en el MISMO orden que `units` (persistencia determinista por
    chunk_index aunque se procesen en paralelo).
    """
    if batch_fn is None:
        batch_fn = _extract_propositions_batch
    sem = threading.BoundedSemaphore(max_parallel)
    results: list = [None] * len(units)

    def work(i, chapter_id, batch):
        with sem:
            return i, batch_fn(
                session,
                doc_id,
                batch,
                doc_path,
                verbose,
                model_size=model_size,
                chapter_id=chapter_id,
            )

    # El pool admite hasta 2×N workers; el semáforo es quien limita la
    # concurrencia real de llamadas LLM a N.
    pool_size = min(len(units), max_parallel * 2)
    with ThreadPoolExecutor(max_workers=pool_size) as ex:
        futures = [ex.submit(work, i, sp, b) for i, (sp, b) in enumerate(units)]
        for fut in futures:
            i, props = fut.result()
            results[i] = props
    return results


def _locate_span_in_batch(chunks, batch_text: str, offsets, text_span: str):
    """Localiza `text_span` en el texto concatenado del lote y lo asigna al
    chunk que lo contiene.

    `offsets` es una lista de (chunk_id, start, end) — rangos de cada chunk
    en `batch_text`. Devuelve (chunk_id, char_start, char_end, line_start,
    line_end) relativos al chunk (1-based), o (None,)*5 si el span no
    aparece (el LLM parafraseó) — la proposición se descarta (chunk_id es
    NOT NULL en kag_propositions).
    """
    if not text_span:
        return (None, None, None, None, None)
    char_start, char_end, _ls, _le = _locate_span_in_chunk(batch_text, text_span)
    if char_start is None:
        return (None, None, None, None, None)
    for chunk, (_cid, cstart, cend) in zip(chunks, offsets):
        if cstart <= char_start < cend:
            rel_start = char_start - cstart
            rel_end = char_end - cstart
            line_start = chunk["content"][:rel_start].count("\n") + 1
            line_end = chunk["content"][:rel_end].count("\n") + 1
            return (chunk["id"], rel_start, rel_end, line_start, line_end)
    return (None, None, None, None, None)


def _extract_propositions_batch(
    session, doc_id, chunks, doc_path, verbose, model_size="large", chapter_id=""
):
    """Extrae proposiciones de un lote de chunks con UNA llamada LLM.

    `chunks`: lista de dicts {id, content, chunk_index, chapter_id}.
    `chapter_id`: capítulo/división del lote (metadata del prompt; "" =
    nivel archivo). Concatena los contenidos (separador \n\n---\n\n) y pasa
    el texto como `chapter_text_content`; el prompt exige el output
    organizado por divisiones ({"divisions": [{"chapter_id",
    "propositions"}]}) y cada proposición se asigna al chunk que contiene su
    text_span (offsets relativos al chunk). `model_size` ("small" |
    "large", default "large" vía KAG_PROPOSITION_MODEL): la respuesta de un
    lote de hasta 400k tokens es mucho mayor en extensión que la de un chunk
    suelto, así que se usa el LLM grande (max_tokens_large). Si el LLM falla
    o devuelve JSON inválido → [] (degradación: la ingesta sigue).
    """
    if model_size not in ("small", "large"):
        model_size = "large"
    settings = load_settings(session)
    retries = int(getattr(settings, "llm_retries", 3) or 3) if settings else 3
    fallback = getattr(settings, "fallback_model", None) if settings else None
    model_name = (
        (
            getattr(settings, "large_model", None)
            if model_size == "large"
            else getattr(settings, "small_model", None)
        )
        if settings
        else None
    )
    system, user_template = _get_prompt_pair(
        session,
        model_name,
        TASK_PROPOSITION_CHUNKING,
        PROPOSITIONAL_SYSTEM_SHORT,
        PROPOSITIONAL_USER_SHORT,
    )
    sep = "\n\n---\n\n"
    batch_text = sep.join(ch["content"] for ch in chunks)
    offsets = []
    cursor = 0
    for ch in chunks:
        start = cursor
        cursor += len(ch["content"])
        offsets.append((ch["id"], start, cursor))
        cursor += len(sep)
    first_idx = chunks[0]["chunk_index"]
    last_idx = chunks[-1]["chunk_index"]
    chunk_index_label = (
        f"{first_idx}-{last_idx}" if first_idx != last_idx else str(first_idx)
    )
    prompt = _fill_prompt(
        user_template,
        source_file=doc_path,
        document_id=doc_id,
        chapter_id=chapter_id or "(archivo completo)",
        chunk_index=chunk_index_label,
        chunk_title=chunks[0].get("chapter_id") or "",
        line_start=1,
        line_end=batch_text.count("\n") + 1,
        chapter_text_content=batch_text,
    )
    try:
        text_out, _model, _used_fallback = call_with_retries(
            session,
            prompt=prompt,
            system=system,
            model_size=model_size,
            response_format={"type": "json_object"},
            retries=retries,
            fallback_model=fallback,
        )
        data = parse_llm_output(text_out)
    except Exception as exc:  # noqa: BLE001 — LLM no disponible: degradación
        if verbose:
            print(f"[KAG] ⚠ Proposiciones fallaron (lote {chunk_index_label}): {exc}")
        return []
    chunk_chapters = {ch["id"]: (ch.get("chapter_id") or "") for ch in chunks}
    out = []
    for div_path, p in _propositions_from_llm(
        data, default_chapter_id=chapter_id or ""
    ):
        statement = str(p.get("statement") or "").strip()
        if not statement:
            continue
        text_span = str(p.get("text_span") or "")
        chunk_id, char_start, char_end, line_start, line_end = _locate_span_in_batch(
            chunks, batch_text, offsets, text_span
        )
        citations = p.get("citations_references") or []
        if not isinstance(citations, list):
            citations = []
        # chapter_id: el del capítulo declarado por el LLM en la división;
        # si falta, el del chunk asignado (o "").
        prop_chapter = div_path or chunk_chapters.get(chunk_id, "")
        out.append(
            {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "chapter_id": prop_chapter,
                "core_idea_id": str(p.get("core_idea_id") or "")[:50],
                "argument_id": str(p.get("argument_id") or "")[:50],
                "statement": statement,
                "text_span": text_span,
                "char_start": char_start,
                "char_end": char_end,
                "line_start": line_start,
                "line_end": line_end,
                "citations": [str(c) for c in citations],
            }
        )
    return out


def _extract_document_batch(
    session, doc_id, chunks, doc_path, verbose, model_size="large", chapter_id=""
):
    """Extrae proposiciones + entidades + relaciones de un lote de chunks
    (paráfrasis) con UNA llamada LLM grande (spec `kag_chapter_propositions`).

    `chunks`: lista de dicts {id, content, chunk_index, chapter_id} donde
    `content` es el texto a procesar (paráfrasis o content si NULL).
    Concatena los textos (separador \n\n---\n\n) y construye offsets por
    chunk (patrón de _locate_span_in_batch). El LLM devuelve
    {"propositions": [{"chunk_index": int, ...}], "entities": [...],
    "relations": [...]}. Asignación de proposiciones a chunks: primario =
    chunk_index declarado por el LLM (mapear chunk_index → chunk id); si
    falta o es inválido, secundario = _locate_span_in_batch sobre el texto
    concatenado. Si ambos fallan → la proposición se descarta (chunk_id es
    NOT NULL en kag_propositions). `chapter_id` de cada proposición = el del
    chunk asignado (o "" si NULL). Devuelve (propositions, entities,
    relations). Si el LLM falla o devuelve JSON inválido → ([], [], [])
    (degradación: la ingesta sigue).
    """
    if model_size not in ("small", "large"):
        model_size = "large"
    settings = load_settings(session)
    retries = int(getattr(settings, "llm_retries", 3) or 3) if settings else 3
    fallback = getattr(settings, "fallback_model", None) if settings else None
    model_name = (
        (
            getattr(settings, "large_model", None)
            if model_size == "large"
            else getattr(settings, "small_model", None)
        )
        if settings
        else None
    )
    system, user_template = _get_prompt_pair(
        session,
        model_name,
        TASK_CHAPTER_PROPOSITIONS,
        CHAPTER_PROPOSITIONS_SYSTEM_SHORT,
        CHAPTER_PROPOSITIONS_USER_SHORT,
    )
    sep = "\n\n---\n\n"
    batch_text = sep.join(ch["content"] for ch in chunks)
    offsets = []
    cursor = 0
    for ch in chunks:
        start = cursor
        cursor += len(ch["content"])
        offsets.append((ch["id"], start, cursor))
        cursor += len(sep)
    # Mapa chunk_index → chunk id (asignación primaria por chunk_index).
    index_to_id = {ch["chunk_index"]: ch["id"] for ch in chunks}
    chunk_chapters = {ch["id"]: (ch.get("chapter_id") or "") for ch in chunks}
    paraphrases_json = json.dumps(
        [
            {"chunk_index": ch["chunk_index"], "paraphrase": ch["content"]}
            for ch in chunks
        ],
        ensure_ascii=False,
    )
    # Metadata del prompt: capítulos cubiertos por el lote.
    chapter_ids = sorted({str(c) for c in chunk_chapters.values() if c})
    chapter_label = ", ".join(chapter_ids) if chapter_ids else "(archivo completo)"
    prompt = _fill_prompt(
        user_template,
        source_file=str(Path(doc_path)).replace("\\", "/"),
        document_id=str(doc_id),
        chapter_id=chapter_label,
        paraphrases_json=paraphrases_json,
    )
    try:
        text_out, _model, _used_fallback = call_with_retries(
            session,
            prompt=prompt,
            system=system,
            model_size=model_size,
            response_format={"type": "json_object"},
            retries=retries,
            fallback_model=fallback,
        )
        data = parse_llm_output(text_out)
    except Exception as exc:  # noqa: BLE001 — LLM no disponible: degradación
        if verbose:
            first_idx = chunks[0]["chunk_index"]
            last_idx = chunks[-1]["chunk_index"]
            label = (
                f"{first_idx}-{last_idx}" if first_idx != last_idx else str(first_idx)
            )
            print(f"[KAG] ⚠ Proposiciones fallaron (lote {label}): {exc}")
        return [], [], []
    if not isinstance(data, dict):
        return [], [], []
    # Proposiciones: primario chunk_index, secundario _locate_span_in_batch.
    out = []
    for p in data.get("propositions") or []:
        if not isinstance(p, dict):
            continue
        statement = str(p.get("statement") or "").strip()
        if not statement:
            continue
        text_span = str(p.get("text_span") or "")
        chunk_id = None
        char_start = char_end = line_start = line_end = None
        try:
            idx = int(p.get("chunk_index"))
        except (TypeError, ValueError):
            idx = None
        if idx is not None and idx in index_to_id:
            chunk_id = index_to_id[idx]
        if chunk_id is None:
            # Fallback: localizar el span en el texto concatenado del lote.
            chunk_id, char_start, char_end, line_start, line_end = (
                _locate_span_in_batch(chunks, batch_text, offsets, text_span)
            )
        if chunk_id is None:
            continue  # sin chunk asignable → descartar (chunk_id NOT NULL)
        citations = p.get("citations_references") or []
        if not isinstance(citations, list):
            citations = []
        out.append(
            {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "chapter_id": chunk_chapters.get(chunk_id, ""),
                "core_idea_id": str(p.get("core_idea_id") or "")[:50],
                "argument_id": str(p.get("argument_id") or "")[:50],
                "statement": statement,
                "text_span": text_span,
                "char_start": char_start,
                "char_end": char_end,
                "line_start": line_start,
                "line_end": line_end,
                "citations": [str(c) for c in citations],
            }
        )
    # Entidades y relaciones (nivel lote/documento).
    entities = []
    for e in data.get("entities") or []:
        if not isinstance(e, dict):
            continue
        name = str(e.get("name") or "").strip()
        if not name:
            continue
        entities.append(
            {
                "name": name,
                "type": str(e.get("type") or "concept"),
                "description": str(e.get("description") or ""),
            }
        )
    relations = []
    for r in data.get("relations") or []:
        if not isinstance(r, dict):
            continue
        src = str(r.get("source") or "").strip()
        tgt = str(r.get("target") or "").strip()
        if not src or not tgt:
            continue
        relations.append(
            {
                "source": src,
                "target": tgt,
                "type": str(r.get("type") or "RELACIONA"),
                "description": str(r.get("description") or ""),
            }
        )
    return out, entities, relations


def _extract_propositions_for_doc(
    session,
    doc_id,
    doc_path,
    verbose,
    batch_size_tokens,
    embed_fn=None,
    model_size="large",
    max_parallel=3,
):
    """Extrae proposiciones de TODOS los chunks del doc (paso aparte del
    chunking, etapa 'chunked').

    Agrupación jerárquica doc→capítulo: los chunks se agrupan por documento
    (uno por .md) y luego por capítulo = chapter_id (detectado por LLM, Fase
    2; por ahora NULL → un único grupo a nivel de documento) SOLO si el
    capítulo tiene > CHAPTER_TOKEN_THRESHOLD (30k) tokens; los capítulos ≤
    umbral se agrupan a nivel de archivo. Cada grupo se parte en lotes por
    presupuesto de tokens (KAG_PROPOSITION_BATCH_SIZE, default 400k); cada
    lote = UNA llamada LLM. Los lotes se disparan en paralelo
    (ThreadPoolExecutor + semáforo, KAG_PROPOSITION_PARALLEL) y se persisten
    en orden de chunk_index (determinista). Cache por content_hash (0026):
    los chunks que ya tienen proposiciones persistidas y cuyo content_hash no
    cambió se saltan (no re-extraer en re-ingestas). `model_size` ("small" |
    "large", default "large" vía KAG_PROPOSITION_MODEL): la respuesta de un
    lote es mucho mayor en extensión que la de un chunk suelto, por eso el
    default es el LLM grande.
    """
    rows = session.execute(
        text(
            "SELECT id, content, chunk_index, chapter_id, content_hash "
            "FROM kag_chunks WHERE doc_id = :id ORDER BY chunk_index"
        ),
        {"id": doc_id},
    ).fetchall()
    if not rows:
        return
    cached_ids = {
        r[0]
        for r in session.execute(
            text("SELECT DISTINCT chunk_id FROM kag_propositions WHERE doc_id = :id"),
            {"id": doc_id},
        ).fetchall()
    }
    pending = []
    for r in rows:
        h = _chunk_content_hash(r.content)
        if r.id in cached_ids and r.content_hash == h:
            continue  # ya extraído con hash idéntico (cache)
        pending.append(
            {
                "id": r.id,
                "content": r.content,
                "chunk_index": r.chunk_index,
                "chapter_id": r.chapter_id,
            }
        )
    if not pending:
        if verbose:
            print("[KAG] ⏭ proposiciones: todos los chunks ya extraídos (cache).")
        return
    # Agrupación doc→capítulo: capítulos >30k tokens = grupo propio; el resto
    # se fusiona a nivel de archivo. Cada grupo se parte en lotes por tokens.
    units: list[tuple[str, list]] = []  # (chapter_id, chunks)
    for group in _group_chunks_by_chapter(pending):
        for batch in _batch_chunks_by_tokens(group["chunks"], batch_size_tokens):
            units.append((group["chapter_id"], batch))
    # Orden de persistencia determinista: por chunk_index (aunque los lotes
    # se procesen en paralelo).
    units.sort(key=lambda u: u[1][0]["chunk_index"])
    if verbose:
        for i, (ch_id, batch) in enumerate(units, 1):
            total = sum(estimate_tokens(c["content"]) for c in batch)
            print(
                f"[KAG] ⚙ proposiciones: lote {i}/{len(units)} "
                f"({len(batch)} chunks, ~{total} tokens, "
                f"capítulo: {ch_id or '(archivo)'})..."
            )
    results = _run_proposition_batches_parallel(
        session,
        doc_id,
        doc_path,
        units,
        verbose,
        model_size=model_size,
        max_parallel=max_parallel,
    )
    for (_ch_id, batch), props in zip(units, results):
        # Degradación: proposiciones sin chunk asignable (span no localizable)
        # se descartan — kag_propositions.chunk_id es NOT NULL.
        props = [p for p in props if p["chunk_id"] is not None]
        _store_propositions(session, doc_id, props, embed_fn=embed_fn)
        for ch in batch:
            session.execute(
                text("UPDATE kag_chunks SET content_hash = :h WHERE id = :id"),
                {"h": _chunk_content_hash(ch["content"]), "id": ch["id"]},
            )


def _extract_document_propositions(
    session,
    doc_id,
    doc_path,
    verbose,
    batch_size_tokens,
    embed_fn=None,
    model_size="large",
    max_parallel=3,
):
    """Extrae proposiciones + entidades + relaciones de TODOS los chunks del
    doc (etapa 'chunked', Fase 5) usando las paráfrasis de la Fase 4.

    Flujo por DOCUMENTO (decisión: proposiciones + entidades por documento,
    tomando todas las paráfrasis de todos los capítulos):
    1. Lee TODAS las paráfrasis de TODOS los capítulos del documento
       (SELECT ... FROM kag_chunks WHERE doc_id = :id ORDER BY chunk_index).
       El texto a procesar por chunk = paraphrase or content (si la
       paráfrasis es NULL, fallback al content — degradación natural si la
       Fase 4 no corrió).
    2. Cache por content_hash (0026): los chunks que ya tienen proposiciones
       persistidas y cuyo content_hash no cambió se saltan.
    3. Agrupación por DOCUMENTO (no por capítulo >30k): todos los chunks
       pendientes se concatenan (con sus paráfrasis) y se parten en lotes
       por presupuesto de tokens (_batch_chunks_by_tokens,
       KAG_PROPOSITION_BATCH_SIZE=400000). Cada lote = UNA llamada LLM.
    4. _extract_document_batch: LLM grande con spec `kag_chapter_propositions`
       → {"propositions": [...], "entities": [...], "relations": [...]}.
       Asignación de proposiciones a chunks: primario = chunk_index
       declarado por el LLM; si falta o es inválido, secundario =
       _locate_span_in_batch. Si ambos fallan → descartar (chunk_id NOT
       NULL).
    5. Persistencia: proposiciones vía _store_propositions (embedding batch);
       entidades + relaciones vía _store_entities_relations (dedup por
       name_norm contra las de spaCy — no duplica).
    6. Paralelismo: lotes independientes → _run_proposition_batches_parallel
       (ThreadPoolExecutor + semáforo, batch_fn=_extract_document_batch).
       Persistencia en orden determinista por chunk_index.
    7. Degradación: LLM falla o JSON inválido en un lote → log + skip del
       lote (la ingesta sigue). Entidades vacías → solo proposiciones.
    """
    rows = session.execute(
        text(
            "SELECT id, chunk_index, content, paraphrase, chapter_id, content_hash "
            "FROM kag_chunks WHERE doc_id = :id ORDER BY chunk_index"
        ),
        {"id": doc_id},
    ).fetchall()
    if not rows:
        return
    cached_ids = {
        r[0]
        for r in session.execute(
            text("SELECT DISTINCT chunk_id FROM kag_propositions WHERE doc_id = :id"),
            {"id": doc_id},
        ).fetchall()
    }
    pending = []
    for r in rows:
        h = _chunk_content_hash(r.content)
        if r.id in cached_ids and r.content_hash == h:
            continue  # ya extraído con hash idéntico (cache)
        pending.append(
            {
                "id": r.id,
                # Texto a procesar: paráfrasis o content si NULL (degradación
                # natural si la Fase 4 no corrió).
                "content": (getattr(r, "paraphrase", None) or r.content),
                # Content original para el content_hash (cache 0026).
                "original_content": r.content,
                "chunk_index": r.chunk_index,
                "chapter_id": r.chapter_id,
            }
        )
    if not pending:
        if verbose:
            print("[KAG] ⏭ proposiciones: todos los chunks ya extraídos (cache).")
        return
    # Agrupación por DOCUMENTO: todos los chunks pendientes se parten en
    # lotes por presupuesto de tokens (cada lote = UNA llamada LLM).
    units: list[tuple[str, list]] = []  # (chapter_id, chunks)
    for batch in _batch_chunks_by_tokens(pending, batch_size_tokens):
        units.append(("", batch))
    # Orden de persistencia determinista: por chunk_index (aunque los lotes
    # se procesen en paralelo).
    units.sort(key=lambda u: u[1][0]["chunk_index"])
    if verbose:
        for i, (_ch_id, batch) in enumerate(units, 1):
            total = sum(estimate_tokens(c["content"]) for c in batch)
            print(
                f"[KAG] ⚙ proposiciones: lote {i}/{len(units)} "
                f"({len(batch)} chunks, ~{total} tokens)..."
            )
    results = _run_proposition_batches_parallel(
        session,
        doc_id,
        doc_path,
        units,
        verbose,
        model_size=model_size,
        max_parallel=max_parallel,
        batch_fn=_extract_document_batch,
    )
    for (_ch_id, batch), (props, entities, relations) in zip(units, results):
        # Degradación: proposiciones sin chunk asignable (chunk_index
        # inválido y span no localizable) se descartan — kag_propositions.
        # chunk_id es NOT NULL.
        props = [p for p in props if p["chunk_id"] is not None]
        _store_propositions(session, doc_id, props, embed_fn=embed_fn)
        # Entidades + relaciones LLM: se persisten contra el primer chunk del
        # lote (dedup por name_norm por doc — no duplica con las de spaCy).
        if entities or relations:
            _store_entities_relations(
                session,
                doc_id,
                batch[0]["id"],
                {"entities": entities, "relations": relations},
                embed_fn=embed_fn,
            )
        for ch in batch:
            session.execute(
                text("UPDATE kag_chunks SET content_hash = :h WHERE id = :id"),
                {"h": _chunk_content_hash(ch["original_content"]), "id": ch["id"]},
            )


# ---------------------------------------------------------------------
# Paráfrasis de chunks (Fase 4, etapa 'paraphrased')
# ---------------------------------------------------------------------


def _paraphrase_chunks(session, doc_id, doc_path, verbose=True):
    """Parafrasea TODOS los chunks del doc (etapa 'paraphrased', Fase 4).

    Agrupación por capítulo: los chunks con chapter_id UUID forman un grupo
    por capítulo; los NULL se agrupan en un único grupo "(sin capítulo)"
    (chapter_id ""). Cada grupo = UNA llamada LLM grande (model_size="large")
    con la spec `kag_chunk_paraphrase` → {"paraphrases": [{"chunk_index":
    int, "paraphrase": str}]}. El prompt recibe `chunks_json` =
    json.dumps([{chunk_index, content}] del grupo).

    Los grupos son independientes → se disparan en paralelo
    (ThreadPoolExecutor + semáforo, KAG_PARAPHRASE_PARALLEL default 3) y se
    persisten en orden determinista por chunk_index. Los chunks sin
    paraphrase en la respuesta del LLM quedan NULL. Degradación: si el LLM
    falla o devuelve JSON inválido en un grupo → log + paraphrase NULL para
    esos chunks (la ingesta sigue). Si no hay chunks → no hace nada.
    """
    rows = session.execute(
        text(
            "SELECT id, chunk_index, content, chapter_id FROM kag_chunks "
            "WHERE doc_id = :id ORDER BY chunk_index"
        ),
        {"id": doc_id},
    ).fetchall()
    if not rows:
        return
    # Agrupación por capítulo (chapter_id UUID o "" para los NULL).
    groups: list[tuple[str, list]] = []  # (chapter_id, chunks)
    for r in rows:
        ch_id = str(r.chapter_id) if r.chapter_id else ""
        if groups and groups[-1][0] == ch_id:
            groups[-1][1].append(
                {"id": r.id, "chunk_index": r.chunk_index, "content": r.content}
            )
        else:
            groups.append(
                (
                    ch_id,
                    [{"id": r.id, "chunk_index": r.chunk_index, "content": r.content}],
                )
            )
    config = resolve_config(session)
    max_parallel = int(get_config_value(config, "KAG_PARAPHRASE_PARALLEL", 3) or 3)
    if max_parallel < 1:
        max_parallel = 3
    sem = threading.BoundedSemaphore(max_parallel)
    results: list = [None] * len(groups)

    def work(i, chapter_id, group_chunks):
        with sem:
            return i, _paraphrase_group(
                session, doc_id, doc_path, chapter_id, group_chunks, verbose
            )

    pool_size = min(len(groups), max_parallel * 2)
    with ThreadPoolExecutor(max_workers=pool_size) as ex:
        futures = [
            ex.submit(work, i, ch_id, chs) for i, (ch_id, chs) in enumerate(groups)
        ]
        for fut in futures:
            i, paraphrases = fut.result()
            results[i] = paraphrases
    # Persistencia determinista por chunk_index (aunque se procesen en
    # paralelo): los chunks sin paraphrase quedan NULL.
    for (_ch_id, group_chunks), paraphrases in zip(groups, results):
        by_index = {int(p.get("chunk_index")): p.get("paraphrase") for p in paraphrases}
        for ch in group_chunks:
            paraphrase = by_index.get(ch["chunk_index"])
            if not paraphrase:
                continue
            session.execute(
                text("UPDATE kag_chunks SET paraphrase = :p WHERE id = :id"),
                {"p": sanitize_text(str(paraphrase)), "id": ch["id"]},
            )
    session.commit()


def _paraphrase_group(session, doc_id, doc_path, chapter_id, chunks, verbose=True):
    """Parafrasea UN grupo de chunks (un capítulo) con UNA llamada LLM grande.

    `chapter_id`: UUID del capítulo o "" (grupo sin capítulo). Devuelve la
    lista de dicts {"chunk_index", "paraphrase"} del LLM; si el LLM falla o
    devuelve JSON inválido → [] (degradación: paraphrase NULL, la ingesta
    sigue).
    """
    try:
        settings = load_settings(session)
    except Exception:  # noqa: BLE001 — sesión falsa en tests
        settings = None
    retries = int(getattr(settings, "llm_retries", 3) or 3) if settings else 3
    fallback = getattr(settings, "fallback_model", None) if settings else None
    model_name = getattr(settings, "large_model", None) if settings else None
    system, user_template = _get_prompt_pair(
        session,
        model_name,
        TASK_CHUNK_PARAPHRASE,
        PARAPHRASE_SYSTEM_SHORT,
        PARAPHRASE_USER_SHORT,
    )
    chunks_json = json.dumps(
        [{"chunk_index": ch["chunk_index"], "content": ch["content"]} for ch in chunks],
        ensure_ascii=False,
    )
    prompt = _fill_prompt(
        user_template,
        source_file=str(Path(doc_path)).replace("\\", "/"),
        document_id=str(doc_id),
        chapter_id=chapter_id or "(sin capítulo)",
        chunks_json=chunks_json,
    )
    try:
        text_out, _model, _used_fallback = call_with_retries(
            session,
            prompt=prompt,
            system=system,
            model_size="large",
            response_format={"type": "json_object"},
            retries=retries,
            fallback_model=fallback,
        )
        data = parse_llm_output(text_out)
        paraphrases = data.get("paraphrases") if isinstance(data, dict) else None
        if not isinstance(paraphrases, list):
            return []
        out = []
        for p in paraphrases:
            if not isinstance(p, dict):
                continue
            try:
                idx = int(p.get("chunk_index"))
            except (TypeError, ValueError):
                continue
            paraphrase = str(p.get("paraphrase") or "").strip()
            if not paraphrase:
                continue
            out.append({"chunk_index": idx, "paraphrase": paraphrase})
        return out
    except Exception as exc:  # noqa: BLE001 — LLM no disponible: degradación
        if verbose:
            print(
                f"[KAG] ⚠ Paráfrasis fallaron (capítulo {chapter_id or '(sin capítulo)'}): "
                f"{exc}"
            )
        return []


# ---------------------------------------------------------------------
# Resumen jerárquico con Qwen 2.5 local
# ---------------------------------------------------------------------

QWEN_SUMMARY_SYSTEM = """You are a summarization assistant. Follow these rules strictly:
- Output exactly ONE sentence, maximum 30 words.
- No preamble, no explanations, no markdown, no bullet points.
- Only facts present in the text. Do not invent.
- Do not start with phrases like "This text..." or "The text describes...".
- Respond in the same language as the text.

Example:
Text: <text>El backpropagation es un algoritmo que ajusta los pesos de una red neuronal calculando el gradiente de la función de pérdida.</text>
Summary: El backpropagation ajusta los pesos de una red neuronal mediante el gradiente de la función de pérdida."""

QWEN_SUMMARY_USER = """<text>
{text}
</text>

Summary:"""


def _split_h1_h2(text: str) -> list:
    """Divide el texto por encabezados H1/H2 (para el map-reduce de long docs)."""
    sections = []
    current = []
    for line in text.splitlines():
        if re.match(r"^#{1,2}\s+", line):
            if current:
                sections.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("\n".join(current))
    return [s.strip() for s in sections if s.strip()]


def summarize_document(session, text, doc_type):
    """Resumen jerárquico con Qwen 2.5 local. Devuelve '' si falla (degradación).

    short: una llamada con el texto truncado a ~16k tokens.
    long:  map-reduce por secciones H1/H2 + llamada reduce.
    """
    try:
        settings = load_settings(session)
        small_model = getattr(settings, "small_model", None) if settings else None
        summary_system, summary_user = _get_prompt_pair(
            session,
            small_model,
            TASK_QWEN_SUMMARY,
            QWEN_SUMMARY_SYSTEM,
            QWEN_SUMMARY_USER,
        )
        if doc_type == "short":
            truncated = text[:SUMMARY_MAX_CHARS]
            return complete_local(
                session,
                summary_user.format(text=truncated),
                system=summary_system,
                max_tokens=200,
            ).strip()
        section_summaries = []
        sections = _split_h1_h2(text)
        if not sections:
            return ""
        max_parallel = get_config_value(
            resolve_config(session), "KAG_SUMMARY_PARALLEL", 3
        )
        if max_parallel and max_parallel > 1 and len(sections) > 1:
            # Fase map en paralelo: cada sección es una llamada complete_local
            # independiente. Se recolectan los resultados indexados para
            # preservar el orden original de _split_h1_h2 (el combined va en
            # el orden del documento). El reduce final sigue siendo secuencial.
            def _map_section(i, sec):
                truncated = sec[:SUMMARY_MAX_CHARS]
                s = complete_local(
                    session,
                    summary_user.format(text=truncated),
                    system=summary_system,
                    max_tokens=60,
                ).strip()
                return i, s

            try:
                with ThreadPoolExecutor(max_workers=max_parallel) as ex:
                    results = list(ex.map(_map_section, range(len(sections)), sections))
                for i, s in results:
                    if s:
                        section_summaries.append(s)
            except Exception as exc:  # noqa: BLE001 — degradación no bloqueante
                print(f"[KAG] ⚠ Fase map paralela no disponible: {exc}")
                # Fallback secuencial: mismo resultado, solo más lento.
                for sec in sections:
                    truncated = sec[:SUMMARY_MAX_CHARS]
                    s = complete_local(
                        session,
                        summary_user.format(text=truncated),
                        system=summary_system,
                        max_tokens=60,
                    ).strip()
                    if s:
                        section_summaries.append(s)
        else:
            for sec in sections:
                truncated = sec[:SUMMARY_MAX_CHARS]
                s = complete_local(
                    session,
                    summary_user.format(text=truncated),
                    system=summary_system,
                    max_tokens=60,
                ).strip()
                if s:
                    section_summaries.append(s)
        if not section_summaries:
            return ""
        combined = "\n".join(f"- {s}" for s in section_summaries)
        return complete_local(
            session,
            summary_user.format(text=combined),
            system=summary_system,
            max_tokens=200,
        ).strip()
    except Exception as exc:  # noqa: BLE001 — degradación no bloqueante
        print(f"[KAG] ⚠ Resumen local no disponible: {exc}")
        return ""


# ---------------------------------------------------------------------
# Figuras (VLM)
# ---------------------------------------------------------------------


def describe_figure(session, image_path, caption):
    """Describe una imagen con el VLM vía complete_vision (data URL base64).

    Devuelve '' si falla (la figura se indexa igual, con description='').
    """
    try:
        mime = "image/png" if str(image_path).lower().endswith(".png") else "image/jpeg"
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        data_url = f"data:{mime};base64,{b64}"
        prompt = "Describe esta figura con precisión."
        if caption:
            prompt += f" Caption: {caption}"
        return complete_vision(session, prompt, image_url=data_url).strip()
    except Exception as exc:  # noqa: BLE001 — degradación no bloqueante
        print(f"[KAG] ⚠ VLM no disponible para {image_path}: {exc}")
        return ""


# ---------------------------------------------------------------------
# Fase 2 — análisis documental (ficha ISO 25964 + capítulos)
# ---------------------------------------------------------------------


def _scope_thematic_from_thematic(ficha: dict) -> str:
    """Deriva el scope_thematic de la ficha (preferred + non-preferred con |).

    A partir de `thematic_areas_iso25964` (lista de dicts con preferred_term y
    non_preferred_terms), produce una cadena plana para búsquedas por texto:
    cada temática contribuye su término preferido y sus no preferidos,
    separados por '|'. Devuelve '' si no hay temáticas.
    """
    areas = ficha.get("thematic_areas_iso25964") or []
    if not isinstance(areas, list):
        return ""
    parts = []
    for area in areas:
        if not isinstance(area, dict):
            continue
        preferred = str(area.get("preferred_term") or "").strip()
        if preferred:
            parts.append(preferred)
        non_preferred = area.get("non_preferred_terms") or []
        if isinstance(non_preferred, list):
            parts.extend(str(t).strip() for t in non_preferred if str(t).strip())
    return " | ".join(parts)


def _enrich_library_of_congress(ficha: dict, verbose: bool = True) -> dict:
    """Enriquece la ficha con URIs de la Biblioteca del Congreso (best-effort).

    Para cada término LCSH de `library_of_congress.lcsh_terms` consulta
    `LibraryOfCongressAPITool.query_subject_heading` y añade la URI al término
    (dict {term, uri}); para `lcc_classification` consulta
    `query_classification_code` y añade lcc_uri. Cualquier error degrada a la
    ficha original (nunca romper la ingesta). Import perezoso de src.kag.tools.
    """
    loc = ficha.get("library_of_congress") or {}
    if not isinstance(loc, dict):
        loc = {}
    try:
        from src.kag.tools import LibraryOfCongressAPITool

        tool = LibraryOfCongressAPITool()
        lcsh_terms = loc.get("lcsh_terms") or []
        if isinstance(lcsh_terms, list):
            enriched = []
            for term in lcsh_terms:
                term_str = (
                    str(term)
                    if not isinstance(term, dict)
                    else str(term.get("term") or "")
                )
                if not term_str:
                    continue
                try:
                    result = tool.query_subject_heading(term_str)
                except Exception as exc:  # noqa: BLE001 — best-effort
                    if verbose:
                        print(f"[KAG] ⚠ LCSH '{term_str}' no enriquecido: {exc}")
                    result = None
                if result and isinstance(result, dict):
                    enriched.append({"term": term_str, "uri": result.get("uri") or ""})
                else:
                    enriched.append({"term": term_str, "uri": ""})
            if enriched:
                loc["lcsh_terms"] = enriched
        lcc = loc.get("lcc_classification") or {}
        if isinstance(lcc, dict):
            label = str(lcc.get("label") or "")
            if label:
                try:
                    result = tool.query_classification_code(label)
                except Exception as exc:  # noqa: BLE001 — best-effort
                    if verbose:
                        print(f"[KAG] ⚠ LCC '{label}' no enriquecido: {exc}")
                    result = None
                if result and isinstance(result, dict):
                    lcc["lcc_uri"] = result.get("lcc_uri") or ""
                    lcc["call_number"] = result.get("lcc_call_number") or lcc.get(
                        "call_number"
                    )
        ficha["library_of_congress"] = loc
    except Exception as exc:  # noqa: BLE001 — degradación natural
        if verbose:
            print(f"[KAG] ⚠ Enriquecimiento LCC/LCSH no disponible: {exc}")
    return ficha


def _index_document_analysis(
    session, doc_id, doc_path, slice_text, verbose=True
) -> dict:
    """Fase 2: ficha documental + capítulos del documento (LLM grande).

    Llama al LLM grande con la spec `kag_document_analysis` pasando el
    contexto COMPLETO del documento (`document_context`, sin truncar) y
    persiste SIEMPRE:
      - `kag_documents.ficha_jsonb` = el JSON completo del análisis (ficha +
        capítulos + scope_thematic derivado).
      - `kag_documents.sections_json` = los capítulos detectados.
      - `kag_chapters` = una fila por capítulo (RETURNING id → UUID).

    La salida del LLM es un índice JERÁRQUICO `index: [{division, chapters}]`;
    si el LLM devuelve `chapters` plano (schema v1.0) o un índice vacío/
    malformado, se usa el fallback a `chapters` plano.

    DECISIÓN DE DISEÑO: `line_start`/`line_end` de los capítulos son RELATIVOS
    AL DOCUMENTO (slice), no al archivo — la Fase 3 segmentará el slice por
    capítulo. Devuelve el dict {chapter_id_str: uuid} para las Fases 3/4/5.

    Degradación crítica: si el LLM falla o devuelve JSON inválido, se persiste
    una ficha con defaults (title, technical_level='intermediate',
    thematic_areas=[], library_of_congress={}, bibtex='', key_entities=[]) y
    UN capítulo único con todo el rango del documento (line_start=1,
    line_end=total, has_images=False). El UPDATE de ficha_jsonb y el INSERT de
    capítulos ocurren SIEMPRE.
    """
    total_lines = len(slice_text.splitlines())
    title = _title_from_md(slice_text, doc_path)
    try:
        settings = load_settings(session)
    except Exception:  # noqa: BLE001 — sesión falsa en tests
        settings = None
    retries = int(getattr(settings, "llm_retries", 3) or 3) if settings else 3
    fallback = getattr(settings, "fallback_model", None) if settings else None
    large_model = getattr(settings, "large_model", None) if settings else None
    try:
        source_file = str(Path(doc_path)).replace("\\", "/")
        system, user_template = _get_prompt_pair(
            session,
            large_model,
            TASK_DOCUMENT_ANALYSIS,
            DOCUMENT_ANALYSIS_SYSTEM_SHORT,
            DOCUMENT_ANALYSIS_USER_SHORT,
        )
        prompt = _fill_prompt(
            user_template,
            source_file=source_file,
            document_id=str(doc_id),
            document_context=slice_text,
        )
        text_out, _model, _used_fallback = call_with_retries(
            session,
            prompt=prompt,
            system=system,
            model_size="large",
            response_format={"type": "json_object"},
            retries=retries,
            fallback_model=fallback,
        )
        data = parse_llm_output(text_out)
        ficha = data.get("ficha") if isinstance(data, dict) else None
        # Índice jerárquico (nuevo schema): index = [{division, chapters: [...]}].
        # Backward compat: chapters plano (schema v1.0).
        chapters = None
        if isinstance(data, dict):
            index = data.get("index")
            if isinstance(index, list) and index:
                chapters = []
                for div in index:
                    if not isinstance(div, dict):
                        continue
                    div_title = str(div.get("division") or "").strip()
                    div_chapters = div.get("chapters")
                    if not isinstance(div_chapters, list):
                        continue
                    for ch in div_chapters:
                        if isinstance(ch, dict):
                            ch = dict(ch)
                            if div_title:
                                ch["division"] = div_title
                            chapters.append(ch)
                # Backward compat: si el índice vino malformado (divisiones sin
                # chapters), reintentar con el schema plano v1.0.
                if not chapters:
                    chapters = data.get("chapters")
            else:
                chapters = data.get("chapters")
        if not isinstance(ficha, dict):
            ficha = None
        if not isinstance(chapters, list):
            chapters = None
    except Exception as exc:  # noqa: BLE001 — LLM no disponible: degradación
        if verbose:
            print(f"[KAG] ⚠ Análisis por LLM falló: {exc}")
        ficha = None
        chapters = None

    # ── Ficha con defaults si el LLM falló ────────────────────────────────
    if not ficha:
        ficha = {
            "title": title,
            "technical_level": "intermediate",
            "thematic_areas_iso25964": [],
            "library_of_congress": {},
            "bibtex": "",
            "key_entities": [],
        }
    else:
        ficha.setdefault("title", title)
        ficha.setdefault("technical_level", "intermediate")
        ficha.setdefault("thematic_areas_iso25964", [])
        ficha.setdefault("library_of_congress", {})
        ficha.setdefault("bibtex", "")
        ficha.setdefault("key_entities", [])

    # ── Capítulos: sanitizar rangos y clamp al slice ───────────────────────
    if not chapters:
        chapters = [
            {
                "chapter_id": "",
                "title": title,
                "line_start": 1,
                "line_end": total_lines,
                "has_images": False,
            }
        ]
    seen_ids = set()
    clean_chapters = []
    for ch in chapters:
        if not isinstance(ch, dict):
            continue
        try:
            ls = int(ch.get("line_start") or 1)
            le = int(ch.get("line_end") or total_lines)
        except (TypeError, ValueError):
            ls, le = 1, total_lines
        if ls > le:
            ls, le = le, ls
        ls = max(1, min(ls, total_lines))
        le = max(1, min(le, total_lines))
        cid = str(ch.get("chapter_id") or "").strip()
        if cid in seen_ids:
            base = cid or "cap"
            n = 2
            while f"{base}_{n}" in seen_ids:
                n += 1
            cid = f"{base}_{n}"
        seen_ids.add(cid)
        has_images = ch.get("has_images")
        if isinstance(has_images, str):
            has_images = has_images.strip().lower() in ("1", "true", "yes")
        else:
            has_images = bool(has_images)
        clean_chapters.append(
            {
                "chapter_id": cid,
                "title": sanitize_text(str(ch.get("title") or ""))[:300] or title,
                "line_start": ls,
                "line_end": le,
                "has_images": has_images,
                "division": str(ch.get("division") or "").strip(),
            }
        )
    if not clean_chapters:
        clean_chapters = [
            {
                "chapter_id": "",
                "title": title,
                "line_start": 1,
                "line_end": total_lines,
                "has_images": False,
            }
        ]

    # ── Enriquecimiento best-effort (LCC/LCSH) + scope_thematic ────────────
    ficha = _enrich_library_of_congress(ficha, verbose=verbose)
    ficha["scope_thematic"] = _scope_thematic_from_thematic(ficha)

    # ── Persistir SIEMPRE: ficha_jsonb + sections_json + kag_chapters ─────
    sections = [
        {
            "chapter_id": ch["chapter_id"],
            "title": ch["title"],
            "line_start": ch["line_start"],
            "line_end": ch["line_end"],
            "has_images": ch["has_images"],
            "division": ch.get("division", ""),
        }
        for ch in clean_chapters
    ]
    session.execute(
        text(
            "UPDATE kag_documents SET ficha_jsonb = :ficha, sections_json = :sections "
            "WHERE id = :doc_id"
        ),
        {
            "ficha": json.dumps(ficha, ensure_ascii=False),
            "sections": json.dumps(sections, ensure_ascii=False),
            "doc_id": doc_id,
        },
    )
    chapter_map: dict[str, str] = {}
    for ch in clean_chapters:
        row = session.execute(
            text(
                "INSERT INTO kag_chapters "
                "(doc_id, chapter_id, title, line_start, line_end, has_images) "
                "VALUES (:doc_id, :chapter_id, :title, :line_start, :line_end, "
                ":has_images) RETURNING id"
            ),
            {
                "doc_id": doc_id,
                "chapter_id": ch["chapter_id"],
                "title": ch["title"],
                "line_start": ch["line_start"],
                "line_end": ch["line_end"],
                "has_images": ch["has_images"],
            },
        ).scalar()
        chapter_map[ch["chapter_id"]] = str(row)
    if verbose:
        print(
            f"[KAG] 📑 {doc_path}: {len(clean_chapters)} capítulo(s) "
            f"y ficha documental persistidos."
        )
    return chapter_map


def _find_caption(md_text: str, image_name: str) -> str:
    """Busca el alt text de `![alt](...image_name...)` en el markdown."""
    m = re.search(rf"!\[([^\]]*)\]\([^)]*{re.escape(image_name)}[^)]*\)", md_text)
    return m.group(1).strip() if m else ""


def _find_chunk_for_image(session, doc_id, image_name):
    """Chunk cuyo contenido referencia la imagen (aproximado, por substring)."""
    row = session.execute(
        text(
            "SELECT id FROM kag_chunks "
            "WHERE doc_id = :doc_id AND content LIKE :pat "
            "ORDER BY chunk_index LIMIT 1"
        ),
        {"doc_id": doc_id, "pat": f"%{image_name}%"},
    ).first()
    return row.id if row else None


def _index_figures(
    session, doc_id, md_path, slice_text, doc_line_start, verbose=True
) -> int:
    """Indexa las figuras del documento con visión condicional + FAQ Reverse HyDE.

    La visión SOLO se llama si al menos un capítulo del documento tiene
    `has_images=True` (decisión del usuario): se leen los capítulos de
    kag_chapters y si ninguno tiene imágenes, se salta (return 0).

    Las imágenes se extraen con `MarkdownImageExtractorTool` (import perezoso)
    sobre el rango físico del documento en el archivo (doc_line_start..
    doc_line_end). Para cada imagen que existe en disco:
      - contexto circundante ±15 líneas del slice (chunk que la referencia por
        substring, como `_find_chunk_for_image`),
      - data URL base64 + `complete_vision` con prompt que pide JSON
        (image_type, dense_visual_description, epistemic_contribution,
        faq_indexing [3-5 preguntas Reverse HyDE], associated_entities),
      - INSERT en kag_figures con todos los campos (migración 0030).

    Las imágenes son independientes → ThreadPoolExecutor (max_workers 3,
    semáforo) como el `_index_figures` clásico; el INSERT se hace en orden
    determinista (sorted por anchor_line). Si el pool falla → secuencial
    (nunca romper). Degradación por imagen: description='' y faq_indexing=[]
    si el VLM falla o el JSON es inválido.
    """
    # 0. Visión condicional: solo si algún capítulo tiene has_images.
    chapters = session.execute(
        text(
            "SELECT id, chapter_id, line_start, line_end, has_images "
            "FROM kag_chapters WHERE doc_id = :id ORDER BY line_start"
        ),
        {"id": doc_id},
    ).fetchall()
    if not any(getattr(ch, "has_images", False) for ch in chapters):
        if verbose:
            print(
                f"[KAG] ⏭ {md_path.name}: ningún capítulo con imágenes — "
                "visión omitida."
            )
        return 0

    # 1. Extraer imágenes del rango físico del documento en el archivo.
    try:
        from src.kag.tools import MarkdownImageExtractorTool

        doc_line_end = doc_line_start + len(slice_text.splitlines()) - 1
        extracted = MarkdownImageExtractorTool().extract_images_from_document(
            str(md_path), str(doc_id), doc_line_start, doc_line_end
        )
    except Exception as exc:  # noqa: BLE001 — degradación natural
        if verbose:
            print(f"[KAG] ⚠ Extracción de imágenes falló: {exc}")
        return 0
    figures = [
        {
            "image_path": f.get("file_path"),
            "anchor_line": int(f.get("anchor_line") or 0),
            "caption": sanitize_text(str(f.get("caption") or "")),
        }
        for f in extracted
        if f.get("exists_on_disk") and f.get("file_path")
    ]
    if not figures:
        return 0
    # Orden determinista por anchor_line (línea física del archivo).
    figures.sort(key=lambda f: f["anchor_line"])

    # 2. Despachar la visión en paralelo (VLM). results[i] ↔ figures[i].
    config = resolve_config(session)
    max_workers = int(get_config_value(config, "KAG_FIGURE_PARALLEL", 3) or 3)
    if max_workers < 1:
        max_workers = 3
    results: list[dict] = [{}] * len(figures)
    try:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(figures))) as ex:
            futures = [
                ex.submit(
                    _describe_figure_vision,
                    session,
                    f,
                    slice_text,
                    doc_line_start,
                    verbose,
                )
                for f in figures
            ]
            for i, fut in enumerate(futures):
                results[i] = fut.result()
    except Exception as exc:  # noqa: BLE001 — degradación: secuencial
        if verbose:
            print(f"[KAG] ⚠ Pool de figuras falló ({exc}) — secuencial.")
        results = [
            _describe_figure_vision(session, f, slice_text, doc_line_start, verbose)
            for f in figures
        ]

    # 3. INSERT por figura en orden determinista (sorted por anchor_line).
    count = 0
    for f, vision in zip(figures, results):
        vision = vision or {}
        session.execute(
            text(
                "INSERT INTO kag_figures "
                "(doc_id, chunk_id, image_path, caption, description, image_type, "
                "dense_visual_description, epistemic_contribution, faq_indexing, "
                "associated_entities, anchor_line) "
                "VALUES (:doc_id, :chunk_id, :image_path, :caption, :description, "
                ":image_type, :dense_visual_description, :epistemic_contribution, "
                ":faq_indexing, :associated_entities, :anchor_line)"
            ),
            {
                "doc_id": doc_id,
                "chunk_id": _find_chunk_for_image(session, doc_id, f["image_path"]),
                "image_path": f["image_path"],
                "caption": f["caption"],
                "description": sanitize_text(
                    str(vision.get("dense_visual_description") or "")
                ),
                "image_type": sanitize_text(str(vision.get("image_type") or ""))[:50],
                "dense_visual_description": sanitize_text(
                    str(vision.get("dense_visual_description") or "")
                ),
                "epistemic_contribution": sanitize_text(
                    str(vision.get("epistemic_contribution") or "")
                ),
                "faq_indexing": json.dumps(
                    vision.get("faq_indexing") or [], ensure_ascii=False
                ),
                "associated_entities": json.dumps(
                    vision.get("associated_entities") or [], ensure_ascii=False
                ),
                "anchor_line": f["anchor_line"],
            },
        )
        count += 1
    if verbose and count:
        print(f"[KAG] 🖼 {count} figuras indexadas para {md_path.name}.")
    return count


def _describe_figure_vision(
    session, figure: dict, slice_text: str, doc_line_start: int, verbose: bool
) -> dict:
    """Describe UNA figura con el VLM (JSON estructurado + FAQ Reverse HyDE).

    Prompt con contexto circundante ±15 líneas del slice (el chunk que
    referencia la imagen por substring). `anchor_line` de la figura es la línea
    del ARCHIVO; `doc_line_start` es la primera línea del documento en el
    archivo, así el contexto se calcula relativo al slice. Pide JSON:
    image_type, dense_visual_description, epistemic_contribution,
    faq_indexing (3-5 preguntas Reverse HyDE), associated_entities.
    Degradación: devuelve {} (description='' y faq_indexing=[] en el INSERT)
    si el VLM falla o el JSON es inválido — nunca romper.
    """
    image_path = figure.get("image_path") or ""
    try:
        mime = "image/png" if str(image_path).lower().endswith(".png") else "image/jpeg"
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        data_url = f"data:{mime};base64,{b64}"
        # Contexto circundante ±15 líneas del slice: la línea del archivo se
        # convierte a índice relativo del slice restando doc_line_start.
        context = ""
        anchor = int(figure.get("anchor_line") or 0)
        if anchor > 0:
            lines = slice_text.splitlines()
            rel = anchor - doc_line_start  # 0-based en el slice
            start = max(0, rel - 15)
            end = min(len(lines), rel + 15)
            context = "\n".join(lines[start:end])
        caption = figure.get("caption") or ""
        prompt = (
            "Analiza esta figura de un documento académico y devuelve JSON estricto "
            "con este schema: "
            '{"image_type": "diagram|chart_or_plot|flowchart|conceptual_illustration|'
            'screenshot|table_image|photograph", "dense_visual_description": str, '
            '"epistemic_contribution": str, "faq_indexing": [str], '
            '"associated_entities": [str]}. '
            "image_type: clasifica la figura en UNA de las categorías. "
            "dense_visual_description: transcripción EXACTA de lo que se ve "
            "(ejes, leyendas, flujos, valores, relaciones espaciales) — sin "
            "interpretar. epistemic_contribution: qué aporta la figura al "
            "conocimiento del documento que el texto no dice. faq_indexing: "
            "3-5 preguntas que esta figura puede responder (Reverse HyDE, para "
            "recuperación por pregunta). associated_entities: entidades que "
            "aparecen en la figura."
        )
        if caption:
            prompt += f" Caption: {caption}"
        if context:
            prompt += f"\n\nContexto circundante:\n{context}"
        text_out = complete_vision(session, prompt, image_url=data_url)
        data = parse_llm_output(text_out)
        if not isinstance(data, dict):
            return {}
        faq = data.get("faq_indexing") or []
        if not isinstance(faq, list):
            faq = []
        entities = data.get("associated_entities") or []
        if not isinstance(entities, list):
            entities = []
        return {
            "image_type": str(data.get("image_type") or ""),
            "dense_visual_description": str(data.get("dense_visual_description") or ""),
            "epistemic_contribution": str(data.get("epistemic_contribution") or ""),
            "faq_indexing": [str(q) for q in faq if str(q).strip()][:5],
            "associated_entities": [str(e) for e in entities if str(e).strip()],
        }
    except Exception as exc:  # noqa: BLE001 — degradación no bloqueante
        if verbose:
            print(f"[KAG] ⚠ VLM no disponible para {image_path}: {exc}")
        return {}


# ---------------------------------------------------------------------
# Flujo de indexación
# ---------------------------------------------------------------------


def _title_from_md(text: str, doc_path: str) -> str:
    """Título: primer H1 del markdown, o el nombre del archivo."""
    m = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    if m:
        return sanitize_text(m.group(1).strip())[:300]
    return sanitize_text(Path(doc_path).stem)[:300]


def _build_skeleton(md_text: str, max_chars: int = 2000) -> str:
    """Esqueleto del archivo: primeras ~2000 chars + encabezados H1/H2/H3.

    Cada encabezado se lista con su número de línea REAL (1-based) en el
    archivo fuente — es la pista estructural que usa el LLM para delimitar
    los documentos apilados.
    """
    head = md_text[:max_chars]
    headers = []
    for i, line in enumerate(md_text.splitlines(), 1):
        if re.match(r"^#{1,3}\s+", line):
            headers.append(f"L{i}: {line.strip()}")
    parts = [head]
    if headers:
        parts.append("Encabezados (H1/H2/H3) con número de línea:")
        parts.extend(headers)
    return "\n".join(parts)


def _index_document_separation(session, md_path, md_text, verbose=True) -> list[dict]:
    """Fase 1: separa el archivo en los documentos (libros/papers) apilados.

    Devuelve SIEMPRE ≥1 documento: [{document_id, title, line_start, line_end,
    language}]. Degradación natural (nunca romper):
      1. LLM grande con la spec kag_document_separation (esqueleto + pistas
         del MultibookFinderTool si detecta ≥2 documentos).
      2. Si el LLM falla o el JSON es inválido → MultibookFinderTool.
      3. Si tampoco → un solo documento con el archivo completo
         (document_id="", comportamiento clásico).
    """
    md_path = Path(md_path)
    total_lines = len(md_text.splitlines())
    lang = detect_language(md_text)
    try:
        source_file = str(md_path.relative_to(DOCS_DIR)).replace("\\", "/")
    except ValueError:
        source_file = md_path.name

    # Pistas del MultibookFinderTool (detección física por ISBN/separadores).
    hints = []
    try:
        from src.kag.tools import MultibookFinderTool

        found = MultibookFinderTool().execute(str(md_path))
        if found:
            hints = found
    except Exception as exc:  # noqa: BLE001 — degradación natural
        if verbose:
            print(f"[KAG] ⚠ MultibookFinderTool no disponible: {exc}")

    # La lista determinista se inyecta al LLM como BASE (contenido del prompt):
    # el LLM la confirma y SOLO añade divisiones si encuentra libros/papers
    # separados no detectados físicamente. NUNCA divide un libro en capítulos
    # (eso es la Fase 2). Si no hay pistas, se indica "(ninguno detectado)".
    if hints:
        deterministic_docs = "\n".join(
            f"- [{h.get('document_id') or '?'}] {h.get('title', '?')}: líneas {h.get('line_start')}-{h.get('line_end')}"
            for h in hints
        )
    else:
        deterministic_docs = "(ninguno detectado)"

    # LLM grande con la spec kag_document_separation.
    documents = None
    try:
        try:
            settings = load_settings(session)
        except Exception:  # noqa: BLE001 — sesión falsa en tests
            settings = None
        retries = int(getattr(settings, "llm_retries", 3) or 3) if settings else 3
        fallback = getattr(settings, "fallback_model", None) if settings else None
        large_model = getattr(settings, "large_model", None) if settings else None
        system, user_template = _get_prompt_pair(
            session,
            large_model,
            TASK_DOCUMENT_SEPARATION,
            DOCUMENT_SEPARATION_SYSTEM_SHORT,
            DOCUMENT_SEPARATION_USER_SHORT,
        )
        prompt = _fill_prompt(
            user_template,
            source_file=source_file,
            skeleton=_build_skeleton(md_text),
            deterministic_documents=deterministic_docs,
        )
        text_out, _model, _used_fallback = call_with_retries(
            session,
            prompt=prompt,
            system=system,
            model_size="large",
            response_format={"type": "json_object"},
            retries=retries,
            fallback_model=fallback,
        )
        data = parse_llm_output(text_out)
        raw = data.get("documents") if isinstance(data, dict) else None
        if isinstance(raw, list) and raw:
            documents = raw
    except Exception as exc:  # noqa: BLE001 — LLM no disponible: degradación
        if verbose:
            print(f"[KAG] ⚠ Separación por LLM falló: {exc}")

    # Sanitización de rangos (swap si line_start > line_end, clamp al archivo).
    # Los documentos deterministas (hints) SIEMPRE se incluyen; los del LLM se
    # añaden solo si no colisionan por rango de líneas con un hint.
    out = []
    seen_ids = set()

    def _append_doc(did, title, ls, le, language):
        if did in seen_ids:
            # Evitar violación del UNIQUE (doc_path, document_id) si el LLM
            # repite ids: sufijo numérico determinista.
            base = did or "doc"
            n = 2
            while f"{base}_{n}" in seen_ids:
                n += 1
            did = f"{base}_{n}"
        seen_ids.add(did)
        out.append(
            {
                "document_id": did,
                "title": title,
                "line_start": ls,
                "line_end": le,
                "language": language,
            }
        )

    # 1) Deterministas: siempre incluidos (rango sanitizado).
    hint_ranges = []
    for h in hints:
        ls = max(1, int(h.get("line_start") or 1))
        le = min(total_lines, int(h.get("line_end") or total_lines))
        if ls > le:
            ls, le = le, ls
        hint_ranges.append((ls, le))
        _append_doc(
            str(h.get("document_id") or ""),
            sanitize_text(str(h.get("title") or ""))[:300]
            or _title_from_md(md_text, source_file),
            ls,
            le,
            lang,
        )

    # 2) LLM: se añaden solo si no colisionan por rango con un hint.
    if documents:
        for d in documents:
            if not isinstance(d, dict):
                continue
            try:
                ls = int(d.get("line_start") or 1)
                le = int(d.get("line_end") or total_lines)
            except (TypeError, ValueError):
                ls, le = 1, total_lines
            if ls > le:
                ls, le = le, ls
            ls = max(1, min(ls, total_lines))
            le = max(1, min(le, total_lines))
            if any(ls <= h_le and h_ls <= le for h_ls, h_le in hint_ranges):
                continue
            _append_doc(
                str(d.get("document_id") or ""),
                sanitize_text(str(d.get("title") or ""))[:300]
                or _title_from_md(md_text, source_file),
                ls,
                le,
                str(d.get("language") or lang) or lang,
            )
    if out:
        return out

    # Degradación: MultibookFinderTool (detección física, sin LLM).
    if hints:
        out = []
        for h in hints:
            ls = max(1, int(h.get("line_start") or 1))
            le = min(total_lines, int(h.get("line_end") or total_lines))
            if ls > le:
                ls, le = le, ls
            out.append(
                {
                    "document_id": str(h.get("document_id") or ""),
                    "title": (
                        sanitize_text(str(h.get("title") or ""))[:300]
                        or _title_from_md(md_text, source_file)
                    ),
                    "line_start": ls,
                    "line_end": le,
                    "language": lang,
                }
            )
        if out:
            return out

    # Degradación final: un solo documento con el archivo completo.
    return [
        {
            "document_id": "",
            "title": _title_from_md(md_text, source_file),
            "line_start": 1,
            "line_end": total_lines,
            "language": lang,
        }
    ]


def index_document(
    session,
    md_path,
    force=False,
    no_summary=False,
    verbose=True,
    llm_entities=False,
    extract_propositions=True,
):
    """Indexa un archivo .md en el KAG (flujo §2.3 del diseño).

    Fase 1: separa el archivo en los documentos apilados (libros, papers,
    artículos) y indexa CADA documento como UNA fila de kag_documents
    (idempotente por (doc_path, document_id) + content_hash del slice;
    --force re-indexa). Devuelve dict resumen AGREGADO del archivo.

    Atomicidad POR ETAPA (máquina de estados, src/kag/stages.py):
      pending -> segmented -> chunked -> figures -> ready

    Cada etapa commitea su trabajo y actualiza `stage`. Si el proceso se
    interrumpe (p. ej. durante la segmentación, que es lenta), al reanudar
    se saltan las etapas ya completadas y se continúa desde la interrumpida.

    `llm_entities=True` usa la extracción LLM por chunk (Together, costosa);
    por defecto se usa la extracción determinista con spaCy (gratis).

    `extract_propositions=True` (default) extrae y persiste proposiciones
    atómicas en `kag_propositions` (capa micro, migración 0024). La
    extracción es un paso APARTE del chunking (etapa 'chunked', sobre TODOS
    los chunks ya persistidos): agrupación doc→capítulo (chapter_id
    detectado por LLM, Fase 2; por ahora NULL → un único grupo a nivel de
    documento; capítulos >30k tokens = grupo propio, el resto a nivel de
    archivo), batching por tokens (KAG_PROPOSITION_BATCH_SIZE, default 400k)
    con UNA llamada LLM por lote, llamadas en paralelo (semáforo,
    KAG_PROPOSITION_PARALLEL) y cache por content_hash (0026) — los chunks
    ya extraídos con hash idéntico se saltan en re-ingestas. Con False el
    hook se salta: comportamiento clásico EXACTO.
    """
    md_path = Path(md_path)
    if not md_path.exists():
        raise FileNotFoundError(f"No existe {md_path}")
    doc_path = str(md_path.relative_to(DOCS_DIR)).replace("\\", "/")
    # Flags de proposiciones desde config (KAG_EXTRACT_PROPOSITIONS,
    # KAG_PROPOSITION_BATCH_SIZE, KAG_PROPOSITION_MODEL,
    # KAG_PROPOSITION_PARALLEL) combinadas con el param de compatibilidad.
    extract_propositions, prop_batch_size, prop_model_size, prop_max_parallel = (
        _proposition_flags(session, extract_propositions)
    )
    # OJO: la variable se llama md_text (NO text) para no sombrear la función
    # text() de SQLAlchemy — un bug previo rompía la ingesta con
    # "'str' object is not callable" en el primer SELECT.
    md_text = sanitize_text(md_path.read_text(encoding="utf-8"))
    # Hash del ARCHIVO completo: el slice de un documento único (document_id="")
    # que cubre todo el archivo coincide con md_text, así que su content_hash
    # ES este hash (se evita re-hashear el slice completo).
    file_hash = hashlib.sha256(md_text.encode("utf-8")).hexdigest()

    # ── Fase 1: separación del archivo en documentos ────────────────────────
    doc_specs = _index_document_separation(session, md_path, md_text, verbose)
    if not doc_specs:  # defensivo: la separación SIEMPRE devuelve ≥1
        doc_specs = [
            {
                "document_id": "",
                "title": _title_from_md(md_text, doc_path),
                "line_start": 1,
                "line_end": len(md_text.splitlines()),
                "language": detect_language(md_text),
            }
        ]

    results = []
    for spec in doc_specs:
        results.append(
            _index_document_slice(
                session,
                md_path,
                doc_path,
                md_text,
                file_hash,
                spec,
                force=force,
                no_summary=no_summary,
                verbose=verbose,
                llm_entities=llm_entities,
                extract_propositions=extract_propositions,
                prop_batch_size=prop_batch_size,
                prop_model_size=prop_model_size,
                prop_max_parallel=prop_max_parallel,
            )
        )

    indexed = sum(1 for r in results if r.get("status") == "indexed")
    skipped = sum(1 for r in results if r.get("status") == "skipped")
    if verbose:
        print(
            f"[KAG] 📚 {doc_path}: {len(doc_specs)} documento(s) — "
            f"{indexed} indexado(s), {skipped} omitido(s)."
        )
    return {
        "status": (
            "indexed" if indexed else (results[0]["status"] if results else "skipped")
        ),
        "doc_path": doc_path,
        "document_count": len(doc_specs),
        "chunk_count": sum(r.get("chunk_count", 0) for r in results),
        "entity_count": sum(r.get("entity_count", 0) for r in results),
        "relation_count": sum(r.get("relation_count", 0) for r in results),
        "figure_count": sum(r.get("figure_count", 0) for r in results),
    }


def _index_document_slice(
    session,
    md_path,
    doc_path,
    md_text,
    file_hash,
    spec,
    force=False,
    no_summary=False,
    verbose=True,
    llm_entities=False,
    extract_propositions=True,
    prop_batch_size=400000,
    prop_model_size="large",
    prop_max_parallel=3,
):
    """Indexa UN documento (slice del archivo) en el KAG (flujo §2.3).

    Idempotente por (doc_path, document_id) + content_hash del slice;
    --force re-indexa. Devuelve dict resumen del documento.
    """
    document_id = str(spec.get("document_id") or "")
    total_lines = len(md_text.splitlines())
    try:
        line_start = int(spec.get("line_start") or 1)
        line_end = int(spec.get("line_end") or total_lines)
    except (TypeError, ValueError):
        line_start, line_end = 1, total_lines
    if line_start > line_end:
        line_start, line_end = line_end, line_start
    line_start = max(1, min(line_start, total_lines))
    line_end = max(1, min(line_end, total_lines))

    # Slice del archivo para este documento. Un documento único (document_id="")
    # que cubre todo el archivo usa md_text completo (y su hash = file_hash).
    if document_id == "" and line_start == 1 and line_end >= total_lines:
        slice_text = md_text
        slice_hash = file_hash
    else:
        slice_text = "\n".join(md_text.splitlines()[line_start - 1 : line_end])
        slice_hash = hashlib.sha256(slice_text.encode("utf-8")).hexdigest()
    slice_tokens = estimate_tokens(slice_text)
    doc_type = "long" if slice_tokens >= LONG_DOC_THRESHOLD else "short"
    lang = detect_language(slice_text)
    title = sanitize_text(str(spec.get("title") or ""))[:300] or _title_from_md(
        slice_text, doc_path
    )
    label = f"{doc_path} [{document_id}]" if document_id else doc_path

    existing = session.execute(
        text(
            "SELECT id, content_hash, status, stage FROM kag_documents "
            "WHERE doc_path = :p AND document_id = :d"
        ),
        {"p": doc_path, "d": document_id},
    ).first()

    # ── Idempotencia / reanudación ─────────────────────────────────────────
    if existing and existing.content_hash == slice_hash and not force:
        if existing.status == "ready":
            null_emb = session.execute(
                text(
                    "SELECT COUNT(*) FROM kag_chunks "
                    "WHERE doc_id = :id AND embedding IS NULL"
                ),
                {"id": existing.id},
            ).scalar()
            if null_emb == 0:
                if verbose:
                    print(f"[KAG] ⏭ {label} ya indexado (hash idéntico).")
                return {"status": "skipped", "doc_path": doc_path}
            # ready pero con embeddings incompletos → re-embeder solo los que
            # faltan (sin re-segmentar ni re-extraer entidades).
            if verbose:
                print(
                    f"[KAG] ♻ {label} ready pero {null_emb} chunks sin "
                    "embedding — re-embebiendo..."
                )
            rows = session.execute(
                text(
                    "SELECT id, content FROM kag_chunks "
                    "WHERE doc_id = :id AND embedding IS NULL"
                ),
                {"id": existing.id},
            ).fetchall()
            from src.embeddings import embed_text

            for r in rows:
                try:
                    emb = embed_text(r.content, input_type="document")
                except Exception as exc:  # noqa: BLE001 — chunk sin embedding
                    if verbose:
                        print(f"[KAG] ⚠ Embedding falló (chunk {r.id}): {exc}")
                    continue
                session.execute(
                    text(
                        "UPDATE kag_chunks SET embedding = CAST(:emb AS vector) "
                        "WHERE id = :id"
                    ),
                    {"emb": embedding_to_sql(emb), "id": r.id},
                )
            session.commit()
            return {"status": "reembedded", "doc_path": doc_path}

        # No está 'ready': reanudar desde la última etapa completada (NO se
        # borra nada — la máquina de estados salta lo ya hecho).
        resume = resume_from(KAG_INGEST_STAGES, existing.stage or "pending")
        if verbose:
            print(
                f"[KAG] ♻ {label} en stage '{existing.stage}' — reanudando "
                f"desde '{resume}'..."
            )
    else:
        # Fuerza o hash cambiado (o fila inexistente): borrar y empezar de cero.
        if existing:
            session.execute(
                text("DELETE FROM kag_documents WHERE id = :id"), {"id": existing.id}
            )
            session.commit()
        resume = "pending"

    doc_id = None
    try:
        if resume == "pending":
            # El INSERT va DENTRO del try: si algo falla temprano (p. ej. descarga
            # de modelos del segmentador), la fila queda con status='failed' y el
            # error visible en la DB en lugar de desaparecer sin rastro.
            doc_id = session.execute(
                text(
                    "INSERT INTO kag_documents "
                    "(doc_path, document_id, line_start, line_end, title, doc_type, "
                    "status, content_hash, token_estimate, stage, language) "
                    "VALUES (:doc_path, :document_id, :line_start, :line_end, :title, "
                    ":doc_type, 'pending', :content_hash, :token_estimate, 'pending', "
                    ":language) RETURNING id"
                ),
                {
                    "doc_path": doc_path,
                    "document_id": document_id,
                    "line_start": line_start,
                    "line_end": line_end,
                    "title": title,
                    "doc_type": doc_type,
                    "content_hash": slice_hash,
                    "token_estimate": slice_tokens,
                    "language": lang,
                },
            ).scalar()
            session.commit()
        else:
            doc_id = existing.id

        # ── Etapa: analysis (ficha documental + capítulos, Fase 2) ─────────
        # El LLM grande detecta la ficha ISO 25964 y los capítulos del
        # documento; se persisten en kag_documents.ficha_jsonb/sections_json y
        # kag_chapters. Los capítulos los usan las Fases 3/4/5.
        if resume in ("pending", "analysis"):
            if resume == "analysis":
                # Re-ejecutar la etapa: limpiar capítulos y ficha previos.
                cleanup_stage(session, "kag_documents", doc_id, "analysis")
            _index_document_analysis(session, doc_id, doc_path, slice_text, verbose)
            set_stage(session, "kag_documents", doc_id, "analysis")

        # ── Etapa: segmented (segmentación + chunks sin embedding) ──────────
        # La segmentación es la etapa LENTA (torch/spacy). Se persisten los
        # chunks con embedding NULL ANTES de embeker: si se interrumpe aquí,
        # al reanudar NO se re-segmenta.
        if verbose:
            print(
                f"[KAG] 📄 {label} | {doc_type} | ~{slice_tokens} tokens | idioma {lang}"
            )
        if resume in ("pending", "analysis", "segmented"):
            if resume == "segmented":
                # Re-ejecutar la etapa: limpiar chunks parciales primero.
                cleanup_stage(session, "kag_documents", doc_id, "segmented")
            from src.kag.segmentador import build_segmenter

            segmenter = build_segmenter(session, lang=lang, verbose=verbose)
            # Coref Stanza: docs cortos lo usan (costo acotado); book stacks lo
            # omiten (prohibitivo: ~11s por segmento).
            use_coref = doc_type == "short"
            # Fase 3: segmentar DENTRO de cada capítulo (kag_chapters, Fase 2).
            # Los capítulos se leen con líneas RELATIVAS al slice (1-based). Si
            # no hay capítulos (degradación de A3), chapters=None → el
            # comportamiento clásico (chapter_id=None).
            capitulos = None
            chapter_rows = session.execute(
                text(
                    "SELECT id, chapter_id, line_start, line_end FROM kag_chapters "
                    "WHERE doc_id = :id ORDER BY line_start"
                ),
                {"id": doc_id},
            ).fetchall()
            if chapter_rows:
                capitulos = [
                    {
                        "chapter_id": str(r.id),
                        "line_start": r.line_start,
                        "line_end": r.line_end,
                    }
                    for r in chapter_rows
                ]
            chunks = chunk_markdown(
                slice_text,
                doc_type,
                segmenter,
                max_tokens=CHUNK_MAX_TOKENS,
                use_coref=use_coref,
                chapters=capitulos,
            )

            chunk_count = 0
            for i, chunk in enumerate(chunks):
                session.execute(
                    text(
                        "INSERT INTO kag_chunks "
                        "(doc_id, chunk_index, chapter_id, content, token_estimate, "
                        "content_hash, embedding) VALUES (:doc_id, :chunk_index, "
                        ":chapter_id, :content, :token_estimate, :content_hash, NULL) "
                    ),
                    {
                        "doc_id": doc_id,
                        "chunk_index": i,
                        "chapter_id": chunk.get("chapter_id"),
                        "content": chunk["content"],
                        "token_estimate": chunk["token_estimate"],
                        "content_hash": _chunk_content_hash(chunk["content"]),
                    },
                )
                chunk_count += 1
            _maintain_word_freq(session, doc_id)
            set_stage(session, "kag_documents", doc_id, "segmented")
            if verbose:
                print(f"[KAG] ✂ {label}: {chunk_count} chunks segmentados.")

        # ── Etapa: paraphrased (paráfrasis de chunks, Fase 4) ───────────────
        # Una llamada LLM grande por capítulo (grupo de chunks); los chunks
        # sin paraphrase (LLM falló o no los devolvió) quedan NULL y la
        # ingesta sigue (degradación natural).
        if resume in ("pending", "analysis", "segmented", "paraphrased"):
            if resume == "paraphrased":
                # Re-ejecutar la etapa: limpiar paráfrasis parciales primero.
                cleanup_stage(session, "kag_documents", doc_id, "paraphrased")
            _paraphrase_chunks(session, doc_id, doc_path, verbose)
            set_stage(session, "kag_documents", doc_id, "paraphrased")

        # ── Etapa: chunked (embeddings + entidades + relaciones) ────────────
        if resume in ("pending", "analysis", "segmented", "paraphrased", "chunked"):
            if resume == "chunked":
                cleanup_stage(session, "kag_documents", doc_id, "chunked")
            # Import perezoso: src.embeddings importa src.db.session (que lee
            # .env al importar) — debe ocurrir DESPUÉS de _fix_db_host().
            from src.embeddings import embed_texts

            # Si se reanuda desde 'chunked' (segmentación ya hecha), el
            # segmenter no está en memoria: reconstruirlo solo para el nlp.
            if "segmenter" not in locals():
                from src.kag.segmentador import build_segmenter

                segmenter = build_segmenter(session, lang=lang, verbose=verbose)

            # Batch de embeddings: una sola llamada al modelo por lote de chunks
            # (en vez de N llamadas individuales — el cuello de botella real de
            # la ingesta). El lote se parte en bloques de EMBED_BATCH_SIZE para
            # no reventar la memoria del modelo.
            EMBED_BATCH_SIZE = 32
            # nlp.pipe multi-core (KAG_ENTITY_N_PROCESS, default 1 — seguro en
            # Windows/macOS donde multiprocessing necesita el guard
            # if __name__ == "__main__"). >1 solo si el entorno lo soporta;
            # extract_entities_deterministic_batch degrada a 1 si falla.
            _config = resolve_config(session)
            entity_n_process = int(
                get_config_value(_config, "KAG_ENTITY_N_PROCESS", 1) or 1
            )
            entity_n_process = max(entity_n_process, 1)
            chunk_rows = session.execute(
                text(
                    "SELECT id, content, chunk_index, chapter_id FROM kag_chunks "
                    "WHERE doc_id = :id ORDER BY chunk_index"
                ),
                {"id": doc_id},
            ).fetchall()
            total_batches = (len(chunk_rows) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE
            for start in range(0, len(chunk_rows), EMBED_BATCH_SIZE):
                batch = chunk_rows[start : start + EMBED_BATCH_SIZE]
                batch_no = start // EMBED_BATCH_SIZE + 1
                if verbose:
                    print(
                        f"[KAG] ⚙ chunked: lote {batch_no}/{total_batches} "
                        f"(chunks {start + 1}-{start + len(batch)} de {len(chunk_rows)})..."
                    )
                try:
                    embs = embed_texts(
                        [r.content for r in batch], input_type="document"
                    )
                except Exception as exc:  # noqa: BLE001 — chunk sin embedding
                    if verbose:
                        print(f"[KAG] ⚠ Embedding falló (lote {start}): {exc}")
                    embs = [None] * len(batch)
                # Extracción determinista con spaCy (gratis): reutiliza el nlp
                # del segmentador (ya cargado) y canonicaliza con el modelo de
                # embeddings de la DB (embed_texts, batch). Con nlp.pipe
                # multi-core (KAG_ENTITY_N_PROCESS, default 1) se procesan
                # TODOS los chunks del lote en paralelo (fuera del GIL para
                # noun chunks, lematización y filtrado sintáctico).
                if llm_entities:
                    batch_data = None
                else:
                    from src.kag.entities import extract_entities_deterministic_batch

                    batch_data = extract_entities_deterministic_batch(
                        segmenter.nlp,
                        [r.content for r in batch],
                        embed_fn=embed_texts,
                        lang=lang,
                        n_process=entity_n_process,
                    )
                for i, row in enumerate(batch):
                    session.execute(
                        text(
                            "UPDATE kag_chunks SET embedding = CAST(:emb AS vector) "
                            "WHERE id = :id"
                        ),
                        {"emb": embedding_to_sql(embs[i]), "id": row.id},
                    )
                    if verbose:
                        print(
                            f"[KAG] ⚙ chunked: chunk {start + i + 1}/{len(chunk_rows)} "
                            "— entidades..."
                        )
                    if llm_entities:
                        data = extract_entities_relations(session, row.content)
                    else:
                        data = batch_data[i]
                    _store_entities_relations(
                        session, doc_id, row.id, data, embed_fn=embed_texts
                    )
            # ── Paso APARTE del chunking: proposiciones + entidades por
            # documento (Fase 5) ────────────────────────────────────────────
            # Opera sobre TODOS los chunks ya persistidos (no chunk por chunk
            # en el bucle de embeddings). Toma las paráfrasis de la Fase 4
            # (paraphrase or content si NULL) y las agrupa por documento en
            # lotes por tokens (KAG_PROPOSITION_BATCH_SIZE, default 400k) con
            # UNA llamada LLM por lote + cache por content_hash (0026): los
            # chunks ya extraídos con hash idéntico se saltan en re-ingestas.
            if extract_propositions:
                _extract_document_propositions(
                    session,
                    doc_id,
                    doc_path,
                    verbose,
                    prop_batch_size,
                    embed_fn=embed_texts,
                    model_size=prop_model_size,
                    max_parallel=prop_max_parallel,
                )
            set_stage(session, "kag_documents", doc_id, "chunked")

        entity_count = session.execute(
            text("SELECT COUNT(*) FROM kag_entities WHERE doc_id = :id"),
            {"id": doc_id},
        ).scalar()
        relation_count = session.execute(
            text("SELECT COUNT(*) FROM kag_relations WHERE doc_id = :id"),
            {"id": doc_id},
        ).scalar()

        # ── Etapa: figures ──────────────────────────────────────────────────
        if resume in (
            "pending",
            "analysis",
            "segmented",
            "paraphrased",
            "chunked",
            "figures",
        ):
            if resume == "figures":
                cleanup_stage(session, "kag_documents", doc_id, "figures")
            if verbose:
                print(f"[KAG] 🖼 figures: indexando figuras de {md_path.name}...")
            figure_count = _index_figures(
                session, doc_id, md_path, slice_text, line_start, verbose
            )
            set_stage(session, "kag_documents", doc_id, "figures")

        # ── Etapa: ready (resumen + cierre) ─────────────────────────────────
        if resume in (
            "pending",
            "analysis",
            "segmented",
            "paraphrased",
            "chunked",
            "figures",
            "ready",
        ):
            # Si se reanuda desde una etapa posterior, los contadores locales
            # no se definieron en esta ejecución: leerlos de la DB.
            if "chunk_count" not in locals():
                chunk_count = session.execute(
                    text("SELECT COUNT(*) FROM kag_chunks WHERE doc_id = :id"),
                    {"id": doc_id},
                ).scalar()
            if "figure_count" not in locals():
                figure_count = session.execute(
                    text("SELECT COUNT(*) FROM kag_figures WHERE doc_id = :id"),
                    {"id": doc_id},
                ).scalar()
            summary = ""
            if not no_summary:
                if verbose:
                    print(
                        "[KAG] 📝 ready: generando resumen jerárquico (Qwen local)..."
                    )
                summary = sanitize_text(
                    summarize_document(session, slice_text, doc_type)
                )
                if verbose:
                    print("[KAG] 📝 ready: resumen listo.")

            session.execute(
                text(
                    "UPDATE kag_documents SET status = 'ready', chunk_count = :cc, "
                    "entity_count = :ec, relation_count = :rc, figure_count = :fc, "
                    "summary = :summary, updated_at = now() WHERE id = :id"
                ),
                {
                    "cc": chunk_count,
                    "ec": entity_count,
                    "rc": relation_count,
                    "fc": figure_count,
                    "summary": summary,
                    "id": doc_id,
                },
            )
            set_stage(session, "kag_documents", doc_id, "ready")

        if verbose:
            print(
                f"[KAG] ✅ {label} indexado: {chunk_count} chunks, "
                f"{entity_count} entidades, {relation_count} relaciones, "
                f"{figure_count} figuras."
            )
        return {
            "status": "indexed",
            "doc_path": doc_path,
            "document_id": document_id,
            "chunk_count": chunk_count,
            "entity_count": entity_count,
            "relation_count": relation_count,
            "figure_count": figure_count,
        }
    except Exception as exc:  # marcar failed y re-lanzar
        session.rollback()
        # El rollback deshizo el INSERT: re-insertar (o actualizar) la fila
        # con status='failed' para que el error quede visible en la DB.
        try:
            session.execute(
                text(
                    "INSERT INTO kag_documents "
                    "(doc_path, document_id, line_start, line_end, title, doc_type, "
                    "status, content_hash, token_estimate, error) "
                    "VALUES (:doc_path, :document_id, :line_start, :line_end, :title, "
                    ":doc_type, 'failed', :content_hash, :token_estimate, :err) "
                    "ON CONFLICT (doc_path, document_id) DO UPDATE SET status = 'failed', "
                    "error = :err, updated_at = now()"
                ),
                {
                    "doc_path": doc_path,
                    "document_id": document_id,
                    "line_start": line_start,
                    "line_end": line_end,
                    "title": title,
                    "doc_type": doc_type,
                    "content_hash": slice_hash,
                    "token_estimate": slice_tokens,
                    "err": str(exc)[:500],
                },
            )
            session.commit()
        except Exception:  # noqa: BLE001 — no bloquea el re-lanzamiento
            pass
        if verbose:
            print(f"[KAG] ❌ Error indexando {label}: {exc}")
            import traceback

            traceback.print_exc(file=sys.stdout)
        raise


def index_all(
    session,
    force=False,
    no_summary=False,
    verbose=True,
    llm_entities=False,
    extract_propositions=True,
):
    """Indexa todos los .md de data/knowledge_repository/docs/."""
    docs = sorted(DOCS_DIR.glob("*.md"))
    if not docs:
        if verbose:
            print("[KAG] 📂 No hay documentos en data/knowledge_repository/docs/.")
        return []
    results = []
    for md in docs:
        try:
            results.append(
                index_document(
                    session,
                    md,
                    force=force,
                    no_summary=no_summary,
                    verbose=verbose,
                    llm_entities=llm_entities,
                    extract_propositions=extract_propositions,
                )
            )
        except Exception as exc:  # noqa: BLE001 — un doc no bloquea el resto
            if verbose:
                print(f"[KAG] ❌ Error indexando {md.name}: {exc}")
    return results


def backfill_entity_embeddings(session, batch_size=64, verbose=True):
    """Rellena name_embedding de entidades que aún lo tienen NULL (migración 0022).

    Idempotente: solo toca filas con name_embedding IS NULL. Embebe los nombres
    en lotes con embed_texts y hace un UPDATE por lote (commit por lote).
    Devuelve el número de entidades actualizadas.
    """
    total = 0
    while True:
        rows = session.execute(
            text(
                "SELECT id, name FROM kag_entities "
                "WHERE name_embedding IS NULL LIMIT :limit"
            ),
            {"limit": batch_size},
        ).fetchall()
        if not rows:
            break
        # Import perezoso: src.embeddings importa src.db.session (que lee
        # .env al importar) — debe ocurrir DESPUÉS de _fix_db_host().
        from src.embeddings import embed_texts

        try:
            embs = embed_texts([r.name for r in rows])
        except Exception as exc:  # noqa: BLE001 — sin embedding: se salta el lote
            if verbose:
                print(f"[KAG] ⚠ Backfill embeddings falló (lote): {exc}")
            break
        for r, emb in zip(rows, embs):
            session.execute(
                text(
                    "UPDATE kag_entities SET name_embedding = CAST(:emb AS vector) "
                    "WHERE id = :id"
                ),
                {"emb": embedding_to_sql(emb), "id": r.id},
            )
        session.commit()
        total += len(rows)
        if verbose:
            print(f"[KAG] ⚙ Backfill embeddings: {total} entidades...")
    if verbose:
        print(f"[KAG] ✅ Backfill embeddings completo: {total} entidades.")
    return total


def backfill_word_freq(session, verbose=True):
    """Reconstruye kag_word_freq para todos los documentos (migración 0023).

    Idempotente: DELETE + INSERT por documento ready. Para los docs ya
    indexados antes de la migración. Devuelve el número de documentos
    procesados.
    """
    rows = session.execute(
        text(
            "SELECT id FROM kag_documents "
            "WHERE status = 'ready' AND EXISTS (SELECT 1 FROM kag_chunks c "
            "WHERE c.doc_id = kag_documents.id) ORDER BY id"
        )
    ).fetchall()
    for r in rows:
        _maintain_word_freq(session, r.id)
    session.commit()
    if verbose:
        print(f"[KAG] ✅ Backfill word_freq completo: {len(rows)} documentos.")
    return len(rows)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def main() -> None:
    # Windows: la consola usa cp1252 y no imprime emojis — forzar UTF-8.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Ingesta KAG del knowledge_repository."
    )
    parser.add_argument(
        "--doc", help="Nombre del .md a indexar (default: todos los de docs/)."
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-indexa aunque el hash no cambió."
    )
    parser.add_argument(
        "--no-summary", action="store_true", help="Omite el resumen local Qwen 2.5."
    )
    parser.add_argument(
        "--llm-entities",
        action="store_true",
        help="Usa la extracción LLM por chunk (Together, costosa) en vez de la "
        "determinista con spaCy (default).",
    )
    parser.add_argument(
        "--no-propositions",
        action="store_true",
        help="Omite la extracción de proposiciones atómicas (comportamiento "
        "clásico EXACTO; default: extraer).",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True, help="Prints descriptivos."
    )
    parser.add_argument(
        "--backfill-embeddings",
        action="store_true",
        help="Rellena name_embedding de entidades sin embedding (migración 0022).",
    )
    parser.add_argument(
        "--backfill-word-freq",
        action="store_true",
        help="Reconstruye kag_word_freq para todos los docs (migración 0023).",
    )
    args = parser.parse_args()

    _fix_db_host()
    from src.db.session import SessionLocal

    session = SessionLocal()
    try:
        if args.backfill_embeddings:
            backfill_entity_embeddings(session, verbose=args.verbose)
        elif args.backfill_word_freq:
            backfill_word_freq(session, verbose=args.verbose)
        elif args.doc:
            index_document(
                session,
                DOCS_DIR / args.doc,
                force=args.force,
                no_summary=args.no_summary,
                verbose=args.verbose,
                llm_entities=args.llm_entities,
                extract_propositions=not args.no_propositions,
            )
        else:
            index_all(
                session,
                force=args.force,
                no_summary=args.no_summary,
                verbose=args.verbose,
                llm_entities=args.llm_entities,
                extract_propositions=not args.no_propositions,
            )
    finally:
        session.close()


if __name__ == "__main__":
    main()
