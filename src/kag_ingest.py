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
import os
import re
import sys
from pathlib import Path

import httpx
from sqlalchemy import bindparam, text

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

# System prompts cortos actuales — fallback EXACTO de hoy cuando no hay
# artefacto compilado (tests sin DB: get_active_prompt devuelve None).
EXTRACT_SYSTEM_SHORT = "Eres un extractor de conocimiento. Devuelve JSON válido."


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


def embedding_to_sql(emb):
    """Convierte una lista de floats a la sintaxis literal de pgvector.

    psycopg2 adapta las listas Python a numeric[], que pgvector no acepta;
    el literal '[0.1,0.2,...]' con CAST AS vector sí funciona.
    """
    if emb is None:
        return None
    return "[" + ",".join(repr(float(x)) for x in emb) + "]"


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
    md_text, doc_type, segmenter, max_tokens=CHUNK_MAX_TOKENS, use_coref=True
):
    """Parte el markdown por encabezados y segmenta cada sección.

    - short: parte por `#`, `##`, `###` preservando section_path.
    - long:  pre-segmenta por `#`, `##` (acota cada llamada al segmentador).

    El segmentador produce cortes semánticos (a veces muy pequeños, ~50-100
    tokens). Para la ingesta KAG fusionamos segmentos adyacentes de la MISMA
    sección hasta `max_tokens` (respetando los cortes del segmentador como
    fronteras duras): reduce el número de chunks (y de llamadas LLM de
    extracción) sin perder los límites semánticos.

    `use_coref=False` omite la resolución de correferencias del segmentador
    (monkeypatch temporal en la instancia): el coref Stanza cuesta ~11s por
    segmento y es prohibitivo en book stacks; los docs cortos sí lo usan.

    Devuelve lista de dicts {"content", "section_path", "token_estimate"}.
    """
    max_level = 2 if doc_type == "long" else 3
    sections = []  # (section_path, content)
    current_path: list = []
    current_lines: list = []

    def flush() -> None:
        content = "\n".join(current_lines).strip()
        if content:
            sections.append((" > ".join(current_path), content))

    for line in md_text.splitlines():
        m = re.match(r"^(#{1,3})\s+(.*)$", line)
        if m and len(m.group(1)) <= max_level:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            current_path = current_path[: level - 1] + [f"{'#' * level} {title}"]
            current_lines = []
        else:
            current_lines.append(line)
    flush()

    if not sections:
        sections = [("", md_text.strip())]

    # Coref opcional: no-op temporal en la instancia (no se modifica el
    # segmentador; el método original se restaura al salir).
    _orig_resolve = getattr(segmenter, "resolve_coreferences", None)
    if not use_coref and _orig_resolve is not None:
        segmenter.resolve_coreferences = lambda segments: segments  # noqa: E731
    try:
        chunks = []
        for section_path, content in sections:
            segments = segmenter.segment_text(content, max_tokens=max_tokens)
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
                            "section_path": section_path,
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
                        "section_path": section_path,
                        "token_estimate": estimate_tokens(buffer),
                    }
                )
        return chunks
    finally:
        if _orig_resolve is not None:
            segmenter.resolve_coreferences = _orig_resolve


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
                    "Authorization": "Bearer ",
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


def _store_entities_relations(session, doc_id, chunk_id, data):
    """Guarda entidades y relaciones de un chunk con dedup por name_norm (por doc).

    Batch: una sola consulta de existentes por doc + un solo INSERT multi-VALUES
    por lote de entidades y otro por lote de relaciones (en vez de N+1
    SELECT/INSERT por entidad). El grafo es incremental a nivel de documento:
    solo se tocan las filas de este doc_id (el trigger de versión 0016
    invalida la caché de adyacencia una vez por statement).
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
        name = str(ent.get("name", "")).strip()
        if not name:
            continue
        name_norm = normalize_entity_name(name)
        if name_norm in entity_ids:
            continue
        entity_ids[name_norm] = None  # placeholder: se rellena con RETURNING
        new_entities.append(
            (
                doc_id,
                chunk_id,
                name,
                name_norm,
                str(ent.get("type", "concept"))[:100],
                str(ent.get("description", "")),
            )
        )
    if new_entities:
        # Un solo INSERT multi-VALUES con UN set de parámetros. Pasar una
        # lista de dicts haría executemany, y psycopg2 no devuelve filas con
        # executemany + RETURNING (ResourceClosedError "does not return rows").
        placeholders = ", ".join(
            f"(:d{i}, :c{i}, :n{i}, :nn{i}, :e{i}, :de{i})"
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
                }
            )
        rows = session.execute(
            text(
                "INSERT INTO kag_entities "
                "(doc_id, chunk_id, name, name_norm, entity_type, description) "
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
        src = normalize_entity_name(str(rel.get("source", "")))
        tgt = normalize_entity_name(str(rel.get("target", "")))
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
                str(rel.get("type", "RELACIONA"))[:200],
                str(rel.get("description", "")),
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
        for sec in _split_h1_h2(text):
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


def _index_figures(session, doc_id, md_path, md_text, verbose=True) -> int:
    """Indexa las figuras de images/[docname]/ si la carpeta existe."""
    docname = md_path.stem
    images_dir = IMAGES_DIR / docname
    if not images_dir.exists():
        return 0
    count = 0
    for img in sorted(images_dir.iterdir()):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        caption = _find_caption(md_text, img.name)
        chunk_id = _find_chunk_for_image(session, doc_id, img.name)
        description = describe_figure(session, str(img), caption)
        session.execute(
            text(
                "INSERT INTO kag_figures "
                "(doc_id, chunk_id, image_path, caption, description) "
                "VALUES (:doc_id, :chunk_id, :image_path, :caption, :description)"
            ),
            {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "image_path": f"images/{docname}/{img.name}",
                "caption": caption,
                "description": description,
            },
        )
        count += 1
    if verbose and count:
        print(f"[KAG] 🖼 {count} figuras indexadas para {docname}.")
    return count


# ---------------------------------------------------------------------
# Flujo de indexación
# ---------------------------------------------------------------------


def _title_from_md(text: str, doc_path: str) -> str:
    """Título: primer H1 del markdown, o el nombre del archivo."""
    m = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    if m:
        return m.group(1).strip()[:300]
    return Path(doc_path).stem[:300]


def index_document(
    session, md_path, force=False, no_summary=False, verbose=True, llm_entities=False
):
    """Indexa un documento .md en el KAG (flujo §2.3 del diseño).

    Idempotente por content_hash; --force re-indexa. Devuelve dict resumen.

    Atomicidad POR ETAPA (máquina de estados, src/kag/stages.py):
      pending -> segmented -> chunked -> figures -> ready

    Cada etapa commitea su trabajo y actualiza `stage`. Si el proceso se
    interrumpe (p. ej. durante la segmentación, que es lenta), al reanudar
    se saltan las etapas ya completadas y se continúa desde la interrumpida.

    `llm_entities=True` usa la extracción LLM por chunk (Together, costosa);
    por defecto se usa la extracción determinista con spaCy (gratis).
    """
    md_path = Path(md_path)
    if not md_path.exists():
        raise FileNotFoundError(f"No existe {md_path}")
    doc_path = str(md_path.relative_to(DOCS_DIR)).replace("\\", "/")
    # OJO: la variable se llama md_text (NO text) para no sombrear la función
    # text() de SQLAlchemy — un bug previo rompía la ingesta con
    # "'str' object is not callable" en el primer SELECT.
    md_text = md_path.read_text(encoding="utf-8")
    content_hash = hashlib.sha256(md_text.encode("utf-8")).hexdigest()
    token_estimate = estimate_tokens(md_text)
    doc_type = "long" if token_estimate >= LONG_DOC_THRESHOLD else "short"

    existing = session.execute(
        text(
            "SELECT id, content_hash, status, stage FROM kag_documents "
            "WHERE doc_path = :p"
        ),
        {"p": doc_path},
    ).first()

    # ── Idempotencia / reanudación ─────────────────────────────────────────
    if existing and existing.content_hash == content_hash and not force:
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
                    print(f"[KAG] ⏭ {doc_path} ya indexado (hash idéntico).")
                return {"status": "skipped", "doc_path": doc_path}
            # ready pero con embeddings incompletos → re-embeder solo los que
            # faltan (sin re-segmentar ni re-extraer entidades).
            if verbose:
                print(
                    f"[KAG] ♻ {doc_path} ready pero {null_emb} chunks sin "
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
                f"[KAG] ♻ {doc_path} en stage '{existing.stage}' — reanudando "
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

    title = _title_from_md(md_text, doc_path)
    doc_id = None
    try:
        if resume == "pending":
            # El INSERT va DENTRO del try: si algo falla temprano (p. ej. descarga
            # de modelos del segmentador), la fila queda con status='failed' y el
            # error visible en la DB en lugar de desaparecer sin rastro.
            doc_id = session.execute(
                text(
                    "INSERT INTO kag_documents "
                    "(doc_path, title, doc_type, status, content_hash, token_estimate, stage) "
                    "VALUES (:doc_path, :title, :doc_type, 'pending', :content_hash, "
                    ":token_estimate, 'pending') RETURNING id"
                ),
                {
                    "doc_path": doc_path,
                    "title": title,
                    "doc_type": doc_type,
                    "content_hash": content_hash,
                    "token_estimate": token_estimate,
                },
            ).scalar()
            session.commit()
        else:
            doc_id = existing.id

        # ── Etapa: segmented (segmentación + chunks sin embedding) ──────────
        # La segmentación es la etapa LENTA (torch/spacy). Se persisten los
        # chunks con embedding NULL ANTES de embeker: si se interrumpe aquí,
        # al reanudar NO se re-segmenta.
        # lang/segmenter se calculan UNA vez: los necesitan tanto la
        # segmentación como la extracción determinista de entidades (nlp).
        lang = detect_language(md_text)
        if verbose:
            print(
                f"[KAG] 📄 {doc_path} | {doc_type} | ~{token_estimate} tokens | idioma {lang}"
            )
        if resume in ("pending", "segmented"):
            if resume == "segmented":
                # Re-ejecutar la etapa: limpiar chunks parciales primero.
                cleanup_stage(session, "kag_documents", doc_id, "segmented")
            from src.kag.segmentador import build_segmenter

            segmenter = build_segmenter(session, lang=lang, verbose=verbose)
            # Coref Stanza: docs cortos lo usan (costo acotado); book stacks lo
            # omiten (prohibitivo: ~11s por segmento).
            use_coref = doc_type == "short"
            chunks = chunk_markdown(
                md_text,
                doc_type,
                segmenter,
                max_tokens=CHUNK_MAX_TOKENS,
                use_coref=use_coref,
            )

            chunk_count = 0
            for i, chunk in enumerate(chunks):
                session.execute(
                    text(
                        "INSERT INTO kag_chunks "
                        "(doc_id, chunk_index, section_path, content, token_estimate, "
                        "embedding) VALUES (:doc_id, :chunk_index, :section_path, "
                        ":content, :token_estimate, NULL) "
                    ),
                    {
                        "doc_id": doc_id,
                        "chunk_index": i,
                        "section_path": chunk["section_path"],
                        "content": chunk["content"],
                        "token_estimate": chunk["token_estimate"],
                    },
                )
                chunk_count += 1
            set_stage(session, "kag_documents", doc_id, "segmented")
            if verbose:
                print(f"[KAG] ✂ {doc_path}: {chunk_count} chunks segmentados.")

        # ── Etapa: chunked (embeddings + entidades + relaciones) ────────────
        if resume in ("pending", "segmented", "chunked"):
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
            chunk_rows = session.execute(
                text(
                    "SELECT id, content FROM kag_chunks "
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
                for i, row in enumerate(batch):
                    session.execute(
                        text(
                            "UPDATE kag_chunks SET embedding = CAST(:emb AS vector) "
                            "WHERE id = :id"
                        ),
                        {"emb": embedding_to_sql(embs[i]), "id": row.id},
                    )
                    if llm_entities:
                        data = extract_entities_relations(session, row.content)
                    else:
                        # Extracción determinista con spaCy (gratis): reutiliza
                        # el nlp del segmentador (ya cargado) y canonicaliza con
                        # el modelo de embeddings de la DB (embed_texts, batch).
                        from src.kag.entities import extract_entities_deterministic

                        data = extract_entities_deterministic(
                            segmenter.nlp,
                            row.content,
                            embed_fn=embed_texts,
                            lang=lang,
                        )
                    _store_entities_relations(session, doc_id, row.id, data)
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
        if resume in ("pending", "segmented", "chunked", "figures"):
            if resume == "figures":
                cleanup_stage(session, "kag_documents", doc_id, "figures")
            figure_count = _index_figures(session, doc_id, md_path, md_text, verbose)
            set_stage(session, "kag_documents", doc_id, "figures")

        # ── Etapa: ready (resumen + cierre) ─────────────────────────────────
        if resume in ("pending", "segmented", "chunked", "figures", "ready"):
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
                summary = summarize_document(session, md_text, doc_type)

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
                f"[KAG] ✅ {doc_path} indexado: {chunk_count} chunks, "
                f"{entity_count} entidades, {relation_count} relaciones, "
                f"{figure_count} figuras."
            )
        return {
            "status": "indexed",
            "doc_path": doc_path,
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
                    "(doc_path, title, doc_type, status, content_hash, "
                    "token_estimate, error) "
                    "VALUES (:doc_path, :title, :doc_type, 'failed', "
                    ":content_hash, :token_estimate, :err) "
                    "ON CONFLICT (doc_path) DO UPDATE SET status = 'failed', "
                    "error = :err, updated_at = now()"
                ),
                {
                    "doc_path": doc_path,
                    "title": title,
                    "doc_type": doc_type,
                    "content_hash": content_hash,
                    "token_estimate": token_estimate,
                    "err": str(exc)[:500],
                },
            )
            session.commit()
        except Exception:  # noqa: BLE001 — no bloquea el re-lanzamiento
            pass
        if verbose:
            print(f"[KAG] ❌ Error indexando {doc_path}: {exc}")
            import traceback

            traceback.print_exc(file=sys.stdout)
        raise


def index_all(session, force=False, no_summary=False, verbose=True, llm_entities=False):
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
                )
            )
        except Exception as exc:  # noqa: BLE001 — un doc no bloquea el resto
            if verbose:
                print(f"[KAG] ❌ Error indexando {md.name}: {exc}")
    return results


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
        "--verbose", action="store_true", default=True, help="Prints descriptivos."
    )
    args = parser.parse_args()

    _fix_db_host()
    from src.db.session import SessionLocal

    session = SessionLocal()
    try:
        if args.doc:
            index_document(
                session,
                DOCS_DIR / args.doc,
                force=args.force,
                no_summary=args.no_summary,
                verbose=args.verbose,
                llm_entities=args.llm_entities,
            )
        else:
            index_all(
                session,
                force=args.force,
                no_summary=args.no_summary,
                verbose=args.verbose,
                llm_entities=args.llm_entities,
            )
    finally:
        session.close()


if __name__ == "__main__":
    main()
