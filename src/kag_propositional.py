"""
Ingesta proposicional del sistema KAG (rediseño profundo — Agente 3).

Pipeline de 6 pasos sobre los `.md` apilados de `data/knowledge_repository/docs/`:

  Paso 0 — MultibookFinderTool: detecta los límites físicos de cada libro/papel
           apilado y calcula el content_hash del slice (idempotencia).
  Paso 1 — Metadatos SLM (ISO 25964 + Library of Congress) -> `documents`.
  Paso 2 — Estructura de capítulos (LLM) -> `document_chapters`.
  Paso 3 — Chunking proposicional jerárquico (LLM, por capítulo) ->
           `propositional_chunks` con embedding vector(768) (batch).
  Paso 4 — Árbol temático secuencial (clustering aglomerativo con conectividad
           bandeada + c-TF-IDF + etiquetado LLM) -> `topic_tree_nodes`.
  Paso 5 — Imágenes (VLM + FAQ Reverse HyDE) -> `document_images`.
  Paso 6 — Cierre: documents.status = 'ready'.

Módulo LIGERO a propósito: scipy/sklearn/embeddings se importan SOLO dentro de
las funciones que los necesitan. Degradación natural en cada paso: si el LLM
falla, se salta con log (la ingesta es trabajo de fondo) y se insertan datos
crudos.

Uso:
    python -m src.kag_propositional                 # procesa todos los .md
    python -m src.kag_propositional --doc foo.md
    python -m src.kag_propositional --force         # re-indexa aunque el hash no cambió
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import bindparam, text

from src.llm.base import call_with_retries, load_settings, parse_llm_output

# Raíz del repositorio de conocimiento (relativa a la raíz del proyecto).
REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "data" / "knowledge_repository" / "docs"
# Manifest de documentos apilados (best-effort, no bloquea la ingesta).
MANIFEST_PATH = REPO_ROOT / "data" / "knowledge_repository" / "stacked_manifest.json"

# ---------------------------------------------------------------------
# Parámetros configurables
# ---------------------------------------------------------------------

# Umbral de distancia coseno para el clustering aglomerativo secuencial.
# Con linkage="average", 0.55 de distancia coseno ≈ 0.45 de similitud media:
# conservador — fusiona párrafos adyacentes del mismo tema sin sobre-fusionar
# secciones distintas (documentado en el módulo, ver _sequential_clusters).
TOPIC_DISTANCE_THRESHOLD = 0.55
# Límite de contexto del LLM por llamada (chars) para el chunking proposicional.
MAX_CONTEXT_CHARS = 12000
# Tamaño del chunk crudo de degradación (~800 tokens ≈ 4 chars/token).
RAW_CHUNK_MAX_CHARS = 800 * 4
# Lote de embeddings (mismo patrón que kag_ingest.index_document).
EMBED_BATCH_SIZE = 32
# Nº de keywords c-TF-IDF por cluster.
TOP_N_KEYWORDS = 8
# Radio de contexto textual alrededor de la imagen (líneas).
CONTEXT_RADIUS_LINES = 15

# ---------------------------------------------------------------------
# Prompts (constantes)
# ---------------------------------------------------------------------

METADATA_SYSTEM = """Eres un indexador bibliográfico especializado en Ciencias Sociales y normas de documentación formal (ISO 25964 y Library of Congress). Tu rol es extraer los metadatos estructurales del documento delimitado por las líneas indicadas. Debes aplicar rigurosamente:
1. Normalización de Tesauros (ISO 25964): Proporciona un mínimo de 3 temáticas principales. 'preferred_term': Descriptor normalizado unívoco. 'non_preferred_terms': Lista exhaustiva de variantes y sinónimos directos. 'scope_note_disambiguation': Nota de alcance que desambigüe el homónimo en el contexto de las ciencias sociales. 'broader_term' (TG), 'narrower_term' (TE) y 'related_terms' (TR).
2. Clasificación de la Biblioteca del Congreso (LCC y LCSH): Proporciona las materias LCSH correspondientes junto a su código LCC representativo (ej: HM, H, HN, JA).
3. Registro BibTeX formal del documento."""

METADATA_USER = """Archivo: {source_file}
ID de Documento: {document_id}
Rango de Líneas: {line_start} a {line_end}

Contenido inicial del documento:
---
{document_head_snippet}
---

Genera el JSON estricto con este schema:
{"source_file": str, "document_id": str, "title": str, "technical_level": "introductory|intermediate|advanced|research", "bibtex": str, "thematic_areas_iso25964": [{"preferred_term": str, "non_preferred_terms": [str], "scope_note_disambiguation": str, "broader_term": str, "narrower_term": str, "related_terms": [str]}], "library_of_congress": {"lcsh_terms": [{"term": str, "uri": str}], "lcc_classification": {"class_code": str, "class_title": str, "uri": str}}, "key_entities": [str]}"""

CHAPTERS_SYSTEM = """Eres un parser de estructura textual. Analiza el archivo Markdown provisto y extrae la segmentación completa de capítulos para el documento especificado. Es mandatorio que cada capítulo contenga el número exacto de línea de inicio y fin dentro del archivo general. Instrucciones: 1. La salida debe reflejar obligatoriamente la jerarquía: archivo -> documento -> capítulos. 2. Identifica los títulos de capítulo (#, ## o mayúsculas canónicas). 3. El 'line_end' del capítulo N debe ser la línea inmediatamente anterior al 'line_start' del capítulo N+1; para el último capítulo, debe corresponder al 'line_end' global del documento. 4. Si existen subsecciones relevantes, mapea sus líneas sin quebrar los rangos del capítulo contenedor."""

CHAPTERS_USER = """Archivo maestro: {source_file}
ID de documento: {document_id}
Límites del documento: {line_start} a {line_end}

Líneas del documento numeradas:
{numbered_text_block}

Devuelve el JSON: {"source_file": str, "document_id": str, "total_chapters": int, "chapters": [{"chapter_index": int, "title": str, "line_start": int, "line_end": int, "main_theme": str, "subsections": [{"title": str, "line_start": int, "line_end": int}]}]}"""

ANALYSIS_SYSTEM = """Eres un indexador bibliográfico especializado en Ciencias Sociales y un parser estructural de documentos académicos. Tu rol es producir, en UNA sola pasada, el análisis documental completo del documento delimitado por las líneas indicadas. Debes aplicar rigurosamente:
1. Ficha bibliográfica: 'title', 'technical_level' (introductory|intermediate|advanced|research), 'bibtex' (registro BibTeX formal).
2. Normalización de Tesauros (ISO 25964): Proporciona un mínimo de 3 temáticas principales. 'preferred_term': Descriptor normalizado unívoco. 'non_preferred_terms': Lista exhaustiva de variantes y sinónimos directos. 'scope_note_disambiguation': Nota de alcance que desambigüe el homónimo en el contexto de las ciencias sociales. 'broader_term' (TG), 'narrower_term' (TE) y 'related_terms' (TR).
3. Clasificación de la Biblioteca del Congreso (LCC y LCSH): Proporciona las materias LCSH correspondientes junto a su código LCC representativo (ej: HM, H, HN, JA).
4. Resumen Ejecutivo Global: 'summary' — síntesis de 2-4 oraciones de la tesis central y la progresión argumental de TODO el documento.
5. Estructura de capítulos: 'chapters[]' con 'chapter_index', 'title', 'line_start', 'line_end', 'main_theme' (tema del capítulo), 'summary' (micro-resumen de 1-2 oraciones del capítulo) y 'subsections[]'. El 'line_end' del capítulo N debe ser la línea inmediatamente anterior al 'line_start' del capítulo N+1; para el último capítulo, debe corresponder al 'line_end' global del documento. Si existen subsecciones relevantes, mapea sus líneas sin quebrar los rangos del capítulo contenedor.
6. Entidades Rectoras: 'key_entities' — entre 15 y 30 conceptos ontológicos nucleares del documento.

La salida debe ser un único objeto JSON atómico que combine ficha, temáticas, clasificación, resumen global, capítulos y entidades."""

ANALYSIS_USER = """Archivo: {source_file}
ID de Documento: {document_id}
Rango de Líneas: {line_start} a {line_end}

Esqueleto del documento (primeras líneas + encabezados H1/H2/H3 con su número de línea real):
---
{document_skeleton}
---

Genera el JSON estricto con este schema:
{"source_file": str, "document_id": str, "title": str, "technical_level": "introductory|intermediate|advanced|research", "bibtex": str, "thematic_areas_iso25964": [{"preferred_term": str, "non_preferred_terms": [str], "scope_note_disambiguation": str, "broader_term": str, "narrower_term": str, "related_terms": [str]}], "library_of_congress": {"lcsh_terms": [{"term": str, "uri": str}], "lcc_classification": {"class_code": str, "class_title": str, "uri": str}}, "key_entities": [str], "summary": str, "chapters": [{"chapter_index": int, "title": str, "line_start": int, "line_end": int, "main_theme": str, "summary": str, "subsections": [{"title": str, "line_start": int, "line_end": int}]}]}"""

PROPOSITIONAL_SYSTEM = """Eres un analista de epistemología y análisis del discurso. Tu objetivo es realizar una extracción proposicional jerárquica sobre el texto de un capítulo. Reglas obligatorias de segmentación:
1. Nivel 1 - Ideas Centrales (Central Claims): Formula las tesis teóricas de alto nivel defendidas en el texto.
2. Nivel 2 - Argumentos (Supporting Arguments): Identifica las premisas lógicas, pruebas empíricas o deducciones conceptuales que sustentan cada tesis.
3. Nivel 3 - Proposiciones Atómicas (Chunks): Descompón cada argumento en sus proposiciones elementales: cada proposición debe ser gramaticalmente independiente y autocontenida (reemplaza anáforas como "éste", "lo anterior", "dicho autor" por el sujeto explícito). 'verbatim_span' debe contener el fragmento de texto exacto del original. REGLA DE REFERENCIAS DUPLICADAS: Si una frase u oración contiene una referencia académica (ej. 'Bourdieu, 1984, p. 52' o 'García & Smith, 2020'), dicha referencia DEBE ser preservada y duplicada en el array 'citations_references' de TODAS las proposiciones atómicas que deriven de ella. Prohibido descartar o separar las citas bibliográficas."""

PROPOSITIONAL_USER = """Archivo: {source_file}
Documento: {document_id}
Capítulo: {chapter_index} - {chapter_title}
Rango: Línea {line_start} a Línea {line_end}

Texto a procesar:
---
{chapter_text_content}
---

Genera el JSON: {"source_file": str, "document_id": str, "chapter_index": int, "chapter_title": str, "core_ideas": [{"core_idea_id": str, "central_claim": str, "supporting_arguments": [{"argument_id": str, "argument_type": "empirical_evidence|theoretical_deduction|methodological_critique|comparative_analysis", "argument_statement": str, "propositional_chunks": [{"chunk_id": str, "proposition": str, "verbatim_span": str, "line_start": int, "line_end": int, "char_start": int, "char_end": int, "citations_references": [str]}]}]}]}"""

TOPIC_LABEL_SYSTEM = """Eres un analista bibliométrico y de modelado de tópicos. Se te presenta una secuencia temporal ordenada de proposiciones atómicas agrupadas mediante agrupamiento aglomerativo secuencial (c-TF-IDF). Tu función es sintetizar el significado de este grupo y asignarle una etiqueta temática precisa que preserve la progresión del texto original. Instrucciones: 1. 'macro_phase_label': Asigna un título representativo y conciso que describa la función de este bloque temático dentro de la obra (por ejemplo: "Fundamentación Ontológica", "Operacionalización Metodológica", "Crítica Empírica"). 2. 'epistemic_summary': Resume en dos oraciones la tesis o progresión conceptual central que abarca este rango específico de proposiciones. 3. Conserva estrictamente los identificadores de chunk de inicio y fin provistos."""

TOPIC_LABEL_USER = """Archivo: {source_file}
Documento: {document_id}
Orden secuencial: Fase {sequential_order}
Palabras representativas c-TF-IDF: {top_ctfidf_keywords}
Rango de chunks: {start_chunk_id} a {end_chunk_id}

Proposiciones constitutivas del cluster:
---
{cluster_statements_text}
---

Genera el JSON: {"source_file": str, "document_id": str, "sequential_order": int, "macro_phase_label": str, "representative_keywords": [str], "start_chunk_id": str, "end_chunk_id": str, "epistemic_summary": str}"""

VISION_SYSTEM = """Eres un asistente de investigación visual especializado en análisis de diagramas científicos, gráficos metodológicos y esquemas teóricos. Tu objetivo es analizar la imagen suministrada junto al contexto textual donde fue citada dentro del documento. Directrices de salida: 1. 'dense_visual_description': Transcribe todo texto, etiquetas de ejes, valores numéricos, leyendas y flujos de cajas o flechas visibles. No utilices generalizaciones como "un gráfico complejo"; desglosa sus componentes exactos. 2. 'epistemic_contribution': Explica con rigor qué fenómeno o mecanismo teórico/metodológico formaliza o comprueba esta imagen. 3. 'faq_indexing': Genera entre 3 y 5 preguntas explícitas que un investigador podría realizar cuya respuesta esté contenida visualmente en la imagen (Reverse HyDE). Evita preguntas genéricas; incluye las variables y relaciones exactas representadas. 4. Extrae las entidades clave directamente referenciadas en el gráfico."""

VISION_USER = """Documento: {document_id}
Línea de inserción: {anchor_line}
Etiqueta Markdown original: {markdown_tag}
Texto de contexto circundante (±15 líneas):
---
{surrounding_text_context}
---

Examina la imagen cargada y devuelve el JSON: {"image_id": str, "document_id": str, "file_path": str, "anchor_line": int, "caption": str, "image_type": "diagram|chart_or_plot|flowchart|conceptual_illustration|screenshot|table_image|photograph", "dense_visual_description": str, "epistemic_contribution": str, "faq_indexing": [str], "associated_entities": [str]}"""

# Task keys de prompt-as-code (specs en src/db/seed_kag_prompts.py).
# El runtime usa get_active_prompt(session, model_name, task_key) y cae a las
# constantes de arriba si no hay artefacto compilado (tests sin DB).
TASK_METADATA = "kag_metadata"
TASK_CHAPTERS = "kag_chapters"
TASK_DOCUMENT_ANALYSIS = "kag_document_analysis"
TASK_PROPOSITIONAL_CHUNKING = "kag_propositional_chunking"
TASK_TOPIC_LABEL = "kag_topic_label"
TASK_VISION_ANALYSIS = "kag_vision_analysis"

# Stopwords para c-TF-IDF (es/en — el corpus es multilingüe). Lista: sklearn
# CountVectorizer NO acepta sets en stop_words (solo str/list/callable).
_CTFIDF_STOPWORDS = [
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
    "como",
    "más",
    "mas",
    "the",
    "and",
    "of",
    "to",
    "in",
    "is",
    "are",
    "a",
    "an",
    "on",
    "for",
    "with",
    "this",
    "that",
    "from",
    "as",
    "at",
    "by",
    "or",
    "be",
    "it",
]

# ---------------------------------------------------------------------
# Helpers de host / texto
# ---------------------------------------------------------------------


def _fix_db_host() -> None:
    """Reemplaza '@db:' por '@localhost:' en DATABASE_URL/DATABASE_URL_ASYNC.

    El host 'db' es la red Docker y no resuelve desde el host; las credenciales
    son las mismas. Debe llamarse ANTES de importar src.db.session.
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


def _fill(template: str, **kwargs) -> str:
    """Rellena placeholders {name} sin tocar las llaves JSON literales del prompt.

    Los prompts contienen schemas JSON con llaves que .format() interpretaría
    como placeholders (KeyError); por eso se usa replace() por clave.
    """
    out = template
    for key, value in kwargs.items():
        out = out.replace("{" + key + "}", str(value))
    return out


def embedding_to_sql(emb):
    """Convierte una lista de floats a la sintaxis literal de pgvector.

    psycopg2 adapta las listas Python a numeric[], que pgvector no acepta;
    el literal '[0.1,0.2,...]' con CAST AS vector sí funciona.
    """
    if emb is None:
        return None
    return "[" + ",".join(repr(float(x)) for x in emb) + "]"


def _parse_embedding(emb):
    """Convierte el literal pgvector (str) o lista a lista de floats."""
    if emb is None:
        return None
    if isinstance(emb, (list, tuple)):
        return [float(x) for x in emb]
    if isinstance(emb, str):
        s = emb.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                return [float(x) for x in s[1:-1].split(",")]
            except ValueError:
                return None
    return None


def _llm_json(session, prompt: str, system: str, model_size: str = "small") -> dict:
    """Llama al LLM (SLM) y parsea JSON. Devuelve {} si falla (degradación).

    Mismo patrón que extract_entities_relations en kag_ingest.py: retries y
    fallback_model salen de session_settings.
    """
    settings = load_settings(session)
    retries = int(getattr(settings, "llm_retries", 3) or 3) if settings else 3
    fallback = getattr(settings, "fallback_model", None) if settings else None
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
        return parse_llm_output(text_out)
    except Exception as exc:  # noqa: BLE001 — LLM no disponible: degradación
        print(f"[KAG-P] ⚠ LLM no disponible: {exc}")
        return {}


def _embed_batch(session, statements: list[str]) -> list:
    """Batch de embeddings: una sola llamada a embed_texts por lote de 32.

    Mismo patrón que kag_ingest.index_document (EMBED_BATCH_SIZE). Si el
    modelo falla, devuelve [None]*n (chunks sin embedding, no rompe la ingesta).
    """
    if not statements:
        return []
    try:
        from src.embeddings import embed_texts

        out: list = []
        for start in range(0, len(statements), EMBED_BATCH_SIZE):
            batch = statements[start : start + EMBED_BATCH_SIZE]
            out.extend(embed_texts(batch, input_type="document"))
        return out
    except Exception as exc:  # noqa: BLE001 — chunk sin embedding
        print(f"[KAG-P] ⚠ Embedding batch falló: {exc}")
        return [None] * len(statements)


def _split_text_blocks(text: str, start_line: int, max_chars: int) -> list[tuple]:
    """Divide `text` en bloques de líneas que no excedan `max_chars`.

    Devuelve [(block_text, abs_line_start, abs_line_end)] con líneas absolutas
    (la primera línea de `text` es `start_line`).
    """
    lines = text.splitlines()
    if not lines:
        return []
    blocks: list[tuple] = []
    current: list[str] = []
    current_chars = 0
    block_start = start_line
    for i, line in enumerate(lines):
        line_chars = len(line) + 1  # +1 por el newline
        if current and current_chars + line_chars > max_chars:
            blocks.append(("\n".join(current), block_start, start_line + i - 1))
            current = [line]
            current_chars = line_chars
            block_start = start_line + i
        else:
            current.append(line)
            current_chars += line_chars
    if current:
        blocks.append(("\n".join(current), block_start, start_line + len(lines) - 1))
    return blocks


def _locate_span(
    text: str,
    span: str,
    line_start: int,
    line_end: int,
    text_start_line: int,
    base_offset: int = 0,
) -> tuple[int, int]:
    """Localiza `span` en `text` (cuyas líneas empiezan en `text_start_line`).

    Devuelve (char_start, char_end) = base_offset + offset dentro de `text`.
    Si el span no aparece (el LLM parafraseó), estima por los límites de línea.
    """
    if span:
        pos = text.find(span)
        if pos != -1:
            return base_offset + pos, base_offset + pos + len(span)
    lines = text.splitlines()
    rel_start = max(0, line_start - text_start_line)
    rel_end = min(len(lines) - 1, line_end - text_start_line)
    if rel_start < len(lines):
        cs = base_offset + sum(len(l) + 1 for l in lines[:rel_start])
        ce = base_offset + sum(len(l) + 1 for l in lines[: rel_end + 1])
        return cs, ce
    return base_offset, base_offset + len(text)


def _surrounding_context(
    doc_text: str,
    doc_start_line: int,
    anchor_line: int,
    radius: int = CONTEXT_RADIUS_LINES,
) -> str:
    """Devuelve las líneas ±radius alrededor de anchor_line (absolutas)."""
    lines = doc_text.splitlines()
    rel = anchor_line - doc_start_line
    start = max(0, rel - radius)
    end = min(len(lines), rel + radius + 1)
    return "\n".join(lines[start:end])


def _scope_thematic_from_thematic(thematic_areas) -> str:
    """Deriva scope_thematic (Branch B: scope_thematic ILIKE ANY(...)) de las
    temáticas ISO 25964: preferred_term + non_preferred_terms separados por '|'."""
    parts: list[str] = []
    for ta in thematic_areas or []:
        if not isinstance(ta, dict):
            continue
        pt = str(ta.get("preferred_term") or "").strip()
        if pt:
            parts.append(pt)
        for npt in ta.get("non_preferred_terms") or []:
            npt = str(npt).strip()
            if npt:
                parts.append(npt)
    return " | ".join(parts)


def _build_document_skeleton(
    doc_text: str, line_start: int, max_chars: int = 2000
) -> str:
    """Construye el esqueleto del documento para la llamada unificada.

    Devuelve las primeras líneas del documento (hasta `max_chars`) más los
    encabezados H1/H2/H3 con su número de línea real (absoluto). El LLM usa
    este esqueleto para segmentar capítulos y ficha sin recibir el texto
    completo (balance: ~34 llamadas/doc -> 1 llamada/doc).
    """
    lines = doc_text.splitlines()
    if not lines:
        return ""
    head: list[str] = []
    head_chars = 0
    for line in lines:
        line_chars = len(line) + 1
        if head and head_chars + line_chars > max_chars:
            break
        head.append(line)
        head_chars += line_chars
    headings = [
        f"{line_start + i}: {line}"
        for i, line in enumerate(lines)
        if line.startswith("#") and len(line) - len(line.lstrip("#")) <= 3
    ]
    parts = ["\n".join(head)]
    if headings:
        parts.append("\nENCABEZADOS:\n" + "\n".join(headings))
    return "\n".join(parts)


# ---------------------------------------------------------------------
# Funciones puras testables (clustering secuencial y c-TF-IDF)
# ---------------------------------------------------------------------


def _sequential_clusters(
    embeddings: list, distance_threshold: float = TOPIC_DISTANCE_THRESHOLD
) -> list[int]:
    """Clustering aglomerativo SECUENCIAL: solo chunks adyacentes se fusionan.

    Técnica elegida: AgglomerativeClustering con connectivity = matriz bandeada
    (diagonal + sub/super-diagonal = 1) y linkage="average". Es la opción
    estadísticamente más precisa frente al greedy merge por similitud par a par:

    - El greedy solo mira la similitud entre chunks contiguos; el average
      linkage considera la distancia media entre TODOS los miembros de dos
      clusters candidatos (más robusto a outliers y a cadenas espurias).
    - La matriz de conectividad bandeada impone la restricción secuencial
      (preserva el orden de lectura) sin necesidad de K-Means/HDBSCAN libres.
    - `distance_threshold` da un criterio de parada natural y configurable.

    Devuelve una lista de etiquetas (label[i] = cluster del chunk i). Si algún
    embedding es None (falló el modelo), degrada a un cluster por chunk.
    """
    n = len(embeddings)
    if n == 0:
        return []
    if n == 1:
        return [0]
    if any(_parse_embedding(e) is None for e in embeddings):
        return list(range(n))

    import numpy as np
    from scipy.sparse import diags
    from sklearn.cluster import AgglomerativeClustering

    X = np.asarray([_parse_embedding(e) for e in embeddings], dtype=float)
    # Normalizar filas a norma unitaria (coseno) con guarda contra ceros.
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = X / norms

    # Conectividad bandeada: chunk i solo puede fusionarse con i-1 o i+1.
    connectivity = diags([np.ones(n - 1), np.ones(n), np.ones(n - 1)], [-1, 0, 1])

    model = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=distance_threshold,
        connectivity=connectivity,
        compute_full_tree=True,
    )
    return model.fit_predict(X).tolist()


def _ctfidf_keywords(statements: list[str], top_n: int = TOP_N_KEYWORDS) -> list[str]:
    """Keywords representativas estilo c-TF-IDF para un cluster de statements.

    Cada statement es un 'documento'; TF-IDF sobre el cluster resalta los
    términos distintivos (los que aparecen en pocos statements pesan más,
    aproximando la frecuencia en el cluster vs. el corpus del documento).
    """
    if not statements:
        return []
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

    vectorizer = CountVectorizer(stop_words=_CTFIDF_STOPWORDS, min_df=1)
    X = vectorizer.fit_transform(statements)
    tfidf = TfidfTransformer().fit_transform(X)
    scores = np.asarray(tfidf.sum(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()
    order = np.argsort(scores)[::-1]
    out: list[str] = []
    for i in order:
        if scores[i] <= 0:
            break
        out.append(str(terms[i]))
        if len(out) >= top_n:
            break
    return out


# ---------------------------------------------------------------------
# Paso 1 — Metadatos SLM (ISO 25964 + LoC)
# ---------------------------------------------------------------------


def _enrich_library_of_congress(data: dict) -> dict:
    """Enriquece la ficha con LCSH/LCC de la API de la Biblioteca del Congreso.

    Best-effort: si la API falla, conserva lo que devolvió el LLM.
    """
    try:
        from src.kag.tools import LibraryOfCongressAPITool

        tool = LibraryOfCongressAPITool()
        loc = data.get("library_of_congress") or {}
        if not isinstance(loc, dict):
            loc = {}
        for term in (loc.get("lcsh_terms") or [])[:3]:
            if not isinstance(term, dict):
                continue
            label = str(term.get("term") or "").strip()
            if not label:
                continue
            result = tool.query_subject_heading(label)
            if result:
                term["uri"] = result.get("uri") or term.get("uri", "")
        lcc = loc.get("lcc_classification") or {}
        if isinstance(lcc, dict):
            class_code = str(lcc.get("class_code") or "").strip()
            if class_code:
                result = tool.query_classification_code(class_code)
                if result:
                    lcc["class_title"] = result.get("lcc_call_number") or lcc.get(
                        "class_title", ""
                    )
                    lcc["uri"] = result.get("lcc_uri") or lcc.get("uri", "")
            loc["lcc_classification"] = lcc
        data["library_of_congress"] = loc
    except Exception as exc:  # noqa: BLE001 — best-effort
        print(f"[KAG-P] ⚠ LoC API no disponible: {exc}")
    return data


def _index_document_analysis(
    session,
    md_path: Path,
    doc_info: dict,
    doc_text: str,
    content_hash: str,
    verbose: bool = False,
) -> tuple[int, list[dict]]:
    """Pasos 1+2 unificados: ficha + capítulos + resumen global + entidades.

    UNA llamada LLM por documento (ANALYSIS_SYSTEM/ANALYSIS_USER) que produce
    la ficha bibliográfica (ISO 25964 + LoC), el resumen ejecutivo global
    (documents.summary), la estructura de capítulos con micro-resúmenes y las
    entidades rectoras. Inserta la fila en `documents` (status='pending') y
    los capítulos en `document_chapters`.

    Degradación crítica: si la llamada unificada falla (data vacío o sin
    `chapters`), se insertan los defaults crudos EXACTOS de hoy — ficha con
    title/technical_level por defecto y capítulo único con todo el rango del
    documento. El INSERT en `documents` y `document_chapters` ocurre SIEMPRE.

    Devuelve (doc_row_id, chapter_rows).
    """
    source_file = str(md_path)
    document_id = doc_info["document_id"]
    line_start = int(doc_info["line_start"])
    line_end = int(doc_info["line_end"])

    settings = load_settings(session)
    small_model = getattr(settings, "small_model", None) if settings else None

    prompt = _fill(
        ANALYSIS_USER,
        source_file=source_file,
        document_id=document_id,
        line_start=line_start,
        line_end=line_end,
        document_skeleton=_build_document_skeleton(doc_text, line_start),
    )
    data = _llm_json(
        session,
        prompt,
        _get_system_prompt(
            session, small_model, TASK_DOCUMENT_ANALYSIS, ANALYSIS_SYSTEM
        ),
        model_size="small",
    )
    data = _enrich_library_of_congress(data)

    # --- Ficha (degradación: defaults crudos exactos de hoy) ---
    title = str(data.get("title") or doc_info.get("title") or md_path.stem)[:300]
    technical_level = str(data.get("technical_level") or "intermediate")
    if technical_level not in ("introductory", "intermediate", "advanced", "research"):
        technical_level = "intermediate"
    thematic = data.get("thematic_areas_iso25964") or []
    if not isinstance(thematic, list):
        thematic = []
    loc = data.get("library_of_congress") or {}
    if not isinstance(loc, dict):
        loc = {}
    key_entities = data.get("key_entities") or []
    if not isinstance(key_entities, list):
        key_entities = []
    summary = str(data.get("summary") or "")

    doc_row_id = session.execute(
        text(
            "INSERT INTO documents "
            "(source_file, document_id, title, technical_level, bibtex, "
            "thematic_areas_iso25964, library_of_congress, key_entities, "
            "scope_thematic, summary, line_start, line_end, status, content_hash) "
            "VALUES (:source_file, :document_id, :title, :technical_level, :bibtex, "
            "CAST(:thematic AS jsonb), CAST(:loc AS jsonb), CAST(:key_entities AS jsonb), "
            ":scope_thematic, :summary, :line_start, :line_end, 'pending', :content_hash) "
            "RETURNING id"
        ),
        {
            "source_file": source_file,
            "document_id": document_id,
            "title": title,
            "technical_level": technical_level,
            "bibtex": str(data.get("bibtex") or ""),
            "thematic": json.dumps(thematic, ensure_ascii=False),
            "loc": json.dumps(loc, ensure_ascii=False),
            "key_entities": json.dumps(key_entities, ensure_ascii=False),
            "scope_thematic": _scope_thematic_from_thematic(thematic),
            "summary": summary,
            "line_start": line_start,
            "line_end": line_end,
            "content_hash": content_hash,
        },
    ).scalar()

    # --- Capítulos (degradación: capítulo único con todo el rango) ---
    chapters = data.get("chapters") or []
    if not isinstance(chapters, list) or not chapters:
        chapters = [
            {
                "chapter_index": 1,
                "title": doc_info.get("title") or "Capítulo único",
                "line_start": line_start,
                "line_end": line_end,
                "main_theme": "",
                "summary": "",
                "subsections": [],
            }
        ]

    chapter_rows: list[dict] = []
    for ch in chapters:
        if not isinstance(ch, dict):
            continue
        ch_index = int(ch.get("chapter_index") or len(chapter_rows) + 1)
        ch_start = int(ch.get("line_start") or line_start)
        ch_end = int(ch.get("line_end") or line_end)
        # Sanitizar rangos dentro de los límites del documento.
        ch_start = max(line_start, min(ch_start, line_end))
        ch_end = max(line_start, min(ch_end, line_end))
        if ch_end < ch_start:
            ch_end = ch_start
        main_theme = str(ch.get("main_theme") or "")
        # Micro-resumen del capítulo; fallback a main_theme (como hoy).
        ch_summary = str(ch.get("summary") or main_theme)
        row_id = session.execute(
            text(
                "INSERT INTO document_chapters "
                "(document_id, chapter_index, title, line_start, line_end, "
                "main_theme, summary, subsections) "
                "VALUES (:document_id, :chapter_index, :title, :line_start, :line_end, "
                ":main_theme, :summary, CAST(:subsections AS jsonb)) RETURNING id"
            ),
            {
                "document_id": doc_row_id,
                "chapter_index": ch_index,
                "title": str(ch.get("title") or f"Capítulo {ch_index}")[:300],
                "line_start": ch_start,
                "line_end": ch_end,
                "main_theme": main_theme,
                "summary": ch_summary,
                "subsections": json.dumps(
                    ch.get("subsections") or [], ensure_ascii=False
                ),
            },
        ).scalar()
        chapter_rows.append(
            {
                "id": row_id,
                "chapter_index": ch_index,
                "title": str(ch.get("title") or ""),
                "line_start": ch_start,
                "line_end": ch_end,
            }
        )
    session.commit()
    if verbose:
        print(f"[KAG-P] 📄 {document_id} ficha + {len(chapter_rows)} capítulos.")
    return doc_row_id, chapter_rows


# ---------------------------------------------------------------------
# Paso 3 — Chunking proposicional jerárquico (LLM, por capítulo)
# ---------------------------------------------------------------------


def _index_propositional_chunks(
    session,
    md_path: Path,
    doc_info: dict,
    doc_text: str,
    doc_row_id: int,
    chapters: list[dict],
    verbose: bool = False,
) -> int:
    """Paso 3: proposiciones atómicas -> propositional_chunks (embedding batch).

    Por capítulo: si el texto excede MAX_CONTEXT_CHARS, se procesa por bloques
    de líneas y se concatenan los resultados. char_start/char_end se calculan
    localizando verbatim_span en el texto del bloque (fallback por líneas).
    Si el LLM falla en un capítulo, degrada a un chunk crudo por bloque de
    ~800 tokens. Los embeddings se calculan en batch (una llamada por lote).
    """
    source_file = str(md_path)
    document_id = doc_info["document_id"]
    doc_line_start = int(doc_info["line_start"])
    doc_lines = doc_text.splitlines()

    settings = load_settings(session)
    small_model = getattr(settings, "small_model", None) if settings else None

    pending: list[dict] = []  # (statement, params) para el batch de embeddings
    for ch in chapters:
        ch_id = ch["id"]
        ch_index = ch["chapter_index"]
        ch_title = ch["title"]
        ch_start = int(ch["line_start"])
        ch_end = int(ch["line_end"])

        rel_start = max(0, ch_start - doc_line_start)
        rel_end = min(len(doc_lines), ch_end - doc_line_start + 1)
        chapter_text = "\n".join(doc_lines[rel_start:rel_end])
        if not chapter_text.strip():
            continue

        context_blocks = _split_text_blocks(chapter_text, ch_start, MAX_CONTEXT_CHARS)
        for block_text, block_start, block_end in context_blocks:
            prompt = _fill(
                PROPOSITIONAL_USER,
                source_file=source_file,
                document_id=document_id,
                chapter_index=ch_index,
                chapter_title=ch_title,
                line_start=block_start,
                line_end=block_end,
                chapter_text_content=block_text,
            )
            data = _llm_json(
                session,
                prompt,
                _get_system_prompt(
                    session,
                    small_model,
                    TASK_PROPOSITIONAL_CHUNKING,
                    PROPOSITIONAL_SYSTEM,
                ),
                model_size="small",
            )
            core_ideas = data.get("core_ideas") if isinstance(data, dict) else None
            if not core_ideas:
                # Degradación: chunk crudo por bloque de ~800 tokens.
                for raw_text, raw_start, raw_end in _split_text_blocks(
                    block_text, block_start, RAW_CHUNK_MAX_CHARS
                ):
                    pending.append(
                        {
                            "chapter_id": ch_id,
                            "core_idea_id": "",
                            "argument_id": "",
                            "statement": raw_text,
                            "text_span": raw_text,
                            "char_start": 0,
                            "char_end": len(raw_text),
                            "line_start": raw_start,
                            "line_end": raw_end,
                            "citations": [],
                        }
                    )
                continue

            # Offset del bloque dentro del capítulo (chars) para char_start/end.
            block_offset = sum(
                len(l) + 1 for l in chapter_text.splitlines()[: block_start - ch_start]
            )
            for ci in core_ideas:
                if not isinstance(ci, dict):
                    continue
                core_idea_id = str(ci.get("core_idea_id") or "")[:50]
                for arg in ci.get("supporting_arguments") or []:
                    if not isinstance(arg, dict):
                        continue
                    argument_id = str(arg.get("argument_id") or "")[:50]
                    for chunk in arg.get("propositional_chunks") or []:
                        if not isinstance(chunk, dict):
                            continue
                        statement = str(chunk.get("proposition") or "").strip()
                        if not statement:
                            continue
                        verbatim = str(chunk.get("verbatim_span") or "")
                        c_line_start = int(chunk.get("line_start") or block_start)
                        c_line_end = int(chunk.get("line_end") or block_end)
                        c_line_start = max(block_start, min(c_line_start, block_end))
                        c_line_end = max(block_start, min(c_line_end, block_end))
                        char_start, char_end = _locate_span(
                            block_text,
                            verbatim,
                            c_line_start,
                            c_line_end,
                            block_start,
                            base_offset=block_offset,
                        )
                        citations = chunk.get("citations_references") or []
                        if not isinstance(citations, list):
                            citations = []
                        pending.append(
                            {
                                "chapter_id": ch_id,
                                "core_idea_id": core_idea_id,
                                "argument_id": argument_id,
                                "statement": statement,
                                "text_span": verbatim,
                                "char_start": char_start,
                                "char_end": char_end,
                                "line_start": c_line_start,
                                "line_end": c_line_end,
                                "citations": [str(c) for c in citations],
                            }
                        )

    # Batch de embeddings: una sola llamada por lote para todos los statements.
    statements = [p["statement"] for p in pending]
    embs = _embed_batch(session, statements)

    count = 0
    for params, emb in zip(pending, embs):
        session.execute(
            text(
                "INSERT INTO propositional_chunks "
                "(document_id, chapter_id, core_idea_id, argument_id, statement, "
                "text_span, char_start, char_end, line_start, line_end, "
                "citation_references, embedding) "
                "VALUES (:document_id, :chapter_id, :core_idea_id, :argument_id, "
                ":statement, :text_span, :char_start, :char_end, :line_start, "
                ":line_end, CAST(:citations AS jsonb), CAST(:emb AS vector))"
            ),
            {
                "document_id": doc_row_id,
                "chapter_id": params["chapter_id"],
                "core_idea_id": params["core_idea_id"],
                "argument_id": params["argument_id"],
                "statement": params["statement"],
                "text_span": params["text_span"],
                "char_start": params["char_start"],
                "char_end": params["char_end"],
                "line_start": params["line_start"],
                "line_end": params["line_end"],
                "citations": json.dumps(params["citations"], ensure_ascii=False),
                "emb": embedding_to_sql(emb),
            },
        )
        count += 1
    session.commit()
    if verbose:
        print(f"[KAG-P] 🧩 {document_id}: {count} chunks proposicionales.")
    return count


# ---------------------------------------------------------------------
# Paso 4 — Árbol temático secuencial (BERTopic restringido)
# ---------------------------------------------------------------------


def _index_topic_tree(
    session, md_path: Path, doc_info: dict, doc_row_id: int, verbose: bool = False
) -> int:
    """Paso 4: clusters secuenciales -> topic_tree_nodes (macro-fases)."""
    source_file = str(md_path)
    document_id = doc_info["document_id"]

    settings = load_settings(session)
    small_model = getattr(settings, "small_model", None) if settings else None

    rows = session.execute(
        text(
            "SELECT id, statement, embedding FROM propositional_chunks "
            "WHERE document_id = :doc_id ORDER BY chapter_id, line_start, id"
        ),
        {"doc_id": doc_row_id},
    ).fetchall()
    if not rows:
        return 0

    embeddings = [r.embedding for r in rows]
    labels = _sequential_clusters(embeddings, TOPIC_DISTANCE_THRESHOLD)

    # Agrupar índices por cluster preservando el orden secuencial.
    clusters: dict[int, list[int]] = {}
    for i, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(i)

    count = 0
    for order, (label, indices) in enumerate(sorted(clusters.items()), start=1):
        cluster_rows = [rows[i] for i in indices]
        statements = [str(r.statement) for r in cluster_rows]
        keywords = _ctfidf_keywords(statements, TOP_N_KEYWORDS)
        start_chunk_id = int(cluster_rows[0].id)
        end_chunk_id = int(cluster_rows[-1].id)

        prompt = _fill(
            TOPIC_LABEL_USER,
            source_file=source_file,
            document_id=document_id,
            sequential_order=order,
            top_ctfidf_keywords=", ".join(keywords),
            start_chunk_id=start_chunk_id,
            end_chunk_id=end_chunk_id,
            cluster_statements_text="\n".join(f"- {s}" for s in statements),
        )
        data = _llm_json(
            session,
            prompt,
            _get_system_prompt(
                session, small_model, TASK_TOPIC_LABEL, TOPIC_LABEL_SYSTEM
            ),
            model_size="small",
        )

        macro_phase_label = str(data.get("macro_phase_label") or f"Fase {order}")[:300]
        epistemic_summary = str(data.get("epistemic_summary") or "")
        llm_keywords = data.get("representative_keywords") or []
        if not isinstance(llm_keywords, list) or not llm_keywords:
            llm_keywords = keywords
        # Conservar estrictamente los ids provistos si el LLM los altera.
        try:
            start_id = int(data.get("start_chunk_id") or start_chunk_id)
        except (TypeError, ValueError):
            start_id = start_chunk_id
        try:
            end_id = int(data.get("end_chunk_id") or end_chunk_id)
        except (TypeError, ValueError):
            end_id = end_chunk_id

        session.execute(
            text(
                "INSERT INTO topic_tree_nodes "
                "(document_id, sequential_order, macro_phase_label, "
                "representative_keywords, start_chunk_id, end_chunk_id, "
                "epistemic_summary) "
                "VALUES (:document_id, :sequential_order, :macro_phase_label, "
                "CAST(:keywords AS jsonb), :start_chunk_id, :end_chunk_id, "
                ":epistemic_summary)"
            ),
            {
                "document_id": doc_row_id,
                "sequential_order": order,
                "macro_phase_label": macro_phase_label,
                "keywords": json.dumps(
                    [str(k) for k in llm_keywords], ensure_ascii=False
                ),
                "start_chunk_id": start_id,
                "end_chunk_id": end_id,
                "epistemic_summary": epistemic_summary,
            },
        )
        count += 1
    session.commit()
    if verbose:
        print(f"[KAG-P] 🌳 {document_id}: {count} nodos del árbol temático.")
    return count


# ---------------------------------------------------------------------
# Paso 5 — Imágenes (VLM + FAQ Indexing)
# ---------------------------------------------------------------------


def _describe_image(session, prompt: str, image_path: str, caption: str) -> dict:
    """Describe una imagen con el VLM vía complete_vision (data URL base64).

    Mismo patrón que describe_figure en kag_ingest.py. Devuelve {} si falla
    (la imagen se registra igual, con descripción vacía).
    """
    vision_model = None
    try:
        from src.llm.together import get_vision_model

        vision_model = get_vision_model(session)
    except Exception:  # noqa: BLE001 — sin DB: fallback a la constante
        pass
    try:
        from src.llm.together import complete_vision

        mime = "image/png" if str(image_path).lower().endswith(".png") else "image/jpeg"
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        data_url = f"data:{mime};base64,{b64}"
        text_out = complete_vision(
            session,
            prompt,
            image_url=data_url,
            system=_get_system_prompt(
                session, vision_model, TASK_VISION_ANALYSIS, VISION_SYSTEM
            ),
            response_format={"type": "json_object"},
        )
        return parse_llm_output(text_out)
    except Exception as exc:  # noqa: BLE001 — degradación no bloqueante
        print(f"[KAG-P] ⚠ VLM no disponible para {image_path}: {exc}")
        return {}


def _index_images(
    session,
    md_path: Path,
    doc_info: dict,
    doc_text: str,
    doc_row_id: int,
    verbose: bool = False,
) -> int:
    """Paso 5: imágenes -> document_images (VLM + FAQ Reverse HyDE)."""
    source_file = str(md_path)
    document_id = doc_info["document_id"]
    line_start = int(doc_info["line_start"])
    line_end = int(doc_info["line_end"])

    try:
        from src.kag.tools import MarkdownImageExtractorTool

        images = MarkdownImageExtractorTool().extract_images_from_document(
            source_file, document_id, line_start, line_end
        )
    except Exception as exc:  # noqa: BLE001 — degradación no bloqueante
        if verbose:
            print(f"[KAG-P] ⚠ Extractor de imágenes falló: {exc}")
        return 0

    count = 0
    for img in images:
        if not img.get("exists_on_disk"):
            continue
        anchor_line = int(img.get("anchor_line") or 0)
        caption = str(img.get("caption") or "")
        markdown_tag = str(img.get("markdown_tag") or "")
        file_path = str(img.get("file_path") or "")
        image_id = f"{document_id}_img_{count + 1:03d}"

        context = _surrounding_context(
            doc_text, line_start, anchor_line, CONTEXT_RADIUS_LINES
        )
        prompt = _fill(
            VISION_USER,
            document_id=document_id,
            anchor_line=anchor_line,
            markdown_tag=markdown_tag,
            surrounding_text_context=context,
        )
        data = _describe_image(session, prompt, file_path, caption)

        faq = data.get("faq_indexing") or []
        if not isinstance(faq, list):
            faq = []
        entities = data.get("associated_entities") or []
        if not isinstance(entities, list):
            entities = []

        session.execute(
            text(
                "INSERT INTO document_images "
                "(document_id, image_id, file_path, anchor_line, caption, image_type, "
                "dense_visual_description, epistemic_contribution, faq_indexing, "
                "associated_entities) "
                "VALUES (:document_id, :image_id, :file_path, :anchor_line, :caption, "
                ":image_type, :dense, :epistemic, CAST(:faq AS jsonb), "
                "CAST(:entities AS jsonb))"
            ),
            {
                "document_id": doc_row_id,
                "image_id": image_id,
                "file_path": file_path,
                "anchor_line": anchor_line,
                "caption": caption,
                "image_type": str(data.get("image_type") or ""),
                "dense": str(data.get("dense_visual_description") or ""),
                "epistemic": str(data.get("epistemic_contribution") or ""),
                "faq": json.dumps([str(q) for q in faq], ensure_ascii=False),
                "entities": json.dumps([str(e) for e in entities], ensure_ascii=False),
            },
        )
        count += 1
    session.commit()
    if verbose and count:
        print(f"[KAG-P] 🖼 {document_id}: {count} imágenes indexadas.")
    return count


# ---------------------------------------------------------------------
# Flujo de indexación (pasos 0 y 6)
# ---------------------------------------------------------------------


def _index_document(
    session, md_path: Path, doc_info: dict, force: bool = False, verbose: bool = False
) -> dict:
    """Procesa un documento detectado por MultibookFinderTool (pasos 1-6)."""
    document_id = doc_info["document_id"]
    line_start = int(doc_info["line_start"])
    line_end = int(doc_info["line_end"])

    lines = md_path.read_text(encoding="utf-8").splitlines()
    doc_text = "\n".join(lines[line_start - 1 : line_end])
    content_hash = hashlib.sha256(doc_text.encode("utf-8")).hexdigest()

    # Idempotencia: mismo hash + status='ready' -> salta (log).
    existing = session.execute(
        text("SELECT id, content_hash, status FROM documents WHERE document_id = :did"),
        {"did": document_id},
    ).first()
    if (
        existing
        and existing.content_hash == content_hash
        and existing.status == "ready"
        and not force
    ):
        if verbose:
            print(f"[KAG-P] ⏭ {document_id} ya indexado (hash idéntico).")
        return {"status": "skipped", "document_id": document_id}

    if existing:
        session.execute(
            text("DELETE FROM documents WHERE id = :id"), {"id": existing.id}
        )
        session.commit()

    doc_row_id, chapters = _index_document_analysis(
        session, md_path, doc_info, doc_text, content_hash, verbose
    )
    chunk_count = _index_propositional_chunks(
        session, md_path, doc_info, doc_text, doc_row_id, chapters, verbose
    )
    topic_count = _index_topic_tree(session, md_path, doc_info, doc_row_id, verbose)
    image_count = _index_images(
        session, md_path, doc_info, doc_text, doc_row_id, verbose
    )

    # Paso 6 — Cierre.
    session.execute(
        text(
            "UPDATE documents SET status = 'ready', updated_at = now() WHERE id = :id"
        ),
        {"id": doc_row_id},
    )
    session.commit()

    if verbose:
        print(
            f"[KAG-P] ✅ {document_id} listo: {len(chapters)} capítulos, "
            f"{chunk_count} chunks, {topic_count} fases, {image_count} imágenes."
        )
    return {
        "status": "indexed",
        "document_id": document_id,
        "chapters": len(chapters),
        "chunks": chunk_count,
        "topic_nodes": topic_count,
        "images": image_count,
    }


def _update_stacked_manifest(md_path: Path, docs: list[dict]) -> None:
    """Escribe/actualiza data/knowledge_repository/stacked_manifest.json.

    Estructura: dict keyed por source_file -> {generated_at, documents[]}.
    Re-ejecutar el mismo .md reemplaza su entrada sin duplicar; los demás
    source_file se preservan. Best-effort: nunca bloquea la ingesta.
    """
    try:
        manifest: dict = {}
        if MANIFEST_PATH.exists():
            try:
                existing = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
                if isinstance(existing, dict):
                    manifest = existing
            except (ValueError, OSError):
                manifest = {}
        entries = [
            {
                "document_id": str(d.get("document_id") or ""),
                "title": str(d.get("title") or ""),
                "line_start": int(d.get("line_start") or 0),
                "line_end": int(d.get("line_end") or 0),
                "total_lines": int(d.get("total_lines") or 0),
                "structural_keyword_hits": int(d.get("structural_keyword_hits") or 0),
            }
            for d in docs
        ]
        manifest[str(md_path)] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "documents": entries,
        }
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST_PATH.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as exc:  # noqa: BLE001 — best-effort, no bloquea la ingesta
        print(f"[KAG-P] ⚠ No se pudo actualizar {MANIFEST_PATH}: {exc}")


def index_stacked_file(
    session, md_path, force: bool = False, verbose: bool = False
) -> list:
    """Paso 0: detecta los documentos apilados y procesa cada uno (pasos 1-6)."""
    md_path = Path(md_path)
    if not md_path.exists():
        raise FileNotFoundError(f"No existe {md_path}")

    from src.kag.tools import MultibookFinderTool

    docs = MultibookFinderTool().execute(str(md_path))
    if not docs:
        if verbose:
            print(f"[KAG-P] 📂 No se detectaron documentos en {md_path.name}.")
        return []

    _update_stacked_manifest(md_path, docs)

    results = []
    for doc in docs:
        try:
            results.append(
                _index_document(session, md_path, doc, force=force, verbose=verbose)
            )
        except Exception as exc:  # noqa: BLE001 — un doc no bloquea el resto
            if verbose:
                print(f"[KAG-P] ❌ Error indexando {doc.get('document_id')}: {exc}")
    return results


def index_all(session, force: bool = False, verbose: bool = False) -> list:
    """Procesa todos los .md de data/knowledge_repository/docs/."""
    docs = sorted(DOCS_DIR.glob("*.md"))
    if not docs:
        if verbose:
            print("[KAG-P] 📂 No hay documentos en data/knowledge_repository/docs/.")
        return []
    results = []
    for md in docs:
        try:
            results.extend(
                index_stacked_file(session, md, force=force, verbose=verbose)
            )
        except Exception as exc:  # noqa: BLE001 — un doc no bloquea el resto
            if verbose:
                print(f"[KAG-P] ❌ Error indexando {md.name}: {exc}")
    return results


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def main() -> None:
    # Windows: la consola usa cp1252 y no imprime emojis — forzar UTF-8.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Ingesta proposicional KAG (documentos, capítulos, chunks, árbol temático, imágenes)."
    )
    parser.add_argument(
        "--doc", help="Nombre del .md a procesar (default: todos los de docs/)."
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-indexa aunque el hash no cambió."
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
            index_stacked_file(
                session, DOCS_DIR / args.doc, force=args.force, verbose=args.verbose
            )
        else:
            index_all(session, force=args.force, verbose=args.verbose)
    finally:
        session.close()


if __name__ == "__main__":
    main()
