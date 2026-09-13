# Diseño — Sistema KAG Pragmático (Vector + Grafo de Entidades + HippoRAG)

**Tipo de documento:** diseño de implementación — capa de conocimiento del pipeline
**Estado:** spec inicial — se actualizará al final de la implementación para reflejar lo construido
**Función:** indexar el `knowledge_repository` (`.md` + imágenes) y responder consultas locales y globales (_multi-hop_) sobre ese conocimiento, con el stack existente (Postgres + pgvector + Together) y la filosofía 0007.

---

## 0. Resumen ejecutivo

Construimos un **KAG pragmático**: nada de Neo4j, nada de Leiden/Louvain, nada de GraphRAG completo. Usamos lo que ya existe (Postgres + pgvector + embeddings locales Jina + LLM vía Together) y añadimos **cinco tablas** (`kag_documents`, `kag_chunks`, `kag_entities`, `kag_relations`, `kag_figures`) + una tabla de configuración del segmentador (`kag_segmenter_settings`) + un **Personalized PageRank** estilo **HippoRAG** (Gutiérrez et al., NeurIPS 2024) para la recuperación multi-hop.

> **Por qué HippoRAG y no GraphRAG:** la activación asociativa en 1–2 saltos (PPR simple sobre el grafo de entidades) supera a GraphRAG en precisión multi-hop con **órdenes de magnitud menos costo computacional** — no hay detección de comunidades ni resúmenes sintéticos por cluster. Para consultas globales usamos **resúmenes por documento** generados con un **LLM local ultra pequeño (Qwen 2.5 quantizado)**, mucho más baratos que los community summaries.

**Segmentación:** usamos el segmentador propio del proyecto (`src/kag/segmentador.py`, `ProgressiveSegmenter`), adaptado para leer su configuración de la base de datos y detectar el idioma del documento (es/en/pt/de/fr) para elegir el modelo spaCy correcto.

**Resúmenes como fuente secundaria:** en la consulta, el orden de recuperación es: (1) keywords/búsqueda vectorial, (2) chunks vía PPR (HippoRAG), (3) **resúmenes de documento** (generados con Qwen 2.5 local) como referencia de segundo plano.

**Filosofía 0007 aplicada:**

- El LLM es la opción SIEMPRE presente: extracción de entidades, resúmenes, entity linking y respuesta final son LLM.
- Lo mecánico (segmentación, embeddings, PPR, ensamblado) es determinista y no depende del LLM.
- Degradación elegante: si el LLM falla en un paso de ingesta, se salta con log (la ingesta es un trabajo de fondo, no una decisión que requiera aceptación humana); si falla en la respuesta, se devuelve el contexto crudo con un mensaje claro.
- Claves y modelos: se leen de la base (`session_settings` → `api_keys`, `embedding_settings`, `llm_models`, `kag_segmenter_settings`), nunca de `.env` ni hardcodeados.

---

## 1. Estructura de datos (migración `0013_kag`)

Cadena de migraciones: `0012_brand_knowledge` → **`0013_kag`** (head).

### 1.1 Tablas KAG

```sql
CREATE TABLE kag_documents (
    id SERIAL PRIMARY KEY,
    doc_path VARCHAR(500) NOT NULL UNIQUE,      -- relativo a data/knowledge_repository/docs/
    title VARCHAR(300) NOT NULL,
    doc_type VARCHAR(10) NOT NULL DEFAULT 'short',  -- 'short' | 'long' (>= 20k tokens)
    status VARCHAR(20) NOT NULL DEFAULT 'pending',  -- 'pending' | 'ready' | 'failed'
    content_hash VARCHAR(64) NOT NULL,          -- sha256 del .md (idempotencia)
    token_estimate INT NOT NULL DEFAULT 0,
    chunk_count INT NOT NULL DEFAULT 0,
    entity_count INT NOT NULL DEFAULT 0,
    relation_count INT NOT NULL DEFAULT 0,
    figure_count INT NOT NULL DEFAULT 0,
    summary TEXT NOT NULL DEFAULT '',           -- resumen Qwen 2.5 local (fuente secundaria)
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE kag_chunks (
    id SERIAL PRIMARY KEY,
    doc_id INT NOT NULL REFERENCES kag_documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    section_path VARCHAR(500) NOT NULL DEFAULT '',  -- ej. '## Introducción > ### Métodos'
    content TEXT NOT NULL,
    token_estimate INT NOT NULL DEFAULT 0,
    embedding vector(768),                          -- jina-embeddings-v5-text-nano (dim 768)
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (doc_id, chunk_index)
);

CREATE TABLE kag_entities (
    id SERIAL PRIMARY KEY,
    doc_id INT NOT NULL REFERENCES kag_documents(id) ON DELETE CASCADE,
    chunk_id INT REFERENCES kag_chunks(id) ON DELETE CASCADE,
    name VARCHAR(300) NOT NULL,
    name_norm VARCHAR(300) NOT NULL,               -- minúsculas + strip (dedup)
    entity_type VARCHAR(100) NOT NULL DEFAULT 'concept',  -- concept|method|law|person|org|figure
    description TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ix_kag_entities_norm ON kag_entities (name_norm);
CREATE INDEX ix_kag_entities_doc ON kag_entities (doc_id);

CREATE TABLE kag_relations (
    id SERIAL PRIMARY KEY,
    doc_id INT NOT NULL REFERENCES kag_documents(id) ON DELETE CASCADE,
    chunk_id INT REFERENCES kag_chunks(id) ON DELETE CASCADE,
    source_entity_id INT NOT NULL REFERENCES kag_entities(id) ON DELETE CASCADE,
    target_entity_id INT NOT NULL REFERENCES kag_entities(id) ON DELETE CASCADE,
    relation_type VARCHAR(200) NOT NULL,            -- ej. 'UTILIZA', 'DEFINE', 'PARTE_DE'
    description TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ix_kag_relations_src ON kag_relations (source_entity_id);
CREATE INDEX ix_kag_relations_dst ON kag_relations (target_entity_id);

CREATE TABLE kag_figures (
    id SERIAL PRIMARY KEY,
    doc_id INT NOT NULL REFERENCES kag_documents(id) ON DELETE CASCADE,
    chunk_id INT REFERENCES kag_chunks(id) ON DELETE CASCADE,
    image_path VARCHAR(500) NOT NULL,               -- relativo a data/knowledge_repository/images/
    caption TEXT NOT NULL DEFAULT '',                -- alt text del .md si existe
    description TEXT NOT NULL DEFAULT '',            -- descripción VLM
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

**Nota sobre la dimensión:** el modelo real (`jina-embeddings-v5-text-nano`) produce vectores de **768** dims. La tabla `artifact_library` existente declara `vector(1024)` — inconsistencia pre-existente del repo, no la tocamos. Las tablas KAG usan `vector(768)` que es la dimensión real del modelo.

### 1.2 Tabla de configuración del segmentador

```sql
CREATE TABLE kag_segmenter_settings (
    id SERIAL PRIMARY KEY,
    nli_model VARCHAR(300) NOT NULL DEFAULT 'facebook/bart-large-mnli',
    segmenter_embedding_model VARCHAR(300) NOT NULL DEFAULT 'sentence-transformers/all-MiniLM-L6-v2',
    spacy_models JSONB NOT NULL DEFAULT '{"es": "es_core_news_md", "en": "en_core_web_md", "pt": "pt_core_news_md", "de": "de_core_news_md", "fr": "fr_core_news_md"}',
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

Singleton (a lo sumo una fila activa), igual que `embedding_settings`. El seed (`src/db/seed_kag.py`) la puebla con los defaults. El segmentador lee de aquí — nunca de constantes hardcodeadas.

### 1.3 Registro del modelo local Qwen 2.5 (tabla `llm_models`)

El seed inserta/actualiza una fila en `llm_models`:

```python
{
    "model_name": "qwen2.5-3b-instruct-q4_k_m",   # etiqueta; el archivo GGUF lo sirve llama.cpp
    "provider": "local",
    "model_size": "small",
    "context_window": 32768,
    "max_output_tokens": 60,
    "temperature_default": 0.1,
    "strengths": ["local", "gratis", "determinista", "resumen"],
    "weaknesses": ["capacidad limitada", "requiere servidor local"],
    "prompt_style": "ChatML estricto, system/user separados, XML delimiters, one-shot",
    "syntax_profile": {
        "api_style": "openai_chat",
        "base_url": "http://localhost:8080/v1",   # llama.cpp server (CRUD-editable)
        "system_role_name": "system",
        "instruction_formatting": {"style": "chatml", "root_tag": "instructions"},
        "structured_output": {"mode": "none"},
        "sampling": {
            "temperature": 0.1, "top_p": 0.9, "repetition_penalty": 1.05,
            "max_tokens": 60, "stop": ["\n", ""]
        },
        "tool_calling": {"mode": "none"},
        "prompt_caching": {"supports_prefix_caching": False},
        "reasoning_mode": {"supports_reasoning": False}
    },
}
```

Y una fila en `api_keys` (provider `local`, key_name `local-qwen`, api_key `""` — sin auth). El cliente local (`complete_local`) lee el modelo activo con `provider='local'` y llama a `{base_url}/chat/completions` (API compatible con OpenAI, como llama.cpp server).

**Prompt engineering de Qwen 2.5 (reglas del usuario, aplicadas en `summarize_document`):**

- ChatML nativo: `system` (reglas/constraints) y `user` (solo payload) separados — nunca texto crudo fusionado.
- Texto fuente delimitado con etiquetas XML (`<text>...</text>`).
- Constraints negativas explícitas en `system`: "no digas 'Este texto...'", "sin markdown", "máximo una oración", "máx 30 palabras".
- One-shot: exactamente un ejemplo de alta calidad.
- Parámetros: `temperature=0.1–0.2`, `top_p=0.85–0.9`, `repetition_penalty=1.05`, `max_tokens=50–60`, `stop=["\n", ""]`.

---

## 2. Fase 1 — Ingesta (`src/kag_ingest.py`)

### 2.1 Layout del repositorio

```
data/knowledge_repository/
├── docs/                          # los .md (artículos, book stacks)
│   ├── paper_nanotech.md
│   └── book_stack_deep_learning.md
└── images/
    └── paper_nanotech/            # carpeta por documento (nombre = .md sin extensión)
        ├── fig1_architecture.jpg
        └── chart_benchmark.png
```

Las imágenes se leen **solo si existen** en `images/[docname]/`. Si la carpeta no existe, la ingesta de figuras se omite silenciosamente.

### 2.2 Segmentación con el segmentador propio (`src/kag/segmentador.py`)

El segmentador (`ProgressiveSegmenter`) se **adapta** (no se reescribe):

1. **Eliminar la dependencia de `reii.config`** — los defaults pasan a constantes de módulo y la configuración real se lee de `kag_segmenter_settings` en la DB.
2. **`detect_language(text)`** — heurística determinista por stopwords (es/en/pt/de/fr). Devuelve el código de idioma.
3. **`build_segmenter(session, text, verbose)`** — factory: detecta idioma → lee `kag_segmenter_settings` → instancia `ProgressiveSegmenter` con el modelo spaCy del idioma detectado, el NLI model y el embedding model de la DB. Cachea por idioma en un dict de módulo (no recargar spaCy/SentenceTransformer/NLI por cada documento).
4. **`chunk_markdown(md_text, doc_type, segmenter, max_tokens)`**:
   - **Short (<20k tokens):** se parte por encabezados Markdown (`#`, `##`, `###`) preservando `section_path`; cada sección se pasa a `segmenter.segment_text(seccion, max_tokens=CHUNK_MAX_TOKENS)` y cada segmento resultante es un chunk.
   - **Long (≥20k tokens):** pre-segmentación por H1/H2 (para acotar cada llamada al segmentador — un book stack de 500k tokens no puede pasar entero por spaCy/coref), y cada sección se segmenta con el segmentador. `section_path` = ruta de encabezados.
   - Cada chunk lleva `section_path` y `token_estimate`.

### 2.3 Flujo de `index_document(session, md_path)`

1. **Leer** el `.md` (utf-8) y calcular `sha256` → `content_hash`.
2. **Idempotencia:** si `kag_documents` ya tiene una fila con ese `doc_path` y el mismo hash → skip (ya indexado). Si el hash cambió → re-indexar (borrar hijos con CASCADE y re-insertar). Flag `--force` para re-indexar a la fuerza.
3. **Clasificar escala:** `token_estimate = len(text) // 4`. `doc_type = 'long'` si ≥ 20k tokens, si no `'short'`.
4. **Chunking** con el segmentador (§2.2).
5. **Embeddings:** `embed_text(chunk.content, input_type="document")` (local, Jina) → `kag_chunks.embedding`.
6. **Extracción LLM** (`extract_entities_relations`): por chunk, el modelo pequeño (Together) extrae entidades y relaciones en JSON:
   ```json
   {
     "entities": [
       {
         "name": "Propagación Hacia Atrás",
         "type": "method",
         "description": "..."
       }
     ],
     "relations": [
       {
         "source": "Red Neuronal",
         "target": "Propagación Hacia Atrás",
         "type": "UTILIZA",
         "description": "..."
       }
     ]
   }
   ```
   - **Dedup por documento:** `name_norm` (minúsculas + strip). Si la entidad ya existe en el doc, se reutiliza su id.
   - Si el LLM falla en un chunk → se salta ese chunk con log (degradación de ingesta, no bloqueante).
7. **Figuras:** si existe `images/[docname]/`, para cada imagen (`*.jpg`, `*.jpeg`, `*.png`):
   - Buscar la referencia `![alt](...)` en el markdown para obtener el caption y el chunk que la referencia.
   - `describe_figure`: VLM vía `complete_vision` con la imagen como **data URL base64** (`data:image/jpeg;base64,...`). Si falla → se guarda con `description = ''` y log.
8. **Resumen jerárquico con Qwen 2.5 local** (`summarize_document`):
   - **Short:** una llamada a `complete_local` con el texto completo (truncado a ~16k tokens).
   - **Long:** map-reduce — una llamada por sección de nivel superior (H1/H2, concatenando sus chunks truncados a ~16k) + una llamada reduce que combina los resúmenes de sección en el resumen del documento.
   - Prompt según las reglas de Qwen 2.5 (§1.3): system con constraints, user con `<text>...</text>`, one-shot, `max_tokens=60` por sección y ~200 para el reduce.
   - Si el servidor local no está disponible → `summary = ''` con log (degradación, no bloqueante). Flag `--no-summary` para omitir.
9. **Actualizar** `kag_documents` (status `ready`, counts) y **print** del resumen de ingesta.

### 2.4 CLI de ingesta

```bash
python -m src.kag_ingest                 # indexa todo lo pendiente/cambiado
python -m src.kag_ingest --doc paper_nanotech.md
python -m src.kag_ingest --force        # re-indexa todo
python -m src.kag_ingest --no-summary
```

---

## 3. Fase 2 — Consulta (`src/kag_query.py`)

### 3.1 Flujo de `ask(session, query)`

1. **Clasificar** (`classify_query`): heurística determinista de keywords.
   - **Global:** "resumen", "conclusiones", "principales", "temas", "overview", "summary", "main topics", "libro", "book", "¿de qué trata?"...
   - **Local:** todo lo demás (preguntas específicas sobre datos, fórmulas, conceptos).
2. **Vector search (keywords):** `embed_text(query, input_type="query")` → top-K chunks por similitud coseno (pgvector `<=>`). K = `top_k` (8) local, `global_top_k` (20) global.
3. **Entity linking:**
   - `extract_query_entities`: LLM (modelo pequeño) extrae las entidades de la pregunta. Si falla → fallback determinista: tokenizar y buscar coincidencias por substring contra `name_norm`.
   - `match_entities`: match exacto por `name_norm`, luego `LIKE '%name%'`.
4. **PPR (HippoRAG)** (`personalized_pagerank`):
   - `build_adjacency`: grafo desde `kag_relations` (simétrico — cada relación aporta arista en ambos sentidos, ponderada por frecuencia).
   - Semilla = entidades matcheadas. Power iteration: `v = (1-α)·seed + α·Mᵀ·v`, `α = 0.15`, máx 50 iteraciones, tol `1e-6`.
   - Top-N entidades por score PPR → sus chunks (`chunks_for_entities`).
5. **Merge + dedup:** chunks vectoriales primero (por similitud), luego chunks de entidades PPR no incluidos (por score PPR).
6. **Subgrafo de tripletas** (`subgraph_triples`): relaciones donde source o target ∈ top entidades PPR, límite 25, con nombres de entidades.
7. **Figuras** (`figures_for_chunks`): figuras de los chunks finales.
8. **Resúmenes como fuente secundaria** (`doc_summaries`): resúmenes de los documentos de los chunks finales — **siempre** (local y global), marcados como referencia de segundo plano, después de keywords y PPR.
9. **Ensamblar contexto** (`assemble_context`):
   ```
   --- FRAGMENTOS RECUPERADOS (keywords + PPR) ---
   [doc: paper_nanotech.md | sección: ## Introducción | chunk 3/14]
   <contenido>

   --- SUBGRAFO DE ENTIDADES ---
   (Red Neuronal) -[UTILIZA]-> (Propagación Hacia Atrás) [doc: ...]

   --- FIGURAS ---
   [fig1_architecture.jpg] <descripción VLM>

   --- RESUMENES DE DOCUMENTO (referencia secundaria) ---
   [paper_nanotech.md] <resumen Qwen 2.5>
   ```
10. **Respuesta** (`generate_answer`): LLM **grande** con el prompt de respuesta (constante en el archivo), `response_format` JSON opcional. Si el LLM falla → devuelve el contexto crudo con nota de degradación.

### 3.2 CLI de consulta

```bash
python -m src.kag_query "¿Qué fórmula usa la propagación hacia atrás?"
python -m src.kag_query --top-k 12 "pregunta"
```

---

## 4. Resiliencia

| Riesgo                                | Mitigación                                                         |
| ------------------------------------- | ------------------------------------------------------------------ |
| LLM extractor falla en un chunk       | Skip del chunk con log; el resto del doc se indexa igual           |
| VLM no disponible / imagen corrupta   | `description = ''`, log; la figura se indexa igual                 |
| Servidor local Qwen 2.5 no disponible | `summary = ''`, log; el doc se indexa igual                        |
| Modelo spaCy del idioma no descargado | Intento de descarga; si falla, fallback al modelo por defecto      |
| Embeddings sin config                 | Falla ruidosa (mismo comportamiento que `src/embeddings.py`)       |
| LLM de respuesta falla                | Devuelve el contexto crudo + nota de degradación                   |
| Re-indexar el mismo doc               | Idempotente por `content_hash`; `--force` para forzar              |
| Consulta sin entidades matcheadas     | PPR con semilla vacía → solo vector search (degradación natural)   |
| `images/[docname]/` no existe         | Se omite la ingesta de figuras silenciosamente                     |
| Documento en idioma no soportado      | `detect_language` devuelve 'es' por defecto (fallback conservador) |

**Reintentos:** todas las llamadas LLM usan `call_with_retries` (tenacity + `fallback_model` de `session_settings`). La visión usa `complete_vision` que ya tiene su propio retry. `complete_local` tiene su propio retry simple (2 intentos) y degrada a `summary=''`.

---

## 5. Archivos y funciones

| Archivo                        | Contenido                                                                                                                                                                                                                                                         |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `alembic/versions/0013_kag.py` | Migración (tablas de §1.1 y §1.2)                                                                                                                                                                                                                                 |
| `src/db/seed_kag.py`           | Seed: `kag_segmenter_settings` + modelo local Qwen 2.5 en `llm_models` + api_key local                                                                                                                                                                            |
| `src/kag/segmentador.py`       | **Adaptado:** sin `reii.config`; `detect_language`, `build_segmenter`, `load_segmenter_config`; `ProgressiveSegmenter`/`ClassicSegmenter` aceptan `nli_model`                                                                                                     |
| `src/kag_ingest.py`            | `estimate_tokens`, `normalize_entity_name`, `chunk_markdown`, `complete_local`, `extract_entities_relations`, `summarize_document`, `describe_figure`, `index_document`, `index_all`, `main`                                                                      |
| `src/kag_query.py`             | `classify_query`, `extract_query_entities`, `match_entities`, `build_adjacency`, `personalized_pagerank`, `vector_search`, `chunks_for_entities`, `subgraph_triples`, `figures_for_chunks`, `doc_summaries`, `assemble_context`, `generate_answer`, `ask`, `main` |
| `tests/test_kag.py`            | Tests de lógica pura (sin DB): chunking, PPR, clasificación, ensamblado, normalización, detección de idioma                                                                                                                                                       |
| `scripts/kag_demo.py`          | CLI de demo: indexa (opcional) + responde una pregunta con prints detallados                                                                                                                                                                                      |

**Estilo deliberado:** código simple y directo, sin dataclasses ni abstracciones innecesarias — fácil de cambiar luego. SQL crudo vía `session.execute(text(...))` (no se toca `src/db/models.py`). Prompts como constantes en los archivos (se pueden migrar a `prompt_templates` después si se quiere prompt-as-code estricto). Prints descriptivos en cada paso (`verbose=True`).

**Conexión a DB desde el host:** el `.env` apunta a host `db` (red Docker). Los scripts que corren en el host (CLIs, demo) deben reemplazar `@db:` por `@localhost:` en `DATABASE_URL`/`DATABASE_URL_ASYNC` **antes** de importar `src.db.session` (mismas credenciales, solo cambia el host). Helper `_fix_db_host()` en cada CLI.

---

## 6. Trabajo futuro (explícitamente fuera de alcance)

- Migrar los prompts a `prompt_templates`/`prompt_artifacts` (prompt-as-code estricto).
- Embeddings por entidad (para entity linking puramente vectorial).
- Resúmenes por comunidad (GraphRAG completo) si el volumen lo justifica.
- Endpoints HTTP (`POST /kag/ask`) — hoy es CLI/script.
- Servir Qwen 2.5 vía Ollama/vLLM (hoy: endpoint OpenAI-compatible, p. ej. llama.cpp server).
