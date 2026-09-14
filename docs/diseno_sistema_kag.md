# Diseño — Sistema KAG Pragmático (Vector + Grafo de Entidades + HippoRAG)

**Tipo de documento:** diseño de implementación — capa de conocimiento del pipeline
**Estado:** implementado y verificado (2026-09-13) — 41/41 tests, migraciones `0013_kag`, `0014_kag_fts` y `0015_kag_hnsw` aplicadas, seed aplicado. Incluye 4 optimizaciones: entity linking anclado con el LLM, desambiguación por copresencia en el grafo, noun chunks con spaCy y búsqueda híbrida densa + FTS + RRF. **Bugs corregidos en verificación con el book stack real (2.8MB):** sombreado de `text()` de SQLAlchemy por la variable local `text` (rompía la ingesta con `'str' object is not callable`), `KeyError` en `EXTRACT_PROMPT` por llaves JSON literales con `.format()`, y `session.rollback()` en el except que deshacía el INSERT (ahora re-inserta con `status='failed'`). **Validación end-to-end (2026-09-13):** fixture temporal `_test_backprop.md` indexado y consultado con éxito (4 chunks, 20 entidades, 16 relaciones; respuesta correcta con cita de fuente). El fixture y sus datos se eliminaron tras validar; la ingesta del book stack real (Handbook of Culture and Psychology, ~712k tokens) quedó corriendo en background.
**Función:** indexar el `knowledge_repository` (`.md` + imágenes) y responder consultas locales y globales (_multi-hop_) sobre ese conocimiento, con el stack existente (Postgres + pgvector + Together) y la filosofía 0007.

---

## 0. Resumen ejecutivo

Construimos un **KAG pragmático**: nada de Neo4j, nada de Leiden/Louvain, nada de GraphRAG completo. Usamos lo que ya existe (Postgres + pgvector + embeddings locales Jina + LLM vía Together) y añadimos **cinco tablas** (`kag_documents`, `kag_chunks`, `kag_entities`, `kag_relations`, `kag_figures`) + una tabla de configuración del segmentador (`kag_segmenter_settings`) + un **Personalized PageRank** estilo **HippoRAG** (Gutiérrez et al., NeurIPS 2024) para la recuperación multi-hop.

> **Por qué HippoRAG y no GraphRAG:** la activación asociativa en 1–2 saltos (PPR simple sobre el grafo de entidades) supera a GraphRAG en precisión multi-hop con **órdenes de magnitud menos costo computacional** — no hay detección de comunidades ni resúmenes sintéticos por cluster. Para consultas globales usamos **resúmenes por documento** generados con un **LLM local ultra pequeño (Qwen 2.5 quantizado)**, mucho más baratos que los community summaries.

**Segmentación:** usamos el segmentador propio del proyecto (`src/kag/segmentador.py`, `ProgressiveSegmenter`), adaptado para leer su configuración de la base de datos (`load_segmenter_config`) y elegir el modelo spaCy del idioma del documento (es/en/pt/de/fr). La detección de idioma (`detect_language`, heurística determinista por stopwords) vive en `src/kag_ingest.py` — módulo ligero, testable sin cargar torch/spacy — y `build_segmenter(session, lang, verbose)` la recibe como parámetro.

**Resúmenes como fuente secundaria:** en la consulta, el orden de recuperación es: (1) keywords/búsqueda vectorial, (2) chunks vía PPR (HippoRAG), (3) **resúmenes de documento** (generados con Qwen 2.5 local) como referencia de segundo plano.

**Filosofía 0007 aplicada:**

- El LLM es la opción SIEMPRE presente: extracción de entidades, resúmenes, entity linking y respuesta final son LLM.
- Lo mecánico (segmentación, embeddings, PPR, ensamblado) es determinista y no depende del LLM.
- Degradación elegante: si el LLM falla en un paso de ingesta, se salta con log (la ingesta es un trabajo de fondo, no una decisión que requiera aceptación humana); si falla en la respuesta, se devuelve el contexto crudo con un mensaje claro.
- Claves y modelos: se leen de la base (`session_settings` → `api_keys`, `embedding_settings`, `llm_models`, `kag_segmenter_settings`), nunca de `.env` ni hardcodeados.

---

## 1. Estructura de datos (migración `0013_kag`)

Cadena de migraciones: `0012_brand_knowledge` → `0013_kag` → `0014_kag_fts` → **`0015_kag_hnsw`** (head).

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

CREATE TRIGGER set_updated_at_kag_documents
BEFORE UPDATE ON kag_documents
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

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

CREATE TRIGGER set_updated_at_kag_segmenter_settings
BEFORE UPDATE ON kag_segmenter_settings
FOR EACH ROW EXECUTE FUNCTION set_updated_at();
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

1. **Eliminar la dependencia de `reii.config`** — los defaults pasan a constantes de módulo (`DEFAULT_NLI_MODEL`, `DEFAULT_SEGMENTER_EMBEDDING_MODEL`, `DEFAULT_SPACY_MODEL`, `DEFAULT_SPACY_MODELS`) y la configuración real se lee de `kag_segmenter_settings` en la DB.
2. **`load_segmenter_config(session)`** — lee la fila activa de `kag_segmenter_settings` (SQL crudo); si no hay fila (o la tabla no existe), devuelve los defaults de módulo. Devuelve dict con `nli_model`, `segmenter_embedding_model` y `spacy_models` (dict por idioma).
3. **`build_segmenter(session, lang='es', verbose=False)`** — factory: lee la config, elige el modelo spaCy del idioma (`spacy_models[lang]`, fallback `DEFAULT_SPACY_MODEL`) e instancia `ProgressiveSegmenter` con el NLI model y el embedding model de la DB. Cachea por idioma en un dict de módulo (`_SEGMENTER_CACHE` — no recargar spaCy/SentenceTransformer/NLI por cada documento).
4. **`detect_language(text)`** — heurística determinista por stopwords (es/en/pt/de/fr), default `'es'`. Vive en `src/kag_ingest.py` (módulo ligero, testable sin cargar torch/spacy), NO en el segmentador; `index_document` la llama y pasa el idioma a `build_segmenter`.
5. **`chunk_markdown(md_text, doc_type, segmenter, max_tokens=CHUNK_MAX_TOKENS, use_coref=True)`** (en `src/kag_ingest.py`):
   - Recibe el segmenter como parámetro (duck-typed — los tests pasan un fake con `segment_text`).
   - **Short (<20k tokens):** se parte por encabezados Markdown (`#`, `##`, `###`) preservando `section_path`; cada sección se pasa a `segmenter.segment_text(seccion, max_tokens=max_tokens)` y cada segmento resultante es un chunk.
   - **Long (≥20k tokens):** pre-segmentación por H1/H2 (para acotar cada llamada al segmentador — un book stack de 500k tokens no puede pasar entero por spaCy/coref), y cada sección se segmenta con el segmentador. `section_path` = ruta de encabezados.
   - **Fusión post-segmentación:** el segmentador produce cortes semánticos a veces muy pequeños (~50-100 tokens). `chunk_markdown` fusiona segmentos adyacentes de la MISMA sección hasta `max_tokens` (800), respetando los cortes del segmentador como fronteras duras. Reduce el número de chunks (y de llamadas LLM de extracción) sin perder los límites semánticos: en el book stack real, ~12.700 segmentos → ~950 chunks.
   - **`use_coref`:** el coref Stanza del segmentador cuesta ~11s por segmento (prohibitivo en book stacks). `index_document` pasa `use_coref=(doc_type == 'short')`: los docs cortos usan coref (costo acotado), los long lo omiten (monkeypatch temporal `resolve_coreferences` en la instancia — el segmentador NO se modifica; el método original se restaura al salir).
   - Cada chunk lleva `section_path` y `token_estimate`.

### 2.3 Flujo de `index_document(session, md_path)`

1. **Leer** el `.md` (utf-8) y calcular `sha256` → `content_hash`.
2. **Idempotencia:** si `kag_documents` ya tiene una fila con ese `doc_path` y el mismo hash → skip (ya indexado). Si el hash cambió → re-indexar (borrar hijos con CASCADE y re-insertar). Flag `--force` para re-indexar a la fuerza.
3. **Clasificar escala:** `token_estimate = len(text) // 4`. `doc_type = 'long'` si ≥ 20k tokens, si no `'short'`.
4. **Chunking** con el segmentador (§2.2).
5. **Embeddings:** `embed_text(chunk.content, input_type="document")` (local, Jina) → `kag_chunks.embedding`. El INSERT usa `embedding_to_sql(emb)` (helper de `src/kag_ingest.py`): convierte la lista de floats al literal pgvector `[0.1,...]` con `CAST(:embedding AS vector)` — psycopg2 adapta las listas Python a `numeric[]`, que pgvector no acepta. Si el embedding falla en un chunk → `emb = None` (chunk sin embedding, con log).
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
   - **Short:** una llamada a `complete_local` con el texto completo (truncado a ~16k tokens), `max_tokens=200`.
   - **Long:** map-reduce — una llamada por sección de nivel superior (H1/H2, concatenando sus chunks truncados a ~16k) con `max_tokens=60` + una llamada reduce que combina los resúmenes de sección en el resumen del documento (`max_tokens=200`).
   - Prompt según las reglas de Qwen 2.5 (§1.3): system con constraints, user con `<text>...</text>`, one-shot.
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
2. **Búsqueda híbrida (keywords):** `embed_text(query, input_type="query")` → `hybrid_search(session, query, q_emb, k)`: búsqueda densa (pgvector `<=>`, coseno, índice HNSW `ix_kag_chunks_embedding` de la migración `0015_kag_hnsw`) + búsqueda léxica FTS (`ts_rank_cd` sobre `content_tsv`, config `'simple'`) fusionadas con **RRF** (`rrf_merge`, k=60). K = `top_k` (8) local, `global_top_k` (20) global. Si `embed_text` falla (p. ej. sin `embedding_settings` activa) → `q_emb=None` y `hybrid_search` degrada a **solo FTS** (rollback + log). Si la migración `0014_kag_fts` no está aplicada → degrada a solo búsqueda densa (rollback + log). Si todo falla → `session.rollback()` (la transacción queda abortada tras el error) y `vec_hits = []` (degradación natural).
3. **Entity linking anclado + copresencia:**
   - `grounded_entity_linking`: pre-filtro determinista (`_noun_chunk_fallback`: noun chunks de spaCy + tokens PROPN; si spaCy no está disponible, tokens sueltos) obtiene un pool de entidades candidatas **reales** del grafo. El LLM (modelo pequeño) selecciona las entidades canónicas **solo entre ese pool** (`GROUNDED_ENTITIES_PROMPT`). Si el LLM falla → devuelve el pool determinista (degradación natural).
   - `match_entities_candidates`: por mención, candidatos del grafo — exacto por `name_norm` primero, luego `LIKE` (hasta 5).
   - `disambiguate_by_cooccurrence`: para cada mención ambigua, elige el candidato que comparte más vecinos (a 1 salto) con las entidades confirmadas (menciones con un solo candidato). Sin confirmadas → primer candidato de cada mención.
4. **PPR (HippoRAG)** (`personalized_pagerank`):
   - Reutiliza `adj` (ya construido en el paso 3 si hay grupos de candidatos).
   - Semilla = entidades desambiguadas. Power iteration: `v = (1-α)·seed + α·Mᵀ·v`, `α = 0.15`, máx 50 iteraciones, tol `1e-6`.
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

### 3.3 Optimizaciones de recuperación y entity linking

Cuatro optimizaciones implementadas sobre el flujo base de §3.1:

**1. Entity linking anclado con el LLM (`grounded_entity_linking`)**

El entity linking original usaba LLM libre + fallback determinista por substring (`LIKE`), lo que producía **falsos positivos** (nombres inventados o variantes que no existen en el grafo) que se propagaban como semilla al PPR. Ahora:

- Pre-filtro determinista (`_noun_chunk_fallback`) obtiene un pool de entidades candidatas **reales** del grafo.
- El LLM (modelo pequeño) **selecciona** entidades canónicas **solo entre ese pool** (prompt `GROUNDED_ENTITIES_PROMPT`).
- El match posterior es **exacto por `name_norm`** — se elimina el `LIKE` de la vía principal.
- Si el LLM falla → devuelve el pool determinista (degradación natural).

**2. Desambiguación por copresencia en el grafo (`match_entities_candidates` + `disambiguate_by_cooccurrence`)**

Una mención puede matchear varias entidades (p. ej. "red" → "Red Neuronal" y "Red de Petri"). `match_entities_candidates` devuelve una lista de listas (candidatos por mención, exacto primero y luego `LIKE` hasta 5). `disambiguate_by_cooccurrence` elige, para cada mención ambigua, el candidato que comparte más vecinos (a 1 salto) con las **entidades confirmadas** (menciones con un solo candidato). Sin entidades confirmadas → conserva el primer candidato de cada mención (no hay señal de copresencia).

**3. Noun chunks con spaCy (`_noun_chunk_fallback` + helpers)**

El fallback determinista por tokens sueltos partía entidades multi-palabra ("red neuronal" → "red" + "neuronal"). Ahora `_noun_chunk_fallback` extrae **frases nominales completas** (noun chunks) + tokens PROPN, con mejor granularidad para entidades multi-palabra. `_get_spacy_nlp(session, lang)` carga spaCy perezosamente con caché por idioma (el módulo sigue siendo ligero) y `_spacy_model_for(session, lang)` resuelve el modelo por idioma desde un dict local `_SPACY_MODELS` (es/en/pt/de/fr), leyendo `kag_segmenter_settings.spacy_models` si existe. Si spaCy no está disponible → degrada al fallback determinista por tokens (`_deterministic_entity_fallback`).

**4. Búsqueda híbrida: densa + FTS + RRF (`fts_search` + `rrf_merge` + `hybrid_search`)**

La búsqueda densa sola no captura **precisión léxica** (acrónimos, códigos, nombres propios, términos exactos). La migración `0014_kag_fts` añade la columna generada `content_tsv tsvector` (config `'simple'`, agnóstica de idioma, sin stemming — ideal para términos exactos) + índice GIN `ix_kag_chunks_content_tsv` sobre `kag_chunks`. En consulta:

- `fts_search(session, query_text, top_k)`: `ts_rank_cd` sobre `content_tsv`.
- `rrf_merge(dense_hits, sparse_hits, k=60, top_k)`: Reciprocal Rank Fusion en Python puro (testeable, sin dependencias).
- `hybrid_search(session, query_text, query_embedding, top_k, rrf_k=60, verbose=False)`: densa + FTS + RRF. Si la migración 0014 no está aplicada → degrada a solo búsqueda densa (rollback + log, solo si `verbose`).

`ask()` usa `hybrid_search` en el paso 2 en vez de `vector_search`.

---

## 4. Resiliencia

| Riesgo                                                    | Mitigación                                                                                                                                                                                                              |
| --------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| LLM extractor falla en un chunk                           | Skip del chunk con log; el resto del doc se indexa igual                                                                                                                                                                |
| VLM no disponible / imagen corrupta                       | `description = ''`, log; la figura se indexa igual                                                                                                                                                                      |
| Servidor local Qwen 2.5 no disponible                     | `summary = ''`, log; el doc se indexa igual                                                                                                                                                                             |
| Modelo spaCy del idioma no disponible                     | Fallback a `DEFAULT_SPACY_MODEL` (`es_core_news_md`) si el idioma no está en `spacy_models`; `spacy.load` falla si el modelo no está instalado                                                                          |
| Embeddings sin config                                     | La consulta degrada a solo FTS (`hybrid_search` con `query_embedding=None`); la ingesta guarda el chunk sin embedding (`NULL`) y el doc se indexa igual. `src/embeddings.py` sigue fallando ruidoso si se llama directo |
| LLM de respuesta falla                                    | Devuelve el contexto crudo + nota de degradación                                                                                                                                                                        |
| Re-indexar el mismo doc                                   | Idempotente por `content_hash`; `--force` para forzar                                                                                                                                                                   |
| Consulta sin entidades matcheadas                         | PPR con semilla vacía → solo búsqueda híbrida (degradación natural)                                                                                                                                                     |
| FTS no disponible (migración 0014 sin aplicar)            | `hybrid_search` detecta la ausencia de `content_tsv`, hace rollback y degrada a solo búsqueda densa (log)                                                                                                               |
| spaCy no disponible en el fallback                        | `_noun_chunk_fallback` degrada al fallback determinista por tokens (`_deterministic_entity_fallback`)                                                                                                                   |
| LLM anclado falla                                         | `grounded_entity_linking` devuelve el pool determinista de candidatos (degradación natural)                                                                                                                             |
| `images/[docname]/` no existe                             | Se omite la ingesta de figuras silenciosamente                                                                                                                                                                          |
| Documento en idioma no soportado                          | `detect_language` devuelve 'es' por defecto (fallback conservador)                                                                                                                                                      |
| Falla temprana en la ingesta (p. ej. descarga de modelos) | El INSERT de `kag_documents` va DENTRO del `try`; el except hace rollback y re-inserta la fila con `status='failed'` y el error visible (ON CONFLICT) — la fila nunca desaparece sin rastro                             |
| Coref Stanza en book stacks                               | `use_coref=False` para docs `long` (monkeypatch temporal en la instancia, sin tocar el segmentador): el coref cuesta ~11s por segmento y es prohibitivo en 500k tokens                                                  |

**Reintentos:** todas las llamadas LLM usan `call_with_retries` (tenacity + `fallback_model` de `session_settings`). La visión usa `complete_vision` que ya tiene su propio retry. `complete_local` tiene su propio retry simple (2 intentos) y lanza excepción si no hay modelo local activo o el servidor no responde — el caller degrada (p. ej. `summary=''`).

---

## 5. Archivos y funciones

| Archivo                             | Contenido                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `alembic/versions/0013_kag.py`      | Migración (tablas de §1.1 y §1.2 + triggers `set_updated_at` en `kag_documents` y `kag_segmenter_settings`)                                                                                                                                                                                                                                                                                                                                                               |
| `alembic/versions/0014_kag_fts.py`  | Migración: columna generada `content_tsv tsvector` (config `'simple'`) + índice GIN `ix_kag_chunks_content_tsv` sobre `kag_chunks` (búsqueda híbrida FTS)                                                                                                                                                                                                                                                                                                                 |
| `alembic/versions/0015_kag_hnsw.py` | Migración: índice HNSW `ix_kag_chunks_embedding` sobre `kag_chunks.embedding` (`vector_cosine_ops`) — acelera la búsqueda vectorial en consulta (sin él, scan secuencial)                                                                                                                                                                                                                                                                                                 |
| `src/db/seed_kag.py`                | Seed: `kag_segmenter_settings` + modelo local Qwen 2.5 en `llm_models` + api_key local (`provider='local'`, `key_name='local-qwen'`, `api_key=''`)                                                                                                                                                                                                                                                                                                                        |
| `src/kag/segmentador.py`            | **Adaptado:** sin `reii.config`; constantes de módulo `DEFAULT_NLI_MODEL`/`DEFAULT_SEGMENTER_EMBEDDING_MODEL`/`DEFAULT_SPACY_MODEL`/`DEFAULT_SPACY_MODELS`; NLI perezoso; `load_segmenter_config`, `build_segmenter(session, lang, verbose)` con caché por idioma; `ProgressiveSegmenter`/`ClassicSegmenter` aceptan `nli_model`                                                                                                                                          |
| `src/kag_ingest.py`                 | `_fix_db_host`, `estimate_tokens`, `normalize_entity_name`, `embedding_to_sql`, `detect_language`, `chunk_markdown`, `complete_local`, `extract_entities_relations`, `_store_entities_relations`, `summarize_document`, `describe_figure`, `_index_figures`, `index_document`, `index_all`, `main`                                                                                                                                                                        |
| `src/kag_query.py`                  | `_fix_db_host`, `classify_query`, `_deterministic_entity_fallback`, `_noun_chunk_fallback`, `_get_spacy_nlp`, `_spacy_model_for`, `grounded_entity_linking`, `match_entities_candidates`, `disambiguate_by_cooccurrence`, `build_adjacency`, `personalized_pagerank`, `vector_search`, `fts_search`, `rrf_merge`, `hybrid_search`, `chunks_for_entities`, `subgraph_triples`, `figures_for_chunks`, `doc_summaries`, `assemble_context`, `generate_answer`, `ask`, `main` |
| `tests/test_kag.py`                 | 41 tests de lógica pura (sin DB, sin torch/spacy): chunking con segmenter fake (incl. fusión hasta max_tokens y `use_coref`), PPR, clasificación, ensamblado, normalización, detección de idioma, `complete_local` (monkeypatch httpx), RRF, desambiguación por copresencia, entity linking anclado, `hybrid_search` (degradación y merge)                                                                                                                                |
| `scripts/kag_demo.py`               | CLI de demo: `--index 'pregunta'` (indexa todo + responde) o modo pregunta directa, con prints detallados                                                                                                                                                                                                                                                                                                                                                                 |
| `scripts/test_together.py`          | Test independiente: envía un prompt a Together AI (modelo pequeño/grande/ambos) y visualiza el resultado, con prints de la config leída de la DB                                                                                                                                                                                                                                                                                                                          |

**Estilo deliberado:** código simple y directo, sin dataclasses ni abstracciones innecesarias — fácil de cambiar luego. SQL crudo vía `session.execute(text(...))` (no se toca `src/db/models.py`). Prompts como constantes en los archivos (se pueden migrar a `prompt_templates` después si se quiere prompt-as-code estricto). Prints descriptivos en cada paso (`verbose=True`).

**Conexión a DB desde el host:** el `.env` apunta a host `db` (red Docker). Los scripts que corren en el host (CLIs, demo) deben reemplazar `@db:` por `@localhost:` en `DATABASE_URL`/`DATABASE_URL_ASYNC` **antes** de importar `src.db.session` (mismas credenciales, solo cambia el host). Helper `_fix_db_host()` en cada CLI (`src/kag_ingest.py`, `src/kag_query.py`, `scripts/kag_demo.py`): si la variable no está en el entorno, la lee del `.env` (pydantic-settings da prioridad a las env vars reales sobre el archivo `.env`).

---

## 6. Trabajo futuro (explícitamente fuera de alcance)

> Ya implementado (fuera de esta lista): entity linking anclado con el LLM (§3.3.1), desambiguación por copresencia (§3.3.2), noun chunks con spaCy (§3.3.3) y búsqueda híbrida densa + FTS + RRF (§3.3.4, migración `0014_kag_fts`).

- Migrar los prompts a `prompt_templates`/`prompt_artifacts` (prompt-as-code estricto).
- Embeddings por entidad (para entity linking puramente vectorial).
- Resúmenes por comunidad (GraphRAG completo) si el volumen lo justifica.
- Endpoints HTTP (`POST /kag/ask`) — hoy es CLI/script.
- Servir Qwen 2.5 vía Ollama/vLLM (hoy: endpoint OpenAI-compatible, p. ej. llama.cpp server).
