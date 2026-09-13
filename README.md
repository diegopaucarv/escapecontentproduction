# Pipeline OS — ESCAPE / Ergalia

Backend del pipeline unificado de producción de contenidos. Este README
cubre cómo correrlo, qué está resuelto y probado, y qué se dejó
deliberadamente como esqueleto y por qué.

## Arranque

```bash
cp .env.example .env
# editar .env: POSTGRES_PASSWORD, JWT_SECRET (generar con
# `python -c "import secrets; print(secrets.token_hex(32))"`)
# Las claves de IA y los modelos NO van en .env: viven en la base de
# datos (tablas api_keys, session_settings, llm_models, prompt_templates).
docker compose up -d --build
docker compose run --rm -e TOGETHER_API_KEY=tgp_v1_... api python -m src.db.seed_llm
docker compose run --rm api python -m src.db.seed_ai
docker compose run --rm api python -m src.llm.compile_prompts
curl localhost:8000/health
```

El seed (`src/db/seed_llm.py`) inserta la clave de Together en `api_keys`
y crea la `session_settings` activa con los modelos pequeño/grande
(`meta-models/Muse-Glimmer-30B` y `deepseek-ai/DeepSeek-V4-Flash-0731`).
El seed (`src/db/seed_ai.py`) registra los modelos en `llm_models` y las
specs de prompts en `prompt_templates`. El compilador
(`src/llm/compile_prompts.py`) transpila las specs a artefactos inmutables
en `prompt_artifacts`. Después se pueden editar desde la API:
`GET/POST/PUT/DELETE /api-keys`, `/settings`, `/llm-models`,
`/prompt-templates` (la clave nunca se devuelve completa, solo enmascarada).

`docker compose up` levanta `db` → `migrate` (corre `alembic upgrade head`
una sola vez y termina) → `api` + `agent_worker` (esperan a que `migrate`
termine con éxito). El esquema ya NO se bootstrapea desde
`sql/001_init.sql` — ese archivo quedó como copia de referencia legible,
la fuente de verdad ejecutable es `alembic/versions/`.

### Hardware: CPU vs GPU

El wheel de `torch` depende del hardware. El Dockerfile lo instala aparte
con ARGs adaptables (default CPU, ~200MB):

```bash
# CPU (default) — sin GPU o no especificado
docker compose up -d --build

# GPU NVIDIA (CUDA 12.6) — setear en .env y rebuild
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
TORCH_PACKAGE=torch==2.12.0+cu126
docker compose up -d --build
```

Fuera de Docker, `python scripts/install_torch.py` detecta `nvidia-smi`
y elige el wheel correcto (`--gpu` / `--cpu` para forzar). El runtime se
adapta solo: `torch.cuda.is_available()` decide device y dtype
(`bfloat16` solo si la GPU lo soporta, si no `float32`).

### Caché de HuggingFace (`HF_HOME`)

El caché de modelos de HuggingFace (el de embeddings, ~480MB) vive en la
variable de entorno estándar `HF_HOME` del sistema — funciona igual en
Linux, Windows y macOS. Docker Compose la lee y monta esa carpeta dentro
del contenedor en `/app/hf_cache` (servicios `api` y `agent_worker`):

```bash
# Linux / macOS
export HF_HOME=/mnt/big_disk/hf_cache

# Windows (PowerShell, persistente a nivel de usuario)
[Environment]::SetEnvironmentVariable('HF_HOME', 'D:\Python\HF_Cache', 'User')
# reiniciar la terminal/editor para que los procesos nuevos la hereden
```

Si `HF_HOME` no está definida, Compose usa `./data/hf_cache` como fallback.
Así el modelo se descarga una sola vez y se reutiliza entre rebuilds, sin
llenar el disco del sistema.

### Cambios de esquema futuros

```bash
alembic revision -m "descripcion"   # crear una migración nueva
alembic upgrade head                 # aplicarla
```

No volver a editar `sql/001_init.sql` ni las migraciones ya aplicadas en
un entorno con datos reales.

### Crear el primer usuario (no hay endpoint de registro, a propósito)

```python
# una sola vez, a mano — evita exponer un /register público sin más control
from src.db.session import SessionLocal
from src.db.models import AppUser
from src.auth import hash_password

with SessionLocal() as s:
    s.add(AppUser(full_name="...", email="...", role="lider",
                   hashed_password=hash_password("...")))
    s.commit()
```

## Qué está resuelto y probado de verdad (no solo "debería funcionar")

Todo lo siguiente se corrió contra Postgres 17 + pgvector real en el
entorno de desarrollo, con casos que deben fallar incluidos, no solo
el camino feliz:

- **Esquema y migraciones** — `alembic upgrade head` → `downgrade base` →
  `upgrade head` de nuevo, sobre una base limpia. ENUMs, CHECK
  constraints, trigger de `updated_at`, columna generada
  `lead_time_minutes` y trigger `pg_notify` probados con INSERT/UPDATE
  reales, incluyendo entradas inválidas (bucket inexistente, brief sin
  segmento) que deben ser rechazadas y lo son.
- **ORM** (`src/db/models.py`) — round-trip real de enums, JSONB, y
  búsqueda por similitud coseno sobre `vector(1024)`.
- **API** (`src/api/main.py`) — probada con `TestClient` contra la base
  real: creación de brief (201 y 422 sin segmento), webhook de
  telemetría, login, y el endpoint de aprobación.
- **RBAC** (`src/auth.py`, `POST /auth/token` + `POST /briefs/{id}/approve`)
  — flujo completo probado: login líder (200), login equipo (200) pero
  **403** al intentar aprobar, líder aprobando (200), password incorrecta
  (401), sin token (401). El endpoint `POST /briefs/{id}/approve` es la
  implementación real de `OWNER_APPROVAL` del diagrama de estados —
  restringido a rol `lider` y solo válido desde estado `revision`
  (409 si el brief no está en revisión).
- **Cierre del bucle RAG** (`src/events/listener.py`) — un evento de
  telemetría real actualiza `retention_24h`/`conversion_30d` en
  `artifact_library`; probado que eventos sucesivos promedian
  correctamente (no sobreescriben). En el camino se encontró y corrigió
  un bug real: asyncpg entrega JSONB como string crudo si no se registra
  un codec — ya está registrado.
- **Embeddings** (`src/embeddings.py`) — inferencia LOCAL de
  `jinaai/jina-embeddings-v5-text-nano` (dim 768) vía
  `transformers.AutoModel` + `trust_remote_code` (la clase custom
  `jina_embeddings_v5` expone `.encode()`). El modelo se descarga desde
  HuggingFace Hub usando el token de HuggingFace del usuario (guardado
  en `api_keys`, provider `huggingface`) y se ejecuta en el contenedor:
  no hay API externa en el hot-path. El `input_type` de Voyage se mapea
  al `prompt_name` de Jina (`query` / `document`, ambos con
  `task='retrieval'`). La clave, el modelo y la dimensión se leen de la
  base (`embedding_settings` -> `api_keys` + `llm_models`); sin config
  falla ruidosa, no silenciosa. El modelo se carga una sola vez por
  proceso (singleton) y el caché de HuggingFace persiste en la carpeta
  de `HF_HOME` del sistema (montada en `/app/hf_cache`; ver "Caché de
  HuggingFace" arriba), no en el disco del sistema.
- **Infraestructura LLM** (`src/llm/together.py`, tablas `api_keys` +
  `session_settings` en `alembic/versions/0003_llm_infra.py`) — la clave
  de Together y los modelos pequeño/grande viven en la base de datos, no
  en `.env` ni en código. CRUD completo por API (`/api-keys`, `/settings`)
  con la clave siempre enmascarada (`tgp_v1_****R6yo`). El cliente
  `complete(session, prompt, model_size="small"|"large")` lee la config
  activa desde la DB y llama a `api.together.xyz` (API compatible con
  OpenAI). Probado con la DB real: seed → `GET /api-keys` → `GET /settings`.
- **Infraestructura de IA modular** (`alembic/versions/0004_ai_modular_infra.py`)
  — registro de modelos (`llm_models`), specs agnósticas de prompts
  (`prompt_templates`), artefactos compilados inmutables (`prompt_artifacts`)
  y settings de embeddings (`embedding_settings`). El compilador
  (`src/llm/compiler.py`) transpila cada spec a un prompt congelado por
  modelo usando el `syntax_profile` de cada modelo como datos (no if/else
  por proveedor) — maneja modelos comerciales y abiertos (Mistral,
  DeepSeek, Llama, Nemotron...). `prompt_artifacts` es INMUTABLE:
  recompilar = INSERT nueva versión + desactivar la anterior.
- **Infraestructura compartida de decisiones** (`src/llm/base.py`, 0007)
  — contrato de decisión único para todos los consumidores:
  `status` (`ok`/`degraded`/`skipped`), `decision_source`
  (`llm`/`deterministic`/`human`) y `requires_user_acceptance`. El LLM es
  la opción SIEMPRE presente: la mayoría de las decisiones se orquestan
  por LLM y se resuelven con SLMs. `load_settings`, `call_with_retries`
  (reintentos + `fallback_model`) y `parse_llm_output` viven acá, junto
  con `ok_result`/`degraded_result`/`skipped_result` — degradar a lo
  determinista SIEMPRE marca `requires_user_acceptance=True`.
- **Refuerzo LLM del semáforo** (`src/llm/reinforcement.py`) — el modelo
  pequeño detecta riesgos que el checklist automático no cubre. Fusión
  CONSERVADORA: solo puede escalar (auto_pass → needs_human_review),
  nunca bajar la severidad. Reintentos (`session_settings.llm_retries`,
  default 3) → fallback (`fallback_model`) → solo-reglas. La degradación
  a solo-reglas es determinista y SIEMPRE requiere aceptación del usuario
  (`requires_user_acceptance=True`), siguiendo el contrato de
  `src/llm/base.py`. JSON schema siempre forzado (`response_format`).
  Probado en vivo contra Together: el refuerzo detectó 2 riesgos no
  cubiertos por el checklist y no bajó el veredicto `fail` de las reglas.
- **Refinador de novedad** (`src/llm/novelty_refinement.py`) — el LLM
  se consulta en la zona gris del enrutamiento (búsqueda vectorial
  inconclusa) y SOLO para juzgar `ángulo no cubierto`; es el camino por
  defecto, no una excepción. Si el LLM no está disponible, se degrada al
  default determinista (ángulo nuevo) y el resultado lo superfície en
  `refinement_status` (`ok`/`degraded`/`skipped`) con
  `requires_user_acceptance=True` — la degradación no se traga
  silenciosamente. El coordinador `route_brief_with_refinement` vincula
  la sesión y pasa el artefacto previo encontrado como `prior_artifacts`
  al refinador.
- **Crítico LLM del checklist** (`src/llm/critic.py`) — consume el spec
  `critic_checklist` para evaluar el borrador contra el checklist de
  publicación (data-driven, variable por bucket). Regla defensiva EN
  CÓDIGO: ítem sin interpretación en `rules` → `no_evaluado`, nunca
  `ok`. Se inyecta en el nodo `critic` del grafo Producer-Critic vía
  `make_critic_checklist_fn(session)`. Degradación CONSERVADORA: si el
  LLM no está disponible, todo el checklist queda `no_evaluado` y el
  veredicto es `needs_human_review` — NUNCA se simula `auto_pass`.
- **Productor guiado por LLM** (`src/llm/producer.py`, 0007) — la
  generación del borrador es prompt-as-code: el productor usa el
  artefacto compilado `producer_draft` (spec en `prompt_templates`,
  compilada a `prompt_artifacts`) y se resuelve con el SLM. Recibe
  `insight_core` + Context Pack + `critic_feedback` de la iteración
  previa. Si el LLM no está disponible o devuelve salida inválida, se
  usa el borrador determinista y `requires_user_acceptance=True` — el
  pipeline se pausa para que el humano decida. `make_producer_fn`
  vincula la sesión para inyectarla en el grafo.
- **Enrutamiento por novedad y Gatekeeper** — lógica pura con tests
  unitarios (`tests/test_novelty_router.py`, `tests/test_gatekeeper.py`).
- **Bucle Producer-Critic** (`src/agents/producer_critic.py`,
  LangGraph 1.2.11) — el grafo compila y corre de verdad: se probó que
  se **pausa** en `interrupt()` para riesgo alto o segmento S5/S6, y se
  **reanuda** correctamente con `Command(resume=...)`. El nodo
  `producer` es guiado por LLM (prompt-as-code vía `producer_draft`,
  inyectado como `producer_fn`); el nodo `critic` llama al crítico LLM
  y al Gatekeeper real. Degradación CONSERVADORA: sin LLM, todo el
  checklist queda `no_evaluado` y el veredicto es `needs_human_review`
  (nunca `auto_pass` simulado). Cualquier degradación determinista
  (productor o crítico) pausa el grafo en `human_review_node` con
  `requires_user_acceptance=True` — el humano debe aceptar la decisión
  antes de continuar. El estado lleva `critic_feedback`, que alimenta al
  productor en la siguiente iteración.
- **Grafo cableado a sesión/endpoints reales** (`src/production/produce_brief.py`, 0007) — `POST /briefs/{id}/produce` dispara el bucle Producer-Critic
  con `make_producer_fn`/`make_critic_checklist_fn` vinculados a la
  sesión real (thread_id = brief_id, checkpointer InMemorySaver
  singleton); si el grafo se pausa (`needs_human_review` o degradación
  determinista), `POST /briefs/{id}/produce/resume` entrega la decisión
  humana al thread pausado; en `auto_pass` la pieza se materializa como
  `ContentArtifact`. Probado con sesión falsa: la degradación real (sin
  settings LLM) pausa con `requires_user_acceptance=True` y el resume
  reanuda con `human_decision` — ver `tests/test_produce_brief.py` y
  `tests/test_api_produce.py`.
- **Auditoría de degradación persistida** (migración 0009) —
  `content_briefs.requires_user_acceptance` persiste si la última
  decisión (alineamiento/refuerzo) fue determinista y quedó pendiente
  de aceptación; se expone en `GET /briefs/{id}` y se sobrescribe en el
  siguiente alineamiento (no se limpia al aprobar: es auditoría).
- **Capa de producción creativa** (migraciones 0005 + 0006, `src/tools/`,
  `src/db/seed_tools.py`, `src/db/seed_templates.py`) — catálogo de 10
  `tool_adapters` (Inkscape, Krita, REAPER, Resolve, ComfyUI, ElevenLabs,
  Veo 3, Canva, Drive, filesystem), 16 `format_specs` (8 tipos × 2 marcas)
  con `tool_chain` y `phases`, 17 `production_templates` estáticos por
  fase, `production_manifests` versionados e inmutables, y `asset_jobs`
  con trazabilidad completa (phase, template, manifest_version, status,
  output_path, error). El orquestador (`src/tools/orchestrator.py`)
  ejecuta la cadena mecánica declarada en `format_specs.phases` +
  `tool_chain` con los parámetros del manifiesto — el LLM escribe datos,
  el código ejecuta. El paso LLM de generación (`producer`, Producer-
  Critic) se ejecuta ANTES de la cadena mecánica y no es una herramienta
  MCP: el orquestador lo filtra explícitamente (`LLM_STEP`). Probado de
  punta a punta contra la DB real: un
  `video_corto` con manifiesto v1 produjo 8 AssetJobs (2 preproducción,
  3 producción, 3 postproducción) todos `done`, y el artefacto pasó a
  `listo` con `storage_path` poblado.
- **API de producción** (Fase 3) — 16 endpoints nuevos: CRUD
  `/tool-adapters`, CRUD `/production-templates`, `/artifacts/{id}/manifests`
  (versionado), `/artifacts/{id}/jobs`, `POST /artifacts/{id}/produce`
  (disparador síncrono) y `GET /tools/validate` (valida que toda cadena
  referencie adapters/templates existentes). El worker de eventos escucha
  `asset_jobs_channel` además de `telemetry_channel`.
- **Generador y refinador de proyectos** (`src/production/generator.py`,
  `src/production/refine.py`, 0007) — el snapshot del proyecto lo escribe
  el LLM con prompt-as-code (artefactos `project_generator` y
  `project_refine`), nunca prompts hardcodeados. El refinador mantiene las
  reglas de negocio en código (qa_checks/constraints) y el LLM solo
  corrige warnings. Ambos devuelven el contrato de decisión de
  `src/llm/base.py`; si el LLM falla, degradan a un resultado
  determinista con `requires_user_acceptance=True` y el endpoint NO
  guarda versión nueva — el pipeline se pausa para que el humano decida.

`pytest tests/ -v` → 251/251 pasan, sin necesitar Postgres (toda la lógica
de agentes está separada de la capa de datos a propósito).

## Infraestructura de IA modular (0004)

### Endpoints nuevos

| Método           | Ruta                       | Descripción                                     |
| ---------------- | -------------------------- | ----------------------------------------------- |
| `GET/POST`       | `/llm-models`              | Lista/crea modelos (registro `llm_models`)      |
| `GET/PUT/DELETE` | `/llm-models/{id}`         | Detalle/edita/borra un modelo                   |
| `GET/POST`       | `/prompt-templates`        | Lista/crea specs de prompts                     |
| `GET/PUT/DELETE` | `/prompt-templates/{id}`   | Detalle/edita/borra una spec                    |
| `GET`            | `/prompt-artifacts`        | Lista artefactos compilados (sin `prompt_text`) |
| `GET`            | `/prompt-artifacts/{id}`   | Detalle con `prompt_text`                       |
| `GET/POST`       | `/embedding-settings`      | Lee/crea la config de embeddings (singleton)    |
| `PUT/DELETE`     | `/embedding-settings/{id}` | Edita/borra la config de embeddings             |
| `POST`           | `/settings/ensure-prompts` | Compila specs → artefactos (idempotente)        |

### Flujo Prompt-as-Code

```
[Spec agnóstica] (prompt_templates)  --compilador determinista-->
[Artefacto inmutable] (prompt_artifacts)  --runtime-->
```

1. **Editar** la spec en `prompt_templates` (CRUD por API).
2. **Compilar** con `POST /settings/ensure-prompts` (o
   `python -m src.llm.compile_prompts`). Si el contenido cambió, se
   INSERTA una versión nueva y se desactiva la anterior — nunca UPDATE.
3. **Ejecutar** — el runtime (`src/llm/reinforcement.py`,
   `src/llm/novelty_refinement.py`, `src/llm/critic.py`,
   `src/llm/producer.py`, `src/production/generator.py`,
   `src/production/refine.py`) lee el artefacto activo y lo usa tal cual.
   Prompt caching de prefijo intacto: el bloque estático no cambia a
   mitad de sesión.

### Reglas de diseño (aprobadas — filosofía 0007)

- **El LLM es la opción SIEMPRE presente** — la mayoría de las decisiones
  se orquestan por LLM y se resuelven con SLMs (modelo pequeño). La
  filosofía anterior ("código determinista por defecto, LLM como
  excepción") quedó INVERTIDA.
- **El usuario es invitado a revisar las decisiones**, sean deterministas
  o LLM, vía el flujo `revision` + `POST /briefs/{id}/approve`
  (OWNER_APPROVAL).
- **La degradación a lo determinista SIEMPRE requiere aceptación del
  usuario** (`requires_user_acceptance=True`) — el pipeline se pausa
  hasta que el humano acepta la decisión determinista.
- **Los roles pueden ser humanos o guiados por LLM**, salvo los pasos
  cruciales (S5/S6, riesgo medio/alto, ruta nueva, producción completa)
  que siguen siendo humanos (OWNER_APPROVAL).
- **El feedback del crítico puede venir del LLM** y alimenta al productor
  en la siguiente iteración (`critic_feedback`).
- **Prompt-as-code en TODAS las llamadas, incluida la generación** — el
  productor usa el artefacto compilado `producer_draft`
  (`src/llm/producer.py`); el generador de proyectos usa
  `project_generator` y el refinador `project_refine`
  (`src/production/generator.py`, `src/production/refine.py`).
- **`alignment_reinforcement`** contiene SOLO la regla que el código no
  puede evaluar: "si detectas un riesgo NO cubierto por el checklist
  automático, escala". Las verificaciones mecánicas (fuente, CTA, S5/S6)
  las impone el código, no el prompt — no hay segunda fuente de verdad.
- **Fusión conservadora**: el LLM nunca baja la severidad. `fail` se
  queda `fail`; `auto_pass` + riesgo LLM → `needs_human_review`.
- **Reintentos y fallback**: `session_settings.llm_retries` (default 3)
  → `fallback_model` → degradación elegante. La degradación a
  solo-reglas es determinista y SIEMPRE requiere aceptación del usuario
  (`requires_user_acceptance=True`).
- **JSON schema siempre forzado** en la llamada al modelo pequeño.
- **`critic_checklist`**: data-driven (checklist como dato). Todo ítem de
  `pipeline_templates.checklist` debe tener su interpretación en `rules`
  (validado en el CRUD con 422 y en el compilador con warning). Ítem sin
  interpretación → `no_evaluado`, nunca `ok` — garantizado en código por
  `src/llm/critic.py::_normalize`, no solo por el prompt. Sin LLM, todo
  el checklist queda `no_evaluado` y el veredicto es `needs_human_review`
  — nunca `auto_pass` simulado.
- **`novelty_scoring`**: el refinamiento LLM de la zona gris es el camino
  por defecto (0.70 ≤ similitud < 0.86). Si el LLM no está disponible,
  se usa el default determinista (ángulo nuevo) y
  `requires_user_acceptance=True` — la degradación se superfície en
  `refinement_status`, no se traga.
- **`prompt_artifacts` es inmutable** — recompilar = INSERT + desactivar
  la anterior.

## Arquitectura de Proyectos — el sistema secuencialmente

El sistema tiene DOS caminos que se complementan:

```
CAMINO DE GOBERNANZA EDITORIAL (qué se publica y por qué)
  brief → alignment (semáforo) → novelty → producer-critic → QA → artifact aprobado

CAMINO TÉCNICO DE PRODUCCIÓN (cómo se materializa)
  project → generación (LLM escribe datos) → edición humana → assets →
  envío a Canva/Resolve → post-producción → entrega
```

### El flujo de un proyecto, paso a paso

1. **`POST /project/new {topic, template_id}`** — crea el proyecto
   (status `borrador`) con su primera versión de snapshot.
2. **`POST /projects/{id}/generate`** — el LLM (modelo pequeño, prompt-
   as-code vía el artefacto `project_generator`) escribe el snapshot JSON
   editable: `guion_vocal`, `prompts_img`, `prompts_video`, `sfx_tags`,
   `copy_por_slide`... informado por el template del formato y las reglas
   de negocio. (El RAG con conocimiento existente se conecta después,
   como hook pluggable.) Si el LLM no está disponible o devuelve salida
   inválida, se degrada a un esqueleto determinista y la respuesta trae
   `decision.requires_user_acceptance=True` — NO se guarda versión nueva
   hasta que el humano acepta (puede editar y aprobar, o reintentar).
3. **Edición humana** — el JSON se edita (UI futura o API) y se aprueba
   con **`POST /projects/{id}/approve {json_editado}`**, que crea una
   versión nueva (historial completo, estilo github).
4. **`POST /projects/{id}/produce`** — el orquestador ejecuta la cadena
   de herramientas del formato (AssetJobs): SVG vía Inkscape, imágenes vía
   ComfyUI, voz vía ElevenLabs, video vía Veo. Cada paso queda trazado en
   `asset_jobs`.
5. **`GET /projects/{id}/postproduction`** — el servicio de
   post-producción genera recomendaciones algorítmicas deterministas
   (cortes de edición, EQ, LUT de marca, aislamiento de voz, loudness)
   desde los templates + manifiesto. El ensamblaje final se envía a
   Canva/DaVinci Resolve con el template de marca (modo IA nativa del
   programa o recomendaciones como fallback).
6. **`POST /projects/{id}/rollback {version}`** — restaura un snapshot
   anterior como versión nueva. Nada se pierde.

### Endpoints de proyectos

| Método             | Ruta                                | Descripción                                                       |
| ------------------ | ----------------------------------- | ----------------------------------------------------------------- |
| `POST`             | `/project/new`                      | Crea proyecto + versión 1                                         |
| `GET/PATCH/DELETE` | `/projects/{id}`                    | Detalle/edita/borra                                               |
| `GET`              | `/projects`                         | Lista (filtros por status/artifact_type)                          |
| `GET`              | `/projects/{id}/versions`           | Historial de versiones (desc)                                     |
| `GET`              | `/projects/{id}/versions/{version}` | Detalle de una versión                                            |
| `POST`             | `/projects/{id}/approve`            | Aprueba JSON editado → versión nueva                              |
| `POST`             | `/projects/{id}/rollback`           | Restaura versión anterior                                         |
| `POST`             | `/projects/{id}/generate`           | LLM escribe el snapshot (prompt-as-code; degrada con aceptación)  |
| `POST`             | `/projects/{id}/refine`             | Edición de copy/prompts/guiones (reglas en código + LLM opcional) |
| `POST`             | `/projects/{id}/produce`            | Ejecuta la cadena de assets                                       |
| `GET`              | `/projects/{id}/postproduction`     | Recomendaciones de post-producción                                |

### Storage externo (HDD fuera de Docker)

`ASSET_STORAGE_PATH` (env var) configura dónde viven los assets pesados
(video crudo, proyectos de Resolve/Reaper, renders). Puede apuntar a un
HDD externo montado en el host; Docker lo monta como volumen. Los
`storage_path` de projects/artifacts/jobs se resuelven contra esta raíz.

### Modelos de visión (otra key, misma filosofía)

El generador de prompts sigue la filosofía Prompt-as-Code. Para visión:
una `api_keys` nueva (key_name `together_vision`, CRUD-editable, nunca en
`.env`), `llm_models` con `model_size='vision'` y specs de prompts de
visión. El compilador los incluye automáticamente.

## Qué sigue abierto, a propósito

El mapa completo del pipeline (rutas A/B/C, Fast/Complete, Kaizen,
Deploy, Learning) está en `docs/pipeline_unificado_produccion_contenidos(1).md`.
Este checklist es el **esquema a seguir**: cada fase del doc, con su estado.

### Fase 0 — Recepción, Brief y Enrutamiento ✅

- [x] `ContentBrief` unificado (bloques A–H, §3)
- [x] `ALIGNMENT` — semáforo 🟢🟡🔴 + refuerzo LLM conservador (§4.2)
- [x] `NOVELTY` — búsqueda vectorial + scoring + zona gris LLM (§4.4)
- [ ] `LAUNCH_DOC` — Documento de Lanzamiento como artefacto persistido
      (hoy el endpoint devuelve el análisis pero no guarda el documento)

### Ruta A — Repetitivo → Kaizen (§5) ❌

- [ ] `ORDER` — órdenes de producción desde el calendario editorial
- [ ] `ORDER_NOTE` — herencia de campos de la fila de calendario
- [ ] `EXECUTE_KAIZEN` — producción con plantilla existente
- [ ] `MEASURE_KPI` — LeadTime, CycleTime, Time-in-Stage, FPQ, Rework Rate
- [ ] `MICRO_KAIZEN` / `MICRO_RITUAL` — micro-experimentos y chequeo exprés
- [ ] `KAIZEN_DECISION` → `UPDATE_REGISTRY` / `ARCHIVE_KAIZEN`

### Ruta B — Solución previa → OBSOLETE_CHECK (§6) ❌

- [ ] `OBSOLETE_CHECK` — ¿evidencia/ángulo obsoleto?
- [ ] `RESEARCH_UPDATE` — verificación de evidencia (estándar Debate
      Informado / anonimización Ergalia)
- [ ] `RESEARCH_NOTE` / `UPDATE_BRIEF`

### Ruta C — Nueva solución → Discovery (§7) ❌

- [ ] Aprendizaje Express — IA synth con Context Packs (canon + guía +
      Artifact Library)
- [ ] `DISCOVERY_NODE` / `DISCOVERY_NOTE` — el "Chispazo" → `insight_core`
- [ ] `TEAM_INPUTS` / `LEADER_REFINE`
- [ ] Test de gancho 15s (n=5, comprensión ≥80%)
- [ ] `DECIDE_ROUTE` (Fast/Complete) / `ATLAS_NOTE`

### Sub-ruta Fast — Sprint 2D + Fast-Probe (§8) ❌

- [ ] `SPRINT2D` — prototipo + prueba con 5 lectores
- [ ] `MVP` / `SETUP_EXP` / `INSTRUMENTATION` (UTMs, tracking)
- [ ] `AUTO_DEPLOY` — publicación piloto programada
- [ ] `FEEDBACK` (24–72h) / `PROBE_EVAL` (árbol de decisión de marca)
- [ ] `PROBE_ITER` (máx 3) / `PROBE_KILL` → postmortem

### Sub-ruta Complete — Full Development (§9) 🔶 (Producer-Critic ✅)

- [ ] `SPEC` / `repurpose_plan` — variantes de formato/canal
- [ ] `PLAN_TECNICO` / `HANDOFF_NOTE` / `RITUAL_DEV`
- [ ] `UX_DESIGN` — lenguaje visual de marca + patrones reutilizables
      → **en diseño** (`docs/diseno_produccion_multiformato.md`)
- [ ] `PROD_DEV` — producción modular por formato (derivación de la pieza
      madre) → **en diseño** (mismo doc)
- [ ] `QA_VALID` — checklist + evidencia + anonimización → **en diseño**
      (mismo doc)
- [ ] Capa de herramientas (OpenClaw + MCP) — `tool_adapters`,
      `asset_jobs`, cadenas por formato → **plan en
      `docs/plan_infraestructura_produccion.md`**
- [ ] `INTEGRATION` / `COMPONENTS_NOTE` (components manifest)
- [x] Bucle Producer-Critic — productor guiado por LLM (prompt-as-code
      vía el artefacto `producer_draft`), crítico con degradación
      conservadora (sin LLM → `needs_human_review`, nunca `auto_pass`
      simulado); cualquier degradación determinista pausa el grafo para
      aceptación humana
- [x] Grafo cableado a sesión/endpoints reales — `POST /briefs/{id}/produce`
      dispara el bucle con `make_producer_fn`/`make_critic_checklist_fn`
      vinculados a la sesión (`src/production/produce_brief.py`);
      `POST /briefs/{id}/produce/resume` entrega la decisión humana al
      thread pausado; en `auto_pass` la pieza se materializa como
      `ContentArtifact`

### Pre-Deploy Gate (§11) 🔶

- [x] `OWNER_APPROVAL` — `POST /briefs/{id}/approve` (rol `lider`)
- [ ] `DATA_KPIS` / `MATERIALS` / `LOGISTICS`
- [ ] `READINESS` — checklist de publicación + visual + OpSec

### Kaizen (§5) ❌ — ver Ruta A

### Deploy (§12) ❌

- [ ] `DEPLOY_FINAL` / `ROLLOUT` (canal ancla → variantes → amplificación)
- [ ] `MONITOR` (24–72h / 7d / 30d / 90d)
- [ ] `RELEASE_NOTE` / `ROLLBACK_PLAN` (crisis) / `LIGHT_DELIVERY`

### Learning & Automation (§13) 🔶

- [x] `FEED_RAG` — telemetría → `artifact_library` (cierre del bucle RAG)
- [ ] `AUTO_POSTMORTEM` / `KB_POST` / `TEMPL_PREFILL`

### Deudas técnicas transversales

- [ ] Checklist automático real en el Critic — verificación mecánica por
      regex/NER como complemento del crítico LLM
- [ ] Checkpointer persistente (`langgraph-checkpoint-postgres`)
- [ ] Media móvil ponderada por recencia en `retention_24h`/`conversion_30d`
- [ ] Endpoint de registro de usuarios (hoy manual, a propósito)

### Deudas técnicas puntuales

- **Primera llamada de embeddings lenta** — la primera vez que se
  ejecuta `embed_text` en un contenedor nuevo, el modelo
  (`jinaai/jina-embeddings-v5-text-nano`, ~480MB) se descarga desde
  HuggingFace Hub y se carga en RAM (~10s en CPU). Después queda
  cacheado en la carpeta de `HF_HOME` (montada en `/app/hf_cache`) y en
  memoria (singleton). Si el contenedor no tiene red al arrancar, la
  primera llamada falla ruidosamente — por diseño.
- **Checklist automático real en el Critic** — hoy el crítico LLM evalúa
  el checklist de forma data-driven (spec `critic_checklist`); la
  verificación mecánica por regex/NER (longitud de CTA, fuente citada,
  anonimización) sigue pendiente como complemento. La spec ya valida que
  todo ítem tenga interpretación.
- **Checkpointer persistente** — el grafo usa `InMemorySaver`. Para que
  un `interrupt()` sobreviva un reinicio del contenedor, cambiar a
  `langgraph-checkpoint-postgres` (comentado en `requirements.txt`).
- **Media móvil ponderada por recencia** en `retention_24h`/
  `conversion_30d` — hoy es un promedio simple de dos valores; con
  volumen real de eventos, un promedio simple diluye señales recientes.
- **Endpoint de registro de usuarios** — deliberadamente no existe;
  altas de usuario son manuales (ver arriba) hasta decidir un flujo de
  invitación con control real.
