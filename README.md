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
- **Embeddings** (`src/embeddings.py`) — envuelve Voyage AI con la firma
  real del SDK instalado (`voyageai==0.5.0`), no una supuesta de memoria.
  No probado con una llamada de red real (sin API key en este entorno);
  la lógica y el manejo de errores (`VOYAGE_API_KEY` ausente → falla
  ruidosa, no silenciosa) sí están.
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
- **Refuerzo LLM del semáforo** (`src/llm/reinforcement.py`) — el modelo
  pequeño detecta riesgos que el checklist automático no cubre. Fusión
  CONSERVADORA: solo puede escalar (auto_pass → needs_human_review),
  nunca bajar la severidad. Reintentos (`session_settings.llm_retries`,
  default 3) → fallback (`fallback_model`) → solo-reglas (degradación
  elegante). JSON schema siempre forzado (`response_format`). Probado en
  vivo contra Together: el refuerzo detectó 2 riesgos no cubiertos por el
  checklist y no bajó el veredicto `fail` de las reglas.
- **Refinador de novedad** (`src/llm/novelty_refinement.py`) — el LLM
  SOLO se consulta en la zona gris del enrutamiento (búsqueda vectorial
  inconclusa) y SOLO para juzgar `ángulo no cubierto`. El scoring
  determinista (`score_novelty`) sigue siendo el default; si el LLM no
  está disponible, se degrada al comportamiento actual.
- **Enrutamiento por novedad y Gatekeeper** — lógica pura con tests
  unitarios (`tests/test_novelty_router.py`, `tests/test_gatekeeper.py`).
- **Esqueleto Producer-Critic** (`src/agents/producer_critic.py`,
  LangGraph 1.2.11) — el grafo compila y corre de verdad: se probó que
  se **pausa** en `interrupt()` para riesgo alto o segmento S5/S6, y se
  **reanuda** correctamente con `Command(resume=...)`. El nodo `critic`
  llama al Gatekeeper real; el nodo `producer` es un stub explícito.

`pytest tests/ -v` → 91/91 pasan, sin necesitar Postgres (toda la lógica
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
   `src/llm/novelty_refinement.py`) lee el artefacto activo y lo usa
   tal cual. Prompt caching de prefijo intacto: el bloque estático no
   cambia a mitad de sesión.

### Reglas de diseño (aprobadas)

- **`alignment_reinforcement`** contiene SOLO la regla que el código no
  puede evaluar: "si detectas un riesgo NO cubierto por el checklist
  automático, escala". Las verificaciones mecánicas (fuente, CTA, S5/S6)
  las impone el código, no el prompt — no hay segunda fuente de verdad.
- **Fusión conservadora**: el LLM nunca baja la severidad. `fail` se
  queda `fail`; `auto_pass` + riesgo LLM → `needs_human_review`.
- **Reintentos y fallback**: `session_settings.llm_retries` (default 3)
  → `fallback_model` → solo-reglas (degradación elegante).
- **JSON schema siempre forzado** en la llamada al modelo pequeño.
- **`critic_checklist`**: data-driven (checklist como dato). Todo ítem de
  `pipeline_templates.checklist` debe tener su interpretación en `rules`
  (validado en el CRUD con 422 y en el compilador con warning). Ítem sin
  interpretación → `no_evaluado`, nunca `ok`.
- **`novelty_scoring`**: el scoring determinista sigue siendo el default.
  El LLM solo se consulta en la zona gris y solo para juzgar `ángulo no
cubierto`.
- **`prompt_artifacts` es inmutable** — recompilar = INSERT + desactivar
  la anterior.

## Qué sigue abierto, a propósito

El mapa completo del pipeline (rutas A/B/C, Fast/Complete, Kaizen,
Deploy, Learning) está en `docs/pipeline_unificado_produccion_contenidos(1).md`.
Lo implementado cubre la **Fase 0** (brief + alineamiento + enrutamiento)
y el esqueleto del bucle Producer-Critic. Las fases siguientes siguen
sin implementar:

- **Ruta B — Solución previa detectada** — `OBSOLETE_CHECK` (¿evidencia/ángulo
  obsoleto?) y `RESEARCH_UPDATE` (verificación de evidencia) del §6 del doc.
- **Ruta C — Nueva solución** — Discovery editorial (el "Chispazo"), test de
  gancho 15s (n=5) y decisión Fast/Complete del §7.
- **Sub-ruta Fast** — Sprint de Diseño y Prueba (2d) + Fast-Probe (7d):
  MVP, setup de piloto, instrumentación, publicación piloto, evaluación
  de KPIs, iteración (máx. 3) y kill con postmortem (§8).
- **Sub-ruta Complete** — SPEC/repurpose_plan, plan técnico, handoff,
  ritual de riesgos, integración y components manifest (§9.1). El bucle
  Producer-Critic (§9.2) es el esqueleto probado; el resto de la secuencia no.
- **Pre-Deploy Gate** — readiness check, materiales y logística (§11).
  Solo existe `OWNER_APPROVAL` (`POST /briefs/{id}/approve`, rol `lider`).
- **Kaizen** — órdenes de producción desde calendario, KPIs de proceso
  (LeadTime, CycleTime, FPQ, Rework Rate), micro-experimentos y micro-rituales (§5).
- **Deploy** — rollout por secuencia de canales, monitoreo 24–72h/7d/30d/90d,
  protocolo de crisis/retiro (§12).
- **Learning & Automation** — postmortem automatizado y prefill de templates
  (§13). El cierre del bucle RAG (telemetría → `artifact_library`) ya existe.

### Deudas técnicas puntuales

- **Nodo `producer` real** — conectar el SDK de Together (o Anthropic) para
  generar/ajustar el borrador a partir de `insight_core` + Context Pack.
  La spec `producer_draft` ya está compilada en `prompt_artifacts`; falta
  el nodo que la consuma. No se implementó porque no hay forma de probarlo
  sin un ciclo real de contenido por el pipeline — ver la evaluación
  crítica original sobre secuenciación.
- **Clave de Voyage** — la infraestructura de embeddings
  (`embedding_settings` + `llm_models` con `voyage-3-large`) existe y
  falla ruidosamente si no hay config; falta que el usuario provea la
  clave y cree la `embedding_settings` activa vía `POST /embedding-settings`.
- **Checklist automático real en el Critic** — hoy `critic_node` marca
  todos los ítems del checklist como `True` (simulado). Antes de
  producción, cada ítem de `pipeline_templates.checklist` necesita una
  verificación real (longitud de CTA, fuente citada, anonimización vía
  regex/NER) en vez de un valor fijo. La spec `critic_checklist` ya
  valida que todo ítem tenga interpretación.
- **Checkpointer persistente** — el grafo usa `InMemorySaver`. Para que
  un `interrupt()` sobreviva un reinicio del contenedor, cambiar a
  `langgraph-checkpoint-postgres` (comentado en `requirements.txt`).
- **Media móvil ponderada por recencia** en `retention_24h`/
  `conversion_30d` — hoy es un promedio simple de dos valores; con
  volumen real de eventos, un promedio simple diluye señales recientes.
- **Endpoint de registro de usuarios** — deliberadamente no existe;
  altas de usuario son manuales (ver arriba) hasta decidir un flujo de
  invitación con control real.
