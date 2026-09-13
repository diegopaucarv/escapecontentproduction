# Pipeline OS — ESCAPE / Ergalia

Backend del pipeline unificado de producción de contenidos. Este README
cubre cómo correrlo, qué está resuelto y probado, y qué se dejó
deliberadamente como esqueleto y por qué.

## Arranque

```bash
cp .env.example .env
# editar .env: POSTGRES_PASSWORD, JWT_SECRET (generar con
# `python -c "import secrets; print(secrets.token_hex(32))"`),
# VOYAGE_API_KEY, ANTHROPIC_API_KEY
docker compose up -d --build
curl localhost:8000/health
```

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

Todo lo siguiente se corrió contra Postgres 16 + pgvector real en el
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
- **RBAC** (`src/auth.py`) — flujo completo probado: login líder (200),
  login equipo (200) pero **403** al intentar aprobar, líder aprobando
  (200), password incorrecta (401), sin token (401). El endpoint
  `POST /briefs/{id}/approve` es la implementación real de
  `OWNER_APPROVAL` del diagrama de estados — restringido a rol `lider`.
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
- **Enrutamiento por novedad y Gatekeeper** — lógica pura con tests
  unitarios (`tests/test_novelty_router.py`, `tests/test_gatekeeper.py`).
- **Esqueleto Producer-Critic** (`src/agents/producer_critic.py`,
  LangGraph 1.2.11) — el grafo compila y corre de verdad: se probó que
  se **pausa** en `interrupt()` para riesgo alto o segmento S5/S6, y se
  **reanuda** correctamente con `Command(resume=...)`. El nodo `critic`
  llama al Gatekeeper real; el nodo `producer` es un stub explícito.

`pytest tests/ -v` → 10/10 pasan, sin necesitar Postgres (toda la lógica
de agentes está separada de la capa de datos a propósito).

## Qué sigue abierto, a propósito

- **Nodo `producer` real** — conectar `langchain-anthropic` (o el SDK de
  Anthropic directo) para generar/ajustar el borrador a partir de
  `insight_core` + Context Pack. No se implementó porque no hay forma de
  probarlo en este entorno sin `ANTHROPIC_API_KEY`, y porque sigue sin
  haber corrido un ciclo real de contenido por el pipeline — ver la
  evaluación crítica original sobre secuenciación.
- **Checklist automático real en el Critic** — hoy `critic_node` marca
  todos los ítems del checklist como `True` (simulado). Antes de
  producción, cada ítem de `pipeline_templates.checklist` necesita una
  verificación real (longitud de CTA, fuente citada, anonimización vía
  regex/NER) en vez de un valor fijo.
- **Checkpointer persistente** — el grafo usa `InMemorySaver`. Para que
  un `interrupt()` sobreviva un reinicio del contenedor, cambiar a
  `langgraph-checkpoint-postgres` (comentado en `requirements.txt`).
- **Media móvil ponderada por recencia** en `retention_24h`/
  `conversion_30d` — hoy es un promedio simple de dos valores; con
  volumen real de eventos, un promedio simple diluye señales recientes.
- **Endpoint de registro de usuarios** — deliberadamente no existe;
  altas de usuario son manuales (ver arriba) hasta decidir un flujo de
  invitación con control real.
