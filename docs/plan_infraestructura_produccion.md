# Plan de Infraestructura — Producción Creativa Local (OpenClaw + MCP)

**Estado:** plan de implementación — derivado de
`docs/diseno_produccion_multiformato.md` (arquitectura) y del estado actual
del backend (Fase 0 + Producer-Critic + infraestructura IA modular).

**Objetivo:** conectar `PROD_DEV`/`QA_VALID` con el software creativo real
(Inkscape, Krita, REAPER, DaVinci Resolve, ComfyUI) y las excepciones cloud
(ElevenLabs, Veo 3, Canva, Google Drive), orquestado localmente vía
OpenClaw + MCP, con trazabilidad completa en Postgres.

---

## 0. Resumen ejecutivo

El backend ya tiene todo lo que decide **QUÉ** producir (Producer-Critic,
`format_specs`, `prompt_artifacts`). Lo que falta es la capa que decide
**CÓMO** materializarlo: la cadena de herramientas por formato, su
ejecución, y la trazabilidad de cada paso (`asset_jobs`).

```
[Backend actual]  →  [Nueva capa de orquestación]  →  [Software creativo]
Producer-Critic      OpenClaw Gateway + tool-agents    Inkscape, Krita, ...
format_specs         MCP servers (stdio)               REAPER, Resolve, ...
prompt_artifacts     asset_jobs (trazabilidad)         ComfyUI, ElevenLabs, ...
```

**Principio rector (del diseño):** los tool-agents NO deciden qué
producir — ejecutan la cadena mecánica de herramientas para materializar
lo que ya pasó por QA. LLM decide contenido, código/herramientas ejecutan
producción, checklist verifica el resultado.

---

## 1. Cambios de esquema (migración 0005)

### 1.1 Tablas nuevas

```sql
-- Catálogo de MCP servers disponibles (data-driven, como pipeline_templates)
CREATE TABLE tool_adapters (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(50) NOT NULL UNIQUE,          -- 'inkscape' | 'krita' | 'reaper' | ...
    mcp_server_name VARCHAR(100) NOT NULL,     -- nombre exacto en mcp.servers de OpenClaw
    execution_mode VARCHAR(50) NOT NULL,       -- 'local' | 'local_orchestration_cloud_inference' | 'remote_cloud'
    requires_license VARCHAR(150) NULL,        -- ej. 'DaVinci Resolve Studio', 'Canva Pro+'
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Un paso ejecutado en una cadena de herramientas para un ContentArtifact
CREATE TABLE asset_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    artifact_id UUID NOT NULL REFERENCES content_artifacts(id) ON DELETE CASCADE,
    tool_adapter_id UUID NOT NULL REFERENCES tool_adapters(id),
    sequence_order INT NOT NULL,               -- posición en la cadena (1, 2, 3...)
    status VARCHAR(20) NOT NULL DEFAULT 'pending',  -- pending | running | done | failed
    input_ref TEXT NULL,                       -- ruta/id del insumo (guion, salida del paso anterior)
    output_path TEXT NULL,                     -- ruta local final del archivo producido
    external_job_id VARCHAR(150) NULL,         -- id del job async si es cloud (Veo 3 polling)
    cost_estimate JSONB NULL,                  -- créditos/costo si aplica (ElevenLabs, Veo3)
    started_at TIMESTAMPTZ NULL,
    completed_at TIMESTAMPTZ NULL,
    error TEXT NULL
);

CREATE INDEX ix_asset_jobs_artifact ON asset_jobs (artifact_id);
CREATE INDEX ix_asset_jobs_status ON asset_jobs (status);
```

### 1.2 Columnas nuevas

```sql
-- format_specs: la cadena de herramientas esperada (data-driven, §8 del diseño)
ALTER TABLE format_specs ADD COLUMN tool_chain JSONB NOT NULL DEFAULT '[]'::jsonb;
-- ej. ["producer", "elevenlabs", "comfyui", "resolve"]

-- content_artifacts: el visual_spec por variante (UX_DESIGN)
ALTER TABLE content_artifacts ADD COLUMN visual_spec JSONB NULL;
ALTER TABLE content_artifacts ADD COLUMN format_spec_id UUID NULL
    REFERENCES format_specs(id);
```

> **Nota:** `format_specs` y `component_library` se crean en esta misma
> migración (eran parte del diseño de producción multi-formato que quedó
> pendiente). `component_library` se añade como tabla nueva también.

### 1.3 Modelos ORM nuevos (`src/db/models.py`)

```python
class ToolAdapter(Base):
    __tablename__ = "tool_adapters"
    id: Mapped[uuid.UUID]
    name: Mapped[str]                 # unique
    mcp_server_name: Mapped[str]
    execution_mode: Mapped[str]
    requires_license: Mapped[str | None]
    is_active: Mapped[bool]

class AssetJob(Base):
    __tablename__ = "asset_jobs"
    id: Mapped[uuid.UUID]
    artifact_id: Mapped[uuid.UUID]    # FK content_artifacts
    tool_adapter_id: Mapped[uuid.UUID]  # FK tool_adapters
    sequence_order: Mapped[int]
    status: Mapped[str]               # pending | running | done | failed
    input_ref: Mapped[str | None]
    output_path: Mapped[str | None]
    external_job_id: Mapped[str | None]
    cost_estimate: Mapped[dict | None]
    started_at: Mapped[datetime | None]
    completed_at: Mapped[datetime | None]
    error: Mapped[str | None]
```

---

## 2. Nuevos módulos backend

### 2.1 `src/tools/` — capa de herramientas (nueva)

```
src/tools/
├── __init__.py
├── registry.py        # lee tool_adapters de la DB, resuelve cadenas
├── orchestrator.py    # ejecuta la cadena de AssetJobs para un artifact
├── mcp_client.py      # cliente MCP genérico (stdio) — habla con OpenClaw
└── chains.py          # selección de cadena por artifact_type (§8 del diseño)
```

**`registry.py`** — data-driven: dado un `artifact_type` + `brand_objective`,
resuelve la cadena de herramientas desde `format_specs.tool_chain` +
`tool_adapters`. Sin if/else por herramienta en el código.

**`orchestrator.py`** — el corazón. Dado un `ContentArtifact` aprobado:

1. Crea los `AssetJob` (uno por paso de la cadena, `sequence_order`).
2. Ejecuta cada paso vía `mcp_client` (o delega a OpenClaw).
3. Actualiza `status`/`output_path`/`cost_estimate`/`error` en cada paso.
4. Cuando el último llega a `done` → `content_artifacts.storage_path` se
   puebla y el artefacto pasa de `borrador` a `listo`.
5. Cada paso cloud (ElevenLabs, Veo 3) registra su costo como evento de
   telemetría (mismo patrón que Together AI).

**`mcp_client.py`** — cliente MCP genérico. Dos modos:

- **Modo directo (dev/test):** habla con los MCP servers vía stdio
  directamente (para probar la cadena sin OpenClaw).
- **Modo OpenClaw (prod):** envía el job a OpenClaw Gateway vía HTTP
  (el gateway tiene los MCP servers configurados y los tool-agents).

### 2.2 `src/agents/tool_agent.py` — el tool-agent

Un agente que ejecuta la cadena de herramientas para un formato. NO decide
qué producir — recibe `(draft, visual_spec, format_spec, artifact_id)` y
ejecuta la cadena. Es el "brazo" que materializa lo que Producer-Critic
aprobó.

```python
def run_tool_chain(session, artifact_id: uuid.UUID) -> dict:
    """Ejecuta la cadena de herramientas del artifact. Idempotente:
    si ya hay AssetJobs done, no re-ejecuta."""
    artifact = session.get(ContentArtifact, artifact_id)
    chain = resolve_chain(session, artifact.artifact_type, artifact.brief.brand_objective)
    jobs = create_asset_jobs(session, artifact, chain)
    for job in jobs:
        execute_job(session, job)   # vía mcp_client / OpenClaw
    return {"artifact_id": str(artifact_id), "jobs": [j.status for j in jobs]}
```

### 2.3 Evento de disparo (LISTEN/NOTIFY)

El backend ya tiene `agent_worker` escuchando `telemetry_channel`. Se
añade un canal nuevo `asset_jobs_channel` (o se reutiliza el patrón):

- Cuando un `ContentArtifact` pasa a `aprobado` (QA_VALID ok), la API
  inserta los `AssetJob` pendientes y hace `NOTIFY asset_jobs_channel`.
- El `agent_worker` (o un worker nuevo `tool_worker`) recibe el notify y
  ejecuta la cadena vía OpenClaw.

**Alternativa más simple:** un endpoint `POST /artifacts/{id}/produce`
que el tool-agent de OpenClaw llama cuando recibe el brief aprobado. El
worker de eventos es para el caso "producción automática"; el endpoint es
para el caso "producción disparada por el agente". Ambos convergen en
`orchestrator.run_tool_chain`.

---

## 3. Integración con OpenClaw

### 3.1 Configuración de MCP servers

OpenClaw se configura con `mcp.servers` (JSON). El plan:

```json
{
  "mcp": {
    "servers": {
      "inkscape": { "command": "inkmcp", "args": [] },
      "krita": { "command": "krita-mcp", "args": [] },
      "reaper": { "command": "reaper-mcp", "args": [] },
      "resolve": { "command": "resolve-mcp", "args": [] },
      "comfyui": { "command": "comfy-mcp", "args": ["--local"] },
      "elevenlabs": { "command": "uvx", "args": ["elevenlabs-mcp"] },
      "veo3": { "command": "veo-mcp-server", "args": [] },
      "gdrive": { "command": "mcp-gdrive", "args": [] },
      "filesystem": {
        "command": "npx",
        "args": [
          "-y",
          "@modelcontextprotocol/server-filesystem",
          "/mnt/disco1",
          "/mnt/disco2"
        ]
      }
    }
  }
}
```

**Regla de diseño:** `tool_adapters.mcp_server_name` debe coincidir
exactamente con la clave en `mcp.servers`. El `registry.py` valida esto
al arrancar (warning si un adapter activo no tiene server configurado).

### 3.2 Skills de OpenClaw

El diseño menciona `skills/` (SKILL.md por flujo: video_corto, carrusel).
El plan:

```
skills/
├── video_corto/SKILL.md    # flujo: guion → voz → assets → montaje
├── carrusel/SKILL.md       # flujo: copy → iconos → composición
├── poster_qr/SKILL.md
├── one_pager/SKILL.md
└── audio/SKILL.md          # broadcast/soundscape
```

Cada SKILL.md documenta la cadena de herramientas del formato, los
parámetros esperados y los criterios de éxito. Es la "receta" que el
tool-agent sigue.

### 3.3 Credenciales

| Servicio       | Mecanismo          | Dónde vive                                               |
| -------------- | ------------------ | -------------------------------------------------------- |
| ElevenLabs     | API key            | `api_keys` (provider `elevenlabs`) — ya existe el patrón |
| Veo 3 / Gemini | API key            | `api_keys` (provider `gemini`)                           |
| Canva          | OAuth (no API key) | Sesión de OpenClaw (token OAuth)                         |
| Google Drive   | OAuth 2.1          | Sesión de OpenClaw (token OAuth)                         |

**Decisión de diseño:** las API keys (ElevenLabs, Gemini) van en
`api_keys` del backend — mismo patrón que Together. Los OAuth (Canva,
Drive) viven en la sesión de OpenClaw porque son mecanismos de credencial
distintos (token de acceso + refresh, no una clave estática). Esto se
documenta en el README.

---

## 4. API endpoints nuevos

| Método           | Ruta                            | Descripción                                       |
| ---------------- | ------------------------------- | ------------------------------------------------- |
| `GET/POST`       | `/tool-adapters`                | Lista/crea adapters (catálogo MCP)                |
| `GET/PUT/DELETE` | `/tool-adapters/{id}`           | Detalle/edita/borra un adapter                    |
| `GET`            | `/artifacts/{id}/jobs`          | Lista los AssetJobs de un artifact (trazabilidad) |
| `POST`           | `/artifacts/{id}/produce`       | Dispara la cadena de herramientas (idempotente)   |
| `GET`            | `/artifacts/{id}/jobs/{job_id}` | Detalle de un job (status, output, costo)         |
| `POST`           | `/jobs/{job_id}/retry`          | Reintenta un job fallido                          |

---

## 5. Flujo completo (video_corto, ESCAPE)

```
1. Producer-Critic aprueba el brief (QA_VALID ok)
2. API crea ContentArtifact (status=borrador, artifact_type=video_corto)
3. API inserta AssetJobs: [elevenlabs(1), comfyui(2), resolve(3)]
4. NOTIFY asset_jobs_channel → tool_worker
5. tool_worker → OpenClaw (o mcp_client directo):
   a. elevenlabs: guion → audio.wav (cloud, costo registrado)
   b. comfyui: visual_spec → frames/clips (GPU local)
   c. resolve: timeline + audio + clips + subtítulos → render mp4
6. Cada job actualiza su status/output_path
7. Último job done → content_artifacts.storage_path = ruta del mp4
8. QA_VALID final sobre el archivo real (verificación mecánica + LLM)
9. RepurposeLink(derived_from) conecta pieza madre → variante
```

---

## 6. Despliegue

| Componente               | Dónde corre                          | Cambio                                           |
| ------------------------ | ------------------------------------ | ------------------------------------------------ |
| `db` (Postgres+pgvector) | Docker Compose                       | migración 0005                                   |
| `api` (FastAPI)          | Docker Compose                       | endpoints nuevos + registry                      |
| `agent_worker`           | Docker Compose                       | + canal asset_jobs (o worker nuevo)              |
| OpenClaw Gateway         | Host (workstation)                   | mcp.servers + skills                             |
| MCP servers              | Host (subprocesos stdio de OpenClaw) | instalación por herramienta                      |
| Apps creativas           | Host (GUI)                           | Inkscape, Krita, REAPER, Resolve Studio, ComfyUI |
| Discos externos          | Host (montados)                      | server-filesystem apuntado a cada mount          |

**Punto clave:** OpenClaw y las apps creativas corren en el HOST, no en
Docker — porque necesitan GUI, GPU y acceso a discos montados. El backend
(Docker) se comunica con OpenClaw por HTTP loopback. Esto ya está
reflejado en el diagrama de despliegue del diseño (§4).

---

## 7. Orden de implementación

### Fase 1 — Esquema y datos (día 1-2) ✅ HECHA

1. Migración 0005: `tool_adapters`, `asset_jobs`, `format_specs.tool_chain`,
   `content_artifacts.visual_spec` + `format_spec_id`, `component_library`.
2. Modelos ORM nuevos.
3. Seed de `tool_adapters` (los 10 del catálogo) + seed de `format_specs`
   con `tool_chain` (los 8 formatos de la tabla §7 del diseño).

### Fase 1.5 — Manifiesto y templates (día 2-3) ✅ HECHA

4. Migración 0006: `production_templates`, `production_manifests`,
   `format_specs.phases`, `asset_jobs.phase`/`template_id`/`manifest_version`.
5. Modelos ORM nuevos (`ProductionTemplate`, `ProductionManifest`).
6. Seed de `production_templates` (los 17 templates de §3) + seed de
   `format_specs.phases` por formato.

### Fase 2 — Capa de herramientas (día 3-5) ✅ HECHA

7. `src/tools/registry.py` — resolución de fases/cadenas (data-driven).
8. `src/tools/manifest.py` — carga/versionado del ProjectManifest.
9. `src/tools/template_engine.py` — manifiesto + template → contrato.
10. `src/tools/mcp_client.py` — cliente MCP genérico (modo directo primero).
11. `src/tools/orchestrator.py` — ejecución de AssetJobs + trazabilidad.
12. Tests unitarios (sin MCP real — mock del cliente).

### Fase 3 — API y eventos (día 6-8) ✅ HECHA

13. Endpoints CRUD `/tool-adapters` + `/production-templates` +
    `/artifacts/{id}/manifests` + `/artifacts/{id}/jobs` + `/produce`.
14. Canal `asset_jobs_channel` en el worker (o endpoint disparador).
15. Tests de integración (TestClient + DB real).

### Fase 4 — Proyectos (0007) ✅ HECHA

16. Migración 0007: `projects` + `project_versions` (rollback estilo github).
17. Modelos ORM (`Project`, `ProjectVersion`) + `ASSET_STORAGE_PATH` en config.
18. Endpoints: `/project/new`, `/projects` CRUD, `/projects/{id}/versions`,
    `/projects/{id}/approve`, `/projects/{id}/rollback`.

### Fase 5 — Flujo agéntico ✅ HECHA

19. `src/production/generator.py` — LLM escribe el snapshot JSON editable
    (topic + template + reglas de negocio). Prompt-as-code (0007): usa el
    artefacto `project_generator`; si el LLM falla, degrada a esqueleto
    determinista con `requires_user_acceptance=True` (sin versión nueva).
20. `src/production/refine.py` — servicio de edición de copy/prompts/guiones
    (reglas de negocio en código; LLM opcional prompt-as-code `project_refine`;
    RAG como hook pluggable después).
21. `src/production/postproduction.py` — recomendaciones algorítmicas
    (cortes, EQ, LUT, voz, loudness) desde templates + manifiesto.
22. Endpoints: `/projects/{id}/generate`, `/projects/{id}/refine`,
    `/projects/{id}/produce`, `/projects/{id}/postproduction`.

### Fase 6 — Visión ✅ HECHA

23. `api_keys` nueva (key_name `together_vision`) + `llm_models` con
    `model_size='vision'` + specs de prompts de visión.
24. Compilador incluye modelos de visión.

### Fase 7 — Integración real de generadores (día 9-12) ⏳

25. Config `mcp.servers` + skills/.
26. Modo OpenClaw del `mcp_client` (HTTP al gateway).
27. Integración real: ComfyUI WebSocket, Veo REST, ElevenLabs REST.
28. Prueba en vivo: carrusel Ergalia de punta a punta (Inkscape → Krita).

### Fase 8 — Pre-procesamiento modular ⏳

29. Scripts de alineación (FFmpeg/OpenCV/Librosa), FCPXML builder —
    módulos pluggable invocados por el proceso central.

### Fase 9 — RAG ⏳

30. Conectar `artifact_library` como contexto en generator/refine.

### Fase 10 — Frontend + WebSocket ⏳

31. React/TypeScript + eventos WebSocket de progreso.

### Fase 11 — QA final sobre archivo real (día 13-15) ⏳

32. Verificaciones mecánicas sobre el archivo (dimensiones, duración,
    tamaño, subtítulos presentes).
33. QA_VALID final integrado al flujo.

---

## 8. Decisiones abiertas (para validar)

1. **¿Canva se mantiene o se excluye?** El diseño lo deja abierto (§13).
   Si "todo local" es estricto → excluir y usar Inkscape+Krita.
2. **¿El tool-agent corre dentro de OpenClaw o como worker del backend?**
   Propuesta: OpenClaw orquesta (tiene los MCP servers); el backend solo
   registra y dispara. Alternativa: el backend habla MCP directo (modo
   directo del mcp_client) y OpenClaw queda como opción.
3. **¿DaVinci Resolve Studio o la edición gratuita?** El diseño exige
   Studio (la gratuita no expone la API de scripting). Es un costo.
4. **¿Veo 3 o Gemini Omni Flash?** El diseño nota que Omni Flash es ahora
   el default recomendado para video. Evaluar antes de fijar.
5. **¿El worker de eventos o el endpoint `/produce` como disparador?**
   Propuesta: ambos — el endpoint para uso manual/agente, el worker para
   automatización.
6. **¿`component_library` se siembra con los patrones de las guías o
   vacía?** Propuesta: sembrar los explícitos de `_lenguaje_visual.md`.

---

## 9. Riesgos y mitigaciones

| Riesgo                                                    | Mitigación                                                                                 |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| MCP servers inestables (Krita timeout, Resolve scripting) | Timeouts configurables por adapter; `asset_jobs.status=failed` con `error`; retry endpoint |
| Costo cloud descontrolado (ElevenLabs, Veo 3)             | `cost_estimate` en cada job; telemetría de costo; límite por brief                         |
| Licencias (Resolve Studio, Canva Pro)                     | `requires_license` en tool_adapters; warning al activar sin licencia                       |
| OpenClaw caído                                            | Modo directo del mcp_client como fallback; jobs quedan `pending` y se reintentan           |
| Discos externos no montados                               | server-filesystem falla ruidoso; job `failed` con error claro                              |
