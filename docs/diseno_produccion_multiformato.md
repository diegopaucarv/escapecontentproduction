# Diseño — Producción Multi-Formato (UX_DESIGN / PROD_DEV / QA_VALID)

**Estado:** propuesto — pendiente de validación
**Alcance:** Sub-ruta Complete del pipeline (§9.1), nodos `UX_DESIGN`,
`PROD_DEV` y `QA_VALID` — la producción modular de una pieza madre en
múltiples formatos/canales para redes sociales.

---

## 0. Principio rector: derivar, no crear

Una campaña multi-formato NO es "escribir N piezas". Es **una pieza madre
que se transforma en N contenedores**. El `insight_core`, la evidencia y
el CTA son el ADN que se conserva; lo que cambia es la estructura, la
longitud, el tono y el lenguaje visual de cada formato.

```
        ┌─────────────────────────────────────────────┐
        │  PIEZA MADRE (insight_core + Context Pack)  │
        └─────────────────────────────────────────────┘
                          │  derivación (no creación)
        ┌─────────┬────────┼─────────┬──────────┬──────┐
        ▼         ▼        ▼         ▼          ▼      ▼
     post     carrusel  video     video     one-    white
     LinkedIn           corto     largo     pager   paper
```

Cada variante se produce con una **receta de derivación** (qué conservar,
qué transformar, qué añadir) — no con un prompt genérico de "escribe un
post sobre X".

---

## 1. El registro de formatos: `format_specs` (tabla nueva, 0005)

La pieza central del diseño. Define, por `(brand_objective, artifact_type)`,
cómo se produce y valida cada formato. Es la misma filosofía data-driven
de `pipeline_templates` y `prompt_templates`: **los datos mandan, no el
código**.

```python
class FormatSpec(Base):
    __tablename__ = "format_specs"
    __table_args__ = (UniqueConstraint("brand_objective", "artifact_type"),)

    id: Mapped[uuid.UUID]
    brand_objective: Mapped[BrandObjective]   # ESCAPE_SOCIAL | ERGALIA_COMERCIAL | HIBRIDO
    artifact_type: Mapped[str]                # post | carrusel | video_corto | ...

    # 1. Estructura del formato (secciones en orden)
    structure: Mapped[list]      # ["hook_15s", "desarrollo", "cta"]

    # 2. Restricciones de longitud/duración
    constraints: Mapped[dict]     # {"max_chars": 280, "slides": 5, "max_duration_s": 60}

    # 3. Receta de derivación desde la pieza madre
    derivation_rules: Mapped[list]  # qué conservar / transformar / añadir

    # 4. Requisitos visuales (lenguaje de marca aplicado al formato)
    visual_requirements: Mapped[dict]  # paleta, tipografía, composición, assets

    # 5. QA checks específicos del formato
    qa_checks: Mapped[list]      # ["cta_unico", "longitud", "fuente_visible", ...]

    is_active: Mapped[bool]
```

### 1.1 Ejemplo — `format_specs` para ESCAPE_SOCIAL

| artifact_type | structure | constraints | derivation_rules (resumen) | qa_checks |
|---|---|---|---|---|
| `post` | hook → desarrollo → cta | `max_chars: 300` | Conserva: insight, evidencia, CTA. Transforma: tono conversacional. Añade: pregunta retórica | `cta_unico`, `longitud`, `fuente_visible`, `gancho_15s_ok` |
| `carrusel` | portada → 3–5 slides → cta | `slides: 5` | Conserva: insight, evidencia. Transforma: una idea por slide. Añade: slide de portada con gancho, slide final CTA | `cta_unico`, `slides_max`, `una_idea_por_slide`, `fuente_visible` |
| `video_corto` | hook 3s → desarrollo 20s → cta 5s | `max_duration_s: 60` | Conserva: insight, CTA. Transforma: guion hablado, ritmo visual. Añade: hook visual, texto en pantalla | `cta_unico`, `duracion`, `hook_visual`, `subtitulos` |
| `video_largo` | hook → contexto → desarrollo → cierre | `max_duration_s: 600` | Conserva: insight, evidencia, CTA. Transforma: argumentación extendida. Añade: estructura narrativa, capítulos | `cta_unico`, `duracion`, `fuente_visible`, `estructura_capitulos` |
| `one_pager` | titular → dato clave → 3 argumentos → cta | `max_pages: 1` | Conserva: insight, evidencia. Transforma: jerarquía visual, cifras destacadas. Añade: bloque de datos | `cta_unico`, `fuente_visible`, `datos_cifrados` |
| `white_paper` | portada → resumen → metodología → hallazgos → referencias | `max_pages: 10` | Conserva: insight, evidencia. Transforma: argumentación formal. Añade: metodología, referencias | `cta_unico`, `fuente_visible`, `referencias`, `anonimizacion_verificada` |

> **Nota:** el checklist base de la marca (`pipeline_templates.checklist`)
> SIEMPRE aplica además de `qa_checks`. El formato añade, no reemplaza.

---

## 2. UX_DESIGN — Lenguaje visual de marca + patrones reutilizables

### 2.1 Qué produce

Un **`visual_spec` por variante**: un JSONB que describe cómo se ve la
pieza (paleta, tipografía, composición, assets, layout por sección). Es
el puente entre el lenguaje visual de la marca (`_lenguaje_visual.md`) y
el artefacto final.

```json
{
  "palette": ["#1A1A2E", "#E94560", "#0F3460"],
  "typography": {"heading": "Montserrat Bold", "body": "Inter Regular"},
  "composition": {"portada": "centrado", "slides": "izquierda-derecha"},
  "assets": ["/components/escape/icono_dato.svg"],
  "layout": {"slide_1": "titular + dato", "slide_2": "argumento + fuente"}
}
```

### 2.2 Cómo se valida

Un **validador mecánico** (`validate_visual_spec`) verifica que el
`visual_spec` cumpla las reglas de marca:

- La paleta está dentro de los colores permitidos de la marca.
- La tipografía es de la familia aprobada.
- La composición es una de las permitidas para el formato.
- Los assets referenciados existen en la biblioteca.

**Regla de diseño:** el validador es código determinista (como el
checklist de `alignment.py`). El LLM puede *proponer* un `visual_spec`,
pero el código *verifica* que cumpla las reglas. Si no cumple → `fail`
con el motivo; si no se puede verificar → `needs_human` (el diseñador
decide).

### 2.3 Patrones reutilizables: `component_library` (tabla nueva)

El `/components/manifest.md` del doc se vuelve una tabla:

```python
class ComponentLibrary(Base):
    __tablename__ = "component_library"

    id: Mapped[uuid.UUID]
    brand_objective: Mapped[BrandObjective]
    artifact_type: Mapped[str]
    component_type: Mapped[str]   # "visual" | "textual" | "estructura"
    name: Mapped[str]             # ej. "icono_dato", "slide_evidencia", "hook_pregunta"
    content: Mapped[dict]         # el patrón reutilizable (JSONB)
    usage_count: Mapped[int]      # cuántas veces se usó (para Kaizen)
    is_active: Mapped[bool]
```

- **UX_DESIGN** consulta los componentes activos de la marca y los
  referencia en el `visual_spec` (no los duplica).
- **Kaizen** (`UPDATE_REGISTRY`) incrementa `usage_count` y puede
  desactivar patrones que no funcionan.
- **Regla:** un componente se reutiliza por referencia (`component_id`),
  nunca copiado — así una mejora de Kaizen propaga a todas las piezas
  futuras.

---

## 3. PROD_DEV — Producción modular por formato

### 3.1 El nodo producer por formato

El nodo `producer` del grafo (hoy stub) se especializa: en vez de un
prompt genérico, usa un **prompt compilado por formato** que fusiona:

1. La spec base `producer_draft` (gancho 15s, CTA único, tono de marca,
   no inventar datos — ya compilada en `prompt_artifacts`).
2. La `derivation_rules` del `format_spec` (la receta del formato).
3. Las `constraints` del `format_spec` (longitud, slides, duración).

El compilador (`src/llm/compiler.py`) ya soporta esto: cada spec se
transpila a un artefacto inmutable por modelo. Se añade el `format_spec`
como capa de datos — **sin if/else por formato en el código**.

### 3.2 La receta de derivación (formato del prompt)

Cada `derivation_rules` se expresa como una lista de instrucciones
estructuradas que el compilador inyecta en el prompt:

```json
[
  "CONSERVAR: insight_core, evidencia, CTA",
  "TRANSFORMAR: longitud a 280 chars, tono conversacional",
  "AÑADIR: pregunta retórica al final del hook",
  "NO INVENTAR: usar solo el Context Pack"
]
```

### 3.3 Context Pack filtrado por formato

El Context Pack (canon + guía + Artifact Library) se **filtra por
formato** antes de pasarlo al producer:

- Un `video_corto` necesita: el dato, el gancho, el CTA.
- Un `white_paper` necesita: el dato, la evidencia completa, las
  referencias, la metodología.

Esto reduce tokens, mejora la adherencia al formato y evita que el LLM
"se pierda" en contexto irrelevante. El filtro es una función determinista
(`filter_context_pack(pack, format_spec)`) que selecciona secciones por
etiqueta.

### 3.4 Flujo del nodo

```
producer_node(state, format_spec, context_pack):
    1. pack_filtrado = filter_context_pack(context_pack, format_spec)
    2. prompt = get_active_prompt(session, large_model, "producer_draft")
       + format_spec.derivation_rules + format_spec.constraints
    3. draft = complete(session, prompt, model_size="large",
                       response_format={"type": "json_object"})
    4. return {"draft": draft, "visual_spec": visual_spec_propuesto}
```

Mismo patrón de resiliencia que los otros consumidores: retries
(`llm_retries`) → fallback (`fallback_model`) → degradación elegante.

---

## 4. QA_VALID — QA editorial: checklist + evidencia + anonimización

### 4.1 Dos capas (consistente con el diseño de alignment)

| Capa | Qué verifica | Cómo | Determinista |
|---|---|---|---|
| **Mecánica (código)** | longitud, CTA único, fuente citada, anonimización, restricciones del formato | regex / NER / conteo | ✅ sí |
| **LLM (crítico)** | juicio editorial: gancho, tono, claridad, coherencia | `critic_checklist` con `qa_checks` del formato | ❌ no (refina) |

**Regla de oro:** la capa mecánica impone lo objetivamente verificable;
el LLM solo añade juicio editorial. Un ítem que la mecánica no puede
verificar → `needs_human`, nunca `ok` (mismo patrón que `alignment.py`).

### 4.2 Verificaciones mecánicas (nuevas, en código)

```python
MECHANICAL_CHECKS = {
    "longitud":        _check_longitud,          # chars <= constraints.max_chars
    "cta_unico":       _check_cta_unico,         # exactamente 1 CTA
    "fuente_visible":  _check_fuente_visible,    # URL/cita presente en el draft
    "anonimizacion":   _check_anonimizacion,     # regex/NER: DNI, email, teléfono, nombres
    "slides_max":      _check_slides_max,        # carrusel: <= constraints.slides
    "duracion":        _check_duracion,          # video: <= constraints.max_duration_s
    "una_idea_por_slide": _check_una_idea,       # carrusel: cada slide un solo punto
    "subtitulos":      _check_subtitulos,        # video: texto en pantalla presente
    "datos_cifrados":  _check_datos_cifrados,    # one-pager: cifras destacadas
    "referencias":     _check_referencias,       # white paper: lista de referencias
}
```

- **Anonimización** es la más delicada: para segmentos S5/S6 o
  `risk_level = alto`, se exige que el draft NO contenga datos personales
  (regex para DNI/email/teléfono + NER para nombres). Si los contiene →
  `fail` con el motivo. Si no se puede confirmar → `needs_human`.
- **Fuente visible**: el claim principal debe citar la `evidence_source`
  del brief (presencia de URL o cita textual). El LLM puede confirmar la
  *coherencia* (que la fuente respalda el claim), pero la *presencia* la
  verifica el código.

### 4.3 El crítico LLM por formato

El nodo `critic` (ya real, `src/llm/critic.py`) recibe:

- El `checklist` = `pipeline_templates.checklist` (base de marca) +
  `format_spec.qa_checks` (específico del formato).
- El `draft` de la variante.
- El `brand_objective`.

La regla defensiva ya implementada aplica: ítem sin interpretación en
`rules` → `no_evaluado`, nunca `ok`. Cada `qa_check` nuevo del formato
debe tener su interpretación en el spec `critic_checklist` (validado en
el CRUD con 422, igual que hoy).

### 4.4 Veredicto final por variante

```
QA_VALID(variante) =
    mecánica (código)  → fail | needs_human | ok
    + LLM (crítico)    → auto_pass | needs_human_review | fail
    ─────────────────────────────────────────────
    fusión conservadora (igual que reinforcement):
    - fail (cualquiera) → fail
    - needs_human (cualquiera) → needs_human
    - ok + auto_pass → ok
```

Cada variante tiene su propio veredicto. La campaña (paquete) solo avanza
a `INTEGRATION` si **todas** las variantes pasan (o están en
`needs_human` con aprobación del líder).

---

## 5. Integración con el grafo Producer-Critic

El grafo actual (`src/agents/producer_critic.py`) se extiende para
producir **una variante por iteración**:

```
START → producer_node(format_spec_i) → critic_node(format_spec_i)
         │                                │
         │ fail (máx 3)                   │ needs_human_review
         ▼                                ▼
      producer_node (reintenta)      human_review_node (interrupt)
         │                                │
         └────────── auto_pass ──────────┘
                        │
                        ▼
              siguiente variante (format_spec_{i+1})
                        │
                        ▼
              INTEGRATION (paquete de campaña)
```

- El `repurpose_plan` del brief (Bloque G) define la lista de
  `format_spec` a producir.
- Cada variante genera su `ContentArtifact` (con `artifact_type`,
  `channel`, `visual_spec`, `storage_path`).
- Los `RepurposeLink` conectan la pieza madre con cada variante
  (`link_type = "derivado"`).

---

## 6. Resumen de tablas nuevas (migración 0005)

| Tabla | Propósito | Relaciones |
|---|---|---|
| `format_specs` | Registro de formatos: estructura, constraints, derivación, visual, QA | `brand_objective` + `artifact_type` (único) |
| `component_library` | Patrones reutilizables (visuales/textuales/estructura) | `brand_objective` + `artifact_type` |

Y se extiende:
- `content_artifacts` → columna `visual_spec` (JSONB) + `format_spec_id` (FK).
- `repurpose_links` → `link_type` ya soporta `derivado` (verificar enum).

---

## 7. Orden de implementación sugerido

1. **Migración 0005** — `format_specs` + `component_library` + columna
   `visual_spec` en `content_artifacts`.
2. **Seed de formatos** — los 6 formatos de la tabla §1.1 para ambas
   marcas (datos puros, como `seed_ai.py`).
3. **Validador mecánico** (`src/agents/qa_validator.py`) — las 10
   verificaciones de §4.2, con tests unitarios (sin DB/red).
4. **Producer por formato** (`src/llm/producer.py`) — consumidor de
   `producer_draft` + `format_spec`, con retries/fallback/degradación.
5. **Extender el grafo** — producir variante por variante, con
   `visual_spec` y QA por formato.
6. **CRUD** — `/format-specs` y `/component-library` (mismo patrón que
   `/llm-models`).

---

## 8. Decisiones abiertas (para validar con el usuario)

1. **¿El `visual_spec` lo propone el LLM o lo arma el diseñador?**
   Propuesta: el LLM propone, el validador mecánico verifica, el
   diseñador aprueba (mismo patrón producer-critic).
2. **¿`component_library` se siembra con los patrones actuales de las
   guías de marca, o se empieza vacía y se llena con Kaizen?**
   Propuesta: sembrar los patrones explícitos de las guías
   (`_lenguaje_visual.md`) y dejar que Kaizen agregue el resto.
3. **¿El Context Pack filtrado por formato es una función determinista o
   una spec más?** Propuesta: determinista (etiquetas de sección), porque
   es lógica de selección, no de lenguaje.
4. **¿El white paper y el one-pager entran en el alcance "redes
   sociales"?** El doc los incluye como formatos de autoridad; se pueden
   producir en la misma campaña o en una campaña paralela.
