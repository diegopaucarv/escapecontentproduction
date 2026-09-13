"""
Modelos ORM (SQLAlchemy 2.0, estilo Mapped/mapped_column) que reflejan
exactamente sql/001_init.sql. Se mantienen sincronizados a mano por ahora;
en cuanto haya una segunda migración, introducir Alembic (ver README) en
vez de seguir editando 001_init.sql directamente.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    CheckConstraint,
    ForeignKey,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ENUM as PgEnum
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func

# La dimensión vive fijada aquí Y en sql/001_init.sql (vector(1024)) a
# propósito: es un cambio de esquema, no de configuración de runtime.
# Si el proveedor de embeddings cambia de dimensión, se hace vía una
# migración nueva (ALTER COLUMN ... TYPE vector(N)) que actualice ambos
# lugares a la vez — nunca solo variando .env.
EMBEDDING_DIM = 1024


class Base(DeclarativeBase):
    pass


class BrandObjective(str, enum.Enum):
    ESCAPE_SOCIAL = "ESCAPE_SOCIAL"
    ERGALIA_COMERCIAL = "ERGALIA_COMERCIAL"
    HIBRIDO = "HIBRIDO"


class ContentBucket(str, enum.Enum):
    difusion_cientifica = "difusion_cientifica"
    comunidad_intelectual = "comunidad_intelectual"
    debate_informado = "debate_informado"
    herramienta_gratuita = "herramienta_gratuita"
    formacion_aplicada = "formacion_aplicada"
    servicio_formal = "servicio_formal"
    caso_autoridad = "caso_autoridad"
    artefacto_fisico_digital = "artefacto_fisico_digital"
    captacion_directa = "captacion_directa"


class RiskLevel(str, enum.Enum):
    bajo = "bajo"
    medio = "medio"
    alto = "alto"


class BriefStatus(str, enum.Enum):
    idea = "idea"
    brief = "brief"
    generando = "generando"
    revision = "revision"
    aprobado = "aprobado"
    produccion = "produccion"
    publicado = "publicado"
    aprendido = "aprendido"
    archivado = "archivado"


class RouteDecision(str, enum.Enum):
    repetitivo = "repetitivo"
    solucion_previa = "solucion_previa"
    nueva_solucion = "nueva_solucion"


class ProductionRoute(str, enum.Enum):
    fast = "fast"
    complete = "complete"


class UserRole(str, enum.Enum):
    lider = "lider"
    equipo = "equipo"
    externo = "externo"


class RepurposeLinkType(str, enum.Enum):
    derived_from = "derived_from"
    expanded_to = "expanded_to"
    translated_to = "translated_to"
    summarized_from = "summarized_from"


def _pg_enum(py_enum, name: str) -> PgEnum:
    """El tipo ya existe en la base (creado por 001_init.sql);
    create_type=False evita que SQLAlchemy intente recrearlo."""
    return PgEnum(py_enum, name=name, create_type=False)


class AppUser(Base):
    __tablename__ = "app_users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    full_name: Mapped[str] = mapped_column(String(150))
    email: Mapped[str] = mapped_column(String(150), unique=True)
    role: Mapped[UserRole] = mapped_column(_pg_enum(UserRole, "user_role_t"))
    brand_scope: Mapped[BrandObjective | None] = mapped_column(
        _pg_enum(BrandObjective, "brand_objective_t"), nullable=True
    )
    hashed_password: Mapped[str | None] = mapped_column(
        String(255), nullable=True
    )  # añadido en 0002 — NULL = usuario sin login (ej. externo)
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class PipelineTemplate(Base):
    __tablename__ = "pipeline_templates"
    __table_args__ = (UniqueConstraint("brand_objective", "content_bucket"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    brand_objective: Mapped[BrandObjective] = mapped_column(
        _pg_enum(BrandObjective, "brand_objective_t")
    )
    content_bucket: Mapped[ContentBucket] = mapped_column(
        _pg_enum(ContentBucket, "content_bucket_t")
    )
    default_route: Mapped[ProductionRoute | None] = mapped_column(
        _pg_enum(ProductionRoute, "production_route_t"), nullable=True
    )
    checklist: Mapped[list] = mapped_column(JSONB, default=list)
    novelty_weights: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now())


class PipelineStage(Base):
    """Tabla de referencia (lookup) — poblada por el seed de 001_init.sql,
    no se escribe desde la app salvo para agregar una etapa nueva."""

    __tablename__ = "pipeline_stages"

    stage_code: Mapped[str] = mapped_column(String(50), primary_key=True)
    stage_label: Mapped[str] = mapped_column(String(150))
    phase_group: Mapped[str] = mapped_column(String(50))


class ContentBrief(Base):
    __tablename__ = "content_briefs"
    __table_args__ = (
        CheckConstraint(
            "segment_client IS NOT NULL OR segment_community IS NOT NULL",
            name="chk_segment_present",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    owner_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("app_users.id"), nullable=True
    )
    status: Mapped[BriefStatus] = mapped_column(
        _pg_enum(BriefStatus, "brief_status_t"), default=BriefStatus.idea
    )

    # Bloque B
    resumen: Mapped[str | None] = mapped_column(Text, nullable=True)
    insight_core: Mapped[str | None] = mapped_column(Text, nullable=True)
    pitch_15s: Mapped[str | None] = mapped_column(Text, nullable=True)
    prior_attempts: Mapped[str | None] = mapped_column(Text, nullable=True)
    risks: Mapped[str | None] = mapped_column(Text, nullable=True)
    novelty_indicators: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    suggested_product_type: Mapped[str | None] = mapped_column(
        String(100), nullable=True
    )
    org_priorities_contrast: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Bloque C
    brand_objective: Mapped[BrandObjective] = mapped_column(
        _pg_enum(BrandObjective, "brand_objective_t")
    )
    phase_number: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    segment_client: Mapped[str | None] = mapped_column(String(2), nullable=True)
    segment_community: Mapped[str | None] = mapped_column(String(2), nullable=True)
    interlocutor_profile: Mapped[str | None] = mapped_column(String(100), nullable=True)
    subprofile: Mapped[str | None] = mapped_column(String(150), nullable=True)
    audience_tier: Mapped[str | None] = mapped_column(String(50), nullable=True)
    need_id: Mapped[str | None] = mapped_column(String(50), nullable=True)
    need: Mapped[str | None] = mapped_column(Text, nullable=True)
    change_hypothesis: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Bloque D
    content_bucket: Mapped[ContentBucket] = mapped_column(
        _pg_enum(ContentBucket, "content_bucket_t")
    )
    service_category: Mapped[str | None] = mapped_column(String(100), nullable=True)
    product_anchor: Mapped[str | None] = mapped_column(String(150), nullable=True)
    entry_offer: Mapped[str | None] = mapped_column(String(150), nullable=True)

    # Bloque E
    artifact_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    channel: Mapped[str | None] = mapped_column(String(50), nullable=True)
    channel_role: Mapped[str | None] = mapped_column(String(50), nullable=True)
    cta: Mapped[str | None] = mapped_column(String(200), nullable=True)
    landing: Mapped[str | None] = mapped_column(String(300), nullable=True)
    funnel_stage: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Bloque F
    language: Mapped[str] = mapped_column(String(10), default="es")
    geography_content: Mapped[str | None] = mapped_column(String(100), nullable=True)
    geography_sales: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Bloque G
    evidence_source: Mapped[str | None] = mapped_column(Text, nullable=True)
    risk_level: Mapped[RiskLevel] = mapped_column(
        _pg_enum(RiskLevel, "risk_level_t"), default=RiskLevel.bajo
    )
    debate_governance: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    validation_required: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    repurpose_plan: Mapped[list | None] = mapped_column(JSONB, nullable=True)

    # Bloque H
    metric_primary: Mapped[str | None] = mapped_column(String(100), nullable=True)
    metric_secondary: Mapped[str | None] = mapped_column(String(100), nullable=True)
    novelty_score: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    route_decision: Mapped[RouteDecision | None] = mapped_column(
        _pg_enum(RouteDecision, "route_decision_t"), nullable=True
    )
    production_route: Mapped[ProductionRoute | None] = mapped_column(
        _pg_enum(ProductionRoute, "production_route_t"), nullable=True
    )
    pipeline_template_id: Mapped[int | None] = mapped_column(
        ForeignKey("pipeline_templates.id"), nullable=True
    )
    # 0007: la última decisión (alineamiento/refuerzo) fue determinista y requiere aceptación explícita del usuario.
    requires_user_acceptance: Mapped[bool] = mapped_column(default=False)

    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now())

    artifacts: Mapped[list["ContentArtifact"]] = relationship(back_populates="brief")


class ContentArtifact(Base):
    __tablename__ = "content_artifacts"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    brief_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("content_briefs.id", ondelete="CASCADE")
    )
    artifact_type: Mapped[str] = mapped_column(String(50))
    channel: Mapped[str] = mapped_column(String(50))
    storage_path: Mapped[str | None] = mapped_column(String(300), nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="borrador")
    published_at: Mapped[datetime | None] = mapped_column(nullable=True)
    utm_campaign: Mapped[str | None] = mapped_column(String(150), nullable=True)
    # Spec de formato aplicada (0005) — qué cadena de herramientas usar.
    format_spec_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("format_specs.id"), nullable=True
    )
    # Proyecto que materializa este artefacto (0008, Fase 5 §20.3).
    project_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("projects.id"), nullable=True
    )
    # Especificación visual de la variante (UX_DESIGN, 0005).
    visual_spec: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now())

    brief: Mapped["ContentBrief"] = relationship(back_populates="artifacts")


class RepurposeLink(Base):
    __tablename__ = "repurpose_links"
    __table_args__ = (
        CheckConstraint(
            "source_artifact_id <> target_artifact_id", name="chk_no_self_link"
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    source_artifact_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("content_artifacts.id", ondelete="CASCADE")
    )
    target_artifact_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("content_artifacts.id", ondelete="CASCADE")
    )
    link_type: Mapped[RepurposeLinkType] = mapped_column(
        _pg_enum(RepurposeLinkType, "repurpose_link_t")
    )
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class ProductionLog(Base):
    __tablename__ = "production_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    brief_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("content_briefs.id", ondelete="CASCADE")
    )
    stage_code: Mapped[str] = mapped_column(ForeignKey("pipeline_stages.stage_code"))
    entered_at: Mapped[datetime] = mapped_column(server_default=func.now())
    exited_at: Mapped[datetime | None] = mapped_column(nullable=True)
    rework_count: Mapped[int] = mapped_column(default=0)
    # lead_time_minutes es GENERATED ALWAYS en la base — solo lectura desde el ORM
    lead_time_minutes: Mapped[int | None] = mapped_column(
        nullable=True, insert_default=None
    )


class ArtifactLibraryRow(Base):
    """Índice vectorial para RAG / Context Packs / búsqueda de duplicado
    (Paso 1 del enrutamiento por novedad, src/agents/novelty_router.py)."""

    __tablename__ = "artifact_library"

    id: Mapped[int] = mapped_column(primary_key=True)
    artifact_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("content_artifacts.id", ondelete="CASCADE"), nullable=True
    )
    brief_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("content_briefs.id", ondelete="CASCADE"), nullable=True
    )
    content_summary: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))
    retention_24h: Mapped[float | None] = mapped_column(nullable=True)
    conversion_30d: Mapped[float | None] = mapped_column(nullable=True)
    indexed_at: Mapped[datetime] = mapped_column(server_default=func.now())


class TelemetryEvent(Base):
    __tablename__ = "telemetry_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    artifact_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("content_artifacts.id", ondelete="SET NULL"), nullable=True
    )
    event_type: Mapped[str] = mapped_column(String(50))
    payload: Mapped[dict] = mapped_column(JSONB, default=dict)
    received_at: Mapped[datetime] = mapped_column(server_default=func.now())


class ApiKey(Base):
    """Claves de API por proveedor (Together, OpenAI, ...). La clave en sí
    es un secreto: se inserta vía seed/env (src/db/seed_llm.py) y NUNCA se
    devuelve completa por la API — solo enmascarada."""

    __tablename__ = "api_keys"
    __table_args__ = (UniqueConstraint("provider", "key_name"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    provider: Mapped[str] = mapped_column(String(50))
    key_name: Mapped[str] = mapped_column(String(100))
    api_key: Mapped[str] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now())


class SessionSettings(Base):
    """Settings de sesión: modelos pequeño/grande + parámetros, referenciando
    UNA api_key. Solo una fila activa a la vez (singleton, ver
    uq_session_settings_active en sql/002_llm_infra.sql)."""

    __tablename__ = "session_settings"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    api_key_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("api_keys.id"))
    small_model: Mapped[str] = mapped_column(String(150))
    large_model: Mapped[str] = mapped_column(String(150))
    # Modelo de respaldo si el pequeño falla tras agotar reintentos (0004).
    fallback_model: Mapped[str | None] = mapped_column(String(150), nullable=True)
    # Reintentos por llamada LLM (0004) — default 3.
    llm_retries: Mapped[int] = mapped_column(default=3)
    temperature_small: Mapped[float] = mapped_column(default=0.7)
    temperature_large: Mapped[float] = mapped_column(default=0.7)
    max_tokens_small: Mapped[int] = mapped_column(default=2048)
    max_tokens_large: Mapped[int] = mapped_column(default=4096)
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now())


class LlmModel(Base):
    """Registro de modelos LLM (0004). CRUD-editable vía /llm-models.

    `syntax_profile` (JSONB) describe CÓMO hablarle a cada modelo/proveedor
    (api_style, system_role_name, instruction_formatting, structured_output,
    tool_calling, prompt_caching, reasoning_mode). El compilador
    (src/llm/compiler.py) lo usa como datos, no como if/else por proveedor.
    """

    __tablename__ = "llm_models"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    model_name: Mapped[str] = mapped_column(String(150), unique=True)
    provider: Mapped[str] = mapped_column(String(50))
    model_size: Mapped[str] = mapped_column(
        String(20)
    )  # 'small' | 'large' | 'embedding'
    context_window: Mapped[int] = mapped_column(default=0)
    max_output_tokens: Mapped[int] = mapped_column(default=0)
    temperature_default: Mapped[float] = mapped_column(default=0.7)
    strengths: Mapped[list] = mapped_column(JSONB, default=list)
    weaknesses: Mapped[list] = mapped_column(JSONB, default=list)
    prompt_style: Mapped[str] = mapped_column(Text, default="")
    syntax_profile: Mapped[dict] = mapped_column(JSONB, default=dict)
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now())


class PromptTemplate(Base):
    """Spec agnóstica de un prompt (0004). CRUD-editable vía /prompt-templates.

    NO es un archivo YAML suelto: vive en la base para que el usuario pueda
    editarlo sin tocar código. El compilador lo transpila a un artefacto
    inmutable por modelo (prompt_artifacts).
    """

    __tablename__ = "prompt_templates"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    task_key: Mapped[str] = mapped_column(String(100), unique=True)
    version: Mapped[str] = mapped_column(String(20), default="1.0")
    intent: Mapped[str] = mapped_column(Text)
    rules: Mapped[list] = mapped_column(JSONB, default=list)
    input_schema: Mapped[dict] = mapped_column(JSONB, default=dict)
    output_schema: Mapped[dict] = mapped_column(JSONB, default=dict)
    few_shot: Mapped[list] = mapped_column(JSONB, default=list)
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now())


class PromptArtifact(Base):
    """Prompt compilado, congelado e INMUTABLE (0004).

    Nunca se hace UPDATE de una fila: recompilar = INSERT nueva versión
    (artifact_version + 1) y desactivar la anterior. Así el runtime siempre
    ejecuta un prompt determinista y auditable, y el prompt caching de
    prefijo no se rompe por cambios a mitad de sesión.
    """

    __tablename__ = "prompt_artifacts"
    __table_args__ = (UniqueConstraint("llm_model_id", "task_key", "artifact_version"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    llm_model_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("llm_models.id"))
    task_key: Mapped[str] = mapped_column(String(100))
    spec_version: Mapped[str] = mapped_column(String(20))
    artifact_version: Mapped[int] = mapped_column(default=1)
    prompt_text: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64))
    compiled_by: Mapped[str] = mapped_column(String(100), default="compiler")
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class EmbeddingSetting(Base):
    """Settings de embeddings (0004). Singleton: a lo sumo una fila activa.

    Referencia la api_key (voyage) y el modelo de embeddings (llm_models con
    model_size='embedding'). src/embeddings.py lee de aquí en vez de .env.
    """

    __tablename__ = "embedding_settings"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    api_key_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("api_keys.id"))
    llm_model_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("llm_models.id"))
    dimension: Mapped[int] = mapped_column(default=1024)
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now())


class FormatSpec(Base):
    """Spec de formato por marca (0005). Data-driven: define la estructura,
    restricciones, reglas de derivación, requisitos visuales, QA checks y la
    cadena de herramientas (tool_chain) para cada (brand_objective,
    artifact_type). El orquestador (src/tools/) la usa como datos, no como
    if/else por formato en el código.
    """

    __tablename__ = "format_specs"
    __table_args__ = (UniqueConstraint("brand_objective", "artifact_type"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    brand_objective: Mapped[BrandObjective] = mapped_column(
        _pg_enum(BrandObjective, "brand_objective_t")
    )
    artifact_type: Mapped[str] = mapped_column(String(50))
    structure: Mapped[dict] = mapped_column(JSONB, default=dict)
    constraints: Mapped[dict] = mapped_column(JSONB, default=dict)
    derivation_rules: Mapped[list] = mapped_column(JSONB, default=list)
    visual_requirements: Mapped[dict] = mapped_column(JSONB, default=dict)
    qa_checks: Mapped[list] = mapped_column(JSONB, default=list)
    tool_chain: Mapped[list] = mapped_column(JSONB, default=list)
    phases: Mapped[dict] = mapped_column(JSONB, default=dict)
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now())


class ComponentLibrary(Base):
    """Patrones reutilizables por marca/formato (0005).

    component_type: 'visual' | 'textual' | 'estructura'. El contenido es
    JSONB libre (ej. un bloque de copy, un patrón de iconografía, una
    plantilla de estructura). usage_count se incrementa al reutilizarse.
    """

    __tablename__ = "component_library"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    brand_objective: Mapped[BrandObjective | None] = mapped_column(
        _pg_enum(BrandObjective, "brand_objective_t"), nullable=True
    )
    artifact_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    component_type: Mapped[str] = mapped_column(String(20))
    name: Mapped[str] = mapped_column(String(100))
    content: Mapped[dict] = mapped_column(JSONB, default=dict)
    usage_count: Mapped[int] = mapped_column(default=0)
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now())


class ToolAdapter(Base):
    """Catálogo de MCP servers disponibles (0005).

    Misma filosofía data-driven que pipeline_templates y format_specs: no
    hardcodear qué herramienta usa cada formato en el código del agente.
    mcp_server_name debe coincidir exactamente con la clave en mcp.servers
    de OpenClaw (el registry lo valida al arrancar).
    """

    __tablename__ = "tool_adapters"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    name: Mapped[str] = mapped_column(String(50), unique=True)
    mcp_server_name: Mapped[str] = mapped_column(String(100))
    execution_mode: Mapped[str] = mapped_column(String(50))
    requires_license: Mapped[str | None] = mapped_column(String(150), nullable=True)
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now())


class AssetJob(Base):
    """Un paso ejecutado en una cadena de herramientas para un
    ContentArtifact (0005) — la trazabilidad entre 'aprobado por QA' y
    'archivo final en disco'.

    status: 'pending' | 'running' | 'done' | 'failed'. cost_estimate
    registra créditos/costo si aplica (ElevenLabs, Veo 3).
    """

    __tablename__ = "asset_jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    artifact_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("content_artifacts.id", ondelete="CASCADE")
    )
    tool_adapter_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tool_adapters.id"))
    sequence_order: Mapped[int] = mapped_column(default=1)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    input_ref: Mapped[str | None] = mapped_column(Text, nullable=True)
    output_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    external_job_id: Mapped[str | None] = mapped_column(String(150), nullable=True)
    cost_estimate: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    phase: Mapped[str | None] = mapped_column(String(20), nullable=True)
    template_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("production_templates.id"), nullable=True
    )
    manifest_version: Mapped[int | None] = mapped_column(nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


class ProductionTemplate(Base):
    """Template estático por tipo de contenido y fase (0006).

    Única fuente de plantillas: la IA las selecciona y parametriza con el
    manifiesto, nunca las edita en runtime. content es JSONB libre (el raw
    embebido si el template no es JSON: otio/ass/cube/svg/txt).
    """

    __tablename__ = "production_templates"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    name: Mapped[str] = mapped_column(String(150), unique=True)
    content_type: Mapped[str] = mapped_column(String(20))
    phase: Mapped[str] = mapped_column(String(20))
    template_format: Mapped[str] = mapped_column(String(20))
    content: Mapped[dict] = mapped_column(JSONB, default=dict)
    version: Mapped[str] = mapped_column(String(20), default="1.0")
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now())


class ProductionManifest(Base):
    """Manifiesto versionado e inmutable por artefacto (0006).

    Única fuente de verdad por artefacto: cada versión es inmutable y se
    identifica por (artifact_id, version). El orquestador lee la última
    versión para materializar la cadena de herramientas.
    """

    __tablename__ = "production_manifests"
    __table_args__ = (UniqueConstraint("artifact_id", "version"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    artifact_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("content_artifacts.id", ondelete="CASCADE")
    )
    version: Mapped[int] = mapped_column()
    manifest: Mapped[dict] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class Project(Base):
    """Camino técnico de producción (0007, diseño §20.2).

    projects guarda solo metadatos + current_version (la versión activa del
    snapshot); el snapshot JSON editable vive versionado e inmutable en
    project_versions. status: 'borrador' | 'en_edicion' | 'aprobado' |
    'en_produccion' | 'listo' | 'fallido'.
    """

    __tablename__ = "projects"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    name: Mapped[str] = mapped_column(String(200))
    topic: Mapped[str] = mapped_column(Text)
    template_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("production_templates.id"), nullable=True
    )
    brand_objective: Mapped[BrandObjective | None] = mapped_column(
        _pg_enum(BrandObjective, "brand_objective_t"), nullable=True
    )
    artifact_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="borrador")
    current_version: Mapped[int] = mapped_column(default=0)
    storage_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now())


class ProjectVersion(Base):
    """Snapshot JSON editable e INMUTABLE por proyecto (0007, §20.2).

    Nunca se hace UPDATE: cada cambio (generación, approve, rollback) es un
    INSERT con versión nueva. El rollback restaura un snapshot anterior
    como versión nueva — el historial nunca se pierde.
    """

    __tablename__ = "project_versions"
    __table_args__ = (UniqueConstraint("project_id", "version"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE")
    )
    version: Mapped[int] = mapped_column()
    snapshot: Mapped[dict] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
