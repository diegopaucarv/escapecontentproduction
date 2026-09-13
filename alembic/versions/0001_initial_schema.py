"""schema inicial completa (equivalente a sql/001_init.sql, ver ese
archivo como copia de referencia legible — esta migración es la fuente
de verdad ejecutable a partir de aquí).

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-09-12
"""
from alembic import op

revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None

UPGRADE_SQL = r"""
-- =====================================================================
-- Pipeline Unificado de Producción de Contenidos — ESCAPE / Ergalia
-- 001_init.sql — esquema base (versionar con Alembic a partir de aquí)
-- =====================================================================

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto; -- gen_random_uuid()

-- ---------------------------------------------------------------------
-- 0. Vocabularios controlados (canon §3, §9.1) — ENUM en vez de VARCHAR
--    libre, para que la base rechace valores que no están en el canon.
-- ---------------------------------------------------------------------

CREATE TYPE brand_objective_t AS ENUM ('ESCAPE_SOCIAL', 'ERGALIA_COMERCIAL', 'HIBRIDO');

CREATE TYPE content_bucket_t AS ENUM (
    'difusion_cientifica', 'comunidad_intelectual', 'debate_informado',
    'herramienta_gratuita', 'formacion_aplicada', 'servicio_formal',
    'caso_autoridad', 'artefacto_fisico_digital', 'captacion_directa'
);

CREATE TYPE segment_client_t AS ENUM ('S1','S2','S3','S4','S5','S6','S7');
CREATE TYPE segment_community_t AS ENUM ('C1','C2','C3','C4');

CREATE TYPE risk_level_t AS ENUM ('bajo', 'medio', 'alto');

CREATE TYPE brief_status_t AS ENUM (
    'idea', 'brief', 'generando', 'revision', 'aprobado',
    'produccion', 'publicado', 'aprendido', 'archivado'
);

CREATE TYPE route_decision_t AS ENUM ('repetitivo', 'solucion_previa', 'nueva_solucion');
CREATE TYPE production_route_t AS ENUM ('fast', 'complete');

CREATE TYPE artifact_type_t AS ENUM (
    'post', 'video_corto', 'video_largo', 'carrusel', 'one_pager',
    'white_paper', 'pdf_recurso', 'broadcast', 'poster_qr',
    'webinar', 'propuesta', 'app_herramienta', 'newsletter', 'hilo'
);

CREATE TYPE repurpose_link_t AS ENUM ('derived_from', 'expanded_to', 'translated_to', 'summarized_from');

CREATE TYPE user_role_t AS ENUM ('lider', 'equipo', 'externo');

-- ---------------------------------------------------------------------
-- 1. Función utilitaria para mantener updated_at sin lógica en la app
-- ---------------------------------------------------------------------

CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ---------------------------------------------------------------------
-- 2. RBAC mínimo (Líder vs Equipo vs Externo — 🟨🟦🟩 del diagrama)
-- ---------------------------------------------------------------------

CREATE TABLE app_users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    full_name VARCHAR(150) NOT NULL,
    email VARCHAR(150) UNIQUE NOT NULL,
    role user_role_t NOT NULL,
    brand_scope brand_objective_t, -- NULL = acceso a ambas marcas
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------
-- 3. Plantillas de pipeline por bucket (config-driven, no hardcodeado)
-- ---------------------------------------------------------------------

CREATE TABLE pipeline_templates (
    id SERIAL PRIMARY KEY,
    brand_objective brand_objective_t NOT NULL,
    content_bucket content_bucket_t NOT NULL,
    default_route production_route_t,
    checklist JSONB NOT NULL DEFAULT '[]'::jsonb, -- ítems del checklist de publicación de marca
    novelty_weights JSONB NOT NULL DEFAULT '{}'::jsonb, -- pesos configurables del §4.4
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (brand_objective, content_bucket)
);

CREATE TRIGGER trg_pipeline_templates_updated_at
    BEFORE UPDATE ON pipeline_templates
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ---------------------------------------------------------------------
-- 4. Content Brief — entidad central (ver pipeline_unificado §3, todos
--    los bloques A-H). Campos de texto largo quedan en TEXT; el cuerpo
--    narrativo completo puede además espejarse como .md en /briefs/.
-- ---------------------------------------------------------------------

CREATE TABLE content_briefs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id UUID REFERENCES app_users(id),
    status brief_status_t NOT NULL DEFAULT 'idea',

    -- Bloque B — origen y contexto
    resumen TEXT,
    insight_core TEXT,
    pitch_15s TEXT,
    prior_attempts TEXT,
    risks TEXT,
    novelty_indicators JSONB,
    suggested_product_type VARCHAR(100),
    org_priorities_contrast TEXT,

    -- Bloque C — jerarquía canónica
    brand_objective brand_objective_t NOT NULL,
    phase_number SMALLINT CHECK (phase_number BETWEEN 1 AND 4),
    segment_client segment_client_t,
    segment_community segment_community_t,
    interlocutor_profile VARCHAR(100),
    subprofile VARCHAR(150),
    audience_tier VARCHAR(50),
    need_id VARCHAR(50),
    need TEXT,
    change_hypothesis TEXT,
    CONSTRAINT chk_segment_present CHECK (segment_client IS NOT NULL OR segment_community IS NOT NULL),

    -- Bloque D — bucket y oferta
    content_bucket content_bucket_t NOT NULL,
    service_category VARCHAR(100),
    product_anchor VARCHAR(150),
    entry_offer VARCHAR(150),

    -- Bloque E — artefacto y canal
    artifact_type artifact_type_t,
    channel VARCHAR(50),
    channel_role VARCHAR(50),
    cta VARCHAR(200),
    landing VARCHAR(300),
    funnel_stage VARCHAR(50),

    -- Bloque F — geografía e idioma
    language VARCHAR(10) DEFAULT 'es',
    geography_content VARCHAR(100),
    geography_sales VARCHAR(100),

    -- Bloque G — gobernanza, evidencia y reutilización
    evidence_source TEXT,
    risk_level risk_level_t NOT NULL DEFAULT 'bajo',
    debate_governance JSONB,
    validation_required JSONB, -- ej. ["factual", "legal", "anonimizacion"]
    repurpose_plan JSONB,      -- lista de variantes planeadas: [{artifact_type, channel}, ...]

    -- Bloque H — métricas y enrutamiento
    metric_primary VARCHAR(100),
    metric_secondary VARCHAR(100),
    novelty_score SMALLINT CHECK (novelty_score BETWEEN 0 AND 10),
    route_decision route_decision_t,
    production_route production_route_t,
    pipeline_template_id INT REFERENCES pipeline_templates(id),

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_briefs_status ON content_briefs(status);
CREATE INDEX idx_briefs_brand_bucket ON content_briefs(brand_objective, content_bucket);
CREATE INDEX idx_briefs_route ON content_briefs(route_decision, production_route);

CREATE TRIGGER trg_content_briefs_updated_at
    BEFORE UPDATE ON content_briefs
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ---------------------------------------------------------------------
-- 5. Content Artifacts — piezas realmente producidas (separado del
--    índice vectorial: aquí vive el metadato operativo/publicable).
-- ---------------------------------------------------------------------

CREATE TABLE content_artifacts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    brief_id UUID NOT NULL REFERENCES content_briefs(id) ON DELETE CASCADE,
    artifact_type artifact_type_t NOT NULL,
    channel VARCHAR(50) NOT NULL,
    storage_path VARCHAR(300), -- ruta en el bucket/objeto (Supabase Storage, Nextcloud, etc.)
    status VARCHAR(30) NOT NULL DEFAULT 'borrador',
    published_at TIMESTAMPTZ,
    utm_campaign VARCHAR(150),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_artifacts_brief ON content_artifacts(brief_id);

CREATE TRIGGER trg_content_artifacts_updated_at
    BEFORE UPDATE ON content_artifacts
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ---------------------------------------------------------------------
-- 6. Repurpose links — trazabilidad real de derivación multi-formato
--    (el mecanismo central de "reutilización" que motivó todo esto)
-- ---------------------------------------------------------------------

CREATE TABLE repurpose_links (
    id SERIAL PRIMARY KEY,
    source_artifact_id UUID NOT NULL REFERENCES content_artifacts(id) ON DELETE CASCADE,
    target_artifact_id UUID NOT NULL REFERENCES content_artifacts(id) ON DELETE CASCADE,
    link_type repurpose_link_t NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT chk_no_self_link CHECK (source_artifact_id <> target_artifact_id)
);

CREATE INDEX idx_repurpose_source ON repurpose_links(source_artifact_id);
CREATE INDEX idx_repurpose_target ON repurpose_links(target_artifact_id);

-- ---------------------------------------------------------------------
-- 7. Artifact Library — índice vectorial para RAG / Context Packs
--    (dimensión configurable: Voyage-3-large=1024 por defecto; ver
--    src/config.py — EMBEDDING_DIM. No asumir 1536 de OpenAI.)
-- ---------------------------------------------------------------------

CREATE TABLE artifact_library (
    id SERIAL PRIMARY KEY,
    artifact_id UUID REFERENCES content_artifacts(id) ON DELETE CASCADE,
    brief_id UUID REFERENCES content_briefs(id) ON DELETE CASCADE,
    content_summary TEXT NOT NULL,
    embedding vector(1024),
    retention_24h DECIMAL(5,2),
    conversion_30d DECIMAL(5,2),
    indexed_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_artifact_library_embedding
    ON artifact_library USING hnsw (embedding vector_cosine_ops);

-- ---------------------------------------------------------------------
-- 8. Production logs — minería de procesos (LeadTime, CycleTime,
--    Time-in-Stage, Rework Rate). lead_time_minutes se calcula solo.
-- ---------------------------------------------------------------------

CREATE TABLE pipeline_stages (
    stage_code VARCHAR(50) PRIMARY KEY,
    stage_label VARCHAR(150) NOT NULL,
    phase_group VARCHAR(50) NOT NULL -- brief | discovery | probe | fulldev | predeploy | deploy | kaizen | learning
);

INSERT INTO pipeline_stages (stage_code, stage_label, phase_group) VALUES
    ('brief', 'Recepción / Brief', 'brief'),
    ('alignment', 'Alineamiento estratégico', 'brief'),
    ('discovery', 'Discovery editorial', 'discovery'),
    ('sprint2d', 'Sprint de diseño y prueba', 'discovery'),
    ('probe', 'Fast-Probe', 'probe'),
    ('fulldev', 'Full Development', 'fulldev'),
    ('predeploy', 'Pre-Deploy Gate', 'predeploy'),
    ('deploy', 'Deploy', 'deploy'),
    ('kaizen', 'Kaizen / Producción recurrente', 'kaizen'),
    ('learning', 'Learning & Automation', 'learning');

CREATE TABLE production_logs (
    id SERIAL PRIMARY KEY,
    brief_id UUID NOT NULL REFERENCES content_briefs(id) ON DELETE CASCADE,
    stage_code VARCHAR(50) NOT NULL REFERENCES pipeline_stages(stage_code),
    entered_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    exited_at TIMESTAMPTZ,
    rework_count INT NOT NULL DEFAULT 0,
    lead_time_minutes INT GENERATED ALWAYS AS (
        CASE WHEN exited_at IS NOT NULL
             THEN CEIL(EXTRACT(EPOCH FROM (exited_at - entered_at)) / 60)::INT
             ELSE NULL END
    ) STORED
);

CREATE INDEX idx_production_logs_brief ON production_logs(brief_id);
CREATE INDEX idx_production_logs_stage ON production_logs(stage_code);

-- ---------------------------------------------------------------------
-- 9. Telemetría entrante (webhooks de UTM / retención / conversión)
--    + notificación async vía LISTEN/NOTIFY (sin Kafka/Redis)
-- ---------------------------------------------------------------------

CREATE TABLE telemetry_events (
    id BIGSERIAL PRIMARY KEY,
    artifact_id UUID REFERENCES content_artifacts(id) ON DELETE SET NULL,
    event_type VARCHAR(50) NOT NULL, -- click | view | retention_24h | conversion_30d | etc.
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    received_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE OR REPLACE FUNCTION notify_telemetry_event()
RETURNS TRIGGER AS $$
BEGIN
    PERFORM pg_notify('telemetry_channel', NEW.id::text);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_telemetry_notify
    AFTER INSERT ON telemetry_events
    FOR EACH ROW EXECUTE FUNCTION notify_telemetry_event();

-- ---------------------------------------------------------------------
-- 10. Datos semilla mínimos para los 9 buckets del canon (ambas marcas)
--     — evita arrancar con la tabla de plantillas vacía.
-- ---------------------------------------------------------------------

INSERT INTO pipeline_templates (brand_objective, content_bucket, default_route, checklist, novelty_weights) VALUES
    ('ESCAPE_SOCIAL', 'difusion_cientifica', 'fast',
        '["fuente_verificable", "gancho_15s_ok", "cta_unico", "anonimizacion_no_aplica"]',
        '{"bucket_nuevo": 3, "formato_nuevo": 2, "canal_nuevo": 2, "angulo_nuevo": 3}'),
    ('ERGALIA_COMERCIAL', 'caso_autoridad', 'complete',
        '["anonimizacion_verificada", "fuente_verificable", "cta_unico", "revision_legal"]',
        '{"bucket_nuevo": 3, "formato_nuevo": 2, "canal_nuevo": 2, "angulo_nuevo": 3}')
ON CONFLICT (brand_objective, content_bucket) DO NOTHING;

"""

DOWNGRADE_SQL = r"""
DROP TABLE IF EXISTS telemetry_events CASCADE;
DROP TABLE IF EXISTS production_logs CASCADE;
DROP TABLE IF EXISTS pipeline_stages CASCADE;
DROP TABLE IF EXISTS artifact_library CASCADE;
DROP TABLE IF EXISTS repurpose_links CASCADE;
DROP TABLE IF EXISTS content_artifacts CASCADE;
DROP TABLE IF EXISTS content_briefs CASCADE;
DROP TABLE IF EXISTS pipeline_templates CASCADE;
DROP TABLE IF EXISTS app_users CASCADE;

DROP FUNCTION IF EXISTS notify_telemetry_event() CASCADE;
DROP FUNCTION IF EXISTS set_updated_at() CASCADE;

DROP TYPE IF EXISTS user_role_t;
DROP TYPE IF EXISTS repurpose_link_t;
DROP TYPE IF EXISTS artifact_type_t;
DROP TYPE IF EXISTS production_route_t;
DROP TYPE IF EXISTS route_decision_t;
DROP TYPE IF EXISTS brief_status_t;
DROP TYPE IF EXISTS risk_level_t;
DROP TYPE IF EXISTS segment_community_t;
DROP TYPE IF EXISTS segment_client_t;
DROP TYPE IF EXISTS content_bucket_t;
DROP TYPE IF EXISTS brand_objective_t;
"""


def upgrade() -> None:
    op.execute(UPGRADE_SQL)


def downgrade() -> None:
    op.execute(DOWNGRADE_SQL)
