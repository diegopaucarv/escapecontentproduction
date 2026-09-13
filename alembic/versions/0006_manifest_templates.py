"""manifiesto versionado + templates estáticos por tipo de contenido y fase

Fase 1.5 del diseño (docs/diseno_produccion_multiformato.md §13.2): añade
la única fuente de plantillas (production_templates), el manifiesto
versionado e inmutable por artefacto (production_manifests), la estructura
de fases por formato (format_specs.phases) y la trazabilidad completa de
cada transformación (asset_jobs.phase/template_id/manifest_version).

Revision ID: 0006_manifest_templates
Revises: 0005_creative_production
Create Date: 2026-09-13
"""

from alembic import op

revision = "0006_manifest_templates"
down_revision = "0005_creative_production"
branch_labels = None
depends_on = None

UPGRADE_SQL = r"""
-- ---------------------------------------------------------------------
-- 1. production_templates — templates estáticos por tipo de contenido y
--    fase (única fuente de plantillas; la IA las selecciona, no las edita)
-- ---------------------------------------------------------------------

CREATE TABLE production_templates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(150) NOT NULL UNIQUE,
    content_type VARCHAR(20) NOT NULL,         -- 'audio' | 'video' | 'grafico'
    phase VARCHAR(20) NOT NULL,                -- 'preproduccion' | 'produccion' | 'postproduccion'
    template_format VARCHAR(20) NOT NULL,      -- 'json' | 'otio' | 'ass' | 'cube' | 'svg' | 'txt'
    content JSONB NOT NULL DEFAULT '{}'::jsonb,
    version VARCHAR(20) NOT NULL DEFAULT '1.0',
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TRIGGER trg_production_templates_updated_at
    BEFORE UPDATE ON production_templates
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ---------------------------------------------------------------------
-- 2. production_manifests — manifiesto versionado e inmutable (única
--    fuente de verdad por artefacto)
-- ---------------------------------------------------------------------

CREATE TABLE production_manifests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    artifact_id UUID NOT NULL REFERENCES content_artifacts(id) ON DELETE CASCADE,
    version INT NOT NULL,
    manifest JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (artifact_id, version)
);

CREATE INDEX ix_production_manifests_artifact ON production_manifests (artifact_id);

-- ---------------------------------------------------------------------
-- 3. format_specs — estructura de fases por formato (marco universal §1)
-- ---------------------------------------------------------------------

ALTER TABLE format_specs ADD COLUMN phases JSONB NOT NULL DEFAULT '{}'::jsonb;

-- ---------------------------------------------------------------------
-- 4. asset_jobs — trazabilidad completa de cada transformación
-- ---------------------------------------------------------------------

ALTER TABLE asset_jobs ADD COLUMN phase VARCHAR(20);
ALTER TABLE asset_jobs ADD COLUMN template_id UUID REFERENCES production_templates(id);
ALTER TABLE asset_jobs ADD COLUMN manifest_version INT;
"""

DOWNGRADE_SQL = r"""
ALTER TABLE asset_jobs DROP COLUMN IF EXISTS manifest_version;
ALTER TABLE asset_jobs DROP COLUMN IF EXISTS template_id;
ALTER TABLE asset_jobs DROP COLUMN IF EXISTS phase;

ALTER TABLE format_specs DROP COLUMN IF EXISTS phases;

DROP TABLE IF EXISTS production_manifests CASCADE;
DROP TABLE IF EXISTS production_templates CASCADE;
"""


def upgrade() -> None:
    op.execute(UPGRADE_SQL)


def downgrade() -> None:
    op.execute(DOWNGRADE_SQL)
