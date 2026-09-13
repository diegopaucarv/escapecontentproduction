"""producción creativa local: format_specs, component_library, tool_adapters, asset_jobs

Capa de orquestación de herramientas (OpenClaw + MCP): catálogo de MCP
servers (tool_adapters), trazabilidad de ejecución (asset_jobs), specs de
formato con cadena de herramientas (format_specs.tool_chain) y biblioteca
de componentes reutilizables (component_library). Además añade
visual_spec y format_spec_id a content_artifacts.

Revision ID: 0005_creative_production
Revises: 0004_ai_modular_infra
Create Date: 2026-09-13
"""

from alembic import op

revision = "0005_creative_production"
down_revision = "0004_ai_modular_infra"
branch_labels = None
depends_on = None

UPGRADE_SQL = r"""
-- ---------------------------------------------------------------------
-- 1. format_specs — spec de formato por marca (data-driven, §2 del diseño)
-- ---------------------------------------------------------------------

CREATE TABLE format_specs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    brand_objective brand_objective_t NOT NULL,
    artifact_type VARCHAR(50) NOT NULL,
    structure JSONB NOT NULL DEFAULT '{}'::jsonb,
    constraints JSONB NOT NULL DEFAULT '{}'::jsonb,
    derivation_rules JSONB NOT NULL DEFAULT '[]'::jsonb,
    visual_requirements JSONB NOT NULL DEFAULT '{}'::jsonb,
    qa_checks JSONB NOT NULL DEFAULT '[]'::jsonb,
    tool_chain JSONB NOT NULL DEFAULT '[]'::jsonb,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (brand_objective, artifact_type)
);

CREATE TRIGGER trg_format_specs_updated_at
    BEFORE UPDATE ON format_specs
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ---------------------------------------------------------------------
-- 2. component_library — patrones reutilizables (visual/textual/estructura)
-- ---------------------------------------------------------------------

CREATE TABLE component_library (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    brand_objective brand_objective_t,
    artifact_type VARCHAR(50),
    component_type VARCHAR(20) NOT NULL,   -- 'visual' | 'textual' | 'estructura'
    name VARCHAR(100) NOT NULL,
    content JSONB NOT NULL DEFAULT '{}'::jsonb,
    usage_count INT NOT NULL DEFAULT 0,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TRIGGER trg_component_library_updated_at
    BEFORE UPDATE ON component_library
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ---------------------------------------------------------------------
-- 3. tool_adapters — catálogo de MCP servers (data-driven)
-- ---------------------------------------------------------------------

CREATE TABLE tool_adapters (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(50) NOT NULL UNIQUE,
    mcp_server_name VARCHAR(100) NOT NULL,
    execution_mode VARCHAR(50) NOT NULL,   -- 'local' | 'local_orchestration_cloud_inference' | 'remote_cloud'
    requires_license VARCHAR(150),
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TRIGGER trg_tool_adapters_updated_at
    BEFORE UPDATE ON tool_adapters
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ---------------------------------------------------------------------
-- 4. asset_jobs — trazabilidad de cada paso de la cadena
-- ---------------------------------------------------------------------

CREATE TABLE asset_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    artifact_id UUID NOT NULL REFERENCES content_artifacts(id) ON DELETE CASCADE,
    tool_adapter_id UUID NOT NULL REFERENCES tool_adapters(id),
    sequence_order INT NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',  -- pending | running | done | failed
    input_ref TEXT,
    output_path TEXT,
    external_job_id VARCHAR(150),
    cost_estimate JSONB,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    error TEXT
);

CREATE INDEX ix_asset_jobs_artifact ON asset_jobs (artifact_id);
CREATE INDEX ix_asset_jobs_status ON asset_jobs (status);

-- ---------------------------------------------------------------------
-- 5. content_artifacts: visual_spec + format_spec_id
-- ---------------------------------------------------------------------

ALTER TABLE content_artifacts ADD COLUMN visual_spec JSONB;
ALTER TABLE content_artifacts ADD COLUMN format_spec_id UUID REFERENCES format_specs(id);
"""

DOWNGRADE_SQL = r"""
ALTER TABLE content_artifacts DROP COLUMN IF EXISTS format_spec_id;
ALTER TABLE content_artifacts DROP COLUMN IF EXISTS visual_spec;

DROP TABLE IF EXISTS asset_jobs CASCADE;
DROP TABLE IF EXISTS tool_adapters CASCADE;
DROP TABLE IF EXISTS component_library CASCADE;
DROP TABLE IF EXISTS format_specs CASCADE;
"""


def upgrade() -> None:
    op.execute(UPGRADE_SQL)


def downgrade() -> None:
    op.execute(DOWNGRADE_SQL)
