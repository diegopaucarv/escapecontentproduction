"""proyectos + versionado de snapshots (rollback)

Fase 4 del diseño (docs/diseno_produccion_multiformato.md §20): añade la
entidad Project (projects) como camino técnico de producción — el snapshot
JSON editable por proyecto con historial inmutable de versiones
(project_versions). El rollback restaura un snapshot anterior como versión
nueva: el historial nunca se pierde.

Revision ID: 0007_projects
Revises: 0006_manifest_templates
Create Date: 2026-09-13
"""

from alembic import op

revision = "0007_projects"
down_revision = "0006_manifest_templates"
branch_labels = None
depends_on = None

UPGRADE_SQL = r"""
-- ---------------------------------------------------------------------
-- 1. projects — camino técnico de producción (§20.2): el snapshot JSON
--    editable vive versionado en project_versions; projects solo guarda
--    metadatos + current_version (la versión activa del snapshot).
-- ---------------------------------------------------------------------

CREATE TABLE projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(200) NOT NULL,
    topic TEXT NOT NULL,
    template_id UUID REFERENCES production_templates(id),
    brand_objective brand_objective_t,
    artifact_type VARCHAR(50),
    status VARCHAR(30) NOT NULL DEFAULT 'borrador',
    current_version INT NOT NULL DEFAULT 0,
    storage_path TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TRIGGER trg_projects_updated_at
    BEFORE UPDATE ON projects
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ---------------------------------------------------------------------
-- 2. project_versions — historial INMUTABLE de snapshots: nunca UPDATE,
--    solo INSERT con versión nueva (rollback = restaurar como versión
--    nueva, el historial nunca se pierde).
-- ---------------------------------------------------------------------

CREATE TABLE project_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    version INT NOT NULL,
    snapshot JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (project_id, version)
);

CREATE INDEX ix_project_versions_project ON project_versions (project_id);
"""

DOWNGRADE_SQL = r"""
DROP TABLE IF EXISTS project_versions CASCADE;
DROP TABLE IF EXISTS projects CASCADE;
"""


def upgrade() -> None:
    op.execute(UPGRADE_SQL)


def downgrade() -> None:
    op.execute(DOWNGRADE_SQL)
