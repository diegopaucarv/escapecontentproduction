"""project_id en content_artifacts — puente proyecto → pipeline (Fase 5)

Fase 5 del diseño (docs/diseno_produccion_multiformato.md §20.3): conecta la
entidad Project (0007) con el pipeline de producción existente. El artefacto
que materializa un proyecto referencia su project_id; el flujo agéntico
(produce_project) crea brief + artefacto + manifiesto y ejecuta la cadena de
herramientas del formato.

Revision ID: 0008_project_artifacts
Revises: 0007_projects
Create Date: 2026-09-13
"""

from alembic import op

revision = "0008_project_artifacts"
down_revision = "0007_projects"
branch_labels = None
depends_on = None

UPGRADE_SQL = r"""
ALTER TABLE content_artifacts ADD COLUMN project_id UUID REFERENCES projects(id);
CREATE INDEX ix_content_artifacts_project ON content_artifacts (project_id);
"""

DOWNGRADE_SQL = r"""
DROP INDEX IF EXISTS ix_content_artifacts_project;
ALTER TABLE content_artifacts DROP COLUMN IF EXISTS project_id;
"""


def upgrade() -> None:
    op.execute(UPGRADE_SQL)


def downgrade() -> None:
    op.execute(DOWNGRADE_SQL)
