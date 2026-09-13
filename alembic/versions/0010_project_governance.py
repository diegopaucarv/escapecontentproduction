"""Gobernanza real en projects — segmento y riesgo declarados (0009/0010)

Cierra el hueco de gobernanza de la Fase 5: produce_project creaba el
brief compañero con S1/riesgo bajo/pre-aprobado hardcodeados, saltándose
Alignment, Novelty Routing, Discovery y el Gatekeeper. Ahora el proyecto
declara segment_client y risk_level reales (ProjectCreate), y
produce_project los usa para el brief + el Gatekeeper — nunca valores
de utilería.

Revision ID: 0010_project_governance
Revises: 0009_brief_requires_acceptance
Create Date: 2026-09-13
"""

from alembic import op

revision = "0010_project_governance"
down_revision = "0009_brief_requires_acceptance"
branch_labels = None
depends_on = None

UPGRADE_SQL = r"""
ALTER TABLE projects ADD COLUMN segment_client VARCHAR(2);
ALTER TABLE projects ADD COLUMN risk_level risk_level_t;
ALTER TABLE projects ADD COLUMN companion_brief_id UUID REFERENCES content_briefs(id);
"""

DOWNGRADE_SQL = r"""
ALTER TABLE projects DROP COLUMN IF EXISTS companion_brief_id;
ALTER TABLE projects DROP COLUMN IF EXISTS risk_level;
ALTER TABLE projects DROP COLUMN IF EXISTS segment_client;
"""


def upgrade() -> None:
    op.execute(UPGRADE_SQL)


def downgrade() -> None:
    op.execute(DOWNGRADE_SQL)
