"""requires_user_acceptance en content_briefs — auditoría de la filosofía 0007

Filosofía 0007: el LLM es la opción siempre presente, y la degradación a
decisiones deterministas SIEMPRE requiere aceptación explícita del usuario.
Este flag persiste si la última decisión (alineamiento/refuerzo) fue
determinista y quedó pendiente de aceptación; sirve para auditoría y para
que el pipeline sepa que debe pausar hasta que el usuario acepte. No se
limpia al aprobar (auditoría); se sobrescribe en el siguiente alineamiento.

Revision ID: 0009_brief_requires_acceptance
Revises: 0008_project_artifacts
Create Date: 2026-09-13
"""

from alembic import op

revision = "0009_brief_requires_acceptance"
down_revision = "0008_project_artifacts"
branch_labels = None
depends_on = None

UPGRADE_SQL = r"""
ALTER TABLE content_briefs ADD COLUMN requires_user_acceptance BOOLEAN NOT NULL DEFAULT FALSE;
"""

DOWNGRADE_SQL = r"""
ALTER TABLE content_briefs DROP COLUMN IF EXISTS requires_user_acceptance;
"""


def upgrade() -> None:
    op.execute(UPGRADE_SQL)


def downgrade() -> None:
    op.execute(DOWNGRADE_SQL)
