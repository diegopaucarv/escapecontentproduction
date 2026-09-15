"""kag — user prompts versionados en prompt-as-code (0021)

Añade el campo `user_template` a `prompt_templates` (spec agnóstica) y a
`prompt_artifacts` (prompt compilado e inmutable), más `user_template_hash`
para la idempotencia del compilador (el artefacto solo se recompila si
cambia el SYSTEM renderizado O el USER template).

Con esto los USER prompts dejan de ser constantes en código: viven en la
spec (parametrizables y versionadas) y se congelan en el artefacto junto al
SYSTEM prompt.

Revision ID: 0021_kag_prompt_user_templates
Revises: 0020_kag_stages
Create Date: 2026-09-14
"""

from alembic import op

revision = "0021_kag_prompt_user_templates"
down_revision = "0020_kag_stages"
branch_labels = None
depends_on = None

UPGRADE_SQL = r"""
ALTER TABLE prompt_templates ADD COLUMN user_template TEXT NOT NULL DEFAULT '';
ALTER TABLE prompt_artifacts ADD COLUMN user_template TEXT NOT NULL DEFAULT '';
ALTER TABLE prompt_artifacts ADD COLUMN user_template_hash VARCHAR(64) NOT NULL DEFAULT '';
"""

DOWNGRADE_SQL = r"""
ALTER TABLE prompt_artifacts DROP COLUMN IF EXISTS user_template_hash;
ALTER TABLE prompt_artifacts DROP COLUMN IF EXISTS user_template;
ALTER TABLE prompt_templates DROP COLUMN IF EXISTS user_template;
"""


def upgrade() -> None:
    op.execute(UPGRADE_SQL)


def downgrade() -> None:
    op.execute(DOWNGRADE_SQL)
