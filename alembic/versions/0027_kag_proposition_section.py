"""kag — section_path por proposición para agrupación por capítulo (0027)

Añade `section_path` a `kag_propositions` (VARCHAR nullable): la proposición
se asocia a su capítulo/sección directamente (agrupación doc→capítulo de la
extracción), no solo vía chunk_id. La extracción (src/kag_ingest.py) la
rellena con el section_path de la división del output del LLM (o el del
chunk asignado).

Revision ID: 0027_kag_proposition_section
Revises: 0026_kag_chunk_content_hash
Create Date: 2026-09-15
"""

from alembic import op

revision = "0027_kag_proposition_section"
down_revision = "0026_kag_chunk_content_hash"
branch_labels = None
depends_on = None

UPGRADE_SQL = r"""
ALTER TABLE kag_propositions ADD COLUMN IF NOT EXISTS section_path VARCHAR(512);
CREATE INDEX IF NOT EXISTS ix_kag_propositions_doc_section
    ON kag_propositions (doc_id, section_path);
"""

DOWNGRADE_SQL = r"""
DROP INDEX IF EXISTS ix_kag_propositions_doc_section;
ALTER TABLE kag_propositions DROP COLUMN IF EXISTS section_path;
"""


def upgrade() -> None:
    op.execute(UPGRADE_SQL)


def downgrade() -> None:
    op.execute(DOWNGRADE_SQL)
