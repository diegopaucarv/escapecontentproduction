"""kag — eliminación del pipeline proposicional (0025)

Unificación KAG: se elimina el pipeline proposicional (src/kag_propositional.py
y src/kag_agents.py) y sus 5 tablas. El pipeline clásico expandido
(src/kag_ingest.py + src/kag_query.py) ahora cubre la capa micro con
`kag_propositions` (0024) y el modo audited con los prompts kag_synthesis/
kag_contradictions/kag_sufficiency/kag_answer.

Revision ID: 0025_kag_drop_propositional
Revises: 0024_kag_propositions
Create Date: 2026-09-15
"""

from alembic import op

revision = "0025_kag_drop_propositional"
down_revision = "0024_kag_propositions"
branch_labels = None
depends_on = None

UPGRADE_SQL = r"""
DROP TABLE IF EXISTS topic_tree_nodes;
DROP TABLE IF EXISTS document_images;
DROP TABLE IF EXISTS propositional_chunks;
DROP TABLE IF EXISTS document_chapters;
DROP TABLE IF EXISTS documents;
"""

DOWNGRADE_SQL = r"""
-- No-op: las tablas proposicionales no se recrean (el pipeline se eliminó).
"""


def upgrade() -> None:
    op.execute(UPGRADE_SQL)


def downgrade() -> None:
    op.execute(DOWNGRADE_SQL)
