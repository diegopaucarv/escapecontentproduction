"""kag — document_id a TEXT (0031)

El LLM de separación (Fase 1, kag_document_separation) genera `document_id`
libremente y a veces usa el nombre completo del archivo como prefijo
(p. ej. '100-The Handbook of Culture and Psychology -- ... -- Oxford
University Pre_doc_001'), que excede VARCHAR(100) y rompe el INSERT con
StringDataRightTruncation. En PostgreSQL no hay diferencia de rendimiento
entre VARCHAR(n) y TEXT (ambos varlena); como el id lo genera el LLM sin
límite garantizado, se pasa a TEXT (sin constraint de longitud). El código
sanitiza/trunca el id con hash (ver _sanitize_document_id en
src/kag_ingest.py) para mantener ids legibles y estables.

Revision ID: 0031_kag_document_id_width
Revises: 0030_kag_figures_vision
Create Date: 2026-09-15
"""

from alembic import op

revision = "0031_kag_document_id_width"
down_revision = "0030_kag_figures_vision"
branch_labels = None
depends_on = None

UPGRADE_SQL = """
ALTER TABLE kag_documents ALTER COLUMN document_id TYPE TEXT;
"""

DOWNGRADE_SQL = """
ALTER TABLE kag_documents ALTER COLUMN document_id TYPE VARCHAR(100);
"""


def upgrade() -> None:
    op.execute(UPGRADE_SQL)


def downgrade() -> None:
    op.execute(DOWNGRADE_SQL)
