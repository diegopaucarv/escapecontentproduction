"""kag — ensanchar document_id a VARCHAR(255) (0031)

El LLM de separación (Fase 1, kag_document_separation) genera `document_id`
libremente y a veces usa el nombre completo del archivo como prefijo
(p. ej. '100-The Handbook of Culture and Psychology -- ... -- Oxford
University Pre_doc_001'), que excede VARCHAR(100) y rompe el INSERT con
StringDataRightTruncation. Se ensancha la columna a VARCHAR(255) (el código
también sanitiza/trunca el id con hash, ver _sanitize_document_id en
src/kag_ingest.py — este ALTER es la red de seguridad de la DB).

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
ALTER TABLE kag_documents ALTER COLUMN document_id TYPE VARCHAR(255);
"""

DOWNGRADE_SQL = """
ALTER TABLE kag_documents ALTER COLUMN document_id TYPE VARCHAR(100);
"""


def upgrade() -> None:
    op.execute(UPGRADE_SQL)


def downgrade() -> None:
    op.execute(DOWNGRADE_SQL)
