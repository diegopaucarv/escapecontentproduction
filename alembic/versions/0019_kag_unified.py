"""kag — llamada unificada de análisis documental (0019)

Consolida la ficha documental (ISO 25964 + Library of Congress), la
estructura de capítulos, el resumen ejecutivo global y las entidades
rectoras en UNA llamada LLM por documento (antes: dos llamadas separadas
en los pasos 1 y 2 de la ingesta proposicional).

Cambios de esquema:

  - documents.summary: resumen ejecutivo global del documento (sustituye
    la fase reduce de Qwen 2.5). Columna nueva con default '' para que
    los INSERTs existentes sigan funcionando.
  - document_images.faq_indexing: índice GIN (jsonb_path_ops) sobre el
    JSONB de preguntas Reverse HyDE — consultas de FAQ por documento.

Revision ID: 0019_kag_unified
Revises: 0018_kag_propositional
Create Date: 2026-09-14
"""

from alembic import op

revision = "0019_kag_unified"
down_revision = "0018_kag_propositional"
branch_labels = None
depends_on = None

UPGRADE_SQL = r"""
-- ---------------------------------------------------------------------
-- documents.summary: resumen ejecutivo global (fase reduce unificada)
-- ---------------------------------------------------------------------
ALTER TABLE documents ADD COLUMN summary TEXT NOT NULL DEFAULT '';

-- ---------------------------------------------------------------------
-- document_images.faq_indexing: GIN para consultas Reverse HyDE
-- ---------------------------------------------------------------------
CREATE INDEX ix_document_images_faq_gin
    ON document_images USING gin (faq_indexing jsonb_path_ops);
"""

DOWNGRADE_SQL = r"""
DROP INDEX IF EXISTS ix_document_images_faq_gin;
ALTER TABLE documents DROP COLUMN IF EXISTS summary;
"""


def upgrade() -> None:
    op.execute(UPGRADE_SQL)


def downgrade() -> None:
    op.execute(DOWNGRADE_SQL)
