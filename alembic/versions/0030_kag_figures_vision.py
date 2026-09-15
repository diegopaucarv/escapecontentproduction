"""kag — visión de figuras con FAQ Reverse HyDE (Fase 2b) (0030)

La Fase 2b (`_index_figures` nuevo) describe cada figura con el VLM y
persiste, además de la descripción clásica, la clasificación tipológica, la
contribución epistémica, preguntas FAQ Reverse HyDE (para recuperación por
pregunta) y las entidades asociadas. Esta migración amplía `kag_figures`
(tabla de 0013: id, doc_id, chunk_id, image_path, caption, description,
created_at):

  - `image_type` (VARCHAR(50)): diagram | chart_or_plot | flowchart |
    conceptual_illustration | screenshot | table_image | photograph.
  - `dense_visual_description` (TEXT): transcripción exacta de ejes,
    leyendas, flujos y elementos visuales (lo que el ojo ve).
  - `epistemic_contribution` (TEXT): qué aporta la figura al conocimiento
    del documento (no redundante con el texto).
  - `faq_indexing` (JSONB): 3-5 preguntas Reverse HyDE que la figura puede
    responder (recuperación por pregunta).
  - `associated_entities` (JSONB): entidades que aparecen en la figura.
  - `anchor_line` (INT): línea física del archivo donde se referencia la
    imagen (0 si no se pudo resolver).

Índice GIN sobre `faq_indexing` (jsonb_path_ops) para acelerar la
recuperación por pregunta (operador @>).

Revision ID: 0030_kag_figures_vision
Revises: 0029_kag_document_rows
Create Date: 2026-09-15
"""

from alembic import op

revision = "0030_kag_figures_vision"
down_revision = "0029_kag_document_rows"
branch_labels = None
depends_on = None

UPGRADE_SQL = """
ALTER TABLE kag_figures ADD COLUMN IF NOT EXISTS image_type VARCHAR(50) NOT NULL DEFAULT '';
ALTER TABLE kag_figures ADD COLUMN IF NOT EXISTS dense_visual_description TEXT NOT NULL DEFAULT '';
ALTER TABLE kag_figures ADD COLUMN IF NOT EXISTS epistemic_contribution TEXT NOT NULL DEFAULT '';
ALTER TABLE kag_figures ADD COLUMN IF NOT EXISTS faq_indexing JSONB NOT NULL DEFAULT '[]'::jsonb;
ALTER TABLE kag_figures ADD COLUMN IF NOT EXISTS associated_entities JSONB NOT NULL DEFAULT '[]'::jsonb;
ALTER TABLE kag_figures ADD COLUMN IF NOT EXISTS anchor_line INT NOT NULL DEFAULT 0;

CREATE INDEX IF NOT EXISTS ix_kag_figures_faq_gin
    ON kag_figures USING gin (faq_indexing jsonb_path_ops);
"""

DOWNGRADE_SQL = """
DROP INDEX IF EXISTS ix_kag_figures_faq_gin;

ALTER TABLE kag_figures DROP COLUMN IF EXISTS anchor_line;
ALTER TABLE kag_figures DROP COLUMN IF EXISTS associated_entities;
ALTER TABLE kag_figures DROP COLUMN IF EXISTS faq_indexing;
ALTER TABLE kag_figures DROP COLUMN IF EXISTS epistemic_contribution;
ALTER TABLE kag_figures DROP COLUMN IF EXISTS dense_visual_description;
ALTER TABLE kag_figures DROP COLUMN IF EXISTS image_type;
"""


def upgrade() -> None:
    op.execute(UPGRADE_SQL)


def downgrade() -> None:
    op.execute(DOWNGRADE_SQL)
