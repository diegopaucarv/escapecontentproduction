"""kag — atomicidad por etapa de ingesta (stage machine) (0020)

Añade la columna `stage` a `kag_documents` y `documents` para que la ingesta
sea atómica POR ETAPA (no solo por documento): cada etapa commitea su trabajo
y actualiza `stage`; al reanudar, se saltan las etapas ya completadas.

Etapas de `kag_documents` (pipeline legacy, src/kag_ingest.py):
  pending   -> fila insertada, nada persistido
  segmented -> chunks insertados (embedding NULL) — la segmentación (lenta)
               queda persistida y NO se repite al reanudar
  chunked   -> embeddings + entidades + relaciones
  figures   -> figuras indexadas
  ready     -> resumen + status='ready'

Etapas de `documents` (pipeline proposicional, src/kag_propositional.py):
  detected  -> MultibookFinderTool lo detectó (manifest), sin fila en DB
  analysis  -> ficha + capítulos (documents + document_chapters)
  chunks    -> propositional_chunks + embeddings
  topic_tree-> topic_tree_nodes
  images    -> document_images
  ready     -> status='ready'

Revision ID: 0020_kag_stages
Revises: 0019_kag_unified
Create Date: 2026-09-14
"""

from alembic import op

revision = "0020_kag_stages"
down_revision = "0019_kag_unified"
branch_labels = None
depends_on = None

UPGRADE_SQL = r"""
ALTER TABLE kag_documents ADD COLUMN stage VARCHAR(20) NOT NULL DEFAULT 'pending';
ALTER TABLE documents ADD COLUMN stage VARCHAR(20) NOT NULL DEFAULT 'detected';
"""

DOWNGRADE_SQL = r"""
ALTER TABLE documents DROP COLUMN IF EXISTS stage;
ALTER TABLE kag_documents DROP COLUMN IF EXISTS stage;
"""


def upgrade() -> None:
    op.execute(UPGRADE_SQL)


def downgrade() -> None:
    op.execute(DOWNGRADE_SQL)
