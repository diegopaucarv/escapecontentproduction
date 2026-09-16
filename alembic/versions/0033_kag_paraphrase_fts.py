"""kag — FTS de paráfrasis sobre kag_chunks (0033)

Añade la columna generada paraphrase_tsv (tsvector con config 'simple',
agnóstica de idioma) sobre la columna paraphrase (Fase 4,
kag_chunk_paraphrase) y un índice GIN para la búsqueda léxica de
paráfrasis. Se fusiona en src/kag_query.py::hybrid_search como canal
adicional (KAG_PARAPHRASE_CHANNEL) junto a la densa (pgvector), la FTS de
content_tsv y el canal semántico de proposiciones.

Config 'simple' a propósito: el corpus es multilingüe (es/en/pt/de/fr) y
sin stemming es ideal para términos exactos — el mismo criterio que la
migración 0014 (content_tsv).

Revision ID: 0033_kag_paraphrase_fts
Revises: 0032_kag_summary_index
Create Date: 2026-09-15
"""

from alembic import op

revision = "0033_kag_paraphrase_fts"
down_revision = "0032_kag_summary_index"
branch_labels = None
depends_on = None

UPGRADE_SQL = r"""
ALTER TABLE kag_chunks
    ADD COLUMN IF NOT EXISTS paraphrase_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('simple', coalesce(paraphrase, ''))) STORED;

CREATE INDEX IF NOT EXISTS ix_kag_chunks_paraphrase_tsv
    ON kag_chunks USING GIN (paraphrase_tsv);
"""

DOWNGRADE_SQL = r"""
DROP INDEX IF EXISTS ix_kag_chunks_paraphrase_tsv;
ALTER TABLE kag_chunks DROP COLUMN IF EXISTS paraphrase_tsv;
"""


def upgrade() -> None:
    op.execute(UPGRADE_SQL)


def downgrade() -> None:
    op.execute(DOWNGRADE_SQL)
