"""kag — búsqueda híbrida FTS (tsvector + GIN) sobre kag_chunks (0014)

Añade la columna generada content_tsv (tsvector con config 'simple',
agnóstica de idioma) y un índice GIN para la búsqueda léxica
(ts_rank_cd, BM25-like) que se fusiona con la búsqueda densa (pgvector)
mediante Reciprocal Rank Fusion (RRF) en src/kag_query.py::hybrid_search.

Config 'simple' a propósito: el corpus es multilingüe (es/en/pt/de/fr) y
sin stemming es ideal para términos exactos (acrónimos, códigos, nombres
propios) — el caso de uso de la precisión léxica.

Revision ID: 0014_kag_fts
Revises: 0013_kag
Create Date: 2026-09-13
"""

from alembic import op

revision = "0014_kag_fts"
down_revision = "0013_kag"
branch_labels = None
depends_on = None

UPGRADE_SQL = r"""
ALTER TABLE kag_chunks
    ADD COLUMN IF NOT EXISTS content_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('simple', coalesce(content, ''))) STORED;

CREATE INDEX IF NOT EXISTS ix_kag_chunks_content_tsv
    ON kag_chunks USING GIN (content_tsv);
"""

DOWNGRADE_SQL = r"""
DROP INDEX IF EXISTS ix_kag_chunks_content_tsv;
ALTER TABLE kag_chunks DROP COLUMN IF EXISTS content_tsv;
"""


def upgrade() -> None:
    op.execute(UPGRADE_SQL)


def downgrade() -> None:
    op.execute(DOWNGRADE_SQL)
