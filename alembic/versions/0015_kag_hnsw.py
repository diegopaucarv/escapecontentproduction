"""kag — índice HNSW sobre kag_chunks.embedding (0015)

Acelera la búsqueda vectorial (pgvector <=>, coseno) en consulta. Sin
índice, vector_search hace un scan secuencial sobre todos los vectores;
con HNSW (vector_cosine_ops) la búsqueda es aproximada pero órdenes de
magnitud más rápida en corpus grandes.

No cambia CUÁNDO se calculan los embeddings (siguen siendo una vez al
indexar, ver src/kag_ingest.py::index_document) — solo acelera la
recuperación en src/kag_query.py::vector_search.

Revision ID: 0015_kag_hnsw
Revises: 0014_kag_fts
Create Date: 2026-09-13
"""

from alembic import op

revision = "0015_kag_hnsw"
down_revision = "0014_kag_fts"
branch_labels = None
depends_on = None

UPGRADE_SQL = r"""
CREATE INDEX IF NOT EXISTS ix_kag_chunks_embedding
    ON kag_chunks USING hnsw (embedding vector_cosine_ops);
"""

DOWNGRADE_SQL = r"""
DROP INDEX IF EXISTS ix_kag_chunks_embedding;
"""


def upgrade() -> None:
    op.execute(UPGRADE_SQL)


def downgrade() -> None:
    op.execute(DOWNGRADE_SQL)
