"""kag — embeddings de nombre de entidad para entity linking (0022)

Añade `name_embedding` (vector(768), mismo modelo que kag_chunks.embedding)
a kag_entities + índice HNSW (vector_cosine_ops). Habilita el entity linking
por similitud coseno en src/kag_query.py::match_entities_candidates: cuando
el match léxico (exacto/LIKE) falla — p. ej. query en otro idioma que el
grafo — se busca la entidad más cercana por embedding.

Los valores se pueblan al indexar (src/kag_ingest.py::_store_entities_relations)
y con el backfill `python -m src.kag_ingest --backfill-embeddings` para las
filas existentes.

Revision ID: 0022_kag_entity_embeddings
Revises: 0021_kag_prompt_user_templates
Create Date: 2026-09-15
"""

from alembic import op

revision = "0022_kag_entity_embeddings"
down_revision = "0021_kag_prompt_user_templates"
branch_labels = None
depends_on = None

UPGRADE_SQL = r"""
ALTER TABLE kag_entities ADD COLUMN name_embedding vector(768);
CREATE INDEX IF NOT EXISTS ix_kag_entities_name_embedding
    ON kag_entities USING hnsw (name_embedding vector_cosine_ops);
"""

DOWNGRADE_SQL = r"""
DROP INDEX IF EXISTS ix_kag_entities_name_embedding;
ALTER TABLE kag_entities DROP COLUMN IF EXISTS name_embedding;
"""


def upgrade() -> None:
    op.execute(UPGRADE_SQL)


def downgrade() -> None:
    op.execute(DOWNGRADE_SQL)
