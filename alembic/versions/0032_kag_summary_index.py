"""kag — índice de resúmenes jerárquicos (0032)

Persiste los resúmenes que `summarize_document` ya calcula (fase map por
secciones H1/H2 + reduce final) en `summary_index`, con dos niveles:
'document' (resumen final, padre) y 'section' (resumen de cada sección,
hijo con parent_id → fila documento). Cada fila lleva embedding vector(768)
(recuperación semántica) y tsv generado con to_tsvector('simple') — el
corpus es multilingüe (en/es) y el FTS del código usa 'simple' a propósito
(ver fts_search en src/kag_query.py). La jerarquía real del sistema son
capítulos (kag_chapters.chapter_id, TEXT), no section_path (eliminado).

Revision ID: 0032_kag_summary_index
Revises: 0031_kag_document_id_width
Create Date: 2026-09-15
"""

from alembic import op

revision = "0032_kag_summary_index"
down_revision = "0031_kag_document_id_width"
branch_labels = None
depends_on = None

UPGRADE_SQL = r"""
CREATE TABLE IF NOT EXISTS summary_index (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id INT NOT NULL REFERENCES kag_documents(id) ON DELETE CASCADE,
    level TEXT NOT NULL CHECK (level IN ('section', 'document')),
    chapter_id TEXT,
    parent_id UUID REFERENCES summary_index(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    embedding vector(768),
    tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('simple', text)) STORED,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_summary_index_doc_id
    ON summary_index (doc_id);
CREATE INDEX IF NOT EXISTS ix_summary_index_embedding
    ON summary_index USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS ix_summary_index_tsv
    ON summary_index USING gin (tsv);
"""

DOWNGRADE_SQL = r"""
DROP TABLE IF EXISTS summary_index;
"""


def upgrade() -> None:
    op.execute(UPGRADE_SQL)


def downgrade() -> None:
    op.execute(DOWNGRADE_SQL)
