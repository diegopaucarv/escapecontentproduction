"""kag — proposiciones atómicas para la expansión del clásico (0024)

Crea `kag_propositions` (proposiciones atómicas autocontenidas extraídas por
chunk en la ingesta clásica — capa micro del pipeline híbrido de dos niveles
de grano: chunks macro + proposiciones micro). Cada proposición referencia su
chunk y documento de origen, conserva el span textual exacto con offsets
absolutos (char/línea) y las referencias académicas duplicadas de la frase
origen (regla de referencias duplicadas del proposicional).

La extracción la hace el hook de ingesta (src/kag_ingest.py, Agente B) con la
spec `kag_proposition_chunking`; el embedding (vector(768),
jina-embeddings-v5-text-nano) habilita la recuperación semántica de la capa
micro en src/kag_query.py (Agente C).

Revision ID: 0024_kag_propositions
Revises: 0023_kag_word_freq
Create Date: 2026-09-15
"""

from alembic import op

revision = "0024_kag_propositions"
down_revision = "0023_kag_word_freq"
branch_labels = None
depends_on = None

UPGRADE_SQL = r"""
CREATE TABLE IF NOT EXISTS kag_propositions (
    id SERIAL PRIMARY KEY,
    doc_id INT NOT NULL REFERENCES kag_documents(id) ON DELETE CASCADE,
    chunk_id INT NOT NULL REFERENCES kag_chunks(id) ON DELETE CASCADE,
    core_idea_id VARCHAR(50),
    argument_id VARCHAR(50),
    statement TEXT NOT NULL,
    text_span TEXT,
    char_start INT,
    char_end INT,
    line_start INT,
    line_end INT,
    citation_references JSONB NOT NULL DEFAULT '[]'::jsonb,
    embedding vector(768),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_kag_propositions_doc_id
    ON kag_propositions (doc_id);
CREATE INDEX IF NOT EXISTS ix_kag_propositions_chunk_id
    ON kag_propositions (chunk_id);
CREATE INDEX IF NOT EXISTS ix_kag_propositions_embedding
    ON kag_propositions USING hnsw (embedding vector_cosine_ops);
"""

DOWNGRADE_SQL = r"""
DROP TABLE IF EXISTS kag_propositions;
"""


def upgrade() -> None:
    op.execute(UPGRADE_SQL)


def downgrade() -> None:
    op.execute(DOWNGRADE_SQL)
