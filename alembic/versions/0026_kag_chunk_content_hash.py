"""kag — content_hash por chunk para cache de proposiciones (0026)

Añade `content_hash` a `kag_chunks` (sha256 hex del contenido del chunk).
La extracción de proposiciones (src/kag_ingest.py, etapa chunked) lo usa
como cache: si el chunk ya tiene proposiciones persistidas y su
content_hash no cambió, no se re-extrae (evita re-extraer en re-ingestas).

Revision ID: 0026_kag_chunk_content_hash
Revises: 0025_kag_drop_propositional
Create Date: 2026-09-15
"""

from alembic import op

revision = "0026_kag_chunk_content_hash"
down_revision = "0025_kag_drop_propositional"
branch_labels = None
depends_on = None

UPGRADE_SQL = r"""
ALTER TABLE kag_chunks ADD COLUMN IF NOT EXISTS content_hash VARCHAR(64) NOT NULL DEFAULT '';
CREATE INDEX IF NOT EXISTS ix_kag_chunks_doc_content_hash
    ON kag_chunks (doc_id, content_hash);
"""

DOWNGRADE_SQL = r"""
DROP INDEX IF EXISTS ix_kag_chunks_doc_content_hash;
ALTER TABLE kag_chunks DROP COLUMN IF EXISTS content_hash;
"""


def upgrade() -> None:
    op.execute(UPGRADE_SQL)


def downgrade() -> None:
    op.execute(DOWNGRADE_SQL)
