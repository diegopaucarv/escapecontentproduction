"""kag — índice de frecuencia de palabras + idioma del documento (0023)

Añade `kag_word_freq` (frecuencia de palabras POR DOCUMENTO, mantenida en
ingestión — misma característica que los otros índices: se recalcula cuando
se añade o modifica un documento) y la columna `language` en kag_documents
(persistida al indexar, detectada con detect_language).

El índice alimenta al crítico LLM en src/kag_query.py::_corpus_common_words:
palabras demasiado frecuentes en el corpus no sirven como términos exactos
(matchearían demasiados chunks). El crítico recibe la lista como contexto y
decide qué términos proponer, traducidos a todos los idiomas soportados.

Revision ID: 0023_kag_word_freq
Revises: 0022_kag_entity_embeddings
Create Date: 2026-09-15
"""

from alembic import op

revision = "0023_kag_word_freq"
down_revision = "0022_kag_entity_embeddings"
branch_labels = None
depends_on = None

UPGRADE_SQL = r"""
CREATE TABLE IF NOT EXISTS kag_word_freq (
    doc_id INT NOT NULL REFERENCES kag_documents(id) ON DELETE CASCADE,
    word TEXT NOT NULL,
    nentry INT NOT NULL DEFAULT 0,          -- ocurrencias de la palabra en el doc
    PRIMARY KEY (doc_id, word)
);
CREATE INDEX IF NOT EXISTS ix_kag_word_freq_word ON kag_word_freq (word);

ALTER TABLE kag_documents ADD COLUMN IF NOT EXISTS language VARCHAR(10) NOT NULL DEFAULT '';
"""

DOWNGRADE_SQL = r"""
DROP TABLE IF EXISTS kag_word_freq;
ALTER TABLE kag_documents DROP COLUMN IF EXISTS language;
"""


def upgrade() -> None:
    op.execute(UPGRADE_SQL)


def downgrade() -> None:
    op.execute(DOWNGRADE_SQL)
