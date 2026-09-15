"""kag — metadatos documentales y relación directa capítulos→chunks (0028)

Fundación de datos del pipeline documental nuevo (Fases 1-5):

  - `kag_chapters`: capítulos detectados por LLM (Fase 2, kag_document_analysis)
    con line_start/line_end sobre el archivo fuente. La relación nueva es
    kag_chapters → kag_chunks.chapter_id → kag_propositions.chapter_id
    (se ELIMINA el section_path determinista de chunk_markdown).
    de un capítulo (Fase 3 futura); por ahora quedan con chapter_id = NULL y
    el pipeline degrada con gracia (chapter_title = "").
  - `kag_chunks.paraphrase` (TEXT nullable): paráfrasis del chunk (Fase 4,
    kag_chunk_paraphrase).
  - `kag_propositions.chapter_id` (UUID nullable): proposiciones asociadas al
    capítulo (Fase 5, kag_chapter_propositions).
  - `kag_documents.ficha_jsonb` / `sections_json` (JSONB): ficha documental y
    esqueleto de secciones (Fase 2).

Revision ID: 0028_kag_document_meta
Revises: 0027_kag_proposition_section
Create Date: 2026-09-15
"""

from alembic import op

revision = "0028_kag_document_meta"
down_revision = "0027_kag_proposition_section"
branch_labels = None
depends_on = None

UPGRADE_SQL = r"""
CREATE TABLE IF NOT EXISTS kag_chapters (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id INT NOT NULL REFERENCES kag_documents(id) ON DELETE CASCADE,
    chapter_id VARCHAR(100) NOT NULL,
    title VARCHAR(500),
    line_start INT,
    line_end INT,
    summary TEXT,
    has_images BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_kag_chapters_doc_chapter
    ON kag_chapters (doc_id, chapter_id);

ALTER TABLE kag_chunks ADD COLUMN IF NOT EXISTS chapter_id UUID
    REFERENCES kag_chapters(id) ON DELETE SET NULL;
ALTER TABLE kag_chunks ADD COLUMN IF NOT EXISTS paraphrase TEXT;
ALTER TABLE kag_chunks DROP COLUMN IF EXISTS section_path;

ALTER TABLE kag_propositions ADD COLUMN IF NOT EXISTS chapter_id UUID
    REFERENCES kag_chapters(id) ON DELETE SET NULL;
ALTER TABLE kag_propositions DROP COLUMN IF EXISTS section_path;

ALTER TABLE kag_documents ADD COLUMN IF NOT EXISTS ficha_jsonb JSONB;
ALTER TABLE kag_documents ADD COLUMN IF NOT EXISTS sections_json JSONB;
"""

DOWNGRADE_SQL = r"""
ALTER TABLE kag_documents DROP COLUMN IF EXISTS sections_json;
ALTER TABLE kag_documents DROP COLUMN IF EXISTS ficha_jsonb;

ALTER TABLE kag_propositions ADD COLUMN IF NOT EXISTS section_path VARCHAR(512);
ALTER TABLE kag_propositions DROP COLUMN IF EXISTS chapter_id;

ALTER TABLE kag_chunks ADD COLUMN IF NOT EXISTS section_path VARCHAR(500)
    NOT NULL DEFAULT '';
ALTER TABLE kag_chunks DROP COLUMN IF EXISTS paraphrase;
ALTER TABLE kag_chunks DROP COLUMN IF EXISTS chapter_id;

DROP INDEX IF EXISTS ix_kag_chapters_doc_chapter;
DROP TABLE IF EXISTS kag_chapters;
"""


def upgrade() -> None:
    op.execute(UPGRADE_SQL)


def downgrade() -> None:
    op.execute(DOWNGRADE_SQL)
