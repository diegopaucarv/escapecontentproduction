"""kag — filas por documento dentro del archivo (Fase 1, separación) (0029)

La Fase 1 (kag_document_separation) detecta los documentos (libros, papers,
artículos) apilados en un único .md y cada uno pasa a ser UNA fila de
kag_documents. Esta migración añade las columnas de identidad del documento
dentro del archivo y cambia la unicidad de doc_path → (doc_path, document_id):

  - `document_id` (VARCHAR(100), default ''): id estable del documento dentro
    del archivo. '' = un solo documento por archivo (comportamiento clásico).
  - `line_start` / `line_end` (INT): rango físico de líneas del documento en
    el archivo fuente (1-based, inclusivo).
  - La UNIQUE de doc_path (kag_documents_doc_path_key, creada en 0013 vía
    `doc_path VARCHAR(500) NOT NULL UNIQUE`) se reemplaza por
    kag_documents_doc_path_document_id_key UNIQUE (doc_path, document_id).

Revision ID: 0029_kag_document_rows
Revises: 0028_kag_document_meta
Create Date: 2026-09-15
"""

from alembic import op

revision = "0029_kag_document_rows"
down_revision = "0028_kag_document_meta"
branch_labels = None
depends_on = None

# Nombre del constraint UNIQUE generado por `doc_path VARCHAR(500) NOT NULL
# UNIQUE` en la migración 0013 (convención de PostgreSQL: <tabla>_<columna>_key).
DOC_PATH_UNIQUE = "kag_documents_doc_path_key"
DOC_PATH_DOCUMENT_ID_UNIQUE = "kag_documents_doc_path_document_id_key"

UPGRADE_SQL = f"""
ALTER TABLE kag_documents ADD COLUMN IF NOT EXISTS document_id VARCHAR(100) NOT NULL DEFAULT '';
ALTER TABLE kag_documents ADD COLUMN IF NOT EXISTS line_start INT NOT NULL DEFAULT 0;
ALTER TABLE kag_documents ADD COLUMN IF NOT EXISTS line_end INT NOT NULL DEFAULT 0;

ALTER TABLE kag_documents DROP CONSTRAINT IF EXISTS {DOC_PATH_UNIQUE};
ALTER TABLE kag_documents ADD CONSTRAINT {DOC_PATH_DOCUMENT_ID_UNIQUE} UNIQUE (doc_path, document_id);

CREATE INDEX IF NOT EXISTS ix_kag_documents_doc_path_document_id
    ON kag_documents (doc_path, document_id);
"""

DOWNGRADE_SQL = f"""
DROP INDEX IF EXISTS ix_kag_documents_doc_path_document_id;

ALTER TABLE kag_documents DROP CONSTRAINT IF EXISTS {DOC_PATH_DOCUMENT_ID_UNIQUE};
ALTER TABLE kag_documents ADD CONSTRAINT {DOC_PATH_UNIQUE} UNIQUE (doc_path);

ALTER TABLE kag_documents DROP COLUMN IF EXISTS line_end;
ALTER TABLE kag_documents DROP COLUMN IF EXISTS line_start;
ALTER TABLE kag_documents DROP COLUMN IF EXISTS document_id;
"""


def upgrade() -> None:
    op.execute(UPGRADE_SQL)


def downgrade() -> None:
    op.execute(DOWNGRADE_SQL)
