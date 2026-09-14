"""kag — capa proposicional (documentos, capítulos, chunks atómicos, imágenes, árbol temático) (0018)

Crea las tablas de la capa proposicional del rediseño KAG (nombres literales
del diseño — SIN prefijo kag_):

  - documents: ficha documental (ISO 25964 + Library of Congress) con los
    límites físicos del MultibookFinderTool (line_start/line_end) y el
    scope_thematic usado en Branch B (`scope_thematic ILIKE ANY(...)`).
  - document_chapters: capítulos del documento; summary alimenta
    escalate_to_parent_context.
  - propositional_chunks: proposiciones atómicas autocontenidas con
    embedding vector(768) (jina-embeddings-v5-text-nano, misma dim que
    kag_chunks) + índice HNSW vector_cosine_ops.
  - document_images: imágenes con descripción visual densa y FAQ Reverse HyDE.
  - topic_tree_nodes: nodos del árbol temático (macro-fases) con
    start/end_chunk_id apuntando a propositional_chunks.

Además añade `is_vision` a llm_models: identifica modelos de visión (VLM)
para lectura de imágenes (seed en src/db/seed_vision.py).

Revision ID: 0018_kag_propositional
Revises: 0017_kag_trgm
Create Date: 2026-09-14
"""

from alembic import op

revision = "0018_kag_propositional"
down_revision = "0017_kag_trgm"
branch_labels = None
depends_on = None

UPGRADE_SQL = r"""
-- ---------------------------------------------------------------------
-- documents: ficha documental (ISO 25964 + Library of Congress)
-- ---------------------------------------------------------------------
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    source_file VARCHAR(500) NOT NULL,               -- ruta del .md maestro
    document_id VARCHAR(100) NOT NULL UNIQUE,        -- ej. '{stem}_doc_001' (MultibookFinderTool)
    title VARCHAR(300) NOT NULL,
    technical_level VARCHAR(20) NOT NULL DEFAULT 'intermediate',  -- introductory|intermediate|advanced|research
    bibtex TEXT NOT NULL DEFAULT '',
    thematic_areas_iso25964 JSONB NOT NULL DEFAULT '[]',  -- [{preferred_term, non_preferred_terms[], scope_note_disambiguation, broader_term, narrower_term, related_terms[]}]
    library_of_congress JSONB NOT NULL DEFAULT '{}',      -- {lcsh_terms[], lcc_classification{}}
    key_entities JSONB NOT NULL DEFAULT '[]',
    scope_thematic TEXT NOT NULL DEFAULT '',         -- Branch B: scope_thematic ILIKE ANY(...)
    line_start INT NOT NULL DEFAULT 0,               -- límites físicos del MultibookFinderTool
    line_end INT NOT NULL DEFAULT 0,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',   -- pending|ready|failed
    content_hash VARCHAR(64) NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TRIGGER set_updated_at_documents
BEFORE UPDATE ON documents
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ---------------------------------------------------------------------
-- document_chapters: capítulos del documento
-- ---------------------------------------------------------------------
CREATE TABLE document_chapters (
    id SERIAL PRIMARY KEY,
    document_id INT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chapter_index INT NOT NULL,
    title VARCHAR(300) NOT NULL,
    line_start INT NOT NULL,
    line_end INT NOT NULL,
    main_theme TEXT NOT NULL DEFAULT '',
    summary TEXT NOT NULL DEFAULT '',                -- escalate_to_parent_context: SELECT id, summary, line_start, line_end
    subsections JSONB NOT NULL DEFAULT '[]',
    UNIQUE (document_id, chapter_index)
);

-- ---------------------------------------------------------------------
-- propositional_chunks: proposiciones atómicas autocontenidas
-- ---------------------------------------------------------------------
CREATE TABLE propositional_chunks (
    id SERIAL PRIMARY KEY,
    document_id INT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chapter_id INT NOT NULL REFERENCES document_chapters(id) ON DELETE CASCADE,
    core_idea_id VARCHAR(50) NOT NULL DEFAULT '',    -- ej. CI_01
    argument_id VARCHAR(50) NOT NULL DEFAULT '',     -- ej. ARG_01_A
    statement TEXT NOT NULL,                         -- proposición atómica autocontenida
    text_span TEXT NOT NULL DEFAULT '',               -- verbatim_span exacto del original
    char_start INT NOT NULL DEFAULT 0,
    char_end INT NOT NULL DEFAULT 0,
    line_start INT NOT NULL DEFAULT 0,
    line_end INT NOT NULL DEFAULT 0,
    citation_references JSONB NOT NULL DEFAULT '[]',  -- array de strings (referencias duplicadas en cada átomo)
    embedding vector(768),                            -- jina-embeddings-v5-text-nano (dim 768, misma que kag_chunks)
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ix_propositional_chunks_doc ON propositional_chunks (document_id);
CREATE INDEX ix_propositional_chunks_chapter ON propositional_chunks (chapter_id);
CREATE INDEX ix_propositional_chunks_embedding
    ON propositional_chunks USING hnsw (embedding vector_cosine_ops);

-- ---------------------------------------------------------------------
-- document_images: imágenes con descripción visual densa
-- ---------------------------------------------------------------------
CREATE TABLE document_images (
    id SERIAL PRIMARY KEY,
    document_id INT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    image_id VARCHAR(100) NOT NULL DEFAULT '',
    file_path VARCHAR(500) NOT NULL,
    anchor_line INT NOT NULL DEFAULT 0,
    caption TEXT NOT NULL DEFAULT '',
    image_type VARCHAR(50) NOT NULL DEFAULT '',      -- diagram|chart_or_plot|flowchart|conceptual_illustration|screenshot|table_image|photograph
    dense_visual_description TEXT NOT NULL DEFAULT '',
    epistemic_contribution TEXT NOT NULL DEFAULT '',
    faq_indexing JSONB NOT NULL DEFAULT '[]',         -- 3-5 preguntas Reverse HyDE
    associated_entities JSONB NOT NULL DEFAULT '[]',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ix_document_images_doc ON document_images (document_id);

-- ---------------------------------------------------------------------
-- topic_tree_nodes: nodos del árbol temático (macro-fases)
-- ---------------------------------------------------------------------
CREATE TABLE topic_tree_nodes (
    id SERIAL PRIMARY KEY,
    document_id INT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    sequential_order INT NOT NULL,
    macro_phase_label VARCHAR(300) NOT NULL DEFAULT '',
    representative_keywords JSONB NOT NULL DEFAULT '[]',
    start_chunk_id INT REFERENCES propositional_chunks(id) ON DELETE SET NULL,
    end_chunk_id INT REFERENCES propositional_chunks(id) ON DELETE SET NULL,
    epistemic_summary TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, sequential_order)
);
CREATE INDEX ix_topic_tree_nodes_doc ON topic_tree_nodes (document_id);

-- ---------------------------------------------------------------------
-- llm_models.is_vision: identifica modelos de visión (VLM)
-- ---------------------------------------------------------------------
ALTER TABLE llm_models ADD COLUMN is_vision BOOLEAN NOT NULL DEFAULT FALSE;
"""

DOWNGRADE_SQL = r"""
ALTER TABLE llm_models DROP COLUMN IF EXISTS is_vision;
DROP TABLE IF EXISTS topic_tree_nodes;
DROP TABLE IF EXISTS document_images;
DROP TABLE IF EXISTS propositional_chunks;
DROP TABLE IF EXISTS document_chapters;
DROP TABLE IF EXISTS documents;
"""


def upgrade() -> None:
    op.execute(UPGRADE_SQL)


def downgrade() -> None:
    op.execute(DOWNGRADE_SQL)
