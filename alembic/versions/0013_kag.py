"""kag — sistema de conocimiento (vector + grafo de entidades + HippoRAG) (0013)

Crea las tablas del sistema KAG (§1.1 y §1.2 del diseño):
kag_documents, kag_chunks (embedding vector(768)), kag_entities, kag_relations,
kag_figures y kag_segmenter_settings. El seed (src/db/seed_kag.py) puebla la
configuración del segmentador y registra el modelo local Qwen 2.5.

Revision ID: 0013_kag
Revises: 0012_brand_knowledge
Create Date: 2026-09-13
"""

from alembic import op

revision = "0013_kag"
down_revision = "0012_brand_knowledge"
branch_labels = None
depends_on = None

UPGRADE_SQL = r"""
CREATE TABLE kag_documents (
    id SERIAL PRIMARY KEY,
    doc_path VARCHAR(500) NOT NULL UNIQUE,      -- relativo a data/knowledge_repository/docs/
    title VARCHAR(300) NOT NULL,
    doc_type VARCHAR(10) NOT NULL DEFAULT 'short',  -- 'short' | 'long' (>= 20k tokens)
    status VARCHAR(20) NOT NULL DEFAULT 'pending',  -- 'pending' | 'ready' | 'failed'
    content_hash VARCHAR(64) NOT NULL,          -- sha256 del .md (idempotencia)
    token_estimate INT NOT NULL DEFAULT 0,
    chunk_count INT NOT NULL DEFAULT 0,
    entity_count INT NOT NULL DEFAULT 0,
    relation_count INT NOT NULL DEFAULT 0,
    figure_count INT NOT NULL DEFAULT 0,
    summary TEXT NOT NULL DEFAULT '',           -- resumen Qwen 2.5 local (fuente secundaria)
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TRIGGER set_updated_at_kag_documents
BEFORE UPDATE ON kag_documents
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE TABLE kag_chunks (
    id SERIAL PRIMARY KEY,
    doc_id INT NOT NULL REFERENCES kag_documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    section_path VARCHAR(500) NOT NULL DEFAULT '',  -- ej. '## Introducción > ### Métodos'
    content TEXT NOT NULL,
    token_estimate INT NOT NULL DEFAULT 0,
    embedding vector(768),                          -- jina-embeddings-v5-text-nano (dim 768)
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (doc_id, chunk_index)
);

CREATE TABLE kag_entities (
    id SERIAL PRIMARY KEY,
    doc_id INT NOT NULL REFERENCES kag_documents(id) ON DELETE CASCADE,
    chunk_id INT REFERENCES kag_chunks(id) ON DELETE CASCADE,
    name VARCHAR(300) NOT NULL,
    name_norm VARCHAR(300) NOT NULL,               -- minúsculas + strip (dedup)
    entity_type VARCHAR(100) NOT NULL DEFAULT 'concept',  -- concept|method|law|person|org|figure
    description TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ix_kag_entities_norm ON kag_entities (name_norm);
CREATE INDEX ix_kag_entities_doc ON kag_entities (doc_id);

CREATE TABLE kag_relations (
    id SERIAL PRIMARY KEY,
    doc_id INT NOT NULL REFERENCES kag_documents(id) ON DELETE CASCADE,
    chunk_id INT REFERENCES kag_chunks(id) ON DELETE CASCADE,
    source_entity_id INT NOT NULL REFERENCES kag_entities(id) ON DELETE CASCADE,
    target_entity_id INT NOT NULL REFERENCES kag_entities(id) ON DELETE CASCADE,
    relation_type VARCHAR(200) NOT NULL,            -- ej. 'UTILIZA', 'DEFINE', 'PARTE_DE'
    description TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ix_kag_relations_src ON kag_relations (source_entity_id);
CREATE INDEX ix_kag_relations_dst ON kag_relations (target_entity_id);

CREATE TABLE kag_figures (
    id SERIAL PRIMARY KEY,
    doc_id INT NOT NULL REFERENCES kag_documents(id) ON DELETE CASCADE,
    chunk_id INT REFERENCES kag_chunks(id) ON DELETE CASCADE,
    image_path VARCHAR(500) NOT NULL,               -- relativo a data/knowledge_repository/images/
    caption TEXT NOT NULL DEFAULT '',                -- alt text del .md si existe
    description TEXT NOT NULL DEFAULT '',            -- descripción VLM
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE kag_segmenter_settings (
    id SERIAL PRIMARY KEY,
    nli_model VARCHAR(300) NOT NULL DEFAULT 'facebook/bart-large-mnli',
    segmenter_embedding_model VARCHAR(300) NOT NULL DEFAULT 'sentence-transformers/all-MiniLM-L6-v2',
    spacy_models JSONB NOT NULL DEFAULT '{"es": "es_core_news_md", "en": "en_core_web_md", "pt": "pt_core_news_md", "de": "de_core_news_md", "fr": "fr_core_news_md"}',
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TRIGGER set_updated_at_kag_segmenter_settings
BEFORE UPDATE ON kag_segmenter_settings
FOR EACH ROW EXECUTE FUNCTION set_updated_at();
"""

DOWNGRADE_SQL = r"""
DROP TABLE IF EXISTS kag_segmenter_settings;
DROP TABLE IF EXISTS kag_figures;
DROP TABLE IF EXISTS kag_relations;
DROP TABLE IF EXISTS kag_entities;
DROP TABLE IF EXISTS kag_chunks;
DROP TABLE IF EXISTS kag_documents;
"""


def upgrade() -> None:
    op.execute(UPGRADE_SQL)


def downgrade() -> None:
    op.execute(DOWNGRADE_SQL)
