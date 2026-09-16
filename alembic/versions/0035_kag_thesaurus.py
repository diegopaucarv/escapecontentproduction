"""kag — capa tesauro del grafo (0035)

Añade kag_thesaurus_terms (términos LCSH/LCC/ISO 25964 con relaciones
broader/narrower/related) y kag_document_thesaurus (documento ↔ términos).
La ingesta (Fase 2) las puebla desde la ficha documental
(_store_document_thesaurus, src/kag_ingest.py); build_adjacency
(src/kag_query.py) las fusiona al grafo como nodos "d:{doc_id}" y
"t:{term_id}" para que el PPR propague masa a través de documentos y
términos controlados.

El trigger de versión (kag_bump_graph_version, migración 0016) se aplica
también a ambas tablas: al cambiar términos o links, la caché de adyacencia
se invalida igual que con kag_relations.

Revision ID: 0035_kag_thesaurus
Revises: 0034_kag_proposition_links
Create Date: 2026-09-15
"""

from alembic import op

revision = "0035_kag_thesaurus"
down_revision = "0034_kag_proposition_links"
branch_labels = None
depends_on = None

UPGRADE_SQL = r"""
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS kag_thesaurus_terms (
    id SERIAL PRIMARY KEY,
    term TEXT NOT NULL,
    term_norm TEXT NOT NULL,
    source TEXT NOT NULL CHECK (source IN ('lcsh', 'lcc', 'iso25964')),
    broader JSONB NOT NULL DEFAULT '[]',
    narrower JSONB NOT NULL DEFAULT '[]',
    related JSONB NOT NULL DEFAULT '[]',
    UNIQUE (term_norm, source)
);

CREATE INDEX IF NOT EXISTS ix_kag_thesaurus_terms_norm_trgm
    ON kag_thesaurus_terms USING gin (term_norm gin_trgm_ops);

CREATE TABLE IF NOT EXISTS kag_document_thesaurus (
    doc_id INT NOT NULL REFERENCES kag_documents(id) ON DELETE CASCADE,
    term_id INT NOT NULL REFERENCES kag_thesaurus_terms(id) ON DELETE CASCADE,
    PRIMARY KEY (doc_id, term_id)
);

CREATE INDEX IF NOT EXISTS ix_kag_document_thesaurus_term
    ON kag_document_thesaurus (term_id);

-- Trigger de versión del grafo: mismo patrón que kag_relations (0016).
DROP TRIGGER IF EXISTS trg_kag_thesaurus_terms_bump_graph
    ON kag_thesaurus_terms;
CREATE TRIGGER trg_kag_thesaurus_terms_bump_graph
AFTER INSERT OR UPDATE OR DELETE ON kag_thesaurus_terms
FOR EACH STATEMENT EXECUTE FUNCTION kag_bump_graph_version();

DROP TRIGGER IF EXISTS trg_kag_document_thesaurus_bump_graph
    ON kag_document_thesaurus;
CREATE TRIGGER trg_kag_document_thesaurus_bump_graph
AFTER INSERT OR UPDATE OR DELETE ON kag_document_thesaurus
FOR EACH STATEMENT EXECUTE FUNCTION kag_bump_graph_version();
"""

DOWNGRADE_SQL = r"""
DROP TRIGGER IF EXISTS trg_kag_document_thesaurus_bump_graph
    ON kag_document_thesaurus;
DROP TRIGGER IF EXISTS trg_kag_thesaurus_terms_bump_graph
    ON kag_thesaurus_terms;
DROP TABLE IF EXISTS kag_document_thesaurus;
DROP TABLE IF EXISTS kag_thesaurus_terms;
-- La extensión pg_trgm NO se dropea: puede ser usada por otras tablas.
"""


def upgrade() -> None:
    op.execute(UPGRADE_SQL)


def downgrade() -> None:
    op.execute(DOWNGRADE_SQL)
