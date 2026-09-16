"""kag — capa proposicional del grafo (0034)

Añade kag_proposition_links: aristas proposición ↔ entidad (una proposición
menciona una entidad). La ingesta (Fase 5) las puebla por coincidencia del
nombre de la entidad en el statement; build_adjacency (src/kag_query.py) las
fusiona al grafo como nodos "p:{id}" conectados a sus entidades (weight 1),
para que el PPR propague masa a través de las proposiciones.

El trigger de versión (kag_bump_graph_version, migración 0016) se aplica
también a esta tabla: al cambiar los links, la caché de adyacencia se
invalida igual que con kag_relations.

Revision ID: 0034_kag_proposition_links
Revises: 0033_kag_paraphrase_fts
Create Date: 2026-09-15
"""

from alembic import op

revision = "0034_kag_proposition_links"
down_revision = "0033_kag_paraphrase_fts"
branch_labels = None
depends_on = None

UPGRADE_SQL = r"""
CREATE TABLE IF NOT EXISTS kag_proposition_links (
    proposition_id INT NOT NULL REFERENCES kag_propositions(id) ON DELETE CASCADE,
    entity_id INT NOT NULL REFERENCES kag_entities(id) ON DELETE CASCADE,
    weight INT NOT NULL DEFAULT 1,
    PRIMARY KEY (proposition_id, entity_id)
);

CREATE INDEX IF NOT EXISTS ix_kag_proposition_links_entity
    ON kag_proposition_links (entity_id);

-- Trigger de versión del grafo: mismo patrón que kag_relations (0016).
DROP TRIGGER IF EXISTS trg_kag_proposition_links_bump_graph
    ON kag_proposition_links;
CREATE TRIGGER trg_kag_proposition_links_bump_graph
AFTER INSERT OR UPDATE OR DELETE ON kag_proposition_links
FOR EACH STATEMENT EXECUTE FUNCTION kag_bump_graph_version();
"""

DOWNGRADE_SQL = r"""
DROP TRIGGER IF EXISTS trg_kag_proposition_links_bump_graph
    ON kag_proposition_links;
DROP TABLE IF EXISTS kag_proposition_links;
"""


def upgrade() -> None:
    op.execute(UPGRADE_SQL)


def downgrade() -> None:
    op.execute(DOWNGRADE_SQL)
