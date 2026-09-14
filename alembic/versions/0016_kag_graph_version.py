"""kag — caché del grafo de entidades por versión (0016)

El grafo de entidades (build_adjacency en src/kag_query.py) se reconstruía
leyendo TODAS las filas de kag_relations en cada consulta. Esta migración
añade una tabla de versión de una sola fila (kag_graph_state) y un trigger
AFTER INSERT/UPDATE/DELETE sobre kag_relations que la incrementa.

Como kag_relations solo cambia en la ingesta (añadir/modificar/borrar un
documento), el grafo se reconstruye únicamente entonces — espejo del índice
HNSW de pgvector, que Postgres mantiene incrementalmente. En consulta,
build_adjacency lee la versión (O(1), una fila) y reutiliza el dict en
memoria si no cambió.

Revision ID: 0016_kag_graph_version
Revises: 0015_kag_hnsw
Create Date: 2026-09-14
"""

from alembic import op

revision = "0016_kag_graph_version"
down_revision = "0015_kag_hnsw"
branch_labels = None
depends_on = None

UPGRADE_SQL = r"""
-- Tabla de versión del grafo: una sola fila (id=1), version BIGINT.
CREATE TABLE IF NOT EXISTS kag_graph_state (
    id SMALLINT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    version BIGINT NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Fila única inicial (ON CONFLICT para idempotencia).
INSERT INTO kag_graph_state (id, version) VALUES (1, 0)
ON CONFLICT (id) DO NOTHING;

-- Función que incrementa la versión (FOR EACH STATEMENT: una vez por
-- statement, no por fila — un lote de relaciones dispara una sola vez).
CREATE OR REPLACE FUNCTION kag_bump_graph_version() RETURNS trigger AS $$
BEGIN
    UPDATE kag_graph_state
       SET version = version + 1, updated_at = now()
     WHERE id = 1;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Trigger AFTER ... ON kag_relations. DROP IF EXISTS primero para que la
-- migración sea re-ejecutable (idempotente).
DROP TRIGGER IF EXISTS trg_kag_relations_bump_graph ON kag_relations;
CREATE TRIGGER trg_kag_relations_bump_graph
AFTER INSERT OR UPDATE OR DELETE ON kag_relations
FOR EACH STATEMENT EXECUTE FUNCTION kag_bump_graph_version();
"""

DOWNGRADE_SQL = r"""
DROP TRIGGER IF EXISTS trg_kag_relations_bump_graph ON kag_relations;
DROP FUNCTION IF EXISTS kag_bump_graph_version();
DROP TABLE IF EXISTS kag_graph_state;
"""


def upgrade() -> None:
    op.execute(UPGRADE_SQL)


def downgrade() -> None:
    op.execute(DOWNGRADE_SQL)
