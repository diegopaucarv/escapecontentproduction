"""kag — pg_trgm + índice GIN sobre kag_entities.name_norm (0017)

El entity linking usa `name_norm LIKE %term%` (comodín a la izquierda), que
invalida los índices B-Tree y fuerza Seq Scan. pg_trgm añade un índice GIN
trigram que acelera los LIKE con comodín en ambos lados (y además habilita
`similarity()` para búsqueda difusa si se quiere después).

No cambia la lógica de consulta — solo acelera el matching de candidatos en
src/kag_query.py (match_entities_candidates, _noun_chunk_fallback).

Revision ID: 0017_kag_trgm
Revises: 0016_kag_graph_version
Create Date: 2026-09-14
"""

from alembic import op

revision = "0017_kag_trgm"
down_revision = "0016_kag_graph_version"
branch_labels = None
depends_on = None

UPGRADE_SQL = r"""
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE INDEX IF NOT EXISTS ix_kag_entities_norm_trgm
    ON kag_entities USING gin (name_norm gin_trgm_ops);
"""

DOWNGRADE_SQL = r"""
DROP INDEX IF EXISTS ix_kag_entities_norm_trgm;
-- La extensión pg_trgm NO se dropea: puede ser usada por otras tablas.
"""


def upgrade() -> None:
    op.execute(UPGRADE_SQL)


def downgrade() -> None:
    op.execute(DOWNGRADE_SQL)
