"""brand_knowledge — canon de marca consultable como Context Pack (0012)

El canon (docs/arquitectura_comercial_y_contenidos_escape_ergalia.md, las
guías de contenido y los _lenguaje_visual.md) vivía como archivos estáticos
que build_context_pack() nunca consultaba: cada pieza que el Producer genera
es una apuesta a que el LLM "adivine" el tono correcto solo por el nombre
del bucket.

Esta migración crea la tabla brand_knowledge: filas consultables etiquetadas
por brand_objective + content_bucket (sección del canon), sin necesitar aún
la Artifact Library vectorial completa. Es trabajo de datos, no de
arquitectura nueva. El seed (src/db/seed_brand_knowledge.py) la puebla con
mockups marcados is_mock=True hasta que los documentos reales existan.

Revision ID: 0012_brand_knowledge
Revises: 0011_kaizen_route
Create Date: 2026-09-13
"""

from alembic import op

revision = "0012_brand_knowledge"
down_revision = "0011_kaizen_route"
branch_labels = None
depends_on = None

UPGRADE_SQL = r"""
CREATE TABLE brand_knowledge (
    id SERIAL PRIMARY KEY,
    brand_objective brand_objective_t NOT NULL,
    content_bucket content_bucket_t,
    section_key VARCHAR(100) NOT NULL,
    section_title VARCHAR(200) NOT NULL,
    content TEXT NOT NULL,
    source_doc VARCHAR(300) NOT NULL,
    is_mock BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (brand_objective, content_bucket, section_key)
);

CREATE TRIGGER set_updated_at_brand_knowledge
BEFORE UPDATE ON brand_knowledge
FOR EACH ROW EXECUTE FUNCTION set_updated_at();
"""

DOWNGRADE_SQL = r"""
DROP TABLE IF EXISTS brand_knowledge;
"""


def upgrade() -> None:
    op.execute(UPGRADE_SQL)


def downgrade() -> None:
    op.execute(DOWNGRADE_SQL)
