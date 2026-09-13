"""Ruta A — Kaizen/Repetitivo: calendar_slots, production_orders, kaizen_cycles

Ruta A del pipeline unificado (docs/pipeline_unificado_produccion_contenidos(1).md §5):
el contenido recurrente de calendario (el "dato incómodo" semanal de Ergalia, los
shorts Lun/Mié/Vie de ESCAPE) es la GRAN MAYORÍA del volumen real de publicación.
Esta migración añade las entidades que formalizan esa ruta:

  1. calendar_slots — las filas del calendario editorial recurrente. Cada slot
     define los campos que la Orden hereda automáticamente (ORDER_NOTE, §5.2):
     brand_objective, content_bucket, artifact_type, channel y owner.
  2. production_orders — la Orden de Producción (ORDER, §5.1). Se crea a partir
     de un calendar_slot + el insight_core/dato de la semana. Solo se completa
     el dato/gancho específico de esa semana; el resto se hereda del slot.
  3. kaizen_cycles — el registro de KAIZEN_DECISION (§5.7): update_registry
     (la plantilla mejora para todo el equipo) o archive (se documenta sin
     cambiar la plantilla base). Ambas salidas convergen en PRE_DEPLOY_ENTRY.

Revision ID: 0011_kaizen_route
Revises: 0010_project_governance
Create Date: 2026-09-13
"""

from alembic import op

revision = "0011_kaizen_route"
down_revision = "0010_project_governance"
branch_labels = None
depends_on = None

UPGRADE_SQL = r"""
-- ---------------------------------------------------------------------
-- 1. calendar_slots — filas del calendario editorial recurrente (§5.2)
--    La Orden hereda de aquí brand_objective, content_bucket, artifact_type,
--    channel y owner. Se puebla con los calendarios operativos reales de
--    ambas marcas (ergalia_mkt_operativo.md Módulo 6, escape_mkt_operativo.md
--    Módulo 4) vía src/db/seed_calendar.py.
-- ---------------------------------------------------------------------

CREATE TABLE calendar_slots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    slot_code VARCHAR(100) NOT NULL UNIQUE,   -- ej. 'escape_lunes_short'
    brand_objective brand_objective_t NOT NULL,
    content_bucket content_bucket_t NOT NULL,
    artifact_type VARCHAR(50) NOT NULL,
    channel VARCHAR(50) NOT NULL,
    default_owner_role VARCHAR(100),           -- ej. 'Editor' (ESCAPE) / 'CM+Diseñador' (Ergalia)
    cadence VARCHAR(20) NOT NULL DEFAULT 'semanal',
    weekday VARCHAR(20),                       -- ej. 'lunes' | 'martes' | 'miercoles' | 'viernes'
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TRIGGER trg_calendar_slots_updated_at
    BEFORE UPDATE ON calendar_slots
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ---------------------------------------------------------------------
-- 2. production_orders — la Orden de Producción (ORDER, §5.1)
--
-- status: 'creada' -> 'en_produccion' -> 'en_gate' -> 'aprobada' ->
--         'publicada' -> 'medida' -> 'archivada'
-- (publicación manual por ahora: 'publicada' se marca a mano cuando la
-- pieza sale al canal; el sistema de publicación automática es la capa
-- DEPLOY, docs/diseno_sistema_publicacion.md, aún no construida.)
-- ---------------------------------------------------------------------

CREATE TABLE production_orders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    calendar_slot VARCHAR(100) NOT NULL REFERENCES calendar_slots(slot_code),
    brand_objective brand_objective_t NOT NULL,
    content_bucket content_bucket_t NOT NULL,
    artifact_type VARCHAR(50) NOT NULL,
    channel VARCHAR(50) NOT NULL,
    owner_id UUID REFERENCES app_users(id),
    scheduled_date DATE NOT NULL,               -- la fecha de la semana
    insight_core TEXT NOT NULL,                -- el dato/gancho específico de esa semana
    status VARCHAR(30) NOT NULL DEFAULT 'creada',
    brief_id UUID REFERENCES content_briefs(id),
    artifact_id UUID REFERENCES content_artifacts(id),
    template_id UUID REFERENCES production_templates(id),
    kpis JSONB NOT NULL DEFAULT '{}'::jsonb,    -- MEASURE_KPI (§5.4)
    kaizen_decision VARCHAR(20),                -- 'update_registry' | 'archive' (§5.7)
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_orders_status ON production_orders(status);
CREATE INDEX idx_orders_slot ON production_orders(calendar_slot);
CREATE INDEX idx_orders_scheduled ON production_orders(scheduled_date);

CREATE TRIGGER trg_production_orders_updated_at
    BEFORE UPDATE ON production_orders
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ---------------------------------------------------------------------
-- 3. kaizen_cycles — KAIZEN_DECISION (§5.7)
-- ---------------------------------------------------------------------

CREATE TABLE kaizen_cycles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    order_id UUID NOT NULL REFERENCES production_orders(id) ON DELETE CASCADE,
    decision VARCHAR(20) NOT NULL,              -- 'update_registry' | 'archive'
    improvement_summary TEXT,
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_kaizen_cycles_order ON kaizen_cycles(order_id);
"""

DOWNGRADE_SQL = r"""
DROP TABLE IF EXISTS kaizen_cycles CASCADE;
DROP TABLE IF EXISTS production_orders CASCADE;
DROP TABLE IF EXISTS calendar_slots CASCADE;
"""


def upgrade() -> None:
    op.execute(UPGRADE_SQL)


def downgrade() -> None:
    op.execute(DOWNGRADE_SQL)
