"""
Seed de la Ruta A — calendario editorial recurrente (calendar_slots).

Inserta/actualiza las filas del calendario semanal/mensual de ambas marcas
(ergalia_mkt_operativo.md Módulo 6, escape_mkt_operativo.md Módulo 4). Cada
slot define los campos que la Orden de Producción hereda automáticamente
(ORDER_NOTE, §5.2 del pipeline): brand_objective, content_bucket,
artifact_type, channel y owner.

Slots reales:
  - ESCAPE: lunes/miércoles/viernes → short → Editor
    (artifact_type video_corto, channel tiktok, bucket difusion_cientifica)
  - Ergalia: martes → Dato incómodo → CM+Diseñador
    (bucket debate_informado, canal linkedin)

Idempotente: upsert por slot_code (clave única). No requiere variables de
entorno — es datos puros.

Uso:
    python -m src.db.seed_calendar
"""

from __future__ import annotations

import argparse

from sqlalchemy import select

from src.db.models import CalendarSlot
from src.db.session import SessionLocal

# ---------------------------------------------------------------------
# Slots del calendario editorial recurrente (§5.2)
# ---------------------------------------------------------------------

CALENDAR_SLOTS = [
    # --- ESCAPE: shorts Lun/Mié/Vie (escape_mkt_operativo.md Módulo 4) ---
    {
        "slot_code": "escape_lunes_short",
        "brand_objective": "ESCAPE_SOCIAL",
        "content_bucket": "difusion_cientifica",
        "artifact_type": "video_corto",
        "channel": "tiktok",
        "default_owner_role": "Editor",
        "cadence": "semanal",
        "weekday": "lunes",
    },
    {
        "slot_code": "escape_miercoles_short",
        "brand_objective": "ESCAPE_SOCIAL",
        "content_bucket": "difusion_cientifica",
        "artifact_type": "video_corto",
        "channel": "tiktok",
        "default_owner_role": "Editor",
        "cadence": "semanal",
        "weekday": "miercoles",
    },
    {
        "slot_code": "escape_viernes_short",
        "brand_objective": "ESCAPE_SOCIAL",
        "content_bucket": "difusion_cientifica",
        "artifact_type": "video_corto",
        "channel": "tiktok",
        "default_owner_role": "Editor",
        "cadence": "semanal",
        "weekday": "viernes",
    },
    # --- Ergalia: martes → Dato incómodo (ergalia_mkt_operativo.md Módulo 6) ---
    {
        "slot_code": "ergalia_martes_dato_incomodo",
        "brand_objective": "ERGALIA_COMERCIAL",
        "content_bucket": "debate_informado",
        "artifact_type": "post",
        "channel": "linkedin",
        "default_owner_role": "CM+Diseñador",
        "cadence": "semanal",
        "weekday": "martes",
    },
]


def _upsert(session, data: dict) -> CalendarSlot:
    """Inserta o actualiza un slot por slot_code (clave única)."""
    row = (
        session.execute(
            select(CalendarSlot).where(CalendarSlot.slot_code == data["slot_code"])
        )
        .scalars()
        .first()
    )
    if row is None:
        row = CalendarSlot(**data)
        session.add(row)
        session.flush()
    else:
        for key, value in data.items():
            setattr(row, key, value)
    return row


def seed(session=None) -> dict:
    """Inserta/actualiza los slots del calendario. Devuelve un resumen.

    Idempotente: upsert por slot_code.
    """
    own_session = session is None
    if own_session:
        session = SessionLocal()
    try:
        rows = [_upsert(session, data) for data in CALENDAR_SLOTS]
        session.commit()
        return {
            "slots": [r.slot_code for r in rows],
            "count": len(rows),
            "brands": sorted({r.brand_objective for r in rows}),
        }
    finally:
        if own_session:
            session.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed del calendario editorial recurrente (Ruta A)."
    )
    parser.parse_args()
    result = seed()
    print("Calendario editorial recurrente listo (calendar_slots):")
    print(f"  Slots: {result['count']}")
    for slot in result["slots"]:
        print(f"    - {slot}")
    print(f"  Marcas: {', '.join(result['brands'])}")


if __name__ == "__main__":
    main()
