"""
Bus de eventos ligero basado en LISTEN/NOTIFY de Postgres.

Por qué no Kafka/Redis: al volumen de esta operación (decenas de piezas
de contenido por semana, no miles de eventos por segundo), sumar un
broker de mensajería es complejidad de operación sin beneficio real.
Postgres ya es la base de datos del sistema; LISTEN/NOTIFY da pub/sub
"gratis" y sin infraestructura nueva. Si el volumen de telemetría crece
en un orden de magnitud (webhooks de ads, por ejemplo), esto es lo
primero que debería migrar a una cola real — hasta entonces, es
deuda técnica aceptada conscientemente, no un descuido.

Este worker escucha `telemetry_channel` (ver el trigger en
sql/001_init.sql / alembic 0001) y cierra el bucle RAG: cada evento de
retención/conversión actualiza la fila correspondiente en
artifact_library, para que la próxima búsqueda de duplicado
(novelty_router.find_closest_prior_artifact) tenga en cuenta qué tan
bien funcionó esa pieza, no solo su similitud semántica.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Awaitable, Callable

import asyncpg

from src.config import get_settings

logger = logging.getLogger("pipeline.events")

Handler = Callable[[str], Awaitable[None]]

# Qué event_type de telemetry_events alimenta qué columna de
# artifact_library. Ver sql/001_init.sql — ambas columnas son promedios
# simples por ahora; una media móvil ponderada por recencia es la
# mejora obvia una vez que haya volumen real de datos para justificarla.
_METRIC_COLUMN_BY_EVENT_TYPE = {
    "retention_24h": "retention_24h",
    "conversion_30d": "conversion_30d",
}


class TelemetryListener:
    def __init__(self, dsn: str, handler: Handler):
        self._dsn = dsn
        self._handler = handler
        self._conn: asyncpg.Connection | None = None

    async def start(self) -> None:
        self._conn = await asyncpg.connect(self._dsn)
        await self._conn.add_listener("telemetry_channel", self._on_notify)
        logger.info("Escuchando telemetry_channel...")
        try:
            while True:
                await asyncio.sleep(3600)  # el listener real reacciona por callback
        finally:
            await self.stop()

    async def stop(self) -> None:
        if self._conn is not None:
            await self._conn.remove_listener("telemetry_channel", self._on_notify)
            await self._conn.close()

    def _on_notify(self, connection, pid, channel, payload: str) -> None:
        asyncio.create_task(self._handler(payload))


async def close_rag_loop(telemetry_event_id: str) -> None:
    """Handler real (reemplaza el placeholder anterior). Se conecta con
    su propia conexión asyncpg de corta duración — este proceso no
    comparte pool con la API, a propósito: son procesos distintos que
    escalan y fallan independientemente."""
    settings = get_settings()
    conn = await asyncpg.connect(settings.database_url_async)
    await conn.set_type_codec(
        "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
    )
    try:
        event = await conn.fetchrow(
            "SELECT artifact_id, event_type, payload FROM telemetry_events WHERE id = $1",
            int(telemetry_event_id),
        )
        if event is None or event["artifact_id"] is None:
            logger.info("telemetry_events.id=%s sin artifact_id asociado, se ignora", telemetry_event_id)
            return

        column = _METRIC_COLUMN_BY_EVENT_TYPE.get(event["event_type"])
        if column is None:
            logger.debug("event_type=%s no mapea a una métrica de artifact_library", event["event_type"])
            return

        value = event["payload"].get("value") if event["payload"] else None
        if value is None:
            logger.warning("telemetry_events.id=%s (%s) sin 'value' en payload", telemetry_event_id, event["event_type"])
            return

        # Promedio simple con el valor existente (si lo hay). Ver nota
        # de módulo sobre migrar a media móvil ponderada más adelante.
        await conn.execute(
            f"""
            UPDATE artifact_library
            SET {column} = COALESCE(({column} + $1) / 2.0, $1)
            WHERE artifact_id = $2
            """,
            float(value),
            event["artifact_id"],
        )
        logger.info(
            "artifact_library actualizada: artifact_id=%s %s=%s",
            event["artifact_id"], column, value,
        )
    finally:
        await conn.close()


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    listener = TelemetryListener(settings.database_url_async, close_rag_loop)
    await listener.start()


if __name__ == "__main__":
    asyncio.run(main())

