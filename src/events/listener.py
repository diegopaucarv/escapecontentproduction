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
sql/001_init.sql) y despacha el manejador correspondiente. Se ejecuta
como el proceso principal del contenedor `agent_worker`.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

import asyncpg

from src.config import get_settings

logger = logging.getLogger("pipeline.events")

Handler = Callable[[str], Awaitable[None]]


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


async def default_handler(telemetry_event_id: str) -> None:
    """Placeholder: aquí se conecta el pipeline RAG en bucle cerrado
    (actualizar retention_24h/conversion_30d en artifact_library según
    el evento recién insertado). Se deja como punto de extensión
    explícito en vez de una implementación falsa que aparente estar
    completa."""
    logger.info("telemetry_events.id=%s recibido, pendiente de procesar", telemetry_event_id)


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    listener = TelemetryListener(settings.database_url_async, default_handler)
    await listener.start()


if __name__ == "__main__":
    asyncio.run(main())
