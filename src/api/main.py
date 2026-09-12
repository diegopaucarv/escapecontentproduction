"""
API mínima. La propuesta original mencionaba "API de ingesta de
telemetría (webhooks)" en el texto y `fastapi`/`uvicorn` en
requirements.txt, pero no existía ningún servicio `api` en el
docker-compose ni un solo archivo de la API — este módulo llena ese
vacío concreto.
"""
from __future__ import annotations

import uuid

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.db.models import ContentBrief, TelemetryEvent
from src.db.session import get_session

app = FastAPI(title="Pipeline Unificado — ESCAPE / Ergalia", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


class TelemetryPayload(BaseModel):
    artifact_id: uuid.UUID | None = None
    event_type: str = Field(..., examples=["click", "view", "retention_24h", "conversion_30d"])
    payload: dict = Field(default_factory=dict)


@app.post("/webhooks/telemetry", status_code=201)
def ingest_telemetry(body: TelemetryPayload, session: Session = Depends(get_session)) -> dict:
    """Punto de entrada único para UTMs/clics/retención. El INSERT dispara
    el trigger `trg_telemetry_notify` (pg_notify) que despierta al worker
    async en src/events/listener.py — no hace falta llamar a nada más
    desde aquí para que el resto del sistema se entere."""
    event = TelemetryEvent(
        artifact_id=body.artifact_id,
        event_type=body.event_type,
        payload=body.payload,
    )
    session.add(event)
    session.commit()
    session.refresh(event)
    return {"id": event.id, "received_at": event.received_at.isoformat()}


class ContentBriefCreate(BaseModel):
    brand_objective: str
    content_bucket: str
    resumen: str
    insight_core: str
    segment_client: str | None = None
    segment_community: str | None = None
    owner_id: uuid.UUID | None = None


@app.post("/briefs", status_code=201)
def create_brief(body: ContentBriefCreate, session: Session = Depends(get_session)) -> dict:
    if not body.segment_client and not body.segment_community:
        raise HTTPException(
            status_code=422,
            detail="Todo Content Brief requiere segment_client o segment_community (canon §9.2).",
        )
    brief = ContentBrief(
        brand_objective=body.brand_objective,
        content_bucket=body.content_bucket,
        resumen=body.resumen,
        insight_core=body.insight_core,
        segment_client=body.segment_client,
        segment_community=body.segment_community,
        owner_id=body.owner_id,
    )
    session.add(brief)
    session.commit()
    session.refresh(brief)
    # Enrutamiento por novedad (src/agents/novelty_router.py) se dispara
    # aquí en una versión real, de forma asíncrona; se deja fuera de este
    # endpoint para no acoplar la API HTTP a la latencia de un embedding.
    return {"id": str(brief.id), "status": brief.status}
