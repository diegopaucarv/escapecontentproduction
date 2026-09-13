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
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from src.agents.alignment import AlignmentInput, run_alignment
from src.auth import (
    authenticate_user,
    create_access_token,
    require_role,
)
from src.db.models import (
    ApiKey,
    AppUser,
    ContentBrief,
    PipelineTemplate,
    SessionSettings,
    TelemetryEvent,
)
from src.db.session import get_session

app = FastAPI(title="Pipeline Unificado — ESCAPE / Ergalia", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/auth/token")
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    session: Session = Depends(get_session),
) -> dict:
    """Login OAuth2 (form: username=email, password). Devuelve un JWT
    firmado con JWT_SECRET. Es el flujo real de RBAC — ver src/auth.py."""
    user = authenticate_user(session, form_data.username, form_data.password)
    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Email o password incorrectos.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(user)
    return {"access_token": token, "token_type": "bearer"}


class TelemetryPayload(BaseModel):
    artifact_id: uuid.UUID | None = None
    event_type: str = Field(
        ..., examples=["click", "view", "retention_24h", "conversion_30d"]
    )
    payload: dict = Field(default_factory=dict)


@app.post("/webhooks/telemetry", status_code=201)
def ingest_telemetry(
    body: TelemetryPayload, session: Session = Depends(get_session)
) -> dict:
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


@app.get("/briefs", response_model=list[dict])
def list_briefs(session: Session = Depends(get_session)) -> list[dict]:
    """Lista los briefs existentes (ordenados por más reciente primero).
    Devuelve solo los campos de identificación para que la lista sea
    ligera; el detalle completo vive en GET /briefs/{brief_id}."""
    rows = (
        session.execute(select(ContentBrief).order_by(ContentBrief.created_at.desc()))
        .scalars()
        .all()
    )
    return [
        {
            "id": str(b.id),
            "status": b.status.value if hasattr(b.status, "value") else b.status,
            "brand_objective": b.brand_objective.value
            if hasattr(b.brand_objective, "value")
            else b.brand_objective,
            "content_bucket": b.content_bucket.value
            if hasattr(b.content_bucket, "value")
            else b.content_bucket,
            "resumen": b.resumen,
            "insight_core": b.insight_core,
            "created_at": b.created_at.isoformat() if b.created_at else None,
            "updated_at": b.updated_at.isoformat() if b.updated_at else None,
        }
        for b in rows
    ]


@app.get("/briefs/{brief_id}", response_model=dict)
def get_brief(brief_id: uuid.UUID, session: Session = Depends(get_session)) -> dict:
    """Lee un Content Brief completo por id — la primera acción del
    pipeline (recepción / brief). El brief puede haber sido editado por
    otro servidor; este endpoint devuelve siempre el estado vigente en
    la base (updated_at refleja la última edición)."""
    brief = session.get(ContentBrief, brief_id)
    if brief is None:
        raise HTTPException(status_code=404, detail=f"Brief {brief_id} no encontrado.")
    return _brief_to_dict(brief)


def _val(v):
    """Convierte un enum de SQLAlchemy a su valor string; pasa el resto tal cual."""
    if hasattr(v, "value"):
        return v.value
    return v


def _brief_to_dict(brief: ContentBrief) -> dict:
    """Serializa un ContentBrief completo (Bloques A-H) a dict JSON.
    Los enums se convierten a su valor string; los JSONB se pasan tal cual."""
    return {
        "id": str(brief.id),
        "owner_id": str(brief.owner_id) if brief.owner_id else None,
        "status": _val(brief.status),
        # Bloque B — origen y contexto
        "resumen": brief.resumen,
        "insight_core": brief.insight_core,
        "pitch_15s": brief.pitch_15s,
        "prior_attempts": brief.prior_attempts,
        "risks": brief.risks,
        "novelty_indicators": brief.novelty_indicators,
        "suggested_product_type": brief.suggested_product_type,
        "org_priorities_contrast": brief.org_priorities_contrast,
        # Bloque C — jerarquía canónica
        "brand_objective": _val(brief.brand_objective),
        "phase_number": brief.phase_number,
        "segment_client": _val(brief.segment_client),
        "segment_community": _val(brief.segment_community),
        "interlocutor_profile": brief.interlocutor_profile,
        "subprofile": brief.subprofile,
        "audience_tier": brief.audience_tier,
        "need_id": brief.need_id,
        "need": brief.need,
        "change_hypothesis": brief.change_hypothesis,
        # Bloque D — bucket y oferta
        "content_bucket": _val(brief.content_bucket),
        "service_category": brief.service_category,
        "product_anchor": brief.product_anchor,
        "entry_offer": brief.entry_offer,
        # Bloque E — artefacto y canal
        "artifact_type": _val(brief.artifact_type),
        "channel": brief.channel,
        "channel_role": brief.channel_role,
        "cta": brief.cta,
        "landing": brief.landing,
        "funnel_stage": brief.funnel_stage,
        # Bloque F — geografía e idioma
        "language": brief.language,
        "geography_content": brief.geography_content,
        "geography_sales": brief.geography_sales,
        # Bloque G — gobernanza, evidencia y reutilización
        "evidence_source": brief.evidence_source,
        "risk_level": _val(brief.risk_level),
        "debate_governance": brief.debate_governance,
        "validation_required": brief.validation_required,
        "repurpose_plan": brief.repurpose_plan,
        # Bloque H — métricas y enrutamiento
        "metric_primary": brief.metric_primary,
        "metric_secondary": brief.metric_secondary,
        "novelty_score": brief.novelty_score,
        "route_decision": _val(brief.route_decision),
        "production_route": _val(brief.production_route),
        "pipeline_template_id": brief.pipeline_template_id,
        "created_at": brief.created_at.isoformat() if brief.created_at else None,
        "updated_at": brief.updated_at.isoformat() if brief.updated_at else None,
    }


@app.post("/briefs/{brief_id}/alignment", response_model=dict)
def align_brief(brief_id: uuid.UUID, session: Session = Depends(get_session)) -> dict:
    """Segunda acción del pipeline: alineamiento estratégico.

    Revisa el brief contra el plan de marca (pipeline_templates: checklist
    + novelty_weights) y devuelve el semáforo 🟢🟡🔴. El enrutamiento por
    novedad (novelty_router) corre de forma asíncrona y puebla
    route_decision/production_route; aquí se usan como contexto.
    """
    brief = session.get(ContentBrief, brief_id)
    if brief is None:
        raise HTTPException(status_code=404, detail=f"Brief {brief_id} no encontrado.")

    template = session.execute(
        select(PipelineTemplate).where(
            PipelineTemplate.brand_objective == brief.brand_objective,
            PipelineTemplate.content_bucket == brief.content_bucket,
        )
    ).scalar_one_or_none()
    if template is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No hay plan de marca (pipeline_template) para "
                f"{_val(brief.brand_objective)}/{_val(brief.content_bucket)}."
            ),
        )

    inputs = AlignmentInput(
        brand_objective=_val(brief.brand_objective),
        content_bucket=_val(brief.content_bucket),
        checklist=template.checklist or [],
        novelty_weights=template.novelty_weights or {},
        evidence_source=brief.evidence_source,
        pitch_15s=brief.pitch_15s,
        cta=brief.cta,
        risk_level=_val(brief.risk_level),
        segment_client=_val(brief.segment_client),
        validation_required=brief.validation_required,
        route_decision=_val(brief.route_decision),
        production_route=_val(brief.production_route),
        novelty_score=brief.novelty_score,
    )
    result = run_alignment(inputs)

    # El veredicto del Gatekeeper decide el estado del brief (§3.9):
    # fail → retrabajo (generando); needs_human_review/auto_pass → revisión
    # (esperando la aprobación del 🟨 líder vía POST /briefs/{id}/approve).
    if _val(brief.status) in ("idea", "brief", "revision", "generando"):
        brief.status = "generando" if result.verdict.value == "fail" else "revision"
        session.commit()

    semaforo = {
        "auto_pass": "🟢",
        "needs_human_review": "🟡",
        "fail": "🔴",
    }[result.verdict.value]

    return {
        "brief_id": str(brief.id),
        "brand_objective": _val(brief.brand_objective),
        "content_bucket": _val(brief.content_bucket),
        "semaforo": semaforo,
        "verdict": result.verdict.value,
        "checklist": [
            {"item": r.item, "status": r.status.value, "reason": r.reason}
            for r in result.checklist_results
        ],
        "failed_items": result.failed_items,
        "human_review_items": result.human_review_items,
        "route": {
            "route_decision": _val(brief.route_decision),
            "production_route": _val(brief.production_route),
            "novelty_score": brief.novelty_score,
        },
        "reasoning": result.reasoning,
    }


@app.post("/briefs/{brief_id}/approve", response_model=dict)
def approve_brief(
    brief_id: uuid.UUID,
    session: Session = Depends(get_session),
    user: AppUser = Depends(require_role("lider")),
) -> dict:
    """OWNER_APPROVAL del Pre-Deploy Gate — decisión humana explícita.

    Solo un 🟨 líder puede aprobar. Mueve el brief de `revision` a
    `aprobado` (ver máquina de estados §3.9 del doc). El Gatekeeper
    (src/agents/gatekeeper.py) nunca puede aprobar en solitario cuando
    el riesgo/segmento/ruta exige revisión humana — este endpoint es
    ese paso humano.
    """
    brief = session.get(ContentBrief, brief_id)
    if brief is None:
        raise HTTPException(status_code=404, detail=f"Brief {brief_id} no encontrado.")
    if _val(brief.status) != "revision":
        raise HTTPException(
            status_code=409,
            detail=(
                f"Solo se puede aprobar un brief en estado 'revision'; "
                f"este está en '{_val(brief.status)}'."
            ),
        )
    brief.status = "aprobado"
    session.commit()
    session.refresh(brief)
    return {
        "brief_id": str(brief.id),
        "status": _val(brief.status),
        "approved_by": str(user.id),
        "approved_at": brief.updated_at.isoformat() if brief.updated_at else None,
    }


@app.post("/briefs", status_code=201)
def create_brief(
    body: ContentBriefCreate, session: Session = Depends(get_session)
) -> dict:
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


# ---------------------------------------------------------------------
# Infraestructura LLM — CRUD de api_keys y session_settings
# ---------------------------------------------------------------------


def _mask_key(api_key: str) -> str:
    """Enmascara la clave: muestra el prefijo y los últimos 4 caracteres.
    La clave completa NUNCA se devuelve por la API."""
    if len(api_key) <= 8:
        return "****"
    return f"{api_key[:7]}****{api_key[-4:]}"


class ApiKeyCreate(BaseModel):
    provider: str = Field(..., examples=["together"])
    key_name: str = Field(..., examples=["together-main"])
    api_key: str = Field(..., min_length=8, examples=["tgp_v1_..."])
    is_active: bool = True


class ApiKeyUpdate(BaseModel):
    provider: str | None = None
    key_name: str | None = None
    api_key: str | None = Field(default=None, min_length=8)
    is_active: bool | None = None


class SettingsCreate(BaseModel):
    api_key_id: uuid.UUID
    small_model: str = Field(..., examples=["meta-models/Muse-Glimmer-30B"])
    large_model: str = Field(..., examples=["deepseek-ai/DeepSeek-V4-Flash-0731"])
    temperature_small: float = 0.7
    temperature_large: float = 0.7
    max_tokens_small: int = 2048
    max_tokens_large: int = 4096
    is_active: bool = True


class SettingsUpdate(BaseModel):
    api_key_id: uuid.UUID | None = None
    small_model: str | None = None
    large_model: str | None = None
    temperature_small: float | None = None
    temperature_large: float | None = None
    max_tokens_small: int | None = None
    max_tokens_large: int | None = None
    is_active: bool | None = None


def _api_key_to_dict(k: ApiKey) -> dict:
    return {
        "id": str(k.id),
        "provider": k.provider,
        "key_name": k.key_name,
        "api_key_masked": _mask_key(k.api_key),
        "is_active": k.is_active,
        "created_at": k.created_at.isoformat() if k.created_at else None,
        "updated_at": k.updated_at.isoformat() if k.updated_at else None,
    }


@app.get("/api-keys", response_model=list[dict])
def list_api_keys(session: Session = Depends(get_session)) -> list[dict]:
    rows = (
        session.execute(select(ApiKey).order_by(ApiKey.created_at.desc()))
        .scalars()
        .all()
    )
    return [_api_key_to_dict(k) for k in rows]


@app.post("/api-keys", status_code=201)
def create_api_key(body: ApiKeyCreate, session: Session = Depends(get_session)) -> dict:
    key = ApiKey(
        provider=body.provider,
        key_name=body.key_name,
        api_key=body.api_key,
        is_active=body.is_active,
    )
    session.add(key)
    session.commit()
    session.refresh(key)
    return _api_key_to_dict(key)


@app.get("/api-keys/{key_id}", response_model=dict)
def get_api_key(key_id: uuid.UUID, session: Session = Depends(get_session)) -> dict:
    key = session.get(ApiKey, key_id)
    if key is None:
        raise HTTPException(status_code=404, detail=f"api_key {key_id} no encontrada.")
    return _api_key_to_dict(key)


@app.put("/api-keys/{key_id}", response_model=dict)
def update_api_key(
    key_id: uuid.UUID, body: ApiKeyUpdate, session: Session = Depends(get_session)
) -> dict:
    key = session.get(ApiKey, key_id)
    if key is None:
        raise HTTPException(status_code=404, detail=f"api_key {key_id} no encontrada.")
    if body.provider is not None:
        key.provider = body.provider
    if body.key_name is not None:
        key.key_name = body.key_name
    if body.api_key is not None:
        key.api_key = body.api_key
    if body.is_active is not None:
        key.is_active = body.is_active
    session.commit()
    session.refresh(key)
    return _api_key_to_dict(key)


@app.delete("/api-keys/{key_id}", status_code=204)
def delete_api_key(key_id: uuid.UUID, session: Session = Depends(get_session)) -> None:
    key = session.get(ApiKey, key_id)
    if key is None:
        raise HTTPException(status_code=404, detail=f"api_key {key_id} no encontrada.")
    session.delete(key)
    session.commit()


def _settings_to_dict(s: SessionSettings) -> dict:
    return {
        "id": str(s.id),
        "api_key_id": str(s.api_key_id),
        "small_model": s.small_model,
        "large_model": s.large_model,
        "temperature_small": float(s.temperature_small),
        "temperature_large": float(s.temperature_large),
        "max_tokens_small": int(s.max_tokens_small),
        "max_tokens_large": int(s.max_tokens_large),
        "is_active": s.is_active,
        "created_at": s.created_at.isoformat() if s.created_at else None,
        "updated_at": s.updated_at.isoformat() if s.updated_at else None,
    }


@app.get("/settings", response_model=dict)
def get_active_settings(session: Session = Depends(get_session)) -> dict:
    """Devuelve la session_settings activa (singleton)."""
    settings = (
        session.execute(
            select(SessionSettings).where(SessionSettings.is_active.is_(True))
        )
        .scalars()
        .first()
    )
    if settings is None:
        raise HTTPException(status_code=404, detail="No hay session_settings activa.")
    return _settings_to_dict(settings)


@app.post("/settings", status_code=201)
def create_settings(
    body: SettingsCreate, session: Session = Depends(get_session)
) -> dict:
    key = session.get(ApiKey, body.api_key_id)
    if key is None:
        raise HTTPException(
            status_code=404, detail=f"api_key {body.api_key_id} no encontrada."
        )
    if body.is_active:
        # Mantiene el singleton: desactiva la activa previa.
        session.execute(
            update(SessionSettings)
            .where(SessionSettings.is_active.is_(True))
            .values(is_active=False)
        )
    settings = SessionSettings(
        api_key_id=body.api_key_id,
        small_model=body.small_model,
        large_model=body.large_model,
        temperature_small=body.temperature_small,
        temperature_large=body.temperature_large,
        max_tokens_small=body.max_tokens_small,
        max_tokens_large=body.max_tokens_large,
        is_active=body.is_active,
    )
    session.add(settings)
    session.commit()
    session.refresh(settings)
    return _settings_to_dict(settings)


@app.put("/settings/{settings_id}", response_model=dict)
def update_settings(
    settings_id: uuid.UUID,
    body: SettingsUpdate,
    session: Session = Depends(get_session),
) -> dict:
    settings = session.get(SessionSettings, settings_id)
    if settings is None:
        raise HTTPException(
            status_code=404, detail=f"session_settings {settings_id} no encontrada."
        )
    if body.api_key_id is not None:
        key = session.get(ApiKey, body.api_key_id)
        if key is None:
            raise HTTPException(
                status_code=404, detail=f"api_key {body.api_key_id} no encontrada."
            )
        settings.api_key_id = body.api_key_id
    if body.small_model is not None:
        settings.small_model = body.small_model
    if body.large_model is not None:
        settings.large_model = body.large_model
    if body.temperature_small is not None:
        settings.temperature_small = body.temperature_small
    if body.temperature_large is not None:
        settings.temperature_large = body.temperature_large
    if body.max_tokens_small is not None:
        settings.max_tokens_small = body.max_tokens_small
    if body.max_tokens_large is not None:
        settings.max_tokens_large = body.max_tokens_large
    if body.is_active is not None:
        settings.is_active = body.is_active
    session.commit()
    session.refresh(settings)
    return _settings_to_dict(settings)


@app.delete("/settings/{settings_id}", status_code=204)
def delete_settings(
    settings_id: uuid.UUID, session: Session = Depends(get_session)
) -> None:
    settings = session.get(SessionSettings, settings_id)
    if settings is None:
        raise HTTPException(
            status_code=404, detail=f"session_settings {settings_id} no encontrada."
        )
    session.delete(settings)
    session.commit()
