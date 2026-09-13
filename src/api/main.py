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
    EmbeddingSetting,
    LlmModel,
    PipelineTemplate,
    PromptArtifact,
    PromptTemplate,
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

    # Refuerzo LLM del semáforo (0004): el modelo pequeño detecta riesgos
    # que el checklist automático no cubre. Fusión CONSERVADORA: solo
    # puede escalar (auto_pass -> needs_human_review), nunca bajar la
    # severidad. Si el LLM no está disponible, degrada a solo-reglas.
    llm_reinforcement = None
    final_verdict = result.verdict.value
    try:
        from src.llm.reinforcement import reinforce_alignment

        brief_data = {
            "resumen": brief.resumen,
            "insight_core": brief.insight_core,
            "pitch_15s": brief.pitch_15s,
            "risk_level": _val(brief.risk_level),
            "segment_client": _val(brief.segment_client),
            "content_bucket": _val(brief.content_bucket),
            "brand_objective": _val(brief.brand_objective),
        }
        llm_reinforcement = reinforce_alignment(
            session,
            brief_data=brief_data,
            rule_verdict=result.verdict.value,
            checklist_results=[
                {"item": r.item, "status": r.status.value}
                for r in result.checklist_results
            ],
        )
        if llm_reinforcement.get("status") == "ok":
            final_verdict = llm_reinforcement["verdict"]
    except Exception:  # noqa: BLE001 — el refuerzo nunca rompe el alineamiento
        llm_reinforcement = {"status": "error", "reason": "reinforcement_failed"}

    # El veredicto del Gatekeeper decide el estado del brief (§3.9):
    # fail → retrabajo (generando); needs_human_review/auto_pass → revisión
    # (esperando la aprobación del 🟨 líder vía POST /briefs/{id}/approve).
    if _val(brief.status) in ("idea", "brief", "revision", "generando"):
        brief.status = "generando" if final_verdict == "fail" else "revision"
        session.commit()

    semaforo = {
        "auto_pass": "🟢",
        "needs_human_review": "🟡",
        "fail": "🔴",
    }[final_verdict]

    return {
        "brief_id": str(brief.id),
        "brand_objective": _val(brief.brand_objective),
        "content_bucket": _val(brief.content_bucket),
        "semaforo": semaforo,
        "verdict": final_verdict,
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
        "llm_reinforcement": llm_reinforcement,
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
    fallback_model: str | None = None
    llm_retries: int = 3
    temperature_small: float = 0.7
    temperature_large: float = 0.7
    max_tokens_small: int = 2048
    max_tokens_large: int = 4096
    is_active: bool = True


class SettingsUpdate(BaseModel):
    api_key_id: uuid.UUID | None = None
    small_model: str | None = None
    large_model: str | None = None
    fallback_model: str | None = None
    llm_retries: int | None = None
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
        "fallback_model": s.fallback_model,
        "llm_retries": int(s.llm_retries),
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
        fallback_model=body.fallback_model,
        llm_retries=body.llm_retries,
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
    if body.fallback_model is not None:
        settings.fallback_model = body.fallback_model
    if body.llm_retries is not None:
        settings.llm_retries = body.llm_retries
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


# ---------------------------------------------------------------------
# Infraestructura de IA modular (0004) — CRUD de modelos, specs y artefactos
# ---------------------------------------------------------------------


class LlmModelCreate(BaseModel):
    model_name: str = Field(..., examples=["meta-models/Muse-Glimmer-30B"])
    provider: str = Field(..., examples=["together"])
    model_size: str = Field(..., examples=["small"])  # small | large | embedding
    context_window: int = 0
    max_output_tokens: int = 0
    temperature_default: float = 0.7
    strengths: list[str] = []
    weaknesses: list[str] = []
    prompt_style: str = ""
    syntax_profile: dict = {}
    is_active: bool = True


class LlmModelUpdate(BaseModel):
    model_name: str | None = None
    provider: str | None = None
    model_size: str | None = None
    context_window: int | None = None
    max_output_tokens: int | None = None
    temperature_default: float | None = None
    strengths: list[str] | None = None
    weaknesses: list[str] | None = None
    prompt_style: str | None = None
    syntax_profile: dict | None = None
    is_active: bool | None = None


class PromptTemplateCreate(BaseModel):
    task_key: str = Field(..., examples=["alignment_reinforcement"])
    version: str = "1.0"
    intent: str = Field(..., examples=["Eres el refuerzo del semáforo..."])
    rules: list[str] = []
    input_schema: dict = {}
    output_schema: dict = {}
    few_shot: list = []
    is_active: bool = True
    # Ítems del checklist de pipeline_templates para validar critic_checklist.
    checklist_items: list[str] | None = None


class PromptTemplateUpdate(BaseModel):
    task_key: str | None = None
    version: str | None = None
    intent: str | None = None
    rules: list[str] | None = None
    input_schema: dict | None = None
    output_schema: dict | None = None
    few_shot: list | None = None
    is_active: bool | None = None
    checklist_items: list[str] | None = None


class EmbeddingSettingsCreate(BaseModel):
    api_key_id: uuid.UUID
    llm_model_id: uuid.UUID
    dimension: int = 1024
    is_active: bool = True


class EmbeddingSettingsUpdate(BaseModel):
    api_key_id: uuid.UUID | None = None
    llm_model_id: uuid.UUID | None = None
    dimension: int | None = None
    is_active: bool | None = None


def _llm_model_to_dict(m: LlmModel) -> dict:
    return {
        "id": str(m.id),
        "model_name": m.model_name,
        "provider": m.provider,
        "model_size": m.model_size,
        "context_window": m.context_window,
        "max_output_tokens": m.max_output_tokens,
        "temperature_default": float(m.temperature_default),
        "strengths": m.strengths or [],
        "weaknesses": m.weaknesses or [],
        "prompt_style": m.prompt_style,
        "syntax_profile": m.syntax_profile or {},
        "is_active": m.is_active,
        "created_at": m.created_at.isoformat() if m.created_at else None,
        "updated_at": m.updated_at.isoformat() if m.updated_at else None,
    }


def _prompt_template_to_dict(t: PromptTemplate) -> dict:
    return {
        "id": str(t.id),
        "task_key": t.task_key,
        "version": t.version,
        "intent": t.intent,
        "rules": t.rules or [],
        "input_schema": t.input_schema or {},
        "output_schema": t.output_schema or {},
        "few_shot": t.few_shot or [],
        "is_active": t.is_active,
        "created_at": t.created_at.isoformat() if t.created_at else None,
        "updated_at": t.updated_at.isoformat() if t.updated_at else None,
    }


def _prompt_artifact_to_dict(a: PromptArtifact, include_text: bool = False) -> dict:
    data = {
        "id": str(a.id),
        "llm_model_id": str(a.llm_model_id),
        "task_key": a.task_key,
        "spec_version": a.spec_version,
        "artifact_version": a.artifact_version,
        "content_hash": a.content_hash,
        "compiled_by": a.compiled_by,
        "is_active": a.is_active,
        "created_at": a.created_at.isoformat() if a.created_at else None,
    }
    if include_text:
        data["prompt_text"] = a.prompt_text
    return data


def _embedding_settings_to_dict(s: EmbeddingSetting) -> dict:
    return {
        "id": str(s.id),
        "api_key_id": str(s.api_key_id),
        "llm_model_id": str(s.llm_model_id),
        "dimension": int(s.dimension),
        "is_active": s.is_active,
        "created_at": s.created_at.isoformat() if s.created_at else None,
        "updated_at": s.updated_at.isoformat() if s.updated_at else None,
    }


def _validate_critic(
    template: PromptTemplate, checklist_items: list[str] | None
) -> None:
    """Valida que todo ítem del checklist tenga interpretación en rules.
    Import lazy: el compilador puede no existir aún en algunos entornos."""
    if template.task_key != "critic_checklist" or not checklist_items:
        return
    try:
        from src.llm.compiler import validate_critic_spec

        missing = validate_critic_spec(template, checklist_items)
    except Exception:  # noqa: BLE001 — si el compilador no está, no bloqueamos
        return
    if missing:
        raise HTTPException(
            status_code=422,
            detail=(
                "Ítems del checklist sin interpretación en rules de critic_checklist: "
                f"{', '.join(missing)}. Añade una regla '<item>: ...' para cada uno."
            ),
        )


@app.get("/llm-models", response_model=list[dict])
def list_llm_models(session: Session = Depends(get_session)) -> list[dict]:
    rows = (
        session.execute(select(LlmModel).order_by(LlmModel.created_at.desc()))
        .scalars()
        .all()
    )
    return [_llm_model_to_dict(m) for m in rows]


@app.post("/llm-models", status_code=201)
def create_llm_model(
    body: LlmModelCreate, session: Session = Depends(get_session)
) -> dict:
    exists = session.execute(
        select(LlmModel).where(LlmModel.model_name == body.model_name)
    ).scalar_one_or_none()
    if exists is not None:
        raise HTTPException(
            status_code=409,
            detail=f"Ya existe un modelo con nombre {body.model_name!r}.",
        )
    model = LlmModel(**body.model_dump())
    session.add(model)
    session.commit()
    session.refresh(model)
    return _llm_model_to_dict(model)


@app.get("/llm-models/{model_id}", response_model=dict)
def get_llm_model(model_id: uuid.UUID, session: Session = Depends(get_session)) -> dict:
    model = session.get(LlmModel, model_id)
    if model is None:
        raise HTTPException(
            status_code=404, detail=f"llm_model {model_id} no encontrado."
        )
    return _llm_model_to_dict(model)


@app.put("/llm-models/{model_id}", response_model=dict)
def update_llm_model(
    model_id: uuid.UUID,
    body: LlmModelUpdate,
    session: Session = Depends(get_session),
) -> dict:
    model = session.get(LlmModel, model_id)
    if model is None:
        raise HTTPException(
            status_code=404, detail=f"llm_model {model_id} no encontrado."
        )
    for field_name, value in body.model_dump(exclude_unset=True).items():
        setattr(model, field_name, value)
    session.commit()
    session.refresh(model)
    return _llm_model_to_dict(model)


@app.delete("/llm-models/{model_id}", status_code=204)
def delete_llm_model(
    model_id: uuid.UUID, session: Session = Depends(get_session)
) -> None:
    model = session.get(LlmModel, model_id)
    if model is None:
        raise HTTPException(
            status_code=404, detail=f"llm_model {model_id} no encontrado."
        )
    session.delete(model)
    session.commit()


@app.get("/prompt-templates", response_model=list[dict])
def list_prompt_templates(session: Session = Depends(get_session)) -> list[dict]:
    rows = (
        session.execute(
            select(PromptTemplate).order_by(PromptTemplate.created_at.desc())
        )
        .scalars()
        .all()
    )
    return [_prompt_template_to_dict(t) for t in rows]


@app.post("/prompt-templates", status_code=201)
def create_prompt_template(
    body: PromptTemplateCreate, session: Session = Depends(get_session)
) -> dict:
    exists = session.execute(
        select(PromptTemplate).where(PromptTemplate.task_key == body.task_key)
    ).scalar_one_or_none()
    if exists is not None:
        raise HTTPException(
            status_code=409,
            detail=f"Ya existe un template con task_key {body.task_key!r}.",
        )
    data = body.model_dump(exclude={"checklist_items"})
    template = PromptTemplate(**data)
    _validate_critic(template, body.checklist_items)
    session.add(template)
    session.commit()
    session.refresh(template)
    return _prompt_template_to_dict(template)


@app.get("/prompt-templates/{template_id}", response_model=dict)
def get_prompt_template(
    template_id: uuid.UUID, session: Session = Depends(get_session)
) -> dict:
    template = session.get(PromptTemplate, template_id)
    if template is None:
        raise HTTPException(
            status_code=404, detail=f"prompt_template {template_id} no encontrado."
        )
    return _prompt_template_to_dict(template)


@app.put("/prompt-templates/{template_id}", response_model=dict)
def update_prompt_template(
    template_id: uuid.UUID,
    body: PromptTemplateUpdate,
    session: Session = Depends(get_session),
) -> dict:
    template = session.get(PromptTemplate, template_id)
    if template is None:
        raise HTTPException(
            status_code=404, detail=f"prompt_template {template_id} no encontrado."
        )
    data = body.model_dump(exclude_unset=True, exclude={"checklist_items"})
    for field_name, value in data.items():
        setattr(template, field_name, value)
    _validate_critic(template, body.checklist_items)
    session.commit()
    session.refresh(template)
    return _prompt_template_to_dict(template)


@app.delete("/prompt-templates/{template_id}", status_code=204)
def delete_prompt_template(
    template_id: uuid.UUID, session: Session = Depends(get_session)
) -> None:
    template = session.get(PromptTemplate, template_id)
    if template is None:
        raise HTTPException(
            status_code=404, detail=f"prompt_template {template_id} no encontrado."
        )
    session.delete(template)
    session.commit()


@app.get("/prompt-artifacts", response_model=list[dict])
def list_prompt_artifacts(session: Session = Depends(get_session)) -> list[dict]:
    """Lista SIN prompt_text (es grande); el detalle individual lo incluye."""
    rows = (
        session.execute(
            select(PromptArtifact).order_by(PromptArtifact.created_at.desc())
        )
        .scalars()
        .all()
    )
    return [_prompt_artifact_to_dict(a) for a in rows]


@app.get("/prompt-artifacts/{artifact_id}", response_model=dict)
def get_prompt_artifact(
    artifact_id: uuid.UUID, session: Session = Depends(get_session)
) -> dict:
    artifact = session.get(PromptArtifact, artifact_id)
    if artifact is None:
        raise HTTPException(
            status_code=404, detail=f"prompt_artifact {artifact_id} no encontrado."
        )
    return _prompt_artifact_to_dict(artifact, include_text=True)


@app.get("/embedding-settings", response_model=dict)
def get_active_embedding_settings(session: Session = Depends(get_session)) -> dict:
    setting = (
        session.execute(
            select(EmbeddingSetting).where(EmbeddingSetting.is_active.is_(True))
        )
        .scalars()
        .first()
    )
    if setting is None:
        raise HTTPException(status_code=404, detail="No hay embedding_settings activa.")
    return _embedding_settings_to_dict(setting)


@app.post("/embedding-settings", status_code=201)
def create_embedding_settings(
    body: EmbeddingSettingsCreate, session: Session = Depends(get_session)
) -> dict:
    key = session.get(ApiKey, body.api_key_id)
    if key is None:
        raise HTTPException(
            status_code=404, detail=f"api_key {body.api_key_id} no encontrada."
        )
    model = session.get(LlmModel, body.llm_model_id)
    if model is None:
        raise HTTPException(
            status_code=404, detail=f"llm_model {body.llm_model_id} no encontrado."
        )
    if body.is_active:
        session.execute(
            update(EmbeddingSetting)
            .where(EmbeddingSetting.is_active.is_(True))
            .values(is_active=False)
        )
    setting = EmbeddingSetting(
        api_key_id=body.api_key_id,
        llm_model_id=body.llm_model_id,
        dimension=body.dimension,
        is_active=body.is_active,
    )
    session.add(setting)
    session.commit()
    session.refresh(setting)
    return _embedding_settings_to_dict(setting)


@app.put("/embedding-settings/{settings_id}", response_model=dict)
def update_embedding_settings(
    settings_id: uuid.UUID,
    body: EmbeddingSettingsUpdate,
    session: Session = Depends(get_session),
) -> dict:
    setting = session.get(EmbeddingSetting, settings_id)
    if setting is None:
        raise HTTPException(
            status_code=404, detail=f"embedding_settings {settings_id} no encontrada."
        )
    if body.api_key_id is not None:
        key = session.get(ApiKey, body.api_key_id)
        if key is None:
            raise HTTPException(
                status_code=404, detail=f"api_key {body.api_key_id} no encontrada."
            )
        setting.api_key_id = body.api_key_id
    if body.llm_model_id is not None:
        model = session.get(LlmModel, body.llm_model_id)
        if model is None:
            raise HTTPException(
                status_code=404, detail=f"llm_model {body.llm_model_id} no encontrado."
            )
        setting.llm_model_id = body.llm_model_id
    if body.dimension is not None:
        setting.dimension = body.dimension
    if body.is_active is not None:
        setting.is_active = body.is_active
    session.commit()
    session.refresh(setting)
    return _embedding_settings_to_dict(setting)


@app.delete("/embedding-settings/{settings_id}", status_code=204)
def delete_embedding_settings(
    settings_id: uuid.UUID, session: Session = Depends(get_session)
) -> None:
    setting = session.get(EmbeddingSetting, settings_id)
    if setting is None:
        raise HTTPException(
            status_code=404, detail=f"embedding_settings {settings_id} no encontrada."
        )
    session.delete(setting)
    session.commit()


@app.post("/settings/ensure-prompts", response_model=dict)
def ensure_prompts(session: Session = Depends(get_session)) -> dict:
    """Compila las specs a artefactos inmutables (idempotente).
    Import lazy: el compilador puede no existir aún en algunos entornos."""
    try:
        from src.llm.compiler import compile_prompts

        return compile_prompts(session)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=f"No se pudo compilar los prompts: {exc}",
        )
