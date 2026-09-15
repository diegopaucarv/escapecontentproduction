"""
API mínima. La propuesta original mencionaba "API de ingesta de
telemetría (webhooks)" en el texto y `fastapi`/`uvicorn` en
requirements.txt, pero no existía ningún servicio `api` en el
docker-compose ni un solo archivo de la API — este módulo llena ese
vacío concreto.
"""

from __future__ import annotations

import uuid
from datetime import date
from pathlib import Path

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
from src.config import get_settings
from src.db.models import (
    ApiKey,
    AppUser,
    AssetJob,
    CalendarSlot,
    ContentArtifact,
    ContentBrief,
    EmbeddingSetting,
    KaizenCycle,
    LlmModel,
    PipelineTemplate,
    ProductionManifest,
    ProductionOrder,
    ProductionTemplate,
    Project,
    ProjectVersion,
    PromptArtifact,
    PromptTemplate,
    SessionSettings,
    TelemetryEvent,
    ToolAdapter,
)
from src.db.session import get_session
from src.orders.flow import (
    create_order as kaizen_create_order,
)
from src.orders.flow import (
    kaizen_decision,
    order_note,
    publish_order,
    record_kpis,
    record_micro_kaizen,
)
from src.orders.flow import (
    produce_order as kaizen_produce_order,
)
from src.production.flow import produce_project
from src.production.generator import generate_snapshot
from src.production.postproduction import postproduction_recommendations
from src.production.produce_brief import resume_produce, run_produce
from src.production.refine import refine_snapshot, resolve_format_spec
from src.tools import (
    OrchestrationError,
    Orchestrator,
    create_manifest,
    list_manifests,
    load_manifest,
    validate_tool_chain,
)

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
        # 0007: la última decisión (alineamiento/refuerzo) fue determinista
        # y requiere aceptación explícita del usuario (auditoría).
        "requires_user_acceptance": brief.requires_user_acceptance,
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

    Filosofía 0007 (LLM-first): el LLM es la opción SIEMPRE presente y
    orquesta la decisión; el usuario es invitado a revisarla. Si el
    refuerzo LLM degrada a determinista (o falla), la decisión requiere
    aceptación explícita del usuario vía el flujo `revision` +
    POST /briefs/{id}/approve (OWNER_APPROVAL).
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

    # Filosofía 0007: el LLM es la opción SIEMPRE presente. Solo cuando el
    # refuerzo responde "ok" la decisión puede venir del LLM (o del humano
    # si el propio LLM lo indica); en cualquier otro caso (degradado,
    # skipped o error) la decisión es determinista por construcción y
    # requiere aceptación explícita del usuario.
    if llm_reinforcement and llm_reinforcement.get("status") == "ok":
        decision_source = llm_reinforcement.get("decision_source", "llm")
        requires_user_acceptance = bool(
            llm_reinforcement.get("requires_user_acceptance", False)
        )
    else:
        # Sin refuerzo LLM (degradado, skipped o error): la decisión es
        # determinista por construcción -> requiere aceptación del usuario.
        decision_source = "deterministic"
        requires_user_acceptance = True

    # Filosofía 0007: el flag se PERSISTE en el brief para auditoría —
    # si la última decisión fue determinista (degradación), el pipeline
    # sabe que debe pausar hasta la aceptación del usuario.
    brief.requires_user_acceptance = requires_user_acceptance

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
        "decision_source": decision_source,
        "requires_user_acceptance": requires_user_acceptance,
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


class ResumeDecision(BaseModel):
    """Decisión humana al reanudar el grafo Producer-Critic pausado."""

    decision: str = Field(..., pattern="^(approve|reject)$")


@app.post("/briefs/{brief_id}/produce", response_model=dict)
def produce_brief(
    brief_id: uuid.UUID,
    session: Session = Depends(get_session),
) -> dict:
    """Ejecuta el bucle Producer-Critic (0007) con sesión real.

    Requiere un brief 'aprobado' (OWNER_APPROVAL previo). Construye el
    estado del grafo desde el brief + su plan de marca y lo invoca con
    thread_id = brief_id (el interrupt/resume queda asociado al brief).
    Si el grafo se pausa (needs_human_review o degradación determinista),
    devuelve la interrupción; el humano la resuelve vía
    POST /briefs/{id}/produce/resume. Si auto-pasa, materializa la pieza
    como ContentArtifact (status 'borrador').
    """
    brief = session.get(ContentBrief, brief_id)
    if brief is None:
        raise HTTPException(status_code=404, detail=f"Brief {brief_id} no encontrado.")
    if _val(brief.status) != "aprobado":
        raise HTTPException(
            status_code=409,
            detail=(
                f"Solo se puede producir un brief 'aprobado'; "
                f"este está en '{_val(brief.status)}'."
            ),
        )
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

    result = run_produce(session, brief, template)

    # auto_pass sin interrupción: la pieza queda lista como artefacto.
    if result.get("verdict") == "auto_pass" and "__interrupt__" not in result:
        artifact = ContentArtifact(
            brief_id=brief.id,
            artifact_type=_val(brief.artifact_type) or "post",
            channel=brief.channel or "social",
            status="borrador",
        )
        session.add(artifact)
        session.commit()

    return {
        "brief_id": str(brief.id),
        "verdict": result.get("verdict"),
        "draft": result.get("draft"),
        "iteration": result.get("iteration"),
        "requires_user_acceptance": result.get("requires_user_acceptance"),
        "checklist_results": result.get("checklist_results"),
        "critic_feedback": result.get("critic_feedback"),
        "human_decision": result.get("human_decision"),
        "interrupted": "__interrupt__" in result,
        "interrupt": (
            result["__interrupt__"][0].value if "__interrupt__" in result else None
        ),
    }


@app.post("/briefs/{brief_id}/produce/resume", response_model=dict)
def resume_brief_production(
    brief_id: uuid.UUID,
    body: ResumeDecision,
    session: Session = Depends(get_session),
) -> dict:
    """Reanuda el grafo Producer-Critic pausado en human_review_node.

    La decisión humana ('approve' | 'reject') se entrega vía
    Command(resume=...) al mismo thread (thread_id = brief_id). Solo
    tiene sentido tras una pausa; si no hay pausa, LangGraph continúa
    desde el último checkpoint.
    """
    brief = session.get(ContentBrief, brief_id)
    if brief is None:
        raise HTTPException(status_code=404, detail=f"Brief {brief_id} no encontrado.")
    result = resume_produce(session, str(brief.id), body.decision)
    return {
        "brief_id": str(brief.id),
        "human_decision": result.get("human_decision"),
        "verdict": result.get("verdict"),
        "interrupted": "__interrupt__" in result,
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
    _auto_compile_prompts(session)
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
    _auto_compile_prompts(session)
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


def _auto_compile_prompts(session: Session) -> None:
    """Recompila artefactos tras crear/editar un prompt template.

    Idempotente: compile_prompts solo crea versiones nuevas si el contenido
    cambió. Un fallo de compilación no debe romper el CRUD — se ignora.
    """
    try:
        from src.llm.compiler import compile_prompts

        compile_prompts(session)
    except Exception:  # noqa: BLE001 — el CRUD no debe fallar por el compile
        pass


# ---------------------------------------------------------------------
# Fase 3 — API de producción: tool_adapters, production_templates,
# manifiestos, jobs y endpoint disparador /produce (diseño §17, pts. 13-15)
# ---------------------------------------------------------------------


class ToolAdapterCreate(BaseModel):
    name: str = Field(..., examples=["inkscape"])
    mcp_server_name: str = Field(..., examples=["inkscape"])
    execution_mode: str = Field(..., examples=["local"])
    requires_license: str | None = None
    is_active: bool = True


class ToolAdapterUpdate(BaseModel):
    name: str | None = None
    mcp_server_name: str | None = None
    execution_mode: str | None = None
    requires_license: str | None = None
    is_active: bool | None = None


class ProductionTemplateCreate(BaseModel):
    name: str = Field(..., examples=["Storyboard_Spec"])
    content_type: str = Field(..., examples=["video"])
    phase: str = Field(..., examples=["preproduccion"])
    template_format: str = Field(..., examples=["json"])
    content: dict = Field(default_factory=dict)
    version: str = "1.0"
    is_active: bool = True


class ProductionTemplateUpdate(BaseModel):
    name: str | None = None
    content_type: str | None = None
    phase: str | None = None
    template_format: str | None = None
    content: dict | None = None
    version: str | None = None
    is_active: bool | None = None


class ManifestCreate(BaseModel):
    manifest: dict = Field(..., examples=[{"project": {"artifact_id": "..."}}])


class ProduceRequest(BaseModel):
    manifest_version: int | None = None
    manifest: dict | None = None


def _tool_adapter_to_dict(a: ToolAdapter) -> dict:
    return {
        "id": str(a.id),
        "name": a.name,
        "mcp_server_name": a.mcp_server_name,
        "execution_mode": a.execution_mode,
        "requires_license": a.requires_license,
        "is_active": a.is_active,
        "created_at": a.created_at.isoformat() if a.created_at else None,
        "updated_at": a.updated_at.isoformat() if a.updated_at else None,
    }


def _production_template_to_dict(t: ProductionTemplate) -> dict:
    return {
        "id": str(t.id),
        "name": t.name,
        "content_type": t.content_type,
        "phase": t.phase,
        "template_format": t.template_format,
        "content": t.content or {},
        "version": t.version,
        "is_active": t.is_active,
        "created_at": t.created_at.isoformat() if t.created_at else None,
        "updated_at": t.updated_at.isoformat() if t.updated_at else None,
    }


def _production_manifest_to_dict(m: ProductionManifest) -> dict:
    return {
        "id": str(m.id),
        "artifact_id": str(m.artifact_id),
        "version": m.version,
        "manifest": m.manifest or {},
        "created_at": m.created_at.isoformat() if m.created_at else None,
    }


def _asset_job_to_dict(j: AssetJob) -> dict:
    return {
        "id": str(j.id),
        "artifact_id": str(j.artifact_id),
        "tool_adapter_id": str(j.tool_adapter_id),
        "sequence_order": j.sequence_order,
        "status": j.status,
        "phase": j.phase,
        "template_id": str(j.template_id) if j.template_id else None,
        "manifest_version": j.manifest_version,
        "input_ref": j.input_ref,
        "output_path": j.output_path,
        "external_job_id": j.external_job_id,
        "cost_estimate": j.cost_estimate or {},
        "error": j.error,
        "started_at": j.started_at.isoformat() if j.started_at else None,
        "completed_at": j.completed_at.isoformat() if j.completed_at else None,
    }


def _get_artifact_or_404(session: Session, artifact_id: uuid.UUID) -> ContentArtifact:
    artifact = session.get(ContentArtifact, artifact_id)
    if artifact is None:
        raise HTTPException(
            status_code=404, detail=f"artefacto {artifact_id} no encontrado."
        )
    return artifact


# ---------------------------------------------------------------------
# CRUD /tool-adapters
# ---------------------------------------------------------------------


@app.get("/tool-adapters", response_model=list[dict])
def list_tool_adapters(session: Session = Depends(get_session)) -> list[dict]:
    rows = (
        session.execute(select(ToolAdapter).order_by(ToolAdapter.created_at.desc()))
        .scalars()
        .all()
    )
    return [_tool_adapter_to_dict(a) for a in rows]


@app.post("/tool-adapters", status_code=201)
def create_tool_adapter(
    body: ToolAdapterCreate, session: Session = Depends(get_session)
) -> dict:
    exists = session.execute(
        select(ToolAdapter).where(ToolAdapter.name == body.name)
    ).scalar_one_or_none()
    if exists is not None:
        raise HTTPException(
            status_code=409,
            detail=f"Ya existe un tool_adapter con name {body.name!r}.",
        )
    adapter = ToolAdapter(**body.model_dump())
    session.add(adapter)
    session.commit()
    session.refresh(adapter)
    return _tool_adapter_to_dict(adapter)


@app.get("/tool-adapters/{adapter_id}", response_model=dict)
def get_tool_adapter(
    adapter_id: uuid.UUID, session: Session = Depends(get_session)
) -> dict:
    adapter = session.get(ToolAdapter, adapter_id)
    if adapter is None:
        raise HTTPException(
            status_code=404, detail=f"tool_adapter {adapter_id} no encontrado."
        )
    return _tool_adapter_to_dict(adapter)


@app.patch("/tool-adapters/{adapter_id}", response_model=dict)
def update_tool_adapter(
    adapter_id: uuid.UUID,
    body: ToolAdapterUpdate,
    session: Session = Depends(get_session),
) -> dict:
    adapter = session.get(ToolAdapter, adapter_id)
    if adapter is None:
        raise HTTPException(
            status_code=404, detail=f"tool_adapter {adapter_id} no encontrado."
        )
    for field_name, value in body.model_dump(exclude_unset=True).items():
        setattr(adapter, field_name, value)
    session.commit()
    session.refresh(adapter)
    return _tool_adapter_to_dict(adapter)


@app.delete("/tool-adapters/{adapter_id}", status_code=204)
def delete_tool_adapter(
    adapter_id: uuid.UUID, session: Session = Depends(get_session)
) -> None:
    adapter = session.get(ToolAdapter, adapter_id)
    if adapter is None:
        raise HTTPException(
            status_code=404, detail=f"tool_adapter {adapter_id} no encontrado."
        )
    session.delete(adapter)
    session.commit()


# ---------------------------------------------------------------------
# CRUD /production-templates
# ---------------------------------------------------------------------


@app.get("/production-templates", response_model=list[dict])
def list_production_templates(
    content_type: str | None = None,
    phase: str | None = None,
    session: Session = Depends(get_session),
) -> list[dict]:
    stmt = select(ProductionTemplate)
    if content_type is not None:
        stmt = stmt.where(ProductionTemplate.content_type == content_type)
    if phase is not None:
        stmt = stmt.where(ProductionTemplate.phase == phase)
    stmt = stmt.order_by(ProductionTemplate.created_at.desc())
    rows = session.execute(stmt).scalars().all()
    return [_production_template_to_dict(t) for t in rows]


@app.post("/production-templates", status_code=201)
def create_production_template(
    body: ProductionTemplateCreate, session: Session = Depends(get_session)
) -> dict:
    exists = session.execute(
        select(ProductionTemplate).where(ProductionTemplate.name == body.name)
    ).scalar_one_or_none()
    if exists is not None:
        raise HTTPException(
            status_code=409,
            detail=f"Ya existe un production_template con name {body.name!r}.",
        )
    template = ProductionTemplate(**body.model_dump())
    session.add(template)
    session.commit()
    session.refresh(template)
    return _production_template_to_dict(template)


@app.get("/production-templates/{template_id}", response_model=dict)
def get_production_template(
    template_id: uuid.UUID, session: Session = Depends(get_session)
) -> dict:
    template = session.get(ProductionTemplate, template_id)
    if template is None:
        raise HTTPException(
            status_code=404,
            detail=f"production_template {template_id} no encontrado.",
        )
    return _production_template_to_dict(template)


@app.patch("/production-templates/{template_id}", response_model=dict)
def update_production_template(
    template_id: uuid.UUID,
    body: ProductionTemplateUpdate,
    session: Session = Depends(get_session),
) -> dict:
    template = session.get(ProductionTemplate, template_id)
    if template is None:
        raise HTTPException(
            status_code=404,
            detail=f"production_template {template_id} no encontrado.",
        )
    for field_name, value in body.model_dump(exclude_unset=True).items():
        setattr(template, field_name, value)
    session.commit()
    session.refresh(template)
    return _production_template_to_dict(template)


@app.delete("/production-templates/{template_id}", status_code=204)
def delete_production_template(
    template_id: uuid.UUID, session: Session = Depends(get_session)
) -> None:
    template = session.get(ProductionTemplate, template_id)
    if template is None:
        raise HTTPException(
            status_code=404,
            detail=f"production_template {template_id} no encontrado.",
        )
    session.delete(template)
    session.commit()


# ---------------------------------------------------------------------
# Manifiestos por artefacto: /artifacts/{id}/manifests
# ---------------------------------------------------------------------


@app.get("/artifacts/{artifact_id}/manifests", response_model=list[dict])
def list_artifact_manifests(
    artifact_id: uuid.UUID, session: Session = Depends(get_session)
) -> list[dict]:
    _get_artifact_or_404(session, artifact_id)
    rows = list_manifests(session, artifact_id)
    return [_production_manifest_to_dict(m) for m in rows]


@app.post("/artifacts/{artifact_id}/manifests", status_code=201)
def create_artifact_manifest(
    artifact_id: uuid.UUID,
    body: ManifestCreate,
    session: Session = Depends(get_session),
) -> dict:
    _get_artifact_or_404(session, artifact_id)
    manifest = create_manifest(session, artifact_id, body.manifest)
    session.commit()
    session.refresh(manifest)
    return _production_manifest_to_dict(manifest)


@app.get("/artifacts/{artifact_id}/manifests/{version}", response_model=dict)
def get_artifact_manifest(
    artifact_id: uuid.UUID,
    version: int,
    session: Session = Depends(get_session),
) -> dict:
    _get_artifact_or_404(session, artifact_id)
    manifest = load_manifest(session, artifact_id, version)
    if manifest is None:
        raise HTTPException(
            status_code=404,
            detail=f"manifiesto versión {version} del artefacto {artifact_id} no encontrado.",
        )
    return _production_manifest_to_dict(manifest)


# ---------------------------------------------------------------------
# Jobs por artefacto: /artifacts/{id}/jobs
# ---------------------------------------------------------------------


@app.get("/artifacts/{artifact_id}/jobs", response_model=list[dict])
def list_artifact_jobs(
    artifact_id: uuid.UUID, session: Session = Depends(get_session)
) -> list[dict]:
    _get_artifact_or_404(session, artifact_id)
    rows = (
        session.execute(
            select(AssetJob)
            .where(AssetJob.artifact_id == artifact_id)
            .order_by(AssetJob.sequence_order)
        )
        .scalars()
        .all()
    )
    return [_asset_job_to_dict(j) for j in rows]


# ---------------------------------------------------------------------
# Endpoint disparador: POST /artifacts/{id}/produce
# ---------------------------------------------------------------------


@app.post("/artifacts/{artifact_id}/produce", response_model=dict)
def produce_artifact(
    artifact_id: uuid.UUID,
    body: ProduceRequest | None = None,
    session: Session = Depends(get_session),
) -> dict:
    """Dispara la cadena de herramientas del artefacto (diseño §17 pt. 14).

    Si `body.manifest` viene, crea una versión nueva del manifiesto antes de
    orquestar. Los errores de orquestación (sin format_spec, sin manifiesto,
    tool_chain rota) se devuelven como 409; cualquier otro error es 500.
    """
    _get_artifact_or_404(session, artifact_id)
    body = body or ProduceRequest()
    manifest_version = body.manifest_version
    if body.manifest is not None:
        manifest = create_manifest(session, artifact_id, body.manifest)
        session.commit()
        session.refresh(manifest)
        manifest_version = manifest.version
    try:
        orchestrator = Orchestrator(session)
        result = orchestrator.run_tool_chain(artifact_id, manifest_version)
        session.commit()  # el orquestador no commitea a propósito (Fase 2)
        return result
    except OrchestrationError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:  # error inesperado del orquestador
        raise HTTPException(
            status_code=500, detail=f"Error al producir el artefacto: {exc}"
        ) from exc


# ---------------------------------------------------------------------
# Validación de la cadena de herramientas (arranque/debug)
# ---------------------------------------------------------------------


@app.get("/tools/validate", response_model=dict)
def validate_tools(session: Session = Depends(get_session)) -> dict:
    errors = validate_tool_chain(session)
    return {"errors": errors}


# ---------------------------------------------------------------------
# Fase 4 — Proyectos (0007): projects + project_versions (diseño §20)
# ---------------------------------------------------------------------


class ProjectCreate(BaseModel):
    name: str = Field(..., examples=["Corto Escape — IA local"])
    topic: str = Field(..., examples=["IA local vs nube: costes reales"])
    template_id: uuid.UUID | None = None
    brand_objective: str | None = None
    artifact_type: str | None = None
    # Gobernanza real (0009/0010): el proyecto declara el segmento y el
    # riesgo de su caso real; produce_project los usa para el brief
    # compañero y el Gatekeeper — nunca S1/bajo hardcodeados.
    segment_client: str | None = Field(
        default=None, pattern="^S[1-6]$", examples=["S1"]
    )
    risk_level: str | None = Field(
        default=None, pattern="^(bajo|medio|alto)$", examples=["bajo"]
    )


class ProjectUpdate(BaseModel):
    name: str | None = None
    topic: str | None = None
    template_id: uuid.UUID | None = None
    brand_objective: str | None = None
    artifact_type: str | None = None
    status: str | None = None
    segment_client: str | None = Field(
        default=None, pattern="^S[1-6]$", examples=["S1"]
    )
    risk_level: str | None = Field(
        default=None, pattern="^(bajo|medio|alto)$", examples=["bajo"]
    )


class ProjectApprove(BaseModel):
    json_editado: dict = Field(..., examples=[{"guion_vocal": "..."}])


class ProjectRollback(BaseModel):
    version: int = Field(..., examples=[1])


def _project_to_dict(p: Project) -> dict:
    return {
        "id": str(p.id),
        "name": p.name,
        "topic": p.topic,
        "template_id": str(p.template_id) if p.template_id else None,
        "brand_objective": _val(p.brand_objective),
        "artifact_type": p.artifact_type,
        "segment_client": _val(p.segment_client),
        "risk_level": _val(p.risk_level),
        "status": p.status,
        "current_version": p.current_version,
        "storage_path": p.storage_path,
        "created_at": p.created_at.isoformat() if p.created_at else None,
        "updated_at": p.updated_at.isoformat() if p.updated_at else None,
    }


def _project_version_to_dict(v: ProjectVersion) -> dict:
    return {
        "id": str(v.id),
        "project_id": str(v.project_id),
        "version": v.version,
        "snapshot": v.snapshot or {},
        "created_at": v.created_at.isoformat() if v.created_at else None,
    }


def _get_project_or_404(session: Session, project_id: uuid.UUID) -> Project:
    project = session.get(Project, project_id)
    if project is None:
        raise HTTPException(
            status_code=404, detail=f"proyecto {project_id} no encontrado."
        )
    return project


def _get_project_version_or_404(
    session: Session, project_id: uuid.UUID, version: int
) -> ProjectVersion:
    row = session.execute(
        select(ProjectVersion).where(
            ProjectVersion.project_id == project_id,
            ProjectVersion.version == version,
        )
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"versión {version} del proyecto {project_id} no encontrada.",
        )
    return row


def _next_project_version(session: Session, project_id: uuid.UUID) -> int:
    rows = (
        session.execute(
            select(ProjectVersion.version)
            .where(ProjectVersion.project_id == project_id)
            .order_by(ProjectVersion.version.desc())
        )
        .scalars()
        .all()
    )
    return (rows[0] if rows else 0) + 1


@app.post("/project/new", status_code=201)
def create_project(
    body: ProjectCreate, session: Session = Depends(get_session)
) -> dict:
    """Crea un proyecto en estado 'borrador' con su versión 1 de snapshot
    (esqueleto vacío {}; el flujo de generación lo llena después).

    Gobernanza (0009/0010): segment_client y risk_level son obligatorios
    — el proyecto declara el caso real desde el inicio; produce_project
    los usa para el brief compañero y el Gatekeeper.
    """
    if body.template_id is not None:
        template = session.get(ProductionTemplate, body.template_id)
        if template is None:
            raise HTTPException(
                status_code=404,
                detail=f"production_template {body.template_id} no encontrado.",
            )
    if not body.segment_client:
        raise HTTPException(
            status_code=422,
            detail="Todo proyecto requiere segment_client (S1-S6) — canon §9.2.",
        )
    if not body.risk_level:
        raise HTTPException(
            status_code=422,
            detail="Todo proyecto requiere risk_level (bajo|medio|alto).",
        )
    project = Project(
        name=body.name,
        topic=body.topic,
        template_id=body.template_id,
        brand_objective=body.brand_objective,
        artifact_type=body.artifact_type,
        segment_client=body.segment_client,
        risk_level=body.risk_level,
        status="borrador",
        current_version=1,
    )
    session.add(project)
    session.flush()  # asigna id para poder referenciarlo en la versión
    version = ProjectVersion(project_id=project.id, version=1, snapshot={})
    session.add(version)
    session.commit()
    session.refresh(project)
    session.refresh(version)
    return {
        "project": _project_to_dict(project),
        "version": _project_version_to_dict(version),
    }


@app.get("/projects", response_model=list[dict])
def list_projects(
    status: str | None = None,
    artifact_type: str | None = None,
    session: Session = Depends(get_session),
) -> list[dict]:
    stmt = select(Project)
    if status is not None:
        stmt = stmt.where(Project.status == status)
    if artifact_type is not None:
        stmt = stmt.where(Project.artifact_type == artifact_type)
    stmt = stmt.order_by(Project.created_at.desc())
    rows = session.execute(stmt).scalars().all()
    return [_project_to_dict(p) for p in rows]


@app.get("/projects/{project_id}", response_model=dict)
def get_project(project_id: uuid.UUID, session: Session = Depends(get_session)) -> dict:
    project = _get_project_or_404(session, project_id)
    return _project_to_dict(project)


@app.patch("/projects/{project_id}", response_model=dict)
def update_project(
    project_id: uuid.UUID,
    body: ProjectUpdate,
    session: Session = Depends(get_session),
) -> dict:
    project = _get_project_or_404(session, project_id)
    for field_name, value in body.model_dump(exclude_unset=True).items():
        setattr(project, field_name, value)
    session.commit()
    session.refresh(project)
    return _project_to_dict(project)


@app.delete("/projects/{project_id}", status_code=204)
def delete_project(
    project_id: uuid.UUID, session: Session = Depends(get_session)
) -> None:
    project = _get_project_or_404(session, project_id)
    session.delete(project)  # CASCADE borra las versiones
    session.commit()


@app.get("/projects/{project_id}/versions", response_model=list[dict])
def list_project_versions(
    project_id: uuid.UUID, session: Session = Depends(get_session)
) -> list[dict]:
    _get_project_or_404(session, project_id)
    rows = (
        session.execute(
            select(ProjectVersion)
            .where(ProjectVersion.project_id == project_id)
            .order_by(ProjectVersion.version.desc())
        )
        .scalars()
        .all()
    )
    return [_project_version_to_dict(v) for v in rows]


@app.get("/projects/{project_id}/versions/{version}", response_model=dict)
def get_project_version(
    project_id: uuid.UUID,
    version: int,
    session: Session = Depends(get_session),
) -> dict:
    _get_project_or_404(session, project_id)
    row = _get_project_version_or_404(session, project_id, version)
    return _project_version_to_dict(row)


@app.post("/projects/{project_id}/approve", response_model=dict)
def approve_project(
    project_id: uuid.UUID,
    body: ProjectApprove,
    session: Session = Depends(get_session),
    user: AppUser = Depends(require_role("lider")),
) -> dict:
    """Aprueba el JSON editado: lo guarda como versión nueva (max+1),
    actualiza current_version y marca el proyecto 'aprobado' con su
    storage_path resuelto contra ASSET_STORAGE_PATH (§20.8).

    OWNER_APPROVAL del Pre-Deploy Gate (0009): solo un 🟨 líder puede
    aprobar — igual que POST /briefs/{id}/approve. Sin esto, cualquiera
    (o nadie autenticado) podía empujar un proyecto a producción.
    """
    project = _get_project_or_404(session, project_id)
    new_version = _next_project_version(session, project_id)
    version = ProjectVersion(
        project_id=project.id, version=new_version, snapshot=body.json_editado
    )
    session.add(version)
    project.current_version = new_version
    project.status = "aprobado"
    project.storage_path = f"{get_settings().asset_storage_path}/projects/{project_id}"
    session.commit()
    session.refresh(project)
    session.refresh(version)
    return {
        "project": _project_to_dict(project),
        "version": _project_version_to_dict(version),
        "approved_by": str(user.id),
    }


@app.post("/projects/{project_id}/rollback", response_model=dict)
def rollback_project(
    project_id: uuid.UUID,
    body: ProjectRollback,
    session: Session = Depends(get_session),
) -> dict:
    """Restaura un snapshot anterior como versión NUEVA (el historial nunca
    se pierde), actualiza current_version y vuelve a 'en_edicion'."""
    project = _get_project_or_404(session, project_id)
    old = _get_project_version_or_404(session, project_id, body.version)
    new_version = _next_project_version(session, project_id)
    version = ProjectVersion(
        project_id=project.id, version=new_version, snapshot=old.snapshot
    )
    session.add(version)
    project.current_version = new_version
    project.status = "en_edicion"
    session.commit()
    session.refresh(project)
    session.refresh(version)
    return {
        "project": _project_to_dict(project),
        "version": _project_version_to_dict(version),
    }


# ---------------------------------------------------------------------
# Fase 5 — Flujo agéntico de proyectos (0008, diseño §20.3-§20.5):
# generate / refine / produce / postproduction
# ---------------------------------------------------------------------


class ProjectGenerate(BaseModel):
    topic: str | None = None


class ProjectRefine(BaseModel):
    rag_context: str | None = None
    use_llm: bool = False


class ProjectProduce(BaseModel):
    pass


@app.post("/projects/{project_id}/generate", response_model=dict)
def generate_project(
    project_id: uuid.UUID,
    body: ProjectGenerate | None = None,
    session: Session = Depends(get_session),
) -> dict:
    """Genera el snapshot JSON del proyecto vía LLM (compilador, §20.3).

    El LLM escribe datos (el snapshot), nunca edita archivos. Si la decisión
    requiere aceptación del usuario (degradada/saltada, 0007), NO se guarda
    versión nueva: se devuelve el contrato con el snapshot determinista para
    que el humano revise (puede editar y aprobar, o reintentar).
    """
    project = _get_project_or_404(session, project_id)
    template = None
    if project.template_id is not None:
        template = session.get(ProductionTemplate, project.template_id)
    format_spec = resolve_format_spec(
        session, project.brand_objective, project.artifact_type
    )
    result = generate_snapshot(session, project, template, format_spec)

    if result.get("requires_user_acceptance"):
        # 0007: degradación/skip → el pipeline se pausa; no se guarda versión.
        return {
            "project": _project_to_dict(project),
            "decision": result,
        }

    new_version = _next_project_version(session, project_id)
    version = ProjectVersion(
        project_id=project.id, version=new_version, snapshot=result["snapshot"]
    )
    session.add(version)
    project.current_version = new_version
    project.status = "en_edicion"
    session.commit()
    session.refresh(project)
    session.refresh(version)
    return {
        "project": _project_to_dict(project),
        "version": _project_version_to_dict(version),
        "decision": result,
    }


@app.post("/projects/{project_id}/refine", response_model=dict)
def refine_project(
    project_id: uuid.UUID,
    body: ProjectRefine | None = None,
    session: Session = Depends(get_session),
) -> dict:
    """Refina el snapshot actual con reglas deterministas (+ LLM opcional).

    Reglas de negocio en código (qa_checks/constraints de la spec); el LLM
    solo corrige warnings si use_llm=True y degrada si falla. Si la decisión
    requiere aceptación del usuario (degradada/saltada, 0007), NO se guarda
    versión nueva: se devuelve el contrato con el resultado determinista.
    """
    project = _get_project_or_404(session, project_id)
    current = _get_project_version_or_404(session, project_id, project.current_version)
    result = refine_snapshot(
        session,
        project,
        current.snapshot,
        rag_context=body.rag_context if body else None,
        use_llm=body.use_llm if body else False,
    )

    if result.get("requires_user_acceptance"):
        # 0007: degradación/skip → el pipeline se pausa; no se guarda versión.
        return {
            "project": _project_to_dict(project),
            "decision": result,
        }

    new_version = _next_project_version(session, project_id)
    version = ProjectVersion(
        project_id=project.id, version=new_version, snapshot=result["snapshot"]
    )
    session.add(version)
    project.current_version = new_version
    session.commit()
    session.refresh(project)
    session.refresh(version)
    return {
        "project": _project_to_dict(project),
        "version": _project_version_to_dict(version),
        "decision": result,
    }


@app.post("/projects/{project_id}/produce", response_model=dict)
def produce_project_endpoint(
    project_id: uuid.UUID,
    body: ProjectProduce | None = None,
    session: Session = Depends(get_session),
) -> dict:
    """Dispara la cadena de herramientas del proyecto (§20.3).

    Solo proyectos 'aprobado'. Gobernanza (0009/0010): produce_project
    crea el brief compañero con el segmento/riesgo reales del proyecto y
    lo pasa por el Gatekeeper. Si el gate exige revisión humana, el
    pipeline se pausa (requires_user_acceptance=True): se devuelve 409
    con el brief_id para que un 🟨 líder lo apruebe vía
    POST /briefs/{id}/approve y luego re-dispare /produce.

    El orquestador no commitea a propósito (Fase 2): este endpoint
    commitea después de produce_project. Los errores de orquestación se
    mapean a 409; cualquier otro error a 500.
    """
    project = _get_project_or_404(session, project_id)
    if project.status != "aprobado":
        raise HTTPException(
            status_code=409,
            detail="el proyecto debe estar aprobado antes de producir",
        )
    current = _get_project_version_or_404(session, project_id, project.current_version)
    try:
        result = produce_project(session, project, current.snapshot)
        if result.get("requires_user_acceptance"):
            # 0007: el pipeline se pausa — el brief compañero exige
            # aprobación de un 🟨 líder antes de tocar herramientas.
            raise HTTPException(
                status_code=409,
                detail=(
                    f"El brief compañero {result['brief_id']} exige revisión "
                    f"humana (verdict={result['verdict']}): {result['reason']} "
                    "Aprueba el brief vía POST /briefs/{id}/approve y "
                    "re-dispara /produce."
                ),
            )
        session.commit()  # el orquestador no commitea a propósito (Fase 2)
        return {
            "project": _project_to_dict(project),
            "artifact_id": result["artifact_id"],
            "manifest_version": result["manifest_version"],
            "result": result["result"],
        }
    except OrchestrationError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 — error inesperado del orquestador
        raise HTTPException(
            status_code=500, detail=f"Error al producir el proyecto: {exc}"
        ) from exc


@app.get("/projects/{project_id}/postproduction", response_model=dict)
def project_postproduction(
    project_id: uuid.UUID, session: Session = Depends(get_session)
) -> dict:
    """Recomendaciones algorítmicas de post-producción (§20.4, modo 2).

    Deterministas, sin LLM: templates de fase 'postproduccion' agrupados
    por content_type, parametrizados con el snapshot actual.
    """
    project = _get_project_or_404(session, project_id)
    current = _get_project_version_or_404(session, project_id, project.current_version)
    return postproduction_recommendations(session, project, current.snapshot)


# ---------------------------------------------------------------------
# Ruta A — Kaizen/Repetitivo (§5 del pipeline unificado)
# ---------------------------------------------------------------------
# La ruta de MAYOR volumen real: el contenido recurrente de calendario
# (dato incómodo semanal de Ergalia, shorts Lun/Mié/Vie de ESCAPE).
# Secuencia: ORDER → ORDER_NOTE → EXECUTE_KAIZEN → MEASURE_KPI →
# MICRO_KAIZEN/MICRO_RITUAL → KAIZEN_DECISION → PRE_DEPLOY_ENTRY.
# Publicación manual por ahora (la capa DEPLOY automática no existe aún).


class OrderCreate(BaseModel):
    slot_code: str
    scheduled_date: date
    insight_core: str
    owner_id: uuid.UUID | None = None
    segment_client: str | None = None
    risk_level: str | None = None
    evidence_source: str | None = None


class KpisPayload(BaseModel):
    kpis: dict = Field(default_factory=dict)


class MicroKaizenPayload(BaseModel):
    experiment: dict = Field(default_factory=dict)
    ritual: dict = Field(default_factory=dict)


class KaizenDecisionPayload(BaseModel):
    decision: str = Field(..., pattern="^(update_registry|archive)$")
    improvement_summary: str = ""


def _slot_to_dict(slot: CalendarSlot) -> dict:
    return {
        "id": str(slot.id),
        "slot_code": slot.slot_code,
        "brand_objective": _val(slot.brand_objective),
        "content_bucket": _val(slot.content_bucket),
        "artifact_type": slot.artifact_type,
        "channel": slot.channel,
        "default_owner_role": slot.default_owner_role,
        "cadence": slot.cadence,
        "weekday": slot.weekday,
        "is_active": slot.is_active,
    }


def _order_to_dict(order: ProductionOrder) -> dict:
    return {
        "id": str(order.id),
        "calendar_slot": order.calendar_slot,
        "brand_objective": _val(order.brand_objective),
        "content_bucket": _val(order.content_bucket),
        "artifact_type": order.artifact_type,
        "channel": order.channel,
        "owner_id": str(order.owner_id) if order.owner_id else None,
        "scheduled_date": order.scheduled_date.isoformat()
        if order.scheduled_date
        else None,
        "insight_core": order.insight_core,
        "status": _val(order.status),
        "brief_id": str(order.brief_id) if order.brief_id else None,
        "artifact_id": str(order.artifact_id) if order.artifact_id else None,
        "template_id": str(order.template_id) if order.template_id else None,
        "kpis": order.kpis or {},
        "kaizen_decision": order.kaizen_decision,
        "created_at": order.created_at.isoformat() if order.created_at else None,
        "updated_at": order.updated_at.isoformat() if order.updated_at else None,
    }


def _kaizen_cycle_to_dict(cycle: KaizenCycle) -> dict:
    return {
        "id": str(cycle.id),
        "order_id": str(cycle.order_id),
        "decision": cycle.decision,
        "improvement_summary": cycle.improvement_summary,
        "metrics": cycle.metrics or {},
        "created_at": cycle.created_at.isoformat() if cycle.created_at else None,
    }


def _get_order_or_404(session: Session, order_id: uuid.UUID) -> ProductionOrder:
    order = session.get(ProductionOrder, order_id)
    if order is None:
        raise HTTPException(status_code=404, detail=f"Orden {order_id} no encontrada.")
    return order


@app.get("/calendar-slots", response_model=list[dict])
def list_calendar_slots(session: Session = Depends(get_session)) -> list[dict]:
    """Lista las filas del calendario editorial recurrente (Ruta A)."""
    rows = (
        session.execute(select(CalendarSlot).order_by(CalendarSlot.slot_code))
        .scalars()
        .all()
    )
    return [_slot_to_dict(s) for s in rows]


@app.post("/orders", status_code=201)
def create_order(body: OrderCreate, session: Session = Depends(get_session)) -> dict:
    """ORDER (§5.1) — crea una Orden de Producción desde la fila del calendario.

    Hereda brand_objective, content_bucket, artifact_type, channel y owner
    de la fila de calendario (ORDER_NOTE, §5.2); solo se completa
    insight_core y el dato/gancho de esa semana.
    """
    try:
        order = kaizen_create_order(
            session,
            slot_code=body.slot_code,
            scheduled_date=body.scheduled_date,
            insight_core=body.insight_core,
            owner_id=body.owner_id,
            evidence_source=body.evidence_source,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    session.commit()
    session.refresh(order)
    return _order_to_dict(order)


@app.get("/orders", response_model=list[dict])
def list_orders(session: Session = Depends(get_session)) -> list[dict]:
    """Lista las órdenes de producción (más recientes primero)."""
    rows = (
        session.execute(
            select(ProductionOrder).order_by(ProductionOrder.created_at.desc())
        )
        .scalars()
        .all()
    )
    return [_order_to_dict(o) for o in rows]


@app.get("/orders/{order_id}", response_model=dict)
def get_order(order_id: uuid.UUID, session: Session = Depends(get_session)) -> dict:
    """Lee una Orden de Producción completa por id."""
    return _order_to_dict(_get_order_or_404(session, order_id))


@app.get("/orders/{order_id}/note", response_model=dict)
def get_order_note(
    order_id: uuid.UUID, session: Session = Depends(get_session)
) -> dict:
    """ORDER_NOTE (§5.2) — el markdown de /orders/order_<id>.md."""
    order = _get_order_or_404(session, order_id)
    return {"order_id": str(order.id), "note": order_note(order)}


@app.post("/orders/{order_id}/produce", response_model=dict)
def produce_order(order_id: uuid.UUID, session: Session = Depends(get_session)) -> dict:
    """EXECUTE_KAIZEN (§5.3) — producción single-pass con plantilla existente.

    Producer genera → critic evalúa el checklist → Gatekeeper decide. Sin el
    grafo Producer-Critic completo (a propósito). En auto_pass la pieza se
    materializa como ContentArtifact y la orden pasa a 'aprobada'.
    """
    order = _get_order_or_404(session, order_id)
    try:
        return kaizen_produce_order(session, order)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/orders/{order_id}/publish", response_model=dict)
def publish_order_endpoint(
    order_id: uuid.UUID, session: Session = Depends(get_session)
) -> dict:
    """Publicación MANUAL por ahora: marca la orden como 'publicada'.

    La capa DEPLOY automática (docs/diseno_sistema_publicacion.md) aún no
    está construida; este paso es el puente manual hasta que exista.
    """
    order = _get_order_or_404(session, order_id)
    try:
        publish_order(session, order)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    session.refresh(order)
    return _order_to_dict(order)


@app.post("/orders/{order_id}/kpis", response_model=dict)
def record_kpis_endpoint(
    order_id: uuid.UUID, body: KpisPayload, session: Session = Depends(get_session)
) -> dict:
    """MEASURE_KPI (§5.4) — LeadTime, CycleTime, Time-in-Stage, FPQ, Rework Rate.

    Las métricas de proceso se calculan automáticamente; las de contenido
    (guardados, respuestas, retención) se pasan en el body.
    """
    order = _get_order_or_404(session, order_id)
    merged = record_kpis(session, order, body.kpis)
    return {"order_id": str(order.id), "kpis": merged}


@app.post("/orders/{order_id}/kaizen", response_model=dict)
def record_micro_kaizen_endpoint(
    order_id: uuid.UUID,
    body: MicroKaizenPayload,
    session: Session = Depends(get_session),
) -> dict:
    """MICRO_KAIZEN (§5.5) + MICRO_RITUAL (§5.6).

    MICRO_KAIZEN: ajuste incremental (otro horario, otro hook, formato de
    apoyo). MICRO_RITUAL (5–10 min): chequeo exprés de riesgos.
    """
    order = _get_order_or_404(session, order_id)
    kpis = record_micro_kaizen(session, order, body.experiment, body.ritual)
    return {"order_id": str(order.id), "kpis": kpis}


@app.post("/orders/{order_id}/kaizen-decision", response_model=dict)
def kaizen_decision_endpoint(
    order_id: uuid.UUID,
    body: KaizenDecisionPayload,
    session: Session = Depends(get_session),
) -> dict:
    """KAIZEN_DECISION (§5.7) → UPDATE_REGISTRY | ARCHIVE_KAIZEN.

    update_registry: la plantilla mejora para todo el equipo (se actualiza
    /components/manifest.md y /kb/kaizen_<id>.md). archive: se documenta
    igual, sin cambiar la plantilla base. Ambas convergen en PRE_DEPLOY_ENTRY.
    """
    order = _get_order_or_404(session, order_id)
    try:
        return kaizen_decision(session, order, body.decision, body.improvement_summary)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/orders/{order_id}/kaizen-cycles", response_model=list[dict])
def list_order_kaizen_cycles(
    order_id: uuid.UUID, session: Session = Depends(get_session)
) -> list[dict]:
    """Lista los ciclos Kaizen de una orden (KAIZEN_DECISION, §5.7)."""
    _get_order_or_404(session, order_id)
    rows = (
        session.execute(select(KaizenCycle).where(KaizenCycle.order_id == order_id))
        .scalars()
        .all()
    )
    return [_kaizen_cycle_to_dict(c) for c in rows]


# ---------------------------------------------------------------------
# Motor KAG unificado — consulta e ingesta
# ---------------------------------------------------------------------


class KagAskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    top_propositions: int = Field(15, ge=1, le=100)
    max_iterations: int = Field(2, ge=0, le=5)
    mode: str = Field("audited", pattern="^(fast|audited)$")


@app.post("/kag/ask", response_model=dict)
def kag_ask(body: KagAskRequest, session: Session = Depends(get_session)) -> dict:
    """Consulta al motor KAG unificado (§3 del diseño).

    mode="audited" (default, API académica): devuelve el payload
    enriquecido — respuesta final, veredicto de suficiencia, citas
    auditadas (grounded_evidence), tensiones epistémicas, estado de
    fallback y metadatos de las obras consultadas (consulted_documents,
    dedup por document_id). mode="fast": respuesta directa sin auditoría
    (verdict/grounded_evidence/epistemic_tensions vacíos). El import de
    ask es lazy: el módulo KAG carga torch/spacy y no debe pesarse al
    arrancar la API.
    """
    from src.kag_query import ask

    try:
        result = ask(session, body.query, mode=body.mode, verbose=False)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if body.mode == "fast":
        # ask() en modo fast devuelve str: respuesta directa, sin auditoría.
        return {
            "answer": result,
            "verdict": None,
            "grounded_evidence": [],
            "epistemic_tensions": [],
            "used_fallback": False,
            "consulted_documents": [],
        }

    # mode="audited": ask() devuelve dict {answer, verdict,
    # grounded_evidence, epistemic_tensions, used_fallback}.
    consulted: dict[str, dict] = {}
    for ev in result.get("grounded_evidence", []) or []:
        doc_id = ev.get("document_id")
        if doc_id is None:
            continue
        if doc_id not in consulted:
            consulted[doc_id] = {
                "document_id": doc_id,
                "document_title": ev.get("document_title"),
                "bibtex_citation_key": ev.get("bibtex_citation_key"),
            }
    return {
        "answer": result.get("answer"),
        "verdict": result.get("verdict"),
        "grounded_evidence": result.get("grounded_evidence", []),
        "epistemic_tensions": result.get("epistemic_tensions", []),
        "used_fallback": result.get("used_fallback", False),
        "consulted_documents": list(consulted.values()),
    }


class KagIngestRequest(BaseModel):
    file_path: str | None = Field(
        None, description="Ruta del .md a indexar; None = todos los de docs/"
    )
    force: bool = False
    extract_propositions: bool = True


@app.post("/kag/ingest", response_model=dict)
def kag_ingest(
    body: KagIngestRequest,
    session: Session = Depends(get_session),
    user: AppUser = Depends(require_role("lider")),
) -> dict:
    """Ingesta clásica del KAG: indexa un .md concreto o todos los de
    data/knowledge_repository/docs/ (force=True reindexa aunque ya estén
    indexados). extract_propositions=True (default) extrae además las
    proposiciones atómicas (capa micro). Solo un 🟨 líder puede disparar
    la ingesta.
    """
    if body.file_path:
        from src.kag_ingest import index_document

        try:
            results = [
                index_document(
                    session,
                    Path(body.file_path),
                    force=body.force,
                    extract_propositions=body.extract_propositions,
                    verbose=False,
                )
            ]
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    else:
        from src.kag_ingest import index_all

        try:
            results = index_all(
                session,
                force=body.force,
                extract_propositions=body.extract_propositions,
                verbose=False,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"results": results, "count": len(results)}
