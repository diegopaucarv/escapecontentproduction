"""Ruta A — Kaizen/Repetitivo (§5 del pipeline unificado).

La ruta de MAYOR volumen real de publicación: el contenido recurrente de
calendario (el "dato incómodo" semanal de Ergalia, los shorts Lun/Mié/Vie
de ESCAPE). Es la ruta más simple a propósito: no necesita el grafo
Producer-Critic completo ni atomización multi-formato — solo
Brief → Orden → producción con plantilla existente → Gate.

Secuencia (§5.1):
  1. ORDER          — create_order(): Orden desde la fila del calendario.
  2. ORDER_NOTE     — order_note(): /orders/order_<id>.md (hereda campos
                      del slot; solo se completa insight_core de la semana).
  3. EXECUTE_KAIZEN — produce_order(): single-pass producer → critic → gate
                      con plantilla existente (sin el grafo completo).
  4. MEASURE_KPI    — record_kpis(): LeadTime, CycleTime, Time-in-Stage,
                      FPQ, Rework Rate.
  5. MICRO_KAIZEN   — record_micro_kaizen(): ajuste incremental.
  6. MICRO_RITUAL   — record_micro_kaizen(): chequeo exprés de riesgos.
  7. KAIZEN_DECISION— kaizen_decision(): UPDATE_REGISTRY (mejora la
                      plantilla para todo el equipo) o ARCHIVE_KAIZEN
                      (se documenta sin cambiar la plantilla base).
  8. Ambas salidas convergen en PRE_DEPLOY_ENTRY (§11).

Filosofía 0007 (no negociable): el LLM es la opción SIEMPRE presente; la
degradación a determinista SIEMPRE requiere aceptación del usuario
(requires_user_acceptance=True) — la orden se pausa en 'en_gate' y nunca
se simula auto_pass. El Gatekeeper solo puede bloquear o pedir revisión;
nunca aprobar en solitario salvo auto_pass (riesgo bajo + repetitivo).

Regla (§5): ningún contenido repetitivo debe tardar más de 1–3 días entre
ORDER y PRE_DEPLOY_ENTRY.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.agents.gatekeeper import GateInput, GateVerdict, evaluate_gate
from src.db.models import (
    BriefStatus,
    CalendarSlot,
    ContentArtifact,
    ContentBrief,
    KaizenCycle,
    PipelineTemplate,
    ProductionOrder,
    ProductionTemplate,
    RiskLevel,
    RouteDecision,
)
from src.llm.critic import run_critic_checklist
from src.llm.producer import run_producer_generation
from src.production.refine import resolve_format_spec
from src.tools.registry import resolve_template
from src.tools.template_engine import render

# Raíz de artefactos de la Ruta A (data/orders, data/artifacts, data/kb).
# Los tests monkeypatchean DATA_ROOT a un tmp_path.
DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
ORDERS_DIR = DATA_ROOT / "orders"
ARTIFACTS_DIR = DATA_ROOT / "artifacts"
KB_DIR = DATA_ROOT / "kb"
COMPONENTS_MANIFEST = DATA_ROOT / "components" / "manifest.md"

# Estados de la orden (máquina de estados, ver migración 0011_kaizen_route).
STATUS_CREADA = "creada"
STATUS_EN_PRODUCCION = "en_produccion"
STATUS_EN_GATE = "en_gate"
STATUS_APROBADA = "aprobada"
STATUS_PUBLICADA = "publicada"
STATUS_MEDIDA = "medida"
STATUS_ARCHIVADA = "archivada"

# Decisiones de KAIZEN_DECISION (§5.7).
DECISION_UPDATE_REGISTRY = "update_registry"
DECISION_ARCHIVE = "archive"

# Segmento/riesgo por defecto del brief compañero de una orden repetitiva.
# El contenido recurrente de calendario es por definición de audiencia
# general y bajo riesgo (ya se publicó muchas veces en este formato). Si una
# semana concreta es sensible (ej. S5/S6 en Ergalia), el caller lo declara
# explícitamente y el Gatekeeper exigirá revisión humana.
DEFAULT_SEGMENT_CLIENT = "S1"
DEFAULT_RISK_LEVEL = RiskLevel.bajo


def _val(v):
    """Convierte un enum de SQLAlchemy a su valor string; pasa el resto tal cual."""
    return v.value if hasattr(v, "value") else v


def _iso(dt) -> str | None:
    return dt.isoformat() if dt else None


def _parse_ts(value) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


# ----------------------------------------------------------------------
# ORDER (§5.1) + ORDER_NOTE (§5.2)
# ----------------------------------------------------------------------


def _get_slot(session: Session, slot_code: str) -> CalendarSlot:
    slot = (
        session.execute(select(CalendarSlot).where(CalendarSlot.slot_code == slot_code))
        .scalars()
        .first()
    )
    if slot is None:
        raise ValueError(f"calendar_slot '{slot_code}' no existe")
    return slot


def order_note(order: ProductionOrder) -> str:
    """ORDER_NOTE (§5.2) — el markdown de /orders/order_<id>.md.

    Hereda brand_objective, content_bucket, artifact_type, channel y owner
    de la fila de calendario; solo se completa insight_core y el
    dato/gancho específico de esa semana.
    """
    return f"""---
tipo: orden
ruta: kaizen
---

# Orden de Producción — {order.id}

## De la fila de calendario (heredado)
- slot: {order.calendar_slot}
- brand_objective: {_val(order.brand_objective)}
- content_bucket: {_val(order.content_bucket)}
- artifact_type: {order.artifact_type}
- channel: {order.channel}
- owner_id: {order.owner_id or "(sin asignar)"}
- scheduled_date: {order.scheduled_date.isoformat() if order.scheduled_date else ""}

## De esta semana (solo se completa esto)
- insight_core: {order.insight_core}

## Estado
- status: {order.status}
- brief_id: {order.brief_id or "(sin brief)"}
- artifact_id: {order.artifact_id or "(sin artefacto)"}
- template_id: {order.template_id or "(sin plantilla — el primer ciclo Kaizen llena el manifest)"}
- kpis: {order.kpis or {}}
- kaizen_decision: {order.kaizen_decision or "(pendiente)"}
"""


def _write_order_note(order: ProductionOrder) -> None:
    """Escribe /orders/order_<id>.md. Nunca lanza: el artefacto markdown
    es un espejo legible; la fuente de verdad es la fila en la DB."""
    try:
        ORDERS_DIR.mkdir(parents=True, exist_ok=True)
        (ORDERS_DIR / f"order_{order.id}.md").write_text(
            order_note(order), encoding="utf-8"
        )
    except OSError:
        pass


def create_order(
    session: Session,
    slot_code: str,
    scheduled_date: date,
    insight_core: str,
    owner_id: uuid.UUID | None = None,
    evidence_source: str | None = None,
) -> ProductionOrder:
    """ORDER (§5.1) — crea la Orden de Producción desde la fila del calendario.

    Hereda brand_objective, content_bucket, artifact_type, channel y owner
    de la fila de calendario (ORDER_NOTE, §5.2); solo se completa
    insight_core y el dato/gancho de esa semana. La fuente verificable del
    dato (evidence_source) se guarda en kpis — el producer la necesita para
    que el ítem 'fuente_verificable' del checklist pueda pasar el gate.
    """
    slot = _get_slot(session, slot_code)
    order = ProductionOrder(
        calendar_slot=slot.slot_code,
        brand_objective=slot.brand_objective,
        content_bucket=slot.content_bucket,
        artifact_type=slot.artifact_type,
        channel=slot.channel,
        owner_id=owner_id,
        scheduled_date=scheduled_date,
        insight_core=insight_core,
        status=STATUS_CREADA,
    )
    if evidence_source:
        order.kpis = {**dict(order.kpis or {}), "evidence_source": evidence_source}
    session.add(order)
    session.flush()
    _write_order_note(order)
    return order


# ----------------------------------------------------------------------
# EXECUTE_KAIZEN (§5.3) — single-pass producer → critic → gate
# ----------------------------------------------------------------------


def _companion_brief(
    session: Session,
    order: ProductionOrder,
    segment_client: str = DEFAULT_SEGMENT_CLIENT,
    risk_level: RiskLevel = DEFAULT_RISK_LEVEL,
) -> ContentBrief:
    """Brief compañero de la orden (repetitivo → va directo a producción, §3.9).

    Se reutiliza si ya existe (re-produce tras aprobación humana del brief);
    si no, se crea con route_decision=repetitivo. El segmento/riesgo se
    declaran por orden (default S1/bajo); si una semana es sensible, el
    Gatekeeper exigirá revisión humana.
    """
    if order.brief_id is not None:
        brief = session.get(ContentBrief, order.brief_id)
        if brief is not None:
            return brief
    brief = ContentBrief(
        brand_objective=order.brand_objective,
        content_bucket=order.content_bucket,
        resumen=(
            f"Orden {order.id} — {order.calendar_slot} "
            f"({order.scheduled_date.isoformat() if order.scheduled_date else ''})"
        ),
        insight_core=order.insight_core,
        evidence_source=(order.kpis or {}).get("evidence_source"),
        segment_client=segment_client,
        risk_level=risk_level,
        route_decision=RouteDecision.repetitivo,
        artifact_type=order.artifact_type,
        channel=order.channel,
        status=BriefStatus.idea,
    )
    session.add(brief)
    session.flush()
    order.brief_id = brief.id
    return brief


def _resolve_template(session: Session, order: ProductionOrder):
    """Plantilla existente de /components/manifest.md (§5.3).

    Busca la primera plantilla de la format_spec del formato (spec.phases →
    templates). Si el manifest está vacío (primer ciclo Kaizen), devuelve
    None — la producción sigue sin plantilla y UPDATE_REGISTRY llenará la
    primera fila real.
    """
    spec = resolve_format_spec(session, order.brand_objective, order.artifact_type)
    if spec is None:
        return None
    for phase in ("preproduccion", "produccion", "postproduccion"):
        phase_spec = (spec.phases or {}).get(phase)
        for template_name in (phase_spec or {}).get("templates", []) or []:
            template = resolve_template(session, template_name)
            if template is not None:
                return template
    return None


def _materialize_artifact(
    session: Session, order: ProductionOrder, brief: ContentBrief, draft: str
) -> ContentArtifact:
    """Materializa la pieza como ContentArtifact (status 'borrador').

    El borrador se escribe como markdown en data/artifacts/kaizen_<id>.md y
    storage_path lo referencia. Si la plantilla existe, se renderiza con el
    borrador y el contrato queda en visual_spec (trazabilidad de qué
    plantilla se usó).
    """
    format_spec = resolve_format_spec(
        session, order.brand_objective, order.artifact_type
    )
    format_spec = resolve_format_spec(
        session, order.brand_objective, order.artifact_type
    )
    artifact = ContentArtifact(
        brief_id=brief.id,
        artifact_type=order.artifact_type,
        channel=order.channel,
        status="borrador",
        format_spec_id=format_spec.id if format_spec else None,
    )
    session.add(artifact)
    session.flush()

    # Trazabilidad: qué plantilla se usó y con qué contrato (si la hay).
    visual_spec: dict = {"draft": draft}
    if order.template_id is not None:
        template = session.get(ProductionTemplate, order.template_id)
        if template is not None:
            visual_spec["template"] = {
                "name": template.name,
                "version": template.version,
                "contract": render(template, {"draft": draft}),
            }
    artifact.visual_spec = visual_spec

    try:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        path = ARTIFACTS_DIR / f"kaizen_{order.id}.md"
        path.write_text(draft, encoding="utf-8")
        artifact.storage_path = str(path)
    except OSError:
        pass
    return artifact


def produce_order(
    session: Session,
    order: ProductionOrder,
    segment_client: str = DEFAULT_SEGMENT_CLIENT,
    risk_level: RiskLevel = DEFAULT_RISK_LEVEL,
) -> dict:
    """EXECUTE_KAIZEN (§5.3) — producción single-pass con plantilla existente.

    A propósito NO usa el grafo Producer-Critic completo (el usuario lo
    pidió explícitamente): producer genera → critic evalúa el checklist →
    Gatekeeper decide. Pasos:

      1. Brief compañero (route_decision=repetitivo, §3.9).
      2. Plantilla existente de /components/manifest.md (si la hay).
      3. Producer LLM genera el borrador (prompt-as-code producer_draft).
      4. Critic LLM evalúa el checklist de publicación del plan de marca.
      5. Gatekeeper: auto_pass / needs_human_review / fail.

    Filosofía 0007: si el producer o el critic degradan a determinista,
    requires_user_acceptance=True y la orden se pausa en 'en_gate' — nunca
    se simula auto_pass. En auto_pass la pieza se materializa como
    ContentArtifact y la orden pasa a 'aprobada'.
    """
    if _val(order.status) not in (STATUS_CREADA, STATUS_EN_PRODUCCION, STATUS_EN_GATE):
        raise ValueError(f"orden en estado '{_val(order.status)}' no se puede producir")

    # 1. Brief compañero.
    brief = _companion_brief(session, order, segment_client, risk_level)

    # 2. Plantilla existente (manifest). El primer ciclo no tiene plantilla.
    template = _resolve_template(session, order)
    if template is not None:
        order.template_id = template.id

    # 3. Plan de marca → checklist de publicación.
    plan = (
        session.execute(
            select(PipelineTemplate).where(
                PipelineTemplate.brand_objective == order.brand_objective,
                PipelineTemplate.content_bucket == order.content_bucket,
            )
        )
        .scalars()
        .first()
    )
    checklist = list(plan.checklist or []) if plan is not None else []

    # 4. Producer LLM (single-pass, prompt-as-code). En retrabajo se le pasa
    #    el feedback del crítico de la ronda anterior (si lo hay) para que el
    #    borrador converja — sin esto el rework es un reintento ciego.
    brief_data = {
        "insight_core": order.insight_core,
        "brand_objective": _val(order.brand_objective),
        "content_bucket": _val(order.content_bucket),
        "artifact_type": order.artifact_type,
        "channel": order.channel,
    }
    prev_feedback = (order.kpis or {}).get("last_critic_feedback")
    # Context Pack mínimo de la Ruta A: la fuente verificable del dato. Sin
    # ella el producer no puede citar (sus reglas prohíben inventar datos) y
    # el ítem 'fuente_verificable' del checklist nunca pasa el gate.
    context_pack = {}
    evidence_source = (order.kpis or {}).get("evidence_source")
    if evidence_source:
        context_pack["evidence_source"] = evidence_source
    producer_result = run_producer_generation(
        session, brief_data, context_pack, critic_feedback=prev_feedback
    )
    draft = producer_result.get("draft", "")
    requires_user_acceptance = bool(
        producer_result.get("requires_user_acceptance", False)
    )

    # 5. Critic LLM (single-pass, checklist del plan de marca).
    critic_result = run_critic_checklist(
        session, draft, checklist, _val(order.brand_objective)
    )
    checklist_results = critic_result.get("checklist_results", {}) or {}
    requires_user_acceptance = requires_user_acceptance or bool(
        critic_result.get("requires_user_acceptance", False)
    )

    # 6. Gatekeeper — solo puede bloquear o pedir revisión; auto_pass solo
    #    con riesgo bajo + repetitivo + checklist completo.
    gate = evaluate_gate(
        GateInput(
            risk_level=_val(brief.risk_level),
            production_route=None,  # ruta repetitiva
            route_decision=_val(brief.route_decision),
            segment_client=_val(brief.segment_client),
            checklist_items={
                item: (checklist_results.get(item) == "ok") for item in checklist
            },
        )
    )
    verdict = gate.verdict.value

    # Filosofía 0007: la degradación a determinista (producer o critic sin
    # LLM) SIEMPRE pausa en needs_human_review — nunca fail ni auto_pass
    # simulado. El veredicto del gate queda como contexto informativo.
    if requires_user_acceptance:
        verdict = GateVerdict.needs_human_review.value

    # 7. Aplicar veredicto a la máquina de estados de la orden.
    order.status = STATUS_EN_GATE
    if verdict == GateVerdict.auto_pass.value and not requires_user_acceptance:
        artifact = _materialize_artifact(session, order, brief, draft)
        order.artifact_id = artifact.id
        order.status = STATUS_APROBADA
        brief.status = BriefStatus.aprobado
    elif verdict == GateVerdict.fail.value:
        # Retrabajo: la orden vuelve a producción; el brief a 'generando'.
        order.status = STATUS_EN_PRODUCCION
        brief.status = BriefStatus.generando
    else:
        # needs_human_review o degradación determinista (0007): pausa.
        brief.status = BriefStatus.revision
        brief.requires_user_acceptance = requires_user_acceptance

    # Trazabilidad de proceso para MEASURE_KPI (§5.4).
    kpis = dict(order.kpis or {})
    kpis["produced_at"] = _iso(datetime.now(timezone.utc))
    # Feedback para la próxima ronda de retrabajo: los ítems que fallaron.
    # El LLM del crítico no siempre devuelve 'reasoning' (output_schema solo
    # pide verdict + checklist_results), así que el checklist es la señal.
    failed_items = [
        item for item, status in checklist_results.items() if status == "fail"
    ]
    if failed_items:
        kpis["last_critic_feedback"] = (
            "El borrador no cumple el checklist: "
            + ", ".join(failed_items)
            + ". Corrige esos puntos en el nuevo borrador."
        )
    else:
        kpis["last_critic_feedback"] = (
            critic_result.get("reasoning") or critic_result.get("reason") or ""
        )
    if "first_pass_quality" not in kpis:
        kpis["first_pass_quality"] = verdict == GateVerdict.auto_pass.value
    if verdict == GateVerdict.fail.value:
        kpis["rework_count"] = int(kpis.get("rework_count", 0)) + 1
    order.kpis = kpis

    session.commit()
    return {
        "order_id": str(order.id),
        "brief_id": str(brief.id),
        "verdict": verdict,
        "draft": draft,
        "checklist_results": checklist_results,
        "critic_feedback": critic_result.get("reasoning"),
        "requires_user_acceptance": requires_user_acceptance,
        "template_id": str(order.template_id) if order.template_id else None,
        "artifact_id": str(order.artifact_id) if order.artifact_id else None,
        "status": _val(order.status),
    }


# ----------------------------------------------------------------------
# Publicación manual (por ahora) + MEASURE_KPI (§5.4)
# ----------------------------------------------------------------------


def publish_order(session: Session, order: ProductionOrder) -> ProductionOrder:
    """Publicación MANUAL por ahora: marca la orden como 'publicada'.

    La capa DEPLOY automática (docs/diseno_sistema_publicacion.md) aún no
    está construida; este paso es el puente manual hasta que exista.
    """
    if _val(order.status) != STATUS_APROBADA:
        raise ValueError(
            f"solo se puede publicar una orden 'aprobada'; esta está en "
            f"'{_val(order.status)}'"
        )
    order.status = STATUS_PUBLICADA
    session.commit()
    return order


def record_kpis(session: Session, order: ProductionOrder, kpis: dict) -> dict:
    """MEASURE_KPI (§5.4) — métricas de proceso de la orden.

    Calcula automáticamente las que derivan de la orden:
      - lead_time_minutes: ORDER → PRE_DEPLOY_ENTRY (created_at → ahora).
      - cycle_time_minutes: EXECUTE_KAIZEN → ahora (produced_at → ahora).
      - time_in_stage: minutos en 'creada' y en 'produccion/gate'.
      - first_pass_quality: ¿el primer paso por el gate fue auto_pass?
      - rework_rate: rework_count / total de pasos por el gate.
    El resto (métricas de contenido: guardados, respuestas, retención) se
    registra tal cual lo reporta el equipo.
    """
    now = datetime.now(timezone.utc)
    created = order.created_at
    if created is not None and created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    produced_at = _parse_ts((order.kpis or {}).get("produced_at"))
    if produced_at is not None and produced_at.tzinfo is None:
        produced_at = produced_at.replace(tzinfo=timezone.utc)

    computed: dict = {}
    if created is not None:
        computed["lead_time_minutes"] = round((now - created).total_seconds() / 60, 1)
    if produced_at is not None:
        computed["cycle_time_minutes"] = round(
            (now - produced_at).total_seconds() / 60, 1
        )
        computed["time_in_stage"] = {
            "creada": (
                round((produced_at - created).total_seconds() / 60, 1)
                if created is not None
                else None
            ),
            "produccion_gate": round((now - produced_at).total_seconds() / 60, 1),
        }
    computed["first_pass_quality"] = bool((order.kpis or {}).get("first_pass_quality"))
    rework_count = int((order.kpis or {}).get("rework_count", 0))
    computed["rework_rate"] = round(rework_count / (rework_count + 1), 2)

    merged = {**computed, **(kpis or {})}
    order.kpis = merged
    if _val(order.status) in (STATUS_APROBADA, STATUS_PUBLICADA):
        order.status = STATUS_MEDIDA
    session.commit()
    return merged


# ----------------------------------------------------------------------
# MICRO_KAIZEN (§5.5) + MICRO_RITUAL (§5.6)
# ----------------------------------------------------------------------


def record_micro_kaizen(
    session: Session, order: ProductionOrder, experiment: dict, ritual: dict
) -> dict:
    """MICRO_KAIZEN (§5.5) + MICRO_RITUAL (§5.6).

    MICRO_KAIZEN: ajuste incremental (probar otro horario, otro hook, un
    formato de apoyo). MICRO_RITUAL (5–10 min): chequeo exprés de riesgos
    y mitigaciones. Ambos se registran en order.kpis.micro_kaizen como
    historial acumulativo.
    """
    kpis = dict(order.kpis or {})
    micro = list(kpis.get("micro_kaizen", []) or [])
    micro.append(
        {
            "experiment": experiment or {},
            "ritual": ritual or {},
            "recorded_at": _iso(datetime.now(timezone.utc)),
        }
    )
    kpis["micro_kaizen"] = micro
    order.kpis = kpis
    session.commit()
    return kpis


# ----------------------------------------------------------------------
# KAIZEN_DECISION (§5.7) → UPDATE_REGISTRY | ARCHIVE_KAIZEN
# ----------------------------------------------------------------------


def _manifest_row(order: ProductionOrder, cycle: KaizenCycle) -> str:
    """Fila del manifest para este ciclo Kaizen."""
    return (
        f"| {order.calendar_slot} | {_val(order.brand_objective)} | "
        f"{_val(order.content_bucket)} | {order.artifact_type} | "
        f"{cycle.created_at.date().isoformat() if cycle.created_at else ''} | "
        f"kaizen_{cycle.id} |"
    )


def _update_registry(
    session: Session, order: ProductionOrder, cycle: KaizenCycle
) -> None:
    """UPDATE_REGISTRY (§5.7) — la plantilla mejora para todo el equipo.

    Actualiza /components/manifest.md (añade la fila del ciclo; si el
    manifest está vacío, reemplaza la fila placeholder) y escribe
    /kb/kaizen_<id>.md. Nunca lanza: el manifest es un espejo legible; la
    fuente de verdad del ciclo es la fila en kaizen_cycles.
    """
    try:
        COMPONENTS_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        if COMPONENTS_MANIFEST.exists():
            text = COMPONENTS_MANIFEST.read_text(encoding="utf-8")
        else:
            text = ""
        row = _manifest_row(order, cycle)
        placeholder = (
            "| _(vacío — se llena con el primer ciclo Kaizen real)_ | | | | | |"
        )
        if placeholder in text:
            text = text.replace(placeholder, row)
        else:
            # Añade la fila después de la cabecera de la tabla.
            marker = "|---|---|---|---|---|---|"
            if marker in text:
                text = text.replace(marker, marker + "\n" + row, 1)
            else:
                text += "\n" + row
        COMPONENTS_MANIFEST.write_text(text, encoding="utf-8")
    except OSError:
        pass

    _write_kb_doc(order, cycle)


def _archive_kaizen(
    session: Session, order: ProductionOrder, cycle: KaizenCycle
) -> None:
    """ARCHIVE_KAIZEN (§5.7) — se documenta igual, sin cambiar la plantilla base."""
    _write_kb_doc(order, cycle)


def _write_kb_doc(order: ProductionOrder, cycle: KaizenCycle) -> None:
    """Escribe /kb/kaizen_<id>.md (documentación del ciclo)."""
    try:
        KB_DIR.mkdir(parents=True, exist_ok=True)
        (KB_DIR / f"kaizen_{cycle.id}.md").write_text(
            f"""---
tipo: kaizen
decision: {cycle.decision}
order_id: {order.id}
---

# Ciclo Kaizen — {cycle.id}

## Orden
- slot: {order.calendar_slot}
- brand_objective: {_val(order.brand_objective)}
- content_bucket: {_val(order.content_bucket)}
- artifact_type: {order.artifact_type}
- channel: {order.channel}
- scheduled_date: {order.scheduled_date.isoformat() if order.scheduled_date else ""}
- insight_core: {order.insight_core}

## Decisión
- decision: {cycle.decision}
- improvement_summary: {cycle.improvement_summary or "(sin resumen)"}

## Métricas del ciclo
{cycle.metrics or {}}
""",
            encoding="utf-8",
        )
    except OSError:
        pass


def kaizen_decision(
    session: Session,
    order: ProductionOrder,
    decision: str,
    improvement_summary: str = "",
) -> dict:
    """KAIZEN_DECISION (§5.7) — ¿la mejora es significativa?

    - update_registry: se actualiza /components/manifest.md (la plantilla
      mejora para todo el equipo) y /kb/kaizen_<id>.md.
    - archive: se documenta igual, sin cambiar la plantilla base.

    Ambas salidas convergen en PRE_DEPLOY_ENTRY (§11).
    """
    if decision not in (DECISION_UPDATE_REGISTRY, DECISION_ARCHIVE):
        raise ValueError(
            f"decisión inválida '{decision}'; debe ser "
            f"'{DECISION_UPDATE_REGISTRY}' o '{DECISION_ARCHIVE}'"
        )
    cycle = KaizenCycle(
        order_id=order.id,
        decision=decision,
        improvement_summary=improvement_summary or None,
        metrics=dict(order.kpis or {}),
    )
    session.add(cycle)
    session.flush()
    order.kaizen_decision = decision
    if decision == DECISION_UPDATE_REGISTRY:
        _update_registry(session, order, cycle)
    else:
        _archive_kaizen(session, order, cycle)
    session.commit()
    return {
        "order_id": str(order.id),
        "kaizen_cycle_id": str(cycle.id),
        "decision": decision,
        "improvement_summary": improvement_summary,
        "converges_to": "PRE_DEPLOY_ENTRY",
    }
