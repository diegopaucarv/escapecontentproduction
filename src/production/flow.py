"""Puente proyecto → pipeline de producción (Fase 5, §20.3).

Crea el brief compañero (ContentArtifact.brief_id es NOT NULL), el
artefacto con su project_id, el manifiesto versionado desde el snapshot y
ejecuta la cadena de herramientas del formato. El orquestador NO commitea
a propósito: el llamador (endpoint) debe hacer session.commit() después.

Gobernanza (0009/0010): el brief compañero ya NO se crea pre-aprobado con
S1/riesgo bajo hardcodeados. Usa el segmento y el riesgo REALES que el
proyecto declara (ProjectCreate) y corre el Gatekeeper (evaluate_gate).
Solo si el gate auto-pasa se produce; si exige revisión humana, el brief
queda en 'revision' con requires_user_acceptance=True y el pipeline se
pausa hasta que un 🟨 líder lo apruebe vía POST /briefs/{id}/approve
(OWNER_APPROVAL) — el mismo camino que el flujo de briefs real.
"""

from __future__ import annotations

from src.agents.gatekeeper import GateInput, GateVerdict, evaluate_gate
from src.db.models import (
    BrandObjective,
    BriefStatus,
    ContentArtifact,
    ContentBrief,
    ContentBucket,
    ProductionRoute,
    RiskLevel,
    RouteDecision,
)
from src.production.refine import resolve_format_spec
from src.tools import Orchestrator, create_manifest


def _val(v):
    """Convierte un enum de SQLAlchemy a su valor string; pasa el resto tal cual."""
    return v.value if hasattr(v, "value") else v


def _find_companion_brief(session, project) -> ContentBrief | None:
    """Busca el brief compañero ya creado para este proyecto (re-produce
    tras aprobación humana del brief: no se duplica el brief)."""
    if project.companion_brief_id is None:
        return None
    return session.get(ContentBrief, project.companion_brief_id)


def _create_companion_brief(session, project) -> ContentBrief:
    """Crea el brief compañero con los datos REALES del proyecto.

    El segmento y el riesgo vienen de ProjectCreate (nunca S1/bajo
    hardcodeados). La ruta es conservadora por construcción: un proyecto
    es una solución nueva con producción completa, y el Gatekeeper trata
    esa combinación como SIEMPRE sujeta a revisión humana.
    """
    brief = ContentBrief(
        brand_objective=project.brand_objective or BrandObjective.ESCAPE_SOCIAL,
        content_bucket=ContentBucket.caso_autoridad,
        resumen=project.topic,
        insight_core=project.topic,
        segment_client=project.segment_client,
        risk_level=project.risk_level or RiskLevel.bajo,
        route_decision=RouteDecision.nueva_solucion,
        production_route=ProductionRoute.complete,
        # 0007: el brief nace en 'revision' — solo el Gatekeeper (auto_pass)
        # o un 🟨 líder (OWNER_APPROVAL) pueden llevarlo a 'aprobado'.
        status=BriefStatus.revision,
    )
    return brief


def _govern_brief(session, project, brief) -> dict:
    """Corre el Gatekeeper sobre el brief compañero y aplica el veredicto.

    Devuelve el contrato de decisión:
    - auto_pass: el brief pasa a 'aprobado' y se puede producir.
    - needs_human_review: el brief queda en 'revision' con
      requires_user_acceptance=True — el pipeline se pausa.
    - fail: igual que needs_human_review (pausa); el líder decide si
      aprueba igual (OWNER_APPROVAL puede sobrepasar un fail).
    """
    # El proyecto no declara checklist de publicación (fuente_verificable,
    # gancho_15s_ok, cta_unico...): el gate no puede verificar nada, así
    # que la revisión humana es la única salida salvo auto_pass explícito.
    gate_input = GateInput(
        risk_level=_val(brief.risk_level),
        production_route=_val(brief.production_route),
        route_decision=_val(brief.route_decision),
        segment_client=_val(brief.segment_client),
        checklist_items={},
        requires_anonymization=bool(brief.validation_required),
    )
    result = evaluate_gate(gate_input)

    if result.verdict == GateVerdict.auto_pass:
        brief.status = BriefStatus.aprobado
        return {
            "verdict": result.verdict.value,
            "requires_user_acceptance": False,
            "reason": result.reason,
        }

    # needs_human_review o fail: el pipeline se pausa (0007). El flag se
    # persiste para auditoría; el líder aprueba vía POST /briefs/{id}/approve.
    brief.requires_user_acceptance = True
    return {
        "verdict": result.verdict.value,
        "requires_user_acceptance": True,
        "reason": result.reason,
        "failed_items": result.failed_items,
    }


def produce_project(session, project, snapshot) -> dict:
    """Materializa el proyecto: brief + artefacto + manifiesto + cadena.

    Gobernanza real (0009/0010): el brief compañero se crea con el
    segmento/riesgo del proyecto y pasa por el Gatekeeper. Si el gate
    exige revisión humana, NO se ejecuta la cadena: se devuelve el
    contrato de pausa (requires_user_acceptance=True) con el brief_id
    para que el líder lo apruebe y se re-dispare /produce.

    Si OrchestrationError se lanza, se re-lanza (el endpoint lo mapea a
    409). El llamador debe commitear después de esta función.
    """
    # 1. Brief compañero — reutiliza el existente si lo hay (re-produce
    #    tras aprobación humana), si no lo crea con gobernanza real.
    brief = _find_companion_brief(session, project)
    if brief is None:
        brief = _create_companion_brief(session, project)
        session.add(brief)
        session.flush()
        # El id es server_default (gen_random_uuid): se asigna en el flush.
        project.companion_brief_id = brief.id

    if _val(brief.status) != "aprobado":
        governance = _govern_brief(session, project, brief)
        if governance["requires_user_acceptance"]:
            # 0007: el pipeline se pausa; el líder aprueba el brief vía
            # POST /briefs/{id}/approve y luego re-dispara /produce.
            return {
                "project": project,
                "brief_id": str(brief.id),
                "requires_user_acceptance": True,
                "verdict": governance["verdict"],
                "reason": governance["reason"],
            }

    # 2. Artefacto con project_id (0008) y format_spec resuelta por marca.
    format_spec = resolve_format_spec(
        session, project.brand_objective, project.artifact_type
    )
    artifact = ContentArtifact(
        brief_id=brief.id,
        artifact_type=project.artifact_type or "post",
        channel="social",
        status="borrador",
        format_spec_id=format_spec.id if format_spec else None,
        project_id=project.id,
    )
    session.add(artifact)
    session.flush()

    # 3. Manifiesto versionado e inmutable desde el snapshot.
    manifest = create_manifest(session, artifact.id, snapshot)

    # 4. Orquestador — no commitea a propósito (Fase 2).
    project.status = "en_produccion"
    result = Orchestrator(session).run_tool_chain(artifact.id, manifest.version)

    # 5. storage_path del proyecto desde el artefacto materializado.
    project.storage_path = artifact.storage_path

    return {
        "project": project,
        "artifact_id": str(artifact.id),
        "manifest_version": manifest.version,
        "result": result,
    }
