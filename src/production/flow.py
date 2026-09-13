"""Puente proyecto → pipeline de producción (Fase 5, §20.3).

Crea el brief compañero (ContentArtifact.brief_id es NOT NULL), el
artefacto con su project_id, el manifiesto versionado desde el snapshot y
ejecuta la cadena de herramientas del formato. El orquestador NO commitea
a propósito: el llamador (endpoint) debe hacer session.commit() después.
"""

from __future__ import annotations

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


def produce_project(session, project, snapshot) -> dict:
    """Materializa el proyecto: brief + artefacto + manifiesto + cadena.

    Si OrchestrationError se lanza, se re-lanza (el endpoint lo mapea a
    409). El llamador debe commitear después de esta función.
    """
    # 1. Brief compañero — ContentArtifact.brief_id es NOT NULL.
    brief = ContentBrief(
        brand_objective=project.brand_objective or BrandObjective.ESCAPE_SOCIAL,
        content_bucket=ContentBucket.caso_autoridad,
        resumen=project.topic,
        insight_core=project.topic,
        segment_client="S1",
        status=BriefStatus.aprobado,
        risk_level=RiskLevel.bajo,
        route_decision=RouteDecision.nueva_solucion,
        production_route=ProductionRoute.complete,
    )
    session.add(brief)
    session.flush()

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
