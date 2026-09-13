"""Recomendaciones de post-producción deterministas (Fase 5, §20.4).

Modo 2 del diseño: recomendaciones algorítmicas desde los templates de
postproducción + el snapshot del proyecto. Sin LLM: reglas en código.
"""

from __future__ import annotations

from src.tools.registry import list_templates
from src.tools.template_engine import merge_deep


def postproduction_recommendations(session, project, snapshot) -> dict:
    """Recomendaciones agrupadas por content_type (audio/video/grafico).

    Para cada template de fase 'postproduccion' construye una entrada
    {"template", "content_type", "params"} donde params es el content del
    template fusionado con las claves relevantes del snapshot. Si no hay
    templates de postproducción, devuelve grupos vacíos (no lanza).
    """
    recomendaciones: dict[str, list[dict]] = {}
    for template in list_templates(session, phase="postproduccion"):
        params = merge_deep(dict(template.content or {}), snapshot or {})
        entry = {
            "template": template.name,
            "content_type": template.content_type,
            "params": params,
        }
        recomendaciones.setdefault(template.content_type, []).append(entry)

    return {
        "project_id": str(project.id),
        "artifact_type": project.artifact_type,
        "recomendaciones": recomendaciones,
    }
