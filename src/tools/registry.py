"""Resolución data-driven de fases y cadenas de herramientas (0005/0006).

El orquestador (src/tools/orchestrator.py) usa estas funciones como datos,
no como if/else por formato: lee `format_specs.phases` + `tool_chain` y el
catálogo de `tool_adapters` / `production_templates` para materializar la
cadena mecánica declarada en la base.
"""

from __future__ import annotations

from sqlalchemy import select

from src.db.models import FormatSpec, ProductionTemplate, ToolAdapter


def _find_spec(session, brand_objective, artifact_type) -> FormatSpec | None:
    return (
        session.execute(
            select(FormatSpec).where(
                FormatSpec.brand_objective == brand_objective,
                FormatSpec.artifact_type == artifact_type,
            )
        )
        .scalars()
        .first()
    )


def resolve_phases(session, brand_objective, artifact_type) -> dict:
    """Devuelve `spec.phases` (preproduccion/produccion/postproduccion, cada
    una con `steps` y `templates`). Si no existe la spec, devuelve {}."""
    spec = _find_spec(session, brand_objective, artifact_type)
    if spec is None:
        return {}
    return spec.phases or {}


def resolve_tool_chain(session, brand_objective, artifact_type) -> list[str]:
    """Devuelve `spec.tool_chain` (nombres de tool_adapter en orden)."""
    spec = _find_spec(session, brand_objective, artifact_type)
    if spec is None:
        return []
    return list(spec.tool_chain or [])


def resolve_template(session, name) -> ProductionTemplate | None:
    """Busca un template activo por nombre."""
    return (
        session.execute(
            select(ProductionTemplate).where(
                ProductionTemplate.name == name,
                ProductionTemplate.is_active.is_(True),
            )
        )
        .scalars()
        .first()
    )


def list_templates(
    session, content_type: str | None = None, phase: str | None = None
) -> list[ProductionTemplate]:
    """Lista templates con filtros opcionales por content_type y phase."""
    stmt = select(ProductionTemplate)
    if content_type is not None:
        stmt = stmt.where(ProductionTemplate.content_type == content_type)
    if phase is not None:
        stmt = stmt.where(ProductionTemplate.phase == phase)
    return list(session.execute(stmt).scalars().all())


def list_adapters(session, is_active: bool | None = True) -> list[ToolAdapter]:
    """Lista tool_adapters. is_active=None devuelve todos (activos e inactivos)."""
    stmt = select(ToolAdapter)
    if is_active is not None:
        stmt = stmt.where(ToolAdapter.is_active.is_(is_active))
    return list(session.execute(stmt).scalars().all())


def validate_tool_chain(session) -> list[str]:
    """Valida la consistencia de todas las format_specs activas.

    Reporta errores si algún nombre de `tool_chain` (excepto "producer",
    que es el paso LLM ya completado) no existe en `tool_adapters.name`, o
    si algún template referenciado en `phases` no existe en
    `production_templates.name`. Devuelve lista vacía si todo está bien.
    """
    errors: list[str] = []
    specs = list(
        session.execute(select(FormatSpec).where(FormatSpec.is_active.is_(True)))
        .scalars()
        .all()
    )
    adapter_names = {a.name for a in list_adapters(session, is_active=None)}
    template_names = {t.name for t in list_templates(session)}

    for spec in specs:
        brand = getattr(spec.brand_objective, "value", spec.brand_objective)
        label = f"format_spec {brand}/{spec.artifact_type}"
        for step in spec.tool_chain or []:
            if step != "producer" and step not in adapter_names:
                errors.append(
                    f"{label}: tool_chain refiere a tool_adapter inexistente '{step}'"
                )
        for phase, phase_spec in (spec.phases or {}).items():
            for template_name in (phase_spec or {}).get("templates", []) or []:
                if template_name not in template_names:
                    errors.append(
                        f"{label} fase {phase}: template inexistente '{template_name}'"
                    )
    return errors
