"""Manifiesto + template → contrato de entrada para la herramienta (0006).

El contrato es siempre `manifiesto + template → JSON de entrada para la
herramienta` (diseño §4): el template estático se parametriza con las
variables del manifiesto y las variables extra del llamador.
"""

from __future__ import annotations

from src.db.models import ProductionTemplate


def merge_deep(base: dict, override: dict) -> dict:
    """Merge profundo: los dicts se fusionan recursivamente, el resto de
    valores (incluidas listas) se reemplazan."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_deep(result[key], value)
        else:
            result[key] = value
    return result


def render(
    template: ProductionTemplate, manifest: dict, variables: dict | None = None
) -> dict:
    """Combina template.content con las variables del manifiesto y las
    variables extra. Devuelve el contrato de entrada para la herramienta:
    {"template_name", "template_version", "params"}."""
    params = merge_deep(dict(template.content or {}), manifest or {})
    if variables:
        params = merge_deep(params, variables)
    return {
        "template_name": template.name,
        "template_version": template.version,
        "params": params,
    }


def render_for_tool(
    template: ProductionTemplate,
    manifest: dict,
    tool_name: str,
    variables: dict | None = None,
) -> dict:
    """Igual que render pero filtra/mapea params según la herramienta.

    Implementación simple: si el contenido tiene un sub-dict `by_tool` o
    una clave con el nombre de la herramienta, se usa ese sub-dict como
    params; si no, se devuelve todo el contrato.
    """
    contract = render(template, manifest, variables)
    params = contract["params"]
    by_tool = params.get("by_tool")
    if isinstance(by_tool, dict) and tool_name in by_tool:
        contract["params"] = by_tool[tool_name]
        return contract
    if tool_name in params and isinstance(params[tool_name], dict):
        contract["params"] = params[tool_name]
        return contract
    return contract
