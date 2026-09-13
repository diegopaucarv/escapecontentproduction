"""Capa de herramientas (Fase 2) — pipeline de producción manifest-driven.

Exporta las clases y funciones principales de registry, manifest,
template_engine, mcp_client y orchestrator.
"""

from src.tools.manifest import (
    create_manifest,
    latest_version,
    list_manifests,
    load_manifest,
)
from src.tools.mcp_client import (
    DirectToolExecutor,
    McpClient,
    McpClientError,
    build_command,
)
from src.tools.orchestrator import OrchestrationError, Orchestrator
from src.tools.registry import (
    list_adapters,
    list_templates,
    resolve_phases,
    resolve_template,
    resolve_tool_chain,
    validate_tool_chain,
)
from src.tools.template_engine import merge_deep, render, render_for_tool

__all__ = [
    "Orchestrator",
    "OrchestrationError",
    "McpClient",
    "McpClientError",
    "DirectToolExecutor",
    "build_command",
    "resolve_phases",
    "resolve_tool_chain",
    "resolve_template",
    "list_templates",
    "list_adapters",
    "validate_tool_chain",
    "create_manifest",
    "load_manifest",
    "list_manifests",
    "latest_version",
    "render",
    "render_for_tool",
    "merge_deep",
]
