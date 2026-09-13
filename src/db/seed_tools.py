"""
Seed de la capa de producción creativa local (0005).

Inserta/actualiza:
  1. El catálogo de `tool_adapters` (los 10 MCP servers del diseño
     `docs/diseno_produccion_multiformato.md` §1).
  2. Las specs de `format_specs` con su `tool_chain` por marca y formato
     (mismo doc §2).

A diferencia de src/db/seed_llm.py, NO requiere ninguna variable de
entorno: es datos puros. `component_library` se deja vacía a propósito —
se puebla a medida que se reutilizan componentes reales.

Uso:
    python -m src.db.seed_tools
"""

from __future__ import annotations

import argparse

from sqlalchemy import select

from src.db.models import BrandObjective, FormatSpec, ToolAdapter
from src.db.session import SessionLocal

# ---------------------------------------------------------------------
# Catálogo de MCP servers (tabla tool_adapters) — diseño §1
# ---------------------------------------------------------------------

TOOL_ADAPTERS = [
    {
        "name": "inkscape",
        "mcp_server_name": "inkscape",
        "execution_mode": "local",
        "requires_license": "Inkscape 1.x instalado",
    },
    {
        "name": "krita",
        "mcp_server_name": "krita",
        "execution_mode": "local",
        "requires_license": "Plugin Python instalado dentro de Krita",
    },
    {
        "name": "reaper",
        "mcp_server_name": "reaper",
        "execution_mode": "local",
        "requires_license": "REAPER con API de scripting/OSC habilitada",
    },
    {
        "name": "resolve",
        "mcp_server_name": "resolve",
        "execution_mode": "local",
        "requires_license": "DaVinci Resolve Studio",
    },
    {
        "name": "comfyui",
        "mcp_server_name": "comfyui",
        "execution_mode": "local",
        "requires_license": "ComfyUI instalado + al menos un checkpoint",
    },
    {
        "name": "elevenlabs",
        "mcp_server_name": "elevenlabs",
        "execution_mode": "local_orchestration_cloud_inference",
        "requires_license": "ELEVENLABS_API_KEY; créditos de uso",
    },
    {
        "name": "veo3",
        "mcp_server_name": "veo3",
        "execution_mode": "local_orchestration_cloud_inference",
        "requires_license": "GEMINI_API_KEY con acceso a Veo habilitado",
    },
    {
        "name": "canva",
        "mcp_server_name": "canva",
        "execution_mode": "remote_cloud",
        "requires_license": "Canva Pro+",
    },
    {
        "name": "gdrive",
        "mcp_server_name": "gdrive",
        "execution_mode": "remote_cloud",
        "requires_license": "Credenciales OAuth de la cuenta de Drive",
    },
    {
        "name": "filesystem",
        "mcp_server_name": "filesystem",
        "execution_mode": "local",
        "requires_license": None,
    },
]

# ---------------------------------------------------------------------
# Specs de formato (tabla format_specs) — diseño §2
#
# Cada spec se expande por marca (ESCAPE_SOCIAL y ERGALIA_COMERCIAL) con
# requisitos visuales propios. La cadena de herramientas (tool_chain) es
# la misma por artifact_type: las herramientas no cambian con la marca.
# 'producer' es el paso LLM (Producer-Critic) que ya ocurrió antes de la
# cadena mecánica — el orquestador lo trata como paso ya completado.
# ---------------------------------------------------------------------

FORMAT_SPECS = [
    {
        "artifact_type": "post",
        "structure": {
            "hook_15s": "string",
            "cuerpo": "string",
            "cta": "string",
        },
        "constraints": {"max_chars": 280, "cta_unico": True},
        "derivation_rules": [
            "derivable a hilo",
            "derivable a carrusel",
            "derivable a video_corto",
        ],
        "qa_checks": ["fuente_verificable", "gancho_15s_ok", "cta_unico"],
        "tool_chain": ["producer"],
    },
    {
        "artifact_type": "video_corto",
        "structure": {
            "guion": "string",
            "voz": "audio",
            "assets_visuales": "array",
            "montaje": "video",
            "subtitulos": "string",
        },
        "constraints": {"duracion_seg": "15-60", "subtitulos": True},
        "derivation_rules": [
            "derivable desde post",
            "derivable a video_largo",
            "derivable a broadcast",
        ],
        "qa_checks": [
            "fuente_verificable",
            "gancho_15s_ok",
            "cta_unico",
            "subtitulos_presentes",
        ],
        "tool_chain": ["producer", "elevenlabs", "comfyui", "resolve"],
    },
    {
        "artifact_type": "video_largo",
        "structure": {
            "guion_extendido": "string",
            "voz": "audio",
            "b_roll": "array",
            "montaje": "video",
            "color": "string",
            "mezcla": "audio",
        },
        "constraints": {"duracion_seg": "180-600", "subtitulos": True},
        "derivation_rules": [
            "derivable desde video_corto",
            "derivable a broadcast",
            "derivable a newsletter",
        ],
        "qa_checks": [
            "fuente_verificable",
            "gancho_15s_ok",
            "cta_unico",
            "subtitulos_presentes",
        ],
        "tool_chain": ["producer", "elevenlabs", "comfyui", "veo3", "resolve"],
    },
    {
        "artifact_type": "carrusel",
        "structure": {
            "copy_por_slide": "array",
            "iconografia": "array",
            "composicion": "array",
            "ensamblaje": "file",
        },
        "constraints": {"slides": "5-10", "cta_unico": True},
        "derivation_rules": [
            "derivable desde post",
            "derivable a one_pager",
            "derivable a poster_qr",
        ],
        "qa_checks": [
            "fuente_verificable",
            "gancho_15s_ok",
            "cta_unico",
            "legibilidad_slides",
        ],
        "tool_chain": ["producer", "inkscape", "krita"],
    },
    {
        "artifact_type": "poster_qr",
        "structure": {
            "copy": "string",
            "cta": "string",
            "qr_embebido": "file",
            "vector_imprimible": "file",
        },
        "constraints": {"formato_impresion": "A3/A4", "qr_escaneable": True},
        "derivation_rules": [
            "derivable desde carrusel",
            "derivable desde one_pager",
        ],
        "qa_checks": [
            "fuente_verificable",
            "cta_unico",
            "qr_escaneable",
            "resolucion_impresion",
        ],
        "tool_chain": ["producer", "inkscape"],
    },
    {
        "artifact_type": "one_pager",
        "structure": {
            "texto": "string",
            "diagramas": "array",
            "cifras": "array",
            "maquetacion": "file",
        },
        "constraints": {"paginas": 1, "cta_unico": True},
        "derivation_rules": [
            "derivable desde carrusel",
            "derivable a white_paper",
            "derivable a propuesta",
        ],
        "qa_checks": [
            "fuente_verificable",
            "cta_unico",
            "cifras_con_fuente",
        ],
        "tool_chain": ["producer", "inkscape", "krita", "canva"],
    },
    {
        "artifact_type": "white_paper",
        "structure": {
            "texto": "string",
            "diagramas": "array",
            "cifras": "array",
            "maquetacion": "file",
        },
        "constraints": {"paginas": "8-20", "cta_unico": True},
        "derivation_rules": [
            "derivable desde one_pager",
            "derivable a webinar",
            "derivable a newsletter",
        ],
        "qa_checks": [
            "fuente_verificable",
            "cta_unico",
            "cifras_con_fuente",
            "revision_legal",
        ],
        "tool_chain": ["producer", "inkscape", "krita", "canva"],
    },
    {
        "artifact_type": "broadcast",
        "structure": {
            "guion": "string",
            "voz": "audio",
            "soundscape": "audio",
            "mezcla": "audio",
        },
        "constraints": {"duracion_seg": "30-180", "cta_unico": True},
        "derivation_rules": [
            "derivable desde video_corto",
            "derivable desde post",
        ],
        "qa_checks": [
            "fuente_verificable",
            "gancho_15s_ok",
            "cta_unico",
            "nivel_audio_ok",
        ],
        "tool_chain": ["producer", "elevenlabs", "reaper"],
    },
]

# Requisitos visuales por marca (UX_DESIGN). Se pueden editar vía CRUD
# después del seed — esto es solo el punto de partida.
VISUAL_REQUIREMENTS = {
    "ESCAPE_SOCIAL": {
        "post": {"tono": "directo, cientifico-accesible", "formato_imagen": "ninguno"},
        "video_corto": {
            "tono": "directo, cientifico-accesible",
            "estilo": "subtitulos grandes, fondo limpio",
        },
        "video_largo": {
            "tono": "directo, cientifico-accesible",
            "estilo": "b-roll documental, color natural",
        },
        "carrusel": {
            "tono": "directo, cientifico-accesible",
            "estilo": "iconografia plana, alto contraste",
        },
        "poster_qr": {
            "tono": "directo, cientifico-accesible",
            "estilo": "vectorial, QR prominente",
        },
        "one_pager": {
            "tono": "directo, cientifico-accesible",
            "estilo": "diagramas claros, jerarquia tipografica",
        },
        "white_paper": {
            "tono": "directo, cientifico-accesible",
            "estilo": "diagramas claros, citas visibles",
        },
        "broadcast": {
            "tono": "directo, cientifico-accesible",
            "estilo": "voz clara, soundscape sobrio",
        },
    },
    "ERGALIA_COMERCIAL": {
        "post": {"tono": "formal, orientado a servicio", "formato_imagen": "ninguno"},
        "video_corto": {
            "tono": "formal, orientado a servicio",
            "estilo": "subtitulos sobrios, identidad corporativa",
        },
        "video_largo": {
            "tono": "formal, orientado a servicio",
            "estilo": "b-roll corporativo, color de marca",
        },
        "carrusel": {
            "tono": "formal, orientado a servicio",
            "estilo": "iconografia corporativa, paleta de marca",
        },
        "poster_qr": {
            "tono": "formal, orientado a servicio",
            "estilo": "vectorial, QR integrado a la identidad",
        },
        "one_pager": {
            "tono": "formal, orientado a servicio",
            "estilo": "diagramas de servicio, jerarquia clara",
        },
        "white_paper": {
            "tono": "formal, orientado a servicio",
            "estilo": "diagramas de servicio, casos de autoridad",
        },
        "broadcast": {
            "tono": "formal, orientado a servicio",
            "estilo": "voz clara, soundscape corporativo",
        },
    },
}


def _upsert_adapter(session, data: dict) -> ToolAdapter:
    adapter = (
        session.execute(select(ToolAdapter).where(ToolAdapter.name == data["name"]))
        .scalars()
        .first()
    )
    if adapter is None:
        adapter = ToolAdapter(**data)
        session.add(adapter)
        session.flush()
    else:
        for k, v in data.items():
            setattr(adapter, k, v)
    return adapter


def _upsert_format_spec(session, data: dict) -> FormatSpec:
    spec = (
        session.execute(
            select(FormatSpec).where(
                FormatSpec.brand_objective == data["brand_objective"],
                FormatSpec.artifact_type == data["artifact_type"],
            )
        )
        .scalars()
        .first()
    )
    if spec is None:
        spec = FormatSpec(**data)
        session.add(spec)
        session.flush()
    else:
        for k, v in data.items():
            setattr(spec, k, v)
    return spec


def seed(session=None) -> dict:
    """Inserta/actualiza tool_adapters y format_specs. Devuelve un resumen.

    No requiere variables de entorno. `component_library` se deja vacía.
    """
    own_session = session is None
    if own_session:
        session = SessionLocal()
    try:
        adapters = [_upsert_adapter(session, data) for data in TOOL_ADAPTERS]

        specs: list[FormatSpec] = []
        for brand in (BrandObjective.ESCAPE_SOCIAL, BrandObjective.ERGALIA_COMERCIAL):
            visual = VISUAL_REQUIREMENTS[brand.value]
            for base in FORMAT_SPECS:
                data = dict(base)
                data["brand_objective"] = brand
                data["visual_requirements"] = visual.get(base["artifact_type"], {})
                specs.append(_upsert_format_spec(session, data))

        session.commit()
        return {
            "adapters": [str(a.id) for a in adapters],
            "adapter_names": [a.name for a in adapters],
            "format_specs": [str(s.id) for s in specs],
            "format_count": len(specs),
        }
    finally:
        if own_session:
            session.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed de producción creativa local (tool_adapters + format_specs)."
    )
    parser.parse_args()
    result = seed()
    print("Capa de producción creativa local lista:")
    print(f"  Tool adapters ({len(result['adapter_names'])}):")
    for name in result["adapter_names"]:
        print(f"    - {name}")
    print(f"  Format specs ({result['format_count']}):")
    print("    - 8 artifact_types x 2 marcas (ESCAPE_SOCIAL, ERGALIA_COMERCIAL)")


if __name__ == "__main__":
    main()
