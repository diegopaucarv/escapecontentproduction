"""
Seed de la Fase 1.5 — templates estáticos y estructura de fases (0006).

Inserta/actualiza:
  1. Los 17 templates de `production_templates` del diseño
     `docs/diseno_produccion_multiformato.md` §3 (audio/video/gráfico ×
     preproducción/producción/postproducción).
  2. La estructura de fases `format_specs.phases` por (brand, artifact_type)
     del mismo doc §7: cada fase con sus `steps` y los `templates` que la
     materializan.

A diferencia de src/db/seed_llm.py, NO requiere ninguna variable de
entorno: es datos puros. Idempotente: upsert por clave única (name para
templates, (brand_objective, artifact_type) para format_specs).

Uso:
    python -m src.db.seed_templates
"""

from __future__ import annotations

import argparse

from sqlalchemy import select

from src.db.models import BrandObjective, FormatSpec, ProductionTemplate
from src.db.seed_tools import FORMAT_SPECS
from src.db.session import SessionLocal

# ---------------------------------------------------------------------
# Templates estáticos (tabla production_templates) — diseño §3
#
# content es JSONB libre: la estructura del template que el orquestador
# parametriza con el manifiesto. Para formatos no-JSON (otio/ass/cube/
# svg/txt) el content guarda la spec/estructura del template, no el raw.
# ---------------------------------------------------------------------

PRODUCTION_TEMPLATES = [
    # --- Audio: preproducción -----------------------------------------
    {
        "name": "Script_SSML_Template",
        "content_type": "audio",
        "phase": "preproduccion",
        "template_format": "json",
        "content": {
            "tags": ["pausa", "tono", "velocidad", "emocion"],
            "estructura": {
                "pausa": {"tag": "break", "atributos": ["time", "strength"]},
                "tono": {"tag": "prosody", "atributos": ["pitch", "contour"]},
                "velocidad": {"tag": "prosody", "atributos": ["rate"]},
                "emocion": {"tag": "emphasis", "atributos": ["level"]},
            },
            "por_parrafo": True,
            "pronunciaciones_foneticas": True,
            "marcas_de_tiempo": True,
        },
    },
    {
        "name": "Audio_Track_Manifest",
        "content_type": "audio",
        "phase": "preproduccion",
        "template_format": "json",
        "content": {
            "canales": [
                {
                    "canal": "voz",
                    "archivo": "string",
                    "in_point": "float",
                    "out_point": "float",
                },
                {
                    "canal": "bgm",
                    "archivo": "string",
                    "in_point": "float",
                    "out_point": "float",
                },
                {
                    "canal": "sfx",
                    "archivo": "string",
                    "in_point": "float",
                    "out_point": "float",
                },
            ],
            "mapa_de_pistas": True,
            "roles_de_voz": True,
        },
    },
    # --- Audio: producción --------------------------------------------
    {
        "name": "TTS_Voice_Preset",
        "content_type": "audio",
        "phase": "produccion",
        "template_format": "json",
        "content": {
            "stability": 0.5,
            "clarity": 0.75,
            "latency": "low",
            "voice_id": "string",
            "sample_rate": 48000,
            "bit_depth": 24,
        },
    },
    {
        "name": "Raw_Audio_Spec",
        "content_type": "audio",
        "phase": "produccion",
        "template_format": "json",
        "content": {
            "voces": {"canal": "mono", "formato": "wav", "compresion": "sin_comprimir"},
            "musica": {
                "canal": "stereo",
                "formato": "wav",
                "compresion": "sin_comprimir",
            },
            "sfx": {"canal": "stereo", "formato": "wav", "compresion": "sin_comprimir"},
            "sample_rate": 48000,
            "bit_depth": 24,
        },
    },
    # --- Audio: postproducción ----------------------------------------
    {
        "name": "Voice_FX_Chain_Preset",
        "content_type": "audio",
        "phase": "postproduccion",
        "template_format": "json",
        "content": {
            "hpf": {"frecuencia": 80, "tipo": "high_pass"},
            "eq_quirurgico": {"por_microfono": True},
            "de_esser": {"frecuencia": 6500, "tipo": "de_esser"},
            "compresor": {
                "tipo": "vca",
                "ratio": "3:1",
                "attack_ms": 15,
                "release_ms": 100,
            },
        },
    },
    {
        "name": "Sidechain_Ducking_Preset",
        "content_type": "audio",
        "phase": "postproduccion",
        "template_format": "json",
        "content": {
            "umbral_db": -22,
            "atenuacion_db": -12,
            "attack_ms": 20,
            "release_ms": 250,
            "fuente": "bgm",
            "objetivo": "voz",
        },
    },
    {
        "name": "Loudness_Master_Spec",
        "content_type": "audio",
        "phase": "postproduccion",
        "template_format": "json",
        "content": {
            "target_lufs": {"podcast": -16, "youtube": -14},
            "true_peak_dbfs": -1.0,
            "estandar": "EBU R128",
        },
    },
    # --- Video: preproducción -----------------------------------------
    {
        "name": "Storyboard_Spec",
        "content_type": "video",
        "phase": "preproduccion",
        "template_format": "json",
        "content": {
            "mapeo": {
                "inicio": "float",
                "fin": "float",
                "tipo_plano": "string",
                "prompt_visual": "string",
                "texto_pantalla": "string",
                "sfx": "string",
            },
            "escenas": [],
        },
    },
    {
        "name": "Timeline_Template",
        "content_type": "video",
        "phase": "preproduccion",
        "template_format": "otio",
        "content": {
            "formato": "otio",
            "pistas": {
                "V4": "subtitulos",
                "V3": "lower_thirds",
                "V2": "b_roll",
                "V1": "principal",
                "A1": "voz",
                "A2": "sfx",
                "A3": "bgm",
            },
            "aspect_ratio": "9:16",
        },
    },
    # --- Video: producción --------------------------------------------
    {
        "name": "ComfyUI_Workflow_Template",
        "content_type": "video",
        "phase": "produccion",
        "template_format": "json",
        "content": {
            "seeds": {"fija": 42, "modo": "determinista"},
            "modelo_base": "string",
            "loras": [{"nombre": "string", "peso": "float"}],
            "resolucion": {"ancho": 1080, "alto": 1920},
            "pasos": 30,
            "cfg": 7.0,
        },
    },
    {
        "name": "Subtitles_Style_Template",
        "content_type": "video",
        "phase": "produccion",
        "template_format": "ass",
        "content": {
            "tipografia": "string",
            "posicion_y": "float",
            "color": "string",
            "stroke": {"ancho": "float", "color": "string"},
            "sombra": {"offset": "float", "color": "string"},
            "karaoke": False,
            "sincronizacion": "word_level",
        },
    },
    # --- Video: postproducción ----------------------------------------
    {
        "name": "Color_Grade_LUT_Template",
        "content_type": "video",
        "phase": "postproduccion",
        "template_format": "cube",
        "content": {
            "look": "corporativo",
            "formato": "cube",
            "tamano": 33,
            "dominio": {"min": 0.0, "max": 1.0},
            "notas": "LUT base de color de marca; ajustar por escena",
        },
    },
    {
        "name": "Transitions_Rules_Spec",
        "content_type": "video",
        "phase": "postproduccion",
        "template_format": "json",
        "content": {
            "reglas": [
                {"tipo": "corte_limpio", "contexto": "entre_b_roll"},
                {"tipo": "dissolve", "frames": 10, "contexto": "cambio_de_capitulo"},
            ],
            "motion_fx": "string",
        },
    },
    {
        "name": "Render_Export_Preset",
        "content_type": "video",
        "phase": "postproduccion",
        "template_format": "json",
        "content": {
            "codec_video": ["H.264", "HEVC"],
            "resolucion": "1080x1920",
            "fps": 30,
            "codec_audio": "AAC",
            "bitrate_audio_kbps": 320,
            "bitrate_video": "string",
        },
    },
    # --- Gráfico: preproducción ---------------------------------------
    {
        "name": "Canvas_Layout_Spec",
        "content_type": "grafico",
        "phase": "preproduccion",
        "template_format": "json",
        "content": {
            "dimensiones": {
                "thumbnail": {"ancho": 1280, "alto": 720},
                "post": {"ancho": 1080, "alto": 1080},
            },
            "rejilla": {
                "columnas": 12,
                "margenes_seguridad": True,
                "areas_de_enfoque": True,
            },
        },
    },
    # --- Gráfico: producción ------------------------------------------
    {
        "name": "Inkscape_Layer_Template",
        "content_type": "grafico",
        "phase": "produccion",
        "template_format": "svg",
        "content": {
            "formato": "svg",
            "capas": [
                {"id": "#background", "tipo": "fondo"},
                {"id": "#illustration", "tipo": "ilustracion"},
                {"id": "#title_text", "tipo": "texto_titulo"},
                {"id": "#brand_logo", "tipo": "logo_marca"},
            ],
            "ids_fijos": True,
        },
    },
    # --- Gráfico: postproducción --------------------------------------
    {
        "name": "Image_Composite_FX_Spec",
        "content_type": "grafico",
        "phase": "postproduccion",
        "template_format": "json",
        "content": {
            "modos_de_fusion": ["Multiply", "Overlay"],
            "sombra_paralela": {
                "radio": "float",
                "offset": "float",
                "opacidad": "float",
            },
            "compresion": {"webp": {"calidad": 80}, "png": {"sin_perdida": True}},
            "grano": "float",
        },
    },
]

# ---------------------------------------------------------------------
# Estructura de fases por formato (format_specs.phases) — diseño §7
#
# Cada artifact_type define sus fases del marco universal (§1) con los
# pasos (steps) y los templates que las materializan. 'post' no tiene
# templates mecánicos: su tool_chain es solo 'producer' (LLM).
# ---------------------------------------------------------------------

FORMAT_PHASES = {
    "post": {
        "preproduccion": {
            "steps": ["estructura_narrativa"],
            "templates": [],
        },
        "produccion": {
            "steps": ["redaccion"],
            "templates": [],
        },
        "postproduccion": {
            "steps": ["formateo_visual"],
            "templates": [],
        },
    },
    "video_corto": {
        "preproduccion": {
            "steps": [
                "desglose_de_escenas",
                "definicion_aspect_ratio",
                "timeline_spec",
            ],
            "templates": ["Storyboard_Spec", "Timeline_Template"],
        },
        "produccion": {
            "steps": ["render_b_roll", "generacion_subtitulos", "graficos_vectoriales"],
            "templates": [
                "TTS_Voice_Preset",
                "ComfyUI_Workflow_Template",
                "Subtitles_Style_Template",
            ],
        },
        "postproduccion": {
            "steps": ["ensamblaje", "color_y_luts", "transiciones", "render_final"],
            "templates": [
                "Color_Grade_LUT_Template",
                "Transitions_Rules_Spec",
                "Render_Export_Preset",
            ],
        },
    },
    "video_largo": {
        "preproduccion": {
            "steps": [
                "desglose_de_escenas",
                "definicion_aspect_ratio",
                "timeline_spec",
            ],
            "templates": ["Storyboard_Spec", "Timeline_Template"],
        },
        "produccion": {
            "steps": [
                "render_b_roll_generativo",
                "generacion_subtitulos",
                "graficos_vectoriales",
            ],
            "templates": [
                "TTS_Voice_Preset",
                "ComfyUI_Workflow_Template",
                "Subtitles_Style_Template",
            ],
        },
        "postproduccion": {
            "steps": ["ensamblaje", "color_y_luts", "transiciones", "render_final"],
            "templates": [
                "Color_Grade_LUT_Template",
                "Transitions_Rules_Spec",
                "Render_Export_Preset",
            ],
        },
    },
    "carrusel": {
        "preproduccion": {
            "steps": ["layout_grid", "extraccion_textos"],
            "templates": ["Canvas_Layout_Spec"],
        },
        "produccion": {
            "steps": ["copy_por_slide", "iconografia", "composicion"],
            "templates": ["Inkscape_Layer_Template"],
        },
        "postproduccion": {
            "steps": ["composicion_raster", "filtros", "exportacion"],
            "templates": ["Image_Composite_FX_Spec"],
        },
    },
    "poster_qr": {
        "preproduccion": {
            "steps": ["layout_grid", "extraccion_textos"],
            "templates": ["Canvas_Layout_Spec"],
        },
        "produccion": {
            "steps": ["vector_imprimible", "qr_embebido"],
            "templates": ["Inkscape_Layer_Template"],
        },
        "postproduccion": {
            "steps": ["export_imprimible"],
            "templates": ["Image_Composite_FX_Spec"],
        },
    },
    "one_pager": {
        "preproduccion": {
            "steps": ["estructura"],
            "templates": [],
        },
        "produccion": {
            "steps": ["texto", "diagramas"],
            "templates": [],
        },
        "postproduccion": {
            "steps": ["maquetacion"],
            "templates": [],
        },
    },
    "white_paper": {
        "preproduccion": {
            "steps": ["estructura"],
            "templates": [],
        },
        "produccion": {
            "steps": ["texto", "diagramas"],
            "templates": [],
        },
        "postproduccion": {
            "steps": ["maquetacion"],
            "templates": [],
        },
    },
    "broadcast": {
        "preproduccion": {
            "steps": [
                "estructuracion_guion",
                "asignacion_roles_voces",
                "mapa_de_pistas",
            ],
            "templates": ["Script_SSML_Template"],
        },
        "produccion": {
            "steps": ["generacion_voz", "seleccion_bgm_sfx"],
            "templates": ["TTS_Voice_Preset", "Raw_Audio_Spec"],
        },
        "postproduccion": {
            "steps": ["procesamiento_voz", "duck_y_mix", "masterizacion"],
            "templates": [
                "Voice_FX_Chain_Preset",
                "Sidechain_Ducking_Preset",
                "Loudness_Master_Spec",
            ],
        },
    },
}

# Fases válidas del marco universal (§1) en orden.
PHASE_ORDER = ("preproduccion", "produccion", "postproduccion")


def _upsert_template(session, data: dict) -> ProductionTemplate:
    template = (
        session.execute(
            select(ProductionTemplate).where(ProductionTemplate.name == data["name"])
        )
        .scalars()
        .first()
    )
    if template is None:
        template = ProductionTemplate(**data)
        session.add(template)
        session.flush()
    else:
        for k, v in data.items():
            setattr(template, k, v)
    return template


def _upsert_format_phases(
    session, brand, artifact_type: str, phases: dict
) -> FormatSpec:
    spec = (
        session.execute(
            select(FormatSpec).where(
                FormatSpec.brand_objective == brand,
                FormatSpec.artifact_type == artifact_type,
            )
        )
        .scalars()
        .first()
    )
    if spec is None:
        return None
    spec.phases = phases
    return spec


def seed(session=None) -> dict:
    """Inserta/actualiza production_templates y format_specs.phases.

    Devuelve un resumen con los ids y nombres insertados. No requiere
    variables de entorno. Idempotente: upsert por clave única.
    """
    own_session = session is None
    if own_session:
        session = SessionLocal()
    try:
        templates = [_upsert_template(session, data) for data in PRODUCTION_TEMPLATES]

        specs: list[FormatSpec] = []
        for brand in (BrandObjective.ESCAPE_SOCIAL, BrandObjective.ERGALIA_COMERCIAL):
            for base in FORMAT_SPECS:
                artifact_type = base["artifact_type"]
                phases = FORMAT_PHASES.get(artifact_type)
                if phases is None:
                    continue
                spec = _upsert_format_phases(session, brand, artifact_type, phases)
                if spec is not None:
                    specs.append(spec)

        session.commit()
        return {
            "templates": [str(t.id) for t in templates],
            "template_names": [t.name for t in templates],
            "format_specs": [str(s.id) for s in specs],
            "format_count": len(specs),
        }
    finally:
        if own_session:
            session.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed de Fase 1.5 (production_templates + format_specs.phases)."
    )
    parser.parse_args()
    result = seed()
    print("Fase 1.5 (manifiesto y templates) lista:")
    print(f"  Templates ({len(result['template_names'])}):")
    for name in result["template_names"]:
        print(f"    - {name}")
    print(f"  Format specs con phases ({result['format_count']}):")
    print("    - 8 artifact_types x 2 marcas (ESCAPE_SOCIAL, ERGALIA_COMERCIAL)")


if __name__ == "__main__":
    main()
