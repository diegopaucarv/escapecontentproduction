"""
Seed de la infraestructura de IA modular (0004).

Inserta/actualiza:
  1. El registro de modelos en `llm_models` (pequeño, grande, embeddings).
  2. Las specs agnósticas de prompts en `prompt_templates` (los 6 tasks
     iniciales: alignment_reinforcement, critic_checklist, novelty_scoring,
     producer_draft, project_generator, project_refine).

A diferencia de src/db/seed_llm.py, NO requiere ninguna variable de entorno:
es datos puros. Tampoco compila artefactos — eso lo hace el compilador
(src/llm/compile_prompts.py) una vez que los modelos y specs existen.

Uso:
    python -m src.db.seed_ai
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from sqlalchemy import select

from src.db.models import LlmModel, PromptTemplate

# ---------------------------------------------------------------------
# Datos de los modelos (registro llm_models)
# ---------------------------------------------------------------------

# Perfil de sintaxis compartido por los modelos de chat de Together
# (API compatible con OpenAI). El compilador lo usa como datos para
# adaptar el prompt a cada modelo — ver src/llm/compiler.py.
TOGETHER_CHAT_PROFILE = {
    "api_style": "openai_chat",
    "system_role_name": "system",
    "instruction_formatting": {
        "style": "markdown",
        "root_tag": "instructions",
        "enforce_scratchpad_first": False,
        "section_delimiters": {
            "rules": "rules",
            "context": "context",
            "examples": "examples",
        },
    },
    "structured_output": {
        "mode": "json_object",
        "schema_delivery": "system_prompt_append",
    },
    "tool_calling": {"mode": "none", "supports_strict_tools": False},
    "prompt_caching": {
        "supports_prefix_caching": True,
        "cache_marker_type": "auto_prefix",
        "min_cache_tokens": 1024,
    },
    "reasoning_mode": {
        "supports_reasoning": False,
        "thinking_parameter": None,
        "strip_think_tags_in_output": False,
    },
}

MODELS = [
    {
        "model_name": "meta-models/Muse-Glimmer-30B",
        "provider": "together",
        "model_size": "small",
        "context_window": 32768,
        "max_output_tokens": 8192,
        "temperature_default": 0.7,
        "strengths": [
            "rapido",
            "economico",
            "bueno en clasificacion y extraccion estructurada",
        ],
        "weaknesses": ["menos creativo en textos largos", "contexto limitado"],
        "prompt_style": "instrucciones directas, salida JSON estricta, pocos ejemplos",
        "syntax_profile": TOGETHER_CHAT_PROFILE,
    },
    {
        "model_name": "deepseek-ai/DeepSeek-V4-Flash-0731",
        "provider": "together",
        "model_size": "large",
        "context_window": 131072,
        "max_output_tokens": 8192,
        "temperature_default": 0.7,
        "strengths": [
            "razonamiento profundo",
            "sigue instrucciones complejas",
            "generacion creativa",
        ],
        "weaknesses": ["mas lento", "mas caro por token"],
        "prompt_style": "few-shot con ejemplos, descomponer tareas en pasos",
        "syntax_profile": TOGETHER_CHAT_PROFILE,
    },
    {
        "model_name": "jinaai/jina-embeddings-v5-text-nano",
        "provider": "huggingface",
        "model_size": "embedding",
        "context_window": 8192,
        "max_output_tokens": 0,
        "temperature_default": 0.0,
        "strengths": [
            "multilingue",
            "nano: rapido y economico",
            "adapters por tarea (retrieval, text-matching, clustering)",
        ],
        "weaknesses": ["dimension 768 (menor que modelos grandes)"],
        "prompt_style": "",
        "syntax_profile": {},
    },
]

# ---------------------------------------------------------------------
# Specs agnósticas de prompts (tabla prompt_templates)
# ---------------------------------------------------------------------

TEMPLATES = [
    {
        "task_key": "alignment_reinforcement",
        "version": "1.0",
        "intent": (
            "Eres el orquestador por defecto de la decision de alineamiento "
            "estrategico. Refuerzas el semaforo basado en reglas: tu funcion es "
            "detectar riesgos que el checklist automatico no cubre. La degradacion "
            "a solo-reglas (sin LLM) requiere aceptacion del usuario."
        ),
        "rules": [
            (
                "Si detectas un riesgo editorial, reputacional o de exposicion "
                "NO cubierto por el checklist automatico, escala a needs_human_review"
            ),
            "Nunca bajar la severidad del veredicto de reglas",
            (
                "No repetir verificaciones que ya hace el checklist automatico "
                "(fuente, CTA, segmentos S5/S6)"
            ),
            "La degradacion a solo-reglas (sin LLM) requiere aceptacion del usuario",
        ],
        "input_schema": {
            "brief": "object",
            "rule_verdict": "string",
            "checklist_results": "array",
        },
        "output_schema": {
            "verdict": "auto_pass|needs_human_review|fail",
            "reasoning": "string",
            "risks_detected": "array",
        },
        "few_shot": [],
    },
    {
        "task_key": "critic_checklist",
        "version": "1.0",
        "intent": (
            "Eres el critico editorial. Verificas el borrador contra el checklist "
            "de publicacion recibido como dato. Cada item del checklist debe tener "
            "su interpretacion en rules; si un item no tiene interpretacion, "
            "marcalo como no_evaluado, nunca ok. Si el critico no esta disponible, "
            "el fallback deterministico (no_evaluado) requiere aceptacion humana."
        ),
        "rules": [
            "fuente_verificable: el claim principal cita una fuente verificable",
            "cta_unico: exactamente un CTA, una sola accion",
            (
                "anonimizacion_no_aplica: la pieza no maneja datos personales; "
                "no requiere anonimizacion"
            ),
            (
                "anonimizacion_verificada: si la pieza maneja datos personales, "
                "estan anonimizados o seudonimizados"
            ),
            "revision_legal: sin afirmaciones legales riesgosas sin respaldo",
            "gancho_15s_ok: el primer parrafo comunica la promesa en 15 segundos",
            "longitud: dentro del rango del formato",
            (
                "Si un item del checklist no tiene regla de interpretacion, "
                "status = no_evaluado"
            ),
        ],
        "input_schema": {
            "draft": "string",
            "checklist": "array",
            "brand_objective": "string",
        },
        "output_schema": {
            "checklist_results": "array",
            "verdict": "auto_pass|needs_human_review|fail",
        },
        "few_shot": [],
    },
    {
        "task_key": "novelty_scoring",
        "version": "1.0",
        "intent": (
            "Eres el refinador de novedad. Por defecto se te consulta en la zona "
            "gris del enrutamiento. Tu unica tarea es juzgar si el angulo/tema de "
            "esta pieza esta cubierto por piezas anteriores de la marca. Si no estas "
            "disponible, el default deterministico (angulo_nuevo = true) requiere "
            "aceptacion del usuario."
        ),
        "rules": [
            "Compara el angulo de la pieza nueva contra los angulos de las piezas anteriores",
            "Si el angulo ya esta cubierto, angulo_nuevo = false",
            "Si el angulo aporta una perspectiva no vista, angulo_nuevo = true",
            (
                "No puntuar bucket/formato/canal: eso ya lo calcula el codigo determinista"
            ),
        ],
        "input_schema": {"brief": "object", "prior_artifacts": "array"},
        "output_schema": {"angulo_nuevo": "boolean", "reasoning": "string"},
        "few_shot": [],
    },
    {
        "task_key": "producer_draft",
        "version": "1.0",
        "intent": (
            "Eres el productor editorial, el paso de generacion guiado por LLM "
            "por defecto. Generas o refinas el borrador de una pieza a partir del "
            "insight_core y el Context Pack, respetando el tono de la marca e "
            "incorporando el critic_feedback de la iteracion anterior si lo recibes."
        ),
        "rules": [
            "Abrir con el gancho de 15 segundos",
            "Un solo CTA al final",
            "Respetar el tono de la marca (brand_objective)",
            "No inventar datos: usar solo lo del Context Pack",
            (
                "Si recibes critic_feedback de la iteración anterior, "
                "incorpóralo al nuevo borrador"
            ),
        ],
        "input_schema": {
            "insight_core": "string",
            "context_pack": "object",
            "brand_objective": "string",
            "artifact_type": "string",
            "channel": "string",
            "critic_feedback": "string",
        },
        "output_schema": {
            "draft": "string",
            "hook_15s": "string",
            "cta": "string",
        },
        "few_shot": [],
    },
    {
        "task_key": "project_generator",
        "version": "1.0",
        "intent": (
            "Eres el compilador de contenido de Pipeline OS. Escribes el snapshot "
            "JSON editable de un proyecto a partir del topic, el template de "
            "producción y la spec de formato. Nunca editas archivos."
        ),
        "rules": [
            "Escribir SOLO datos (el snapshot JSON)",
            "No inventar reglas nuevas",
            "Incluir los campos que la estructura del formato define",
            "Salida JSON estricta",
        ],
        "input_schema": {
            "topic": "string",
            "template": "object",
            "format_spec": "object",
        },
        "output_schema": {"snapshot": "object"},
        "few_shot": [],
    },
    {
        "task_key": "project_refine",
        "version": "1.0",
        "intent": (
            "Eres el editor de copywriting/prompts/guiones de Pipeline OS. "
            "Corriges SOLO los warnings del snapshot JSON de un proyecto."
        ),
        "rules": [
            "Corregir solo los warnings",
            "No cambiar contenido que no esté en los warnings",
            "Salida JSON estricta",
        ],
        "input_schema": {
            "topic": "string",
            "snapshot": "object",
            "warnings": "array",
        },
        "output_schema": {"snapshot": "object"},
        "few_shot": [],
    },
]


def _fix_db_host() -> None:
    """Reemplaza '@db:' por '@localhost:' en DATABASE_URL/DATABASE_URL_ASYNC.

    El host 'db' es la red Docker y no resuelve desde el host; las credenciales
    son las mismas. Debe llamarse ANTES de importar src.db.session.
    """
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    for var in ("DATABASE_URL", "DATABASE_URL_ASYNC"):
        val = os.environ.get(var, "")
        if not val and env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith(f"{var}="):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
        if "@db:" in val:
            os.environ[var] = val.replace("@db:", "@localhost:")


def _upsert_model(session, data: dict) -> LlmModel:
    model = (
        session.execute(
            select(LlmModel).where(LlmModel.model_name == data["model_name"])
        )
        .scalars()
        .first()
    )
    if model is None:
        model = LlmModel(**data)
        session.add(model)
        session.flush()
    else:
        for k, v in data.items():
            setattr(model, k, v)
    return model


def _upsert_template(session, data: dict) -> PromptTemplate:
    template = (
        session.execute(
            select(PromptTemplate).where(PromptTemplate.task_key == data["task_key"])
        )
        .scalars()
        .first()
    )
    if template is None:
        template = PromptTemplate(**data)
        session.add(template)
        session.flush()
    else:
        for k, v in data.items():
            setattr(template, k, v)
    return template


def seed(session=None) -> dict:
    """Inserta/actualiza modelos y specs. Devuelve un resumen con los ids.

    No requiere variables de entorno. No compila artefactos (eso lo hace
    src/llm/compile_prompts.py).
    """
    own_session = session is None
    if own_session:
        _fix_db_host()
        from src.db.session import SessionLocal

        session = SessionLocal()
    try:
        models = [_upsert_model(session, data) for data in MODELS]
        templates = [_upsert_template(session, data) for data in TEMPLATES]
        session.commit()
        return {
            "models": [str(m.id) for m in models],
            "templates": [str(t.id) for t in templates],
            "model_names": [m.model_name for m in models],
            "task_keys": [t.task_key for t in templates],
        }
    finally:
        if own_session:
            session.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed de infraestructura de IA modular (modelos + specs)."
    )
    parser.parse_args()
    _fix_db_host()
    result = seed()
    print("Infraestructura de IA modular lista:")
    print(f"  Modelos ({len(result['model_names'])}):")
    for name in result["model_names"]:
        print(f"    - {name}")
    print(f"  Specs de prompts ({len(result['task_keys'])}):")
    for key in result["task_keys"]:
        print(f"    - {key}")


if __name__ == "__main__":
    main()
