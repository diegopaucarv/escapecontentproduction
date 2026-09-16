"""
Seed de la infraestructura de visión (0007, diseño §20.6).

Inserta/actualiza:
  1. La api_key de visión en `api_keys` (provider "together", key_name
     "together_vision") — placeholder vacío, CRUD-editable vía
     PATCH /api-keys/{id}. NUNCA en .env.
  2. Los modelos de visión en `llm_models` (model_size="vision"):
     Llama-3.2-11B-Vision-Instruct-Turbo y Qwen2-VL-72B-Instruct.
  3. Las specs agnósticas de prompts de visión en `prompt_templates`
     (vision_describe_asset, vision_prompt_visual).

A diferencia de src/db/seed_llm.py, NO requiere ninguna variable de
entorno: es datos puros. Tampoco compila artefactos — eso lo hace el
compilador (src/llm/compile_prompts.py) una vez que los modelos y specs
existen.

Uso:
    python -m src.db.seed_vision
"""

from __future__ import annotations

import argparse

from sqlalchemy import select

from src.db.models import ApiKey, LlmModel, PromptTemplate

# ---------------------------------------------------------------------
# Api key de visión (tabla api_keys)
# ---------------------------------------------------------------------

# Placeholder: el usuario llena la clave real vía PATCH /api-keys/{id}.
# NUNCA se hardcodea una clave real ni se lee de .env aquí.
VISION_API_KEY = {
    "provider": "together",
    "key_name": "together_vision",
    "api_key": "",
    "is_active": True,
}

# ---------------------------------------------------------------------
# Modelos de visión (registro llm_models)
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

VISION_MODELS = [
    {
        "model_name": "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
        "provider": "together",
        "model_size": "vision",
        "is_vision": True,
        "context_window": 8192,
        "max_output_tokens": 2048,
        "temperature_default": 0.4,
        "strengths": ["vision", "rapido", "economico"],
        "weaknesses": ["contexto limitado"],
        "prompt_style": "instrucciones directas, salida JSON estricta",
        "syntax_profile": TOGETHER_CHAT_PROFILE,
    },
    {
        "model_name": "Qwen/Qwen2-VL-72B-Instruct",
        "provider": "together",
        "model_size": "vision",
        "is_vision": True,
        "context_window": 32768,
        "max_output_tokens": 4096,
        "temperature_default": 0.4,
        "strengths": ["vision", "razonamiento visual profundo", "multilingue"],
        "weaknesses": ["mas lento", "mas caro"],
        "prompt_style": "few-shot con ejemplos",
        "syntax_profile": TOGETHER_CHAT_PROFILE,
    },
]

# ---------------------------------------------------------------------
# Specs agnósticas de prompts de visión (tabla prompt_templates)
# ---------------------------------------------------------------------

VISION_TEMPLATES = [
    {
        "task_key": "vision_describe_asset",
        "version": "1.0",
        "intent": (
            "Eres el analista visual. Describes un asset (imagen/video frame) "
            "con precision para que otros modelos lo usen como contexto."
        ),
        "rules": [
            "Describir composicion, colores, texto visible y estilo",
            "No inventar elementos que no estan en la imagen",
            "Salida JSON estricta",
        ],
        "input_schema": {
            "image_url": "string",
            "pregunta": "string",
        },
        "output_schema": {
            "descripcion": "string",
            "elementos_clave": "array",
            "texto_visible": "string",
        },
        "few_shot": [],
    },
    {
        "task_key": "vision_prompt_visual",
        "version": "1.0",
        "intent": (
            "Eres el director de arte. Generas prompts visuales optimizados "
            "para generadores de imagen/video (ComfyUI, Veo) a partir de una "
            "descripcion y el estilo de marca."
        ),
        "rules": [
            "Incluir estilo de marca si se provee",
            "Especificar composicion, iluminacion y mood",
            "Un solo prompt por salida",
            "Salida JSON estricta",
        ],
        "input_schema": {
            "descripcion": "string",
            "brand_style": "string",
        },
        "output_schema": {
            "prompt_visual": "string",
            "negative_prompt": "string",
            "parametros": "object",
        },
        "few_shot": [],
    },
]


def _upsert_api_key(session, data: dict) -> ApiKey:
    key = (
        session.execute(
            select(ApiKey).where(
                ApiKey.provider == data["provider"],
                ApiKey.key_name == data["key_name"],
            )
        )
        .scalars()
        .first()
    )
    if key is None:
        key = ApiKey(**data)
        session.add(key)
        session.flush()
    else:
        for k, v in data.items():
            setattr(key, k, v)
    return key


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
    """Inserta/actualiza la key de visión, los modelos y las specs.

    Devuelve un resumen con los ids. No requiere variables de entorno.
    No compila artefactos (eso lo hace src/llm/compile_prompts.py).
    """
    own_session = session is None
    if own_session:
        from src.db.session import SessionLocal

        session = SessionLocal()
    try:
        key = _upsert_api_key(session, VISION_API_KEY)
        models = [_upsert_model(session, data) for data in VISION_MODELS]
        templates = [_upsert_template(session, data) for data in VISION_TEMPLATES]
        session.commit()
        return {
            "api_key_id": str(key.id),
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
        description="Seed de infraestructura de visión (key + modelos + specs)."
    )
    parser.parse_args()
    result = seed()
    print("Infraestructura de visión lista:")
    print(
        f"  Api key: {result['api_key_id']} (provider together, key_name together_vision)"
    )
    print(f"  Modelos de visión ({len(result['model_names'])}):")
    for name in result["model_names"]:
        print(f"    - {name}")
    print(f"  Specs de prompts de visión ({len(result['task_keys'])}):")
    for key in result["task_keys"]:
        print(f"    - {key}")


if __name__ == "__main__":
    main()
