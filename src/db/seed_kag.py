"""
Seed del sistema KAG (0013).

Inserta/actualiza:
  1. La configuración del segmentador en `kag_segmenter_settings` (singleton
     activo): modelo NLI, modelo de embeddings del segmentador y modelos
     spaCy por idioma.
  2. El modelo local Qwen 2.5 en `llm_models` (provider='local', servido por
     llama.cpp server en http://localhost:8080/v1).
  3. La api_key local en `api_keys` (provider 'local', key_name 'local-qwen',
     api_key '' — sin auth).

A diferencia de src/db/seed_llm.py, NO requiere variables de entorno: es
datos puros.

Uso:
    python -m src.db.seed_kag
"""

from __future__ import annotations

import argparse

from sqlalchemy import select, text

from src.db.models import ApiKey, LlmModel
from src.db.session import SessionLocal

# ---------------------------------------------------------------------
# Configuración del segmentador (tabla kag_segmenter_settings)
# ---------------------------------------------------------------------

SEGMENTER_SETTINGS = {
    "nli_model": "facebook/bart-large-mnli",
    "segmenter_embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "spacy_models": {
        "es": "es_core_news_md",
        "en": "en_core_web_md",
        "pt": "pt_core_news_md",
        "de": "de_core_news_md",
        "fr": "fr_core_news_md",
    },
}

# ---------------------------------------------------------------------
# Modelo local Qwen 2.5 (registro llm_models)
# ---------------------------------------------------------------------

LOCAL_QWEN_MODEL = {
    "model_name": "qwen2.5-3b-instruct-q4_k_m",
    "provider": "local",
    "model_size": "small",
    "context_window": 32768,
    "max_output_tokens": 60,
    "temperature_default": 0.1,
    "strengths": ["local", "gratis", "determinista", "resumen"],
    "weaknesses": ["capacidad limitada", "requiere servidor local"],
    "prompt_style": "ChatML estricto, system/user separados, XML delimiters, one-shot",
    "syntax_profile": {
        "api_style": "openai_chat",
        "base_url": "http://localhost:8080/v1",  # llama.cpp server (CRUD-editable)
        "system_role_name": "system",
        "instruction_formatting": {"style": "chatml", "root_tag": "instructions"},
        "structured_output": {"mode": "none"},
        "sampling": {
            "temperature": 0.1,
            "top_p": 0.9,
            "repetition_penalty": 1.05,
            "max_tokens": 60,
            "stop": ["\n", ""],
        },
        "tool_calling": {"mode": "none"},
        "prompt_caching": {"supports_prefix_caching": False},
        "reasoning_mode": {"supports_reasoning": False},
    },
}

# Api key local (sin auth — el servidor llama.cpp no requiere clave).
LOCAL_API_KEY = {
    "provider": "local",
    "key_name": "local-qwen",
    "api_key": "",
    "is_active": True,
}


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


def _upsert_segmenter_settings(session, data: dict) -> int:
    """Upsert del singleton activo de kag_segmenter_settings (SQL crudo)."""
    row = session.execute(
        text(
            "SELECT id FROM kag_segmenter_settings "
            "WHERE is_active = TRUE ORDER BY id DESC LIMIT 1"
        )
    ).first()
    if row is None:
        result = session.execute(
            text(
                "INSERT INTO kag_segmenter_settings "
                "(nli_model, segmenter_embedding_model, spacy_models) "
                "VALUES (:nli, :emb, :spacy) RETURNING id"
            ),
            {
                "nli": data["nli_model"],
                "emb": data["segmenter_embedding_model"],
                "spacy": data["spacy_models"],
            },
        )
        return result.scalar()
    session.execute(
        text(
            "UPDATE kag_segmenter_settings "
            "SET nli_model = :nli, segmenter_embedding_model = :emb, "
            "spacy_models = :spacy, updated_at = now() WHERE id = :id"
        ),
        {
            "nli": data["nli_model"],
            "emb": data["segmenter_embedding_model"],
            "spacy": data["spacy_models"],
            "id": row.id,
        },
    )
    return row.id


def seed(session=None) -> dict:
    """Inserta/actualiza la config del segmentador, el modelo local y la key.

    Devuelve un resumen con los ids. No requiere variables de entorno.
    """
    own_session = session is None
    if own_session:
        session = SessionLocal()
    try:
        settings_id = _upsert_segmenter_settings(session, SEGMENTER_SETTINGS)
        key = _upsert_api_key(session, LOCAL_API_KEY)
        model = _upsert_model(session, LOCAL_QWEN_MODEL)
        session.commit()
        return {
            "segmenter_settings_id": str(settings_id),
            "api_key_id": str(key.id),
            "model_id": str(model.id),
            "model_name": model.model_name,
        }
    finally:
        if own_session:
            session.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed del sistema KAG (segmentador + modelo local Qwen 2.5)."
    )
    parser.parse_args()
    result = seed()
    print("Sistema KAG listo:")
    print(f"  Segmenter settings: {result['segmenter_settings_id']}")
    print(
        f"  Api key local: {result['api_key_id']} (provider local, key_name local-qwen)"
    )
    print(f"  Modelo local: {result['model_id']} ({result['model_name']})")


if __name__ == "__main__":
    main()
