"""
Seed de la infraestructura LLM (proveedor Together AI).

Inserta/actualiza:
  1. La api_key de Together en `api_keys` (la clave se lee de la variable
     de entorno TOGETHER_API_KEY — NUNCA se hardcodea en git).
  2. La session_settings activa en `session_settings` con los modelos
     pequeño/grande configurados.

Uso:
    TOGETHER_API_KEY=tgp_v1_... python -m src.db.seed_llm
    TOGETHER_API_KEY=tgp_v1_... python -m src.db.seed_llm --reset

--reset: desactiva la session_settings previa y crea una nueva (mantiene
el singleton de una sola fila activa).
"""

from __future__ import annotations

import argparse
import os
import uuid

from sqlalchemy import select, update

from src.db.models import ApiKey, SessionSettings
from src.db.session import SessionLocal

PROVIDER = "together"
KEY_NAME = "together-main"

# Modelos por defecto de la sesión (configurables luego vía CRUD /settings).
DEFAULT_SMALL_MODEL = "meta-models/Muse-Glimmer-30B"
DEFAULT_LARGE_MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"


def seed(reset: bool = False) -> dict:
    """Inserta/actualiza la api_key y la session_settings. Devuelve un
    resumen con los ids. Requiere TOGETHER_API_KEY en el entorno."""
    api_key_value = os.environ.get("TOGETHER_API_KEY", "").strip()
    if not api_key_value:
        raise SystemExit(
            "Falta TOGETHER_API_KEY en el entorno. "
            "Ej: TOGETHER_API_KEY=tgp_v1_... python -m src.db.seed_llm"
        )

    session = SessionLocal()
    try:
        # 1. Upsert de la api_key (por provider+key_name, único).
        key = (
            session.execute(
                select(ApiKey).where(
                    ApiKey.provider == PROVIDER, ApiKey.key_name == KEY_NAME
                )
            )
            .scalars()
            .first()
        )
        if key is None:
            key = ApiKey(provider=PROVIDER, key_name=KEY_NAME, api_key=api_key_value)
            session.add(key)
            session.flush()
        else:
            key.api_key = api_key_value
            key.is_active = True

        # 2. Session settings: si --reset, desactiva la activa previa.
        if reset:
            session.execute(
                update(SessionSettings)
                .where(SessionSettings.is_active.is_(True))
                .values(is_active=False)
            )

        active = (
            session.execute(
                select(SessionSettings).where(SessionSettings.is_active.is_(True))
            )
            .scalars()
            .first()
        )
        if active is None:
            active = SessionSettings(
                api_key_id=key.id,
                small_model=DEFAULT_SMALL_MODEL,
                large_model=DEFAULT_LARGE_MODEL,
            )
            session.add(active)
            session.flush()
        else:
            active.api_key_id = key.id
            active.small_model = DEFAULT_SMALL_MODEL
            active.large_model = DEFAULT_LARGE_MODEL

        session.commit()
        return {
            "api_key_id": str(key.id),
            "settings_id": str(active.id),
            "provider": PROVIDER,
            "small_model": active.small_model,
            "large_model": active.large_model,
        }
    finally:
        session.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed de infraestructura LLM (Together)."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Desactiva la settings previa y crea una nueva.",
    )
    args = parser.parse_args()
    result = seed(reset=args.reset)
    print("Infraestructura LLM lista:")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
