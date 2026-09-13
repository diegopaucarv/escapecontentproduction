"""
Configuración centralizada. Nada de credenciales embebidas en el código:
todo se lee de variables de entorno (ver .env.example).

Los embeddings (jina-embeddings-v5-text-nano) se ejecutan LOCALMENTE vía
transformers (ver src/embeddings.py): la clave y el modelo se leen de la
base (embedding_settings -> api_keys + llm_models), no de .env. Estos
defaults solo existen para entornos sin DB configurada.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Base de datos
    database_url: str = (
        "postgresql+psycopg2://ergalia_user:changeme@localhost:5432/escape_ergalia_os"
    )
    database_url_async: str = (
        "postgresql://ergalia_user:changeme@localhost:5432/escape_ergalia_os"
    )

    # Embeddings — ver docstring del módulo. Desde 0004 la fuente de
    # verdad es la base (embedding_settings -> api_keys + llm_models);
    # estos defaults solo existen para entornos sin DB configurada.
    embedding_provider: str = "jina"  # "jina" | "openai"
    embedding_model: str = "jinaai/jina-embeddings-v5-text-nano"
    embedding_dim: int = 768  # 768 para jina-embeddings-v5-text-nano
    voyage_api_key: str = ""
    openai_api_key: str = ""

    # Generación (redacción asistida, síntesis de Context Packs)
    anthropic_api_key: str = ""
    generation_model: str = "claude-sonnet-5"

    # Umbrales de enrutamiento (§4.4 del pipeline) — configurables sin tocar código
    novelty_score_threshold: int = 4  # >= este valor => "nueva_solucion"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Autenticación (RBAC — ver src/auth.py)
    jwt_secret: str = ""  # JWT_SECRET en .env — generar con secrets.token_hex(32)


@lru_cache
def get_settings() -> Settings:
    return Settings()
