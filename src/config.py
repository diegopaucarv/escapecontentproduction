"""
Configuración centralizada. Nada de credenciales embebidas en el código:
todo se lee de variables de entorno (ver .env.example).

Decisión explícita que la propuesta original dejaba abierta: proveedor
de embeddings. Anthropic no ofrece un modelo de embeddings propio y
recomienda Voyage AI como partner (ver docs.claude.com); ese es el
default aquí. Si el equipo prefiere OpenAI, basta con cambiar
EMBEDDING_PROVIDER y EMBEDDING_DIM — el resto del código no asume
ningún proveedor específico.
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

    # Embeddings — ver docstring del módulo
    embedding_provider: str = "voyage"  # "voyage" | "openai"
    embedding_model: str = "voyage-3-large"
    embedding_dim: int = 1024  # 1024 para voyage-3-large; 1536 si se cambia a OpenAI text-embedding-3-small
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
