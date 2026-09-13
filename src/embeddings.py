"""
Cliente de embeddings. Único punto del sistema que sabe que el
proveedor es Voyage AI — novelty_router.py y todo lo demás reciben un
`EmbedFn` (str -> list[float]) inyectado, así que cambiar de proveedor
en el futuro no toca lógica de negocio, solo este archivo.

Desde 0004, la configuración (clave + modelo + dimensión) NO vive en
.env: se lee de la base (embedding_settings -> api_keys + llm_models).
La clave de Voyage aún no la ha provisto el usuario; la infraestructura
existe y falla ruidosamente (no silenciosamente) si no hay settings.

Firma verificada contra voyageai==0.5.0 instalado en el entorno de
prueba (voyageai.Client.embed real, no supuesta de memoria):
embed(texts: list[str], model=..., input_type=..., output_dimension=...)
-> EmbeddingsObject con atributo `.embeddings: list[list[float]]`.
"""

from __future__ import annotations

import voyageai
from sqlalchemy import select
from tenacity import retry, stop_after_attempt, wait_exponential

from src.db.models import ApiKey, EmbeddingSetting, LlmModel
from src.db.session import SessionLocal


def _active_embedding_config(session=None) -> tuple[str, str, int]:
    """Lee la embedding_settings activa + su api_key + su modelo.

    Devuelve (api_key, model_name, dimension). Lanza RuntimeError con un
    mensaje claro si falta configuración — falla ruidosa, no silenciosa.
    """
    own_session = session is None
    if own_session:
        session = SessionLocal()
    try:
        setting = (
            session.execute(
                select(EmbeddingSetting).where(EmbeddingSetting.is_active.is_(True))
            )
            .scalars()
            .first()
        )
        if setting is None:
            raise RuntimeError(
                "No hay embedding_settings activa. Créala vía POST /embedding-settings "
                "(referenciando una api_key de Voyage y un llm_model con "
                "model_size='embedding')."
            )
        key = session.get(ApiKey, setting.api_key_id)
        if key is None or not key.is_active:
            raise RuntimeError(
                "La api_key referenciada por embedding_settings no existe o está inactiva."
            )
        model = session.get(LlmModel, setting.llm_model_id)
        if model is None or not model.is_active:
            raise RuntimeError(
                "El llm_model referenciado por embedding_settings no existe o está inactivo."
            )
        return key.api_key, model.model_name, int(setting.dimension)
    finally:
        if own_session:
            session.close()


def _client(api_key: str) -> voyageai.Client:
    return voyageai.Client(api_key=api_key)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def embed_text(text: str, input_type: str = "document") -> list[float]:
    """Implementa el `EmbedFn` que espera src/agents/novelty_router.py.
    `input_type='query'` cuando se embebe el texto de búsqueda de un
    brief nuevo; `'document'` cuando se indexa una pieza ya publicada
    en artifact_library — Voyage distingue ambos casos para mejor
    calidad de recuperación, no es un detalle cosmético."""
    api_key, model_name, dimension = _active_embedding_config()
    result = _client(api_key).embed(
        texts=[text],
        model=model_name,
        input_type=input_type,
        output_dimension=dimension,
    )
    return result.embeddings[0]
