"""
Cliente de embeddings. Único punto del sistema que sabe que el
proveedor es Voyage AI — novelty_router.py y todo lo demás reciben un
`EmbedFn` (str -> list[float]) inyectado, así que cambiar de proveedor
en el futuro no toca lógica de negocio, solo este archivo.

Firma verificada contra voyageai==0.5.0 instalado en el entorno de
prueba (voyageai.Client.embed real, no supuesta de memoria):
embed(texts: list[str], model=..., input_type=..., output_dimension=...)
-> EmbeddingsObject con atributo `.embeddings: list[list[float]]`.
"""
from __future__ import annotations

from functools import lru_cache

import voyageai
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_settings


@lru_cache
def _client() -> voyageai.Client:
    settings = get_settings()
    if not settings.voyage_api_key:
        raise RuntimeError(
            "VOYAGE_API_KEY no configurada. Sin esto, novelty_router no puede "
            "hacer la búsqueda de duplicado (Paso 1) y todo caería en la "
            "rama de puntaje por defecto — falla ruidosa, no silenciosa."
        )
    return voyageai.Client(api_key=settings.voyage_api_key)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def embed_text(text: str, input_type: str = "document") -> list[float]:
    """Implementa el `EmbedFn` que espera src/agents/novelty_router.py.
    `input_type='query'` cuando se embebe el texto de búsqueda de un
    brief nuevo; `'document'` cuando se indexa una pieza ya publicada
    en artifact_library — Voyage distingue ambos casos para mejor
    calidad de recuperación, no es un detalle cosmético."""
    settings = get_settings()
    result = _client().embed(
        texts=[text],
        model=settings.embedding_model,
        input_type=input_type,
        output_dimension=settings.embedding_dim,
    )
    return result.embeddings[0]
