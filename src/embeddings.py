"""
Cliente de embeddings. Único punto del sistema que sabe que el
proveedor es Jina (jina-embeddings-v5-text-nano) — novelty_router.py
y todo lo demás reciben un `EmbedFn` (str -> list[float]) inyectado, así
que cambiar de proveedor en el futuro no toca lógica de negocio, solo
este archivo.

Desde 0004, la configuración (clave + modelo + dimensión) NO vive en
.env: se lee de la base (embedding_settings -> api_keys + llm_models).
La clave se guarda en `api_keys` (provider 'huggingface') y la
embedding_settings activa la referencia. Si no hay settings, falla
ruidosamente (no silenciosamente).

Inferencia LOCAL: el modelo se descarga desde HuggingFace Hub (la
api_key de huggingface se usa como token de autenticación para la
descarga) y se ejecuta con transformers.AutoModel + trust_remote_code
(la clase custom jina_embeddings_v5 expone `.encode()`). No hay
llamadas HTTP a ningún proveedor en el hot-path.

El `input_type` de Voyage se mapea al `prompt_name` de Jina:
  - 'query'    -> prompt_name='query'    (embebe el texto de búsqueda)
  - 'document' -> prompt_name='document' (indexa una pieza publicada)
Ambos con task='retrieval'.
"""

from __future__ import annotations

import threading

from sqlalchemy import select

from src.db.models import ApiKey, EmbeddingSetting, LlmModel
from src.db.session import SessionLocal

# Singleton del modelo en memoria (carga perezosa, una sola vez por proceso).
_model = None
_model_lock = threading.Lock()


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
                "(referenciando una api_key y un llm_model con model_size='embedding')."
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


def _prompt_name(input_type: str) -> str:
    """Mapea el input_type de Voyage al prompt_name de Jina."""
    if input_type == "query":
        return "query"
    return "document"


def _get_model(api_key: str, model_name: str):
    """Carga (una sola vez) el modelo con transformers.AutoModel.

    El modelo requiere trust_remote_code=True (clase custom
    jina_embeddings_v5 que expone `.encode()`). La api_key de
    HuggingFace se pasa como token para autenticar la descarga desde el
    Hub. En CPU se usa float32; en GPU, bfloat16 (recomendado por la
    model card). Falla ruidosamente si la descarga/carga falla.
    """
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        import torch
        from transformers import AutoModel

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # bfloat16 solo si la GPU lo soporta; si no, float32 (funciona en todas).
        dtype = (
            torch.bfloat16
            if device.type == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float32
        )
        _model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=api_key or None,
            dtype=dtype,
        ).to(device)
    return _model


def embed_text(text: str, input_type: str = "document") -> list[float]:
    """Implementa el `EmbedFn` que espera src/agents/novelty_router.py.

    `input_type='query'` cuando se embebe el texto de búsqueda de un
    brief nuevo; `'document'` cuando se indexa una pieza ya publicada
    en artifact_library — Jina distingue ambos casos con `prompt_name`
    (query / document) para mejor calidad de recuperación, no es un
    detalle cosmético.
    """
    api_key, model_name, _dimension = _active_embedding_config()
    model = _get_model(api_key, model_name)
    emb = model.encode(
        texts=[text],
        task="retrieval",
        prompt_name=_prompt_name(input_type),
    )
    vec = emb[0]
    if hasattr(vec, "detach"):  # torch.Tensor -> numpy
        # .float() convierte bfloat16 -> float32: pgvector NO soporta bfloat16
        # ("Got unsupported ScalarType BFloat16") y el modelo corre en bfloat16
        # en GPU. Los valores no cambian, solo el dtype.
        vec = vec.detach().cpu().float().numpy()
    return vec.tolist() if hasattr(vec, "tolist") else list(vec)


def embed_texts(texts: list[str], input_type: str = "document") -> list[list[float]]:
    """Versión batch de embed_text: una sola llamada a model.encode para N textos.

    Útil para canonicalizar muchos candidatos (p. ej. entidades del KAG) sin
    pagar el overhead de N llamadas individuales. Mismo modelo/config que
    embed_text; devuelve una lista de vectores en el mismo orden de entrada.
    """
    if not texts:
        return []
    api_key, model_name, _dimension = _active_embedding_config()
    model = _get_model(api_key, model_name)
    embs = model.encode(
        texts=texts,
        task="retrieval",
        prompt_name=_prompt_name(input_type),
    )
    out = []
    for vec in embs:
        if hasattr(vec, "detach"):  # torch.Tensor -> numpy (bfloat16 -> float32)
            vec = vec.detach().cpu().float().numpy()
        out.append(vec.tolist() if hasattr(vec, "tolist") else list(vec))
    return out
