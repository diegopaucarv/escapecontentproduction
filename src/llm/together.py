"""
Cliente LLM unificado — proveedor Together AI (el que usaremos hasta
nuevo aviso).

Diseño deliberado: la clave y los modelos NO viven en config/env, sino en
la base de datos (api_keys + session_settings). Así el usuario puede hacer
CRUD de la clave y de los modelos pequeño/grande sin tocar código ni
re-desplegar. Este módulo solo lee la config activa y llama a Together
(API compatible con OpenAI: /v1/chat/completions).
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session
from tenacity import retry, stop_after_attempt, wait_exponential

from src.db.models import ApiKey, SessionSettings

TOGETHER_CHAT_URL = "https://api.together.xyz/v1/chat/completions"

# Reintentos por defecto si no hay session_settings.llm_retries disponible.
DEFAULT_LLM_RETRIES = 3


class LLMConfigError(RuntimeError):
    """No hay config LLM activa (falta api_key o session_settings)."""


@dataclass
class LLMConfig:
    api_key: str
    small_model: str
    large_model: str
    temperature_small: float
    temperature_large: float
    max_tokens_small: int
    max_tokens_large: int


def get_active_llm_config(session: Session) -> LLMConfig:
    """Lee la session_settings activa + su api_key. Lanza LLMConfigError
    si falta alguna de las dos."""
    settings = (
        session.execute(
            select(SessionSettings).where(SessionSettings.is_active.is_(True))
        )
        .scalars()
        .first()
    )
    if settings is None:
        raise LLMConfigError(
            "No hay session_settings activa. Créala vía POST /settings."
        )
    key = session.get(ApiKey, settings.api_key_id)
    if key is None or not key.is_active:
        raise LLMConfigError(
            "La api_key referenciada por session_settings no existe o está inactiva."
        )
    return LLMConfig(
        api_key=key.api_key,
        small_model=settings.small_model,
        large_model=settings.large_model,
        temperature_small=float(settings.temperature_small),
        temperature_large=float(settings.temperature_large),
        max_tokens_small=int(settings.max_tokens_small),
        max_tokens_large=int(settings.max_tokens_large),
    )


def _llm_retries(session: Session) -> int:
    """Lee session_settings.llm_retries (0004). Si la sesión no lo soporta
    (tests con sesiones falsas), devuelve el default 3."""
    try:
        settings = (
            session.execute(
                select(SessionSettings).where(SessionSettings.is_active.is_(True))
            )
            .scalars()
            .first()
        )
        if settings is not None and getattr(settings, "llm_retries", None):
            return int(settings.llm_retries)
    except Exception:
        pass
    return DEFAULT_LLM_RETRIES


def _post_with_retry(
    session, body: dict, timeout: float, cfg: LLMConfig
) -> httpx.Response:
    """POST a Together con reintentos (tenacity) según session_settings.

    Compartido por `complete` y `complete_vision` para no duplicar el
    patrón de retry.
    """
    retries = _llm_retries(session)

    @retry(
        stop=stop_after_attempt(retries),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def _post() -> httpx.Response:
        return httpx.post(
            TOGETHER_CHAT_URL,
            headers={
                "Authorization": f"Bearer {cfg.api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=timeout,
        )

    return _post()


def complete(
    session: Session,
    prompt: str,
    model_size: str = "small",
    system: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: float = 900.0,
    response_format: dict | None = None,
) -> str:
    """Llama al modelo pequeño, grande o de visión vía Together y devuelve el texto.

    model_size: "small" | "large" | "vision". Los valores por defecto de
    temperature y max_tokens salen de session_settings; se pueden
    sobreescribir por llamada. `response_format` (0004) fuerza salida
    estructurada (p. ej. {"type": "json_object"}) cuando el modelo lo
    soporta.
    """
    if model_size not in ("small", "large", "vision"):
        raise ValueError(
            f"model_size debe ser 'small', 'large' o 'vision', no {model_size!r}"
        )

    cfg = get_active_llm_config(session)
    model = cfg.small_model if model_size == "small" else cfg.large_model
    temp = (
        temperature
        if temperature is not None
        else (cfg.temperature_small if model_size == "small" else cfg.temperature_large)
    )
    tokens = (
        max_tokens
        if max_tokens is not None
        else (cfg.max_tokens_small if model_size == "small" else cfg.max_tokens_large)
    )

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    body: dict = {
        "model": model,
        "messages": messages,
        "temperature": temp,
        "max_tokens": tokens,
    }
    if response_format is not None:
        body["response_format"] = response_format

    resp = _post_with_retry(session, body, timeout, cfg)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def complete_vision(
    session: Session,
    prompt: str,
    image_url: str,
    model_size: str = "vision",
    system: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: float = 900.0,
    response_format: dict | None = None,
) -> str:
    """Llama a un modelo de visión con una imagen (URL) vía Together.

    Igual que `complete`, pero el mensaje de usuario es una lista de
    contenido multimodal estilo OpenAI: texto + image_url. El modelo se
    resuelve con cfg.large_model (el usuario puede setear el modelo de
    visión como large_model en session_settings).
    """
    if model_size not in ("small", "large", "vision"):
        raise ValueError(
            f"model_size debe ser 'small', 'large' o 'vision', no {model_size!r}"
        )

    cfg = get_active_llm_config(session)
    model = cfg.small_model if model_size == "small" else cfg.large_model
    temp = (
        temperature
        if temperature is not None
        else (cfg.temperature_small if model_size == "small" else cfg.temperature_large)
    )
    tokens = (
        max_tokens
        if max_tokens is not None
        else (cfg.max_tokens_small if model_size == "small" else cfg.max_tokens_large)
    )

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    )

    body: dict = {
        "model": model,
        "messages": messages,
        "temperature": temp,
        "max_tokens": tokens,
    }
    if response_format is not None:
        body["response_format"] = response_format

    resp = _post_with_retry(session, body, timeout, cfg)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]
