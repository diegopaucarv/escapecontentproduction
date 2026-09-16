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

import logging
from dataclasses import dataclass

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session
from tenacity import retry, stop_after_attempt, wait_exponential

from src.db.models import ApiKey, LlmModel, SessionSettings
from src.llm.sanitize import LLMJSONError, sanitize_llm_json

logger = logging.getLogger(__name__)

TOGETHER_CHAT_URL = "https://api.together.xyz/v1/chat/completions"

# Reintentos por defecto si no hay session_settings.llm_retries disponible.
DEFAULT_LLM_RETRIES = 3

# Catch-and-continue: outputs truncados por finish_reason='length'.
# El modelo puede exceder max_tokens con JSON grandes (p. ej. proposiciones
# con text_span verbatim); en vez de descartar el parcial, se re-POST con el
# fragmento en el historial y se pide continuar exactamente desde donde se
# cortó (memoria aditiva).
MAX_CONTINUATION_ROUNDS = 3  # 1 original + 3 continuaciones ≈ 4×max_tokens
MIN_OVERLAP_FOR_DEDUP = 2  # chars mínimos para considerar solapamiento
CONTINUATION_PROMPT = (
    "Tu respuesta anterior fue cortada por el límite de longitud. "
    "Continúa EXACTAMENTE desde donde te quedaste, empezando por la "
    "siguiente palabra. No repitas nada de lo ya escrito."
)


class LLMConfigError(RuntimeError):
    """No hay config LLM activa (falta api_key o session_settings)."""


class LLMContentError(RuntimeError):
    """El modelo devolvió una respuesta sin content (razonamiento cortado)."""


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


def get_vision_model(session: Session) -> str | None:
    """Devuelve el model_name del modelo de visión activo (is_vision=True).

    Consulta `llm_models` filtrando por is_vision=True e is_active=True,
    ordenado por model_size (prefiere 'large' sobre 'small' si hay varios).
    Devuelve None si no hay ningún modelo de visión registrado o si la
    sesión no soporta la consulta (tests con sesiones falsas) — el caller
    degrada a cfg.large_model.
    """
    try:
        row = (
            session.execute(
                select(LlmModel)
                .where(LlmModel.is_vision.is_(True), LlmModel.is_active.is_(True))
                .order_by(LlmModel.model_size)
            )
            .scalars()
            .first()
        )
        return row.model_name if row is not None else None
    except Exception:  # noqa: BLE001 — degradación natural
        return None


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


def _model_reasoning_mode(session, model_name: str) -> dict:
    """Lee `reasoning_mode` del syntax_profile del modelo (llm_models).

    Defensivo: si la sesión no soporta la consulta (tests con sesiones
    falsas) o el modelo no tiene perfil, devuelve {} — el caller degrada a
    sin razonamiento.
    """
    try:
        row = (
            session.execute(select(LlmModel).where(LlmModel.model_name == model_name))
            .scalars()
            .first()
        )
        if row is None:
            return {}
        profile = getattr(row, "syntax_profile", None) or {}
        return (profile.get("reasoning_mode") or {}) or {}
    except Exception:  # noqa: BLE001 — degradación natural
        return {}


def _resolve_thinking(
    reasoning: dict, thinking: dict | bool | None, max_tokens: int | None
) -> tuple[str | None, dict | None]:
    """Resuelve el parámetro de razonamiento y su config para el body.

    - `param` es el nombre del parámetro en el body (p. ej. "thinking").
    - `thinking=None` → usa el default del modelo (reasoning.thinking_config).
    - `thinking=False` o `{"type": "disabled"}` → razonamiento apagado.
    - Guardrail: si `max_tokens` es bajo (menor que el piso del modelo o
      que budget + reserva de content), se desactiva el razonamiento — el
      modelo no alcanzaría a producir content y el reintento duplicaría
      costo/latencia.
    """
    param = reasoning.get("thinking_parameter")
    if not param:
        return None, None
    if thinking is None:
        thinking = reasoning.get("thinking_config")
    if thinking is False:
        # thinking=False → razonamiento APAGADO explícitamente (enviar
        # {"type": "disabled"}). Antes se devolvía (None, None) = "no
        # enviar el parámetro", lo que dejaba el default del modelo
        # (razonamiento ACTIVADO) y los tokens de thinking se comían el
        # presupuesto de max_tokens → JSON truncado (finish_reason='length').
        return param, {"type": "disabled"}
    if not thinking:
        return None, None
    if isinstance(thinking, dict) and thinking.get("type") == "disabled":
        return param, thinking
    if max_tokens is not None:
        min_tokens = int(reasoning.get("min_tokens_for_reasoning") or 1024)
        reserve = int(reasoning.get("min_content_reserve") or 512)
        budget = 0
        if isinstance(thinking, dict):
            budget = int(thinking.get("budget_tokens") or 0)
        if max_tokens < min_tokens or (budget and max_tokens < budget + reserve):
            return param, {"type": "disabled"}
    return param, thinking


def _can_disable_thinking(body: dict) -> bool:
    """¿El body lleva un parámetro de razonamiento que podemos apagar?

    False si el modelo no soporta razonamiento (sin parámetro) o si ya venía
    desactivado (reintentar no ayudaría).
    """
    current = body.get("thinking")
    if current is None:
        return False
    if isinstance(current, dict) and current.get("type") == "disabled":
        return False
    return True


def _is_parseable_json(text: str) -> bool:
    """¿El texto parsea como JSON (con el sanitizer, sin lanzar)?"""
    try:
        sanitize_llm_json(text)
        return True
    except LLMJSONError:
        return False


def _merge_fragments(acc: str, frag: str) -> str:
    """Une acc + frag evitando repetir el solapamiento.

    El modelo suele repetir las últimas palabras al continuar (o reinicia
    el JSON desde el principio). Buscamos el sufijo MÁS LARGO de `acc` que
    sea prefijo de `frag` y lo solapamos. El resultado solo se acepta si
    parsea (ver _continue_truncated_content), así un solapamiento falso
    nunca corrompe el JSON final.
    """
    if not acc:
        return frag
    max_n = min(len(acc), len(frag))
    for n in range(max_n, MIN_OVERLAP_FOR_DEDUP - 1, -1):
        if acc[-n:] == frag[:n]:
            return acc + frag[n:]
    return acc + frag


def _continue_truncated_content(
    session, body: dict, first_fragment: str, timeout: float, cfg, expects_json: bool
) -> str | None:
    """Catch-and-continue: re-POST con el fragmento parcial en el historial.

    Devuelve el JSON acumulado si se completó; None si se agotaron las
    iteraciones sin resultado parseable (el caller decide: escape hatch
    thinking-disabled o LLMContentError).
    """
    joined = first_fragment
    cont_body = dict(body)
    cont_body["messages"] = list(body.get("messages") or []) + [
        {"role": "assistant", "content": first_fragment},
        {"role": "user", "content": CONTINUATION_PROMPT},
    ]
    if _can_disable_thinking(cont_body):
        cont_body["thinking"] = {"type": "disabled"}

    for _ in range(MAX_CONTINUATION_ROUNDS):
        resp = _post_with_retry(session, cont_body, timeout, cfg)
        resp.raise_for_status()
        choice = resp.json().get("choices", [{}])[0]
        msg = choice.get("message", {})
        frag = (msg.get("content") or "").strip()
        finish = choice.get("finish_reason")

        if not frag:
            logger.warning("Continuación sin content (finish_reason=%r).", finish)
            break  # nada que acumular → escape hatch

        joined = _merge_fragments(joined, frag)
        if finish != "length":
            # El modelo terminó naturalmente: el acumulado debe parsear.
            if not expects_json or _is_parseable_json(joined):
                return joined
            break  # terminó pero no parsea → escape hatch

        # finish_reason == 'length': memoria aditiva — el historial crece
        # con el fragmento crudo (no el mergeado) para que el modelo vea
        # exactamente lo que escribió.
        cont_body["messages"] = cont_body["messages"] + [
            {"role": "assistant", "content": frag},
            {"role": "user", "content": CONTINUATION_PROMPT},
        ]

    # Agotadas las iteraciones: aceptar el acumulado solo si parsea.
    if expects_json and _is_parseable_json(joined):
        return joined
    return None


def _extract_content(
    session, data: dict, body: dict, timeout: float, cfg, allow_fallback: bool = True
) -> str:
    """Extrae `content` de la respuesta; reintenta sin razonamiento si falta,
    está truncado (finish_reason='length') o no parsea como JSON esperado
    (response_format=json_object).

    - Content completo (finish != 'length' y JSON parseable si se espera)
      → se devuelve tal cual.
    - finish_reason='length' con content parcial → catch-and-continue
      (_continue_truncated_content): se re-POST con el fragmento en el
      historial y se acumula hasta completar. Si se agotan las
      continuaciones sin resultado parseable → escape hatch thinking-disabled.
    - stop natural con content no-JSON (expects_json) → reintento
      thinking-disabled directo (no hay parcial que continuar).
    - Sin content (razonamiento cortado) → reintento thinking-disabled.

    El reintento con thinking disabled es SOLO el fallback de emergencia:
    el default del sistema es thinking ACTIVO (ver call_with_retries). Si el
    modelo no soporta razonamiento (body sin parámetro thinking) o ya venía
    desactivado, no se reintenta — se lanza LLMContentError directo.
    """
    choice = data.get("choices", [{}])[0]
    msg = choice.get("message", {})
    content = msg.get("content")
    finish_reason = choice.get("finish_reason")
    expects_json = body.get("response_format") == {"type": "json_object"}

    if content and content.strip():
        text = content.strip()

        if finish_reason == "length":
            # Catch-and-continue: hay un parcial que continuar. Se re-POST
            # con el fragmento en el historial (memoria aditiva) ANTES de
            # degradar a thinking-disabled — así no se descarta el trabajo
            # ya generado ni se duplica el costo.
            joined = _continue_truncated_content(
                session, body, text, timeout, cfg, expects_json
            )
            if joined is not None:
                return joined
            # Continuaciones agotadas sin resultado parseable → escape hatch.
            if allow_fallback and _can_disable_thinking(body):
                logger.warning(
                    "Content truncado (finish_reason='length') y continuaciones "
                    "sin resultado parseable. Reintentando con thinking "
                    "desactivado..."
                )
                fallback_body = {**body, "thinking": {"type": "disabled"}}
                resp2 = _post_with_retry(session, fallback_body, timeout, cfg)
                resp2.raise_for_status()
                return _extract_content(
                    session,
                    resp2.json(),
                    fallback_body,
                    timeout,
                    cfg,
                    allow_fallback=False,
                )
            raise LLMContentError(
                f"Content truncado tras continuaciones. "
                f"finish_reason='{finish_reason}', expects_json={expects_json}, "
                f"content[:120]={text[:120]!r}"
            )

        if expects_json and not _is_parseable_json(text):
            # Stop natural pero no-JSON: no hay parcial que continuar → el
            # reintento thinking-disabled es el único recurso.
            if allow_fallback and _can_disable_thinking(body):
                logger.warning(
                    "Content no-JSON (finish_reason=%r, expects_json=%s). "
                    "Reintentando con thinking desactivado...",
                    finish_reason,
                    expects_json,
                )
                fallback_body = {**body, "thinking": {"type": "disabled"}}
                resp2 = _post_with_retry(session, fallback_body, timeout, cfg)
                resp2.raise_for_status()
                return _extract_content(
                    session,
                    resp2.json(),
                    fallback_body,
                    timeout,
                    cfg,
                    allow_fallback=False,
                )
            raise LLMContentError(
                f"Content no-JSON tras reintento. "
                f"finish_reason='{finish_reason}', expects_json={expects_json}, "
                f"content[:120]={text[:120]!r}"
            )
        return text

    is_reasoning_truncated = (
        finish_reason == "length" and bool(msg.get("reasoning_content")) and not content
    )
    if is_reasoning_truncated and allow_fallback and _can_disable_thinking(body):
        logger.warning(
            "Reasoning agotó max_tokens=%s (finish_reason='length'). "
            "Reintentando con thinking desactivado...",
            body.get("max_tokens"),
        )
        fallback_body = {**body, "thinking": {"type": "disabled"}}
        resp2 = _post_with_retry(session, fallback_body, timeout, cfg)
        resp2.raise_for_status()
        return _extract_content(
            session, resp2.json(), fallback_body, timeout, cfg, allow_fallback=False
        )

    raise LLMContentError(
        f"LLM no retornó contenido. finish_reason='{choice.get('finish_reason')}', "
        f"has_reasoning={bool(msg.get('reasoning_content'))}, "
        f"has_content={bool(content)}"
    )


def _model_max_output_tokens(session, model_name: str) -> int:
    """Lee max_output_tokens del modelo en llm_models (0 si no está)."""
    try:
        row = (
            session.execute(select(LlmModel).where(LlmModel.model_name == model_name))
            .scalars()
            .first()
        )
        if row is None:
            return 0
        return int(getattr(row, "max_output_tokens", 0) or 0)
    except Exception:  # noqa: BLE001 — degradación natural
        return 0


def _resolve_max_tokens(session, model: str, model_size: str, cfg) -> int:
    """max_tokens por defecto: el máximo del modelo (llm_models.max_output_tokens)
    si está registrado (>0); si no, el tope de sesión (cfg.max_tokens_*).

    Decisión documentada: `max_tokens=None` es el ÚNICO centinela de "máximo
    del modelo". `max_tokens=0` NO se interpreta como máximo: se envía tal
    cual y la API lo rechaza (error claro de caller). Así la semántica es
    explícita y no hay sorpresas para quien pase 0 a propósito.
    """
    model_max = _model_max_output_tokens(session, model)
    if model_max > 0:
        return model_max
    return cfg.max_tokens_small if model_size == "small" else cfg.max_tokens_large


def complete(
    session: Session,
    prompt: str,
    model_size: str = "small",
    system: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: float = 900.0,
    response_format: dict | None = None,
    thinking: dict | bool | None = None,
) -> str:
    """Llama al modelo pequeño, grande o de visión vía Together y devuelve el texto.

    model_size: "small" | "large" | "vision". Los valores por defecto de
    temperature y max_tokens salen de session_settings; se pueden
    sobreescribir por llamada. `response_format` (0004) fuerza salida
    estructurada (p. ej. {"type": "json_object"}) cuando el modelo lo
    soporta.

    `thinking` controla el razonamiento del modelo (si su syntax_profile
    lo soporta): None = default del modelo, False/{"type": "disabled"} =
    apagado, dict = config (p. ej. {"type": "enabled", "budget_tokens": N}).
    Si `max_tokens` es demasiado bajo para razonar (piso del modelo o
    budget + reserva), el razonamiento se desactiva automáticamente.
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
        else _resolve_max_tokens(session, model, model_size, cfg)
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

    reasoning = _model_reasoning_mode(session, model)
    param, thinking_final = _resolve_thinking(reasoning, thinking, tokens)
    if param:
        body[param] = thinking_final

    resp = _post_with_retry(session, body, timeout, cfg)
    resp.raise_for_status()
    data = resp.json()
    return _extract_content(session, data, body, timeout, cfg)


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
    thinking: dict | bool | None = None,
) -> str:
    """Llama a un modelo de visión con una imagen (URL) vía Together.

    Igual que `complete`, pero el mensaje de usuario es una lista de
    contenido multimodal estilo OpenAI: texto + image_url. El modelo se
    resuelve desde `llm_models` con is_vision=True (e is_active=True),
    priorizando 'large' sobre 'small' si hay varios. Regla del usuario:
    la lectura de imágenes SOLO puede usar un VLM — si no hay modelo
    is_vision=True activo, lanza LLMConfigError (nunca degrada a un
    modelo de chat no-visión). El caller degrada con gracia (p. ej.
    describe_figure devuelve '' y _describe_image devuelve {}).
    """
    if model_size not in ("small", "large", "vision"):
        raise ValueError(
            f"model_size debe ser 'small', 'large' o 'vision', no {model_size!r}"
        )

    cfg = get_active_llm_config(session)
    if model_size == "small":
        model = cfg.small_model
    else:
        vision_model = get_vision_model(session)
        if vision_model is None:
            raise LLMConfigError(
                "No hay modelo is_vision=True activo en llm_models; "
                "la lectura de imágenes requiere un VLM."
            )
        model = vision_model
    temp = (
        temperature
        if temperature is not None
        else (cfg.temperature_small if model_size == "small" else cfg.temperature_large)
    )
    tokens = (
        max_tokens
        if max_tokens is not None
        else _resolve_max_tokens(session, model, model_size, cfg)
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

    reasoning = _model_reasoning_mode(session, model)
    param, thinking_final = _resolve_thinking(reasoning, thinking, tokens)
    if param:
        body[param] = thinking_final

    resp = _post_with_retry(session, body, timeout, cfg)
    resp.raise_for_status()
    data = resp.json()
    return _extract_content(session, data, body, timeout, cfg)
