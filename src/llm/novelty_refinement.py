"""
Refinador de novedad por LLM — zona gris del enrutamiento (0004).

Diseño (aprobado por el usuario):
- El scoring determinista (src/agents/novelty_router.py::score_novelty)
  sigue siendo el default: es gratis, instantáneo y probado.
- El LLM SOLO se consulta en la zona gris (Paso 1 de búsqueda vectorial
  inconcluso) y SOLO para juzgar `ángulo no cubierto` — el único
  componente que el código asume fijo por no tener mejor señal.
- Nunca puntúa bucket/formato/canal: eso ya lo calcula el código.
- Degradación elegante: si el LLM no está disponible, se asume ángulo
  nuevo (comportamiento determinista actual).
"""

from __future__ import annotations

import json

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.models import SessionSettings
from src.llm.reinforcement import _call_with_retries, _parse_llm_output


def _load_settings(session: Session) -> SessionSettings | None:
    return (
        session.execute(
            select(SessionSettings).where(SessionSettings.is_active.is_(True))
        )
        .scalars()
        .first()
    )


def refine_angle_novelty(
    session: Session,
    brief_data: dict,
    prior_artifacts: list,
) -> dict:
    """Juzga si el ángulo de la pieza nueva está cubierto por piezas previas.

    Devuelve {"status": "ok", "angulo_nuevo": bool, "reasoning": str} o
    {"status": "degraded"/"skipped", ...}. Nunca lanza.
    """
    settings = _load_settings(session)
    if settings is None:
        return {"status": "skipped", "reason": "no_settings"}

    small_model = settings.small_model
    fallback = getattr(settings, "fallback_model", None) or None
    retries = int(getattr(settings, "llm_retries", 3) or 3)

    try:
        from src.llm.compiler import get_active_prompt

        artifact = get_active_prompt(session, small_model, "novelty_scoring")
    except Exception:  # noqa: BLE001
        artifact = None
    if artifact is None:
        return {"status": "skipped", "reason": "not_compiled"}

    user_payload = {"brief": brief_data, "prior_artifacts": prior_artifacts}
    user_message = json.dumps(user_payload, ensure_ascii=False)

    try:
        text, model_used, used_fallback = _call_with_retries(
            session,
            prompt=user_message,
            system=artifact.prompt_text,
            model_size="small",
            response_format={"type": "json_object"},
            retries=retries,
            fallback_model=fallback,
        )
    except Exception:  # noqa: BLE001 — LLM no disponible: default determinista
        return {
            "status": "degraded",
            "reason": "llm_unavailable",
            "angulo_nuevo": True,
            "reasoning": "LLM no disponible; se asume ángulo nuevo (default determinista).",
            "model_used": small_model,
            "fallback_used": False,
        }

    parsed = _parse_llm_output(text)
    angulo_nuevo = parsed.get("angulo_nuevo")
    if not isinstance(angulo_nuevo, bool):
        return {
            "status": "degraded",
            "reason": "invalid_output",
            "angulo_nuevo": True,
            "reasoning": "Salida del LLM inválida; se asume ángulo nuevo (default determinista).",
            "model_used": model_used,
            "fallback_used": used_fallback,
        }

    return {
        "status": "ok",
        "angulo_nuevo": angulo_nuevo,
        "reasoning": parsed.get("reasoning", ""),
        "model_used": model_used,
        "fallback_used": used_fallback,
    }
