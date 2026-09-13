"""
Productor guiado por LLM — generación prompt-as-code (0007).

Filosofía (aprobada por el usuario):
- El LLM es la opción SIEMPRE presente: la generación del borrador se
  orquesta por LLM (prompt-as-code vía el artefacto `producer_draft`) y se
  resuelve con el SLM (modelo pequeño).
- Prompt-as-code en TODAS las llamadas, incluida la generación: el prompt
  del productor vive en prompt_templates/prompt_artifacts, nunca hardcodeado.
- Degradación a lo determinista SIEMPRE requiere aceptación del usuario:
  si el LLM no está disponible o devuelve una salida inválida, se usa el
  borrador determinista y `requires_user_acceptance=True` — el pipeline se
  pausa para que el humano decida.
- El usuario es invitado a revisar las decisiones, sean deterministas o LLM.

Contrato de resultado (dict) — ver src/llm/base.py:
{
  "status": "ok" | "degraded" | "skipped",
  "decision_source": "llm" | "deterministic" | "human",
  "requires_user_acceptance": bool,
  "reason": str,                    # solo en degraded/skipped
  "draft": str,
  "hook_15s": str,
  "cta": str,
  "reasoning": str,
  "model_used": str | None,
  "fallback_used": bool,
}
"""

from __future__ import annotations

import json
from typing import Callable

from sqlalchemy.orm import Session

from src.llm.base import (
    call_with_retries,
    degraded_result,
    load_settings,
    ok_result,
    parse_llm_output,
    skipped_result,
)

# Task key del artefacto compilado (prompt-as-code) del productor.
PRODUCER_TASK_KEY = "producer_draft"

# Borrador determinista de emergencia: marca explícitamente que requiere
# aceptación del usuario antes de continuar.
DETERMINISTIC_DRAFT = "[borrador determinista — requiere aceptación del usuario]"


def run_producer_generation(
    session: Session,
    brief_data: dict,
    context_pack: dict,
    critic_feedback: str | None = None,
) -> dict:
    """Genera el borrador de la pieza con el LLM (prompt-as-code).

    Devuelve el contrato de decisión de src/llm/base.py con el campo
    `draft` (y `hook_15s`/`cta`/`reasoning` si el LLM los dio). Nunca
    lanza: ante cualquier fallo degrada (status 'degraded'/'skipped') y
    el borrador determinista SIEMPRE requiere aceptación del usuario.
    """
    settings = load_settings(session)
    if settings is None:
        return skipped_result("no_settings", draft=DETERMINISTIC_DRAFT)

    small_model = settings.small_model
    fallback = getattr(settings, "fallback_model", None) or None
    retries = int(getattr(settings, "llm_retries", 3) or 3)

    # Artefacto compilado (import lazy: el compilador puede no existir aún).
    try:
        from src.llm.compiler import get_active_prompt

        artifact = get_active_prompt(session, small_model, PRODUCER_TASK_KEY)
    except Exception:  # noqa: BLE001
        artifact = None
    if artifact is None:
        return skipped_result("not_compiled", draft=DETERMINISTIC_DRAFT)

    user_payload = {
        "brief": brief_data,
        "context_pack": context_pack,
        "critic_feedback": critic_feedback,
    }
    user_message = json.dumps(user_payload, ensure_ascii=False)

    try:
        text, model_used, used_fallback = call_with_retries(
            session,
            prompt=user_message,
            system=artifact.prompt_text,
            model_size="small",
            response_format={"type": "json_object"},
            retries=retries,
            fallback_model=fallback,
        )
    except Exception:  # noqa: BLE001 — LLM no disponible: borrador determinista
        return degraded_result(
            "llm_unavailable",
            draft=DETERMINISTIC_DRAFT,
            reasoning="LLM no disponible; se usó el borrador determinista.",
            model_used=small_model,
            fallback_used=False,
        )

    parsed = parse_llm_output(text)
    draft = parsed.get("draft")
    if not isinstance(draft, str) or not draft.strip():
        return degraded_result(
            "invalid_output",
            draft=DETERMINISTIC_DRAFT,
            reasoning="Salida del LLM inválida; se usó el borrador determinista.",
            model_used=model_used,
            fallback_used=used_fallback,
        )

    return ok_result(
        draft=draft,
        hook_15s=parsed.get("hook_15s", ""),
        cta=parsed.get("cta", ""),
        reasoning=parsed.get("reasoning", ""),
        model_used=model_used,
        fallback_used=used_fallback,
    )


def make_producer_fn(session: Session) -> Callable[[dict, dict, str | None], dict]:
    """Vincula la sesión a run_producer_generation para inyectarla en el
    grafo Producer-Critic (el nodo espera
    Callable[[dict, dict, str | None], dict])."""

    def _fn(
        brief_data: dict, context_pack: dict, critic_feedback: str | None = None
    ) -> dict:
        return run_producer_generation(
            session, brief_data, context_pack, critic_feedback
        )

    return _fn
