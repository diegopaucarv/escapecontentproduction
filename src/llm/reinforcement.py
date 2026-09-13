"""
Refuerzo LLM del semáforo de alineamiento estratégico (0004).

Diseño (aprobado por el usuario):
- El prompt de `alignment_reinforcement` contiene SOLO la regla que el
  código no puede evaluar: "si detectas un riesgo NO cubierto por el
  checklist automático, escala". Las verificaciones mecánicas (fuente,
  CTA, S5/S6) las impone el código (gatekeeper.py / alignment.py), no el
  prompt — no hay segunda fuente de verdad que mantener a mano.
- Fusión CONSERVADORA: el LLM nunca baja la severidad. auto_pass + LLM
  fail/needs_human_review -> needs_human_review. fail se queda fail.
- Reintentos: session_settings.llm_retries (default 3). Agotados ->
  fallback_model -> agotados -> solo reglas (degradación elegante).
- JSON schema SIEMPRE forzado (response_format json_object) en la llamada
  al modelo pequeño.
"""

from __future__ import annotations

import json

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.models import SessionSettings

# Veredictos válidos del output_schema de alignment_reinforcement.
VALID_VERDICTS = {"auto_pass", "needs_human_review", "fail"}


def _load_settings(session: Session) -> SessionSettings | None:
    return (
        session.execute(
            select(SessionSettings).where(SessionSettings.is_active.is_(True))
        )
        .scalars()
        .first()
    )


def _call_with_retries(
    session: Session,
    *,
    prompt: str,
    system: str,
    model_size: str,
    response_format: dict,
    retries: int,
    fallback_model: str | None = None,
) -> tuple[str, str, bool]:
    """Llama a complete() con reintentos y fallback. Devuelve
    (texto, modelo_usado, usó_fallback). Lanza la última excepción si todo
    falla (el caller decide cómo degradar)."""
    from src.llm.together import complete

    last_error: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            return (
                complete(
                    session,
                    prompt,
                    model_size=model_size,
                    system=system,
                    response_format=response_format,
                ),
                model_size,
                False,
            )
        except Exception as exc:  # noqa: BLE001 — cualquier error de red/API
            last_error = exc

    if fallback_model and fallback_model != model_size:
        try:
            return (
                complete(
                    session,
                    prompt,
                    model_size=fallback_model,
                    system=system,
                    response_format=response_format,
                ),
                fallback_model,
                True,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    raise last_error  # type: ignore[misc]


def _parse_llm_output(text: str) -> dict:
    """Parsea el JSON del LLM. Devuelve dict vacío si no es JSON válido."""
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _fuse(rule_verdict: str, llm_verdict: str | None) -> str:
    """Fusión conservadora: el LLM nunca baja la severidad.

    - fail (reglas) -> fail, pase lo que pase.
    - needs_human_review (reglas) -> needs_human_review (el LLM no lo baja).
    - auto_pass (reglas) -> el LLM puede escalar a needs_human_review/fail;
      si el LLM falla o dice auto_pass, se queda auto_pass.
    """
    if rule_verdict == "fail":
        return "fail"
    if rule_verdict == "needs_human_review":
        return "needs_human_review"
    # rule_verdict == "auto_pass"
    if llm_verdict in ("fail", "needs_human_review"):
        return "needs_human_review"
    return "auto_pass"


def reinforce_alignment(
    session: Session,
    brief_data: dict,
    rule_verdict: str,
    checklist_results: list,
) -> dict:
    """Refuerza el semáforo de reglas con el LLM (modelo pequeño).

    Devuelve un dict con el veredicto final fusionado. Nunca lanza: ante
    cualquier fallo degrada a solo-reglas (status 'degraded' o 'skipped').
    """
    settings = _load_settings(session)
    if settings is None:
        return {"status": "skipped", "reason": "no_settings"}

    small_model = settings.small_model
    fallback = getattr(settings, "fallback_model", None) or None
    retries = int(getattr(settings, "llm_retries", 3) or 3)

    # Artefacto compilado (import lazy: el compilador puede no existir aún).
    try:
        from src.llm.compiler import get_active_prompt

        artifact = get_active_prompt(session, small_model, "alignment_reinforcement")
    except Exception:  # noqa: BLE001
        artifact = None
    if artifact is None:
        return {"status": "skipped", "reason": "not_compiled"}

    user_payload = {
        "brief": brief_data,
        "rule_verdict": rule_verdict,
        "checklist_results": checklist_results,
    }
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
    except Exception:  # noqa: BLE001 — LLM no disponible: solo reglas
        return {
            "status": "degraded",
            "reason": "llm_unavailable",
            "verdict": rule_verdict,
            "model_used": small_model,
            "fallback_used": False,
        }

    parsed = _parse_llm_output(text)
    llm_verdict = parsed.get("verdict")
    if llm_verdict not in VALID_VERDICTS:
        # Veredicto inválido/ausente: conservador, pero sin escalar de más.
        llm_verdict = None

    final = _fuse(rule_verdict, llm_verdict)
    return {
        "status": "ok",
        "verdict": final,
        "llm_verdict": llm_verdict,
        "reasoning": parsed.get("reasoning", ""),
        "risks_detected": parsed.get("risks_detected", []),
        "model_used": model_used,
        "fallback_used": used_fallback,
    }
