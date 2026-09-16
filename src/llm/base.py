"""
Infraestructura compartida de decisiones agénticas (0007).

Filosofía (aprobada por el usuario):
- El LLM es la opción SIEMPRE presente: la mayoría de las decisiones se
  orquestan por LLM y se resuelven con SLMs (modelo pequeño).
- El usuario es invitado a revisar las decisiones, sean deterministas o LLM.
- La degradación a lo determinista SIEMPRE requiere aceptación del usuario:
  `requires_user_acceptance=True` — el pipeline no continúa sin su visto bueno.
- Los roles pueden ser humanos o guiados por LLM, salvo los pasos cruciales
  (OWNER_APPROVAL: S5/S6, riesgo medio/alto, ruta nueva, producción completa).
- Prompt-as-code en TODAS las llamadas, incluida la generación (producer).

Contrato de resultado de decisión (dict) — TODOS los consumidores lo siguen:
{
  "status": "ok" | "degraded" | "skipped",
  "decision_source": "llm" | "deterministic" | "human",
  "requires_user_acceptance": bool,   # True si se degradó a determinista
  "reason": str,                       # solo en degraded/skipped
  "reasoning": str,                    # explicación (LLM o determinista)
  "model_used": str | None,
  "fallback_used": bool,
  ...campos específicos de la tarea (verdict, checklist_results, draft...)
}
"""

from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.models import SessionSettings
from src.llm.sanitize import LLMJSONError, LLMTruncatedError, sanitize_llm_json

logger = logging.getLogger(__name__)

# Origen de la decisión.
DECISION_SOURCE_LLM = "llm"
DECISION_SOURCE_DETERMINISTIC = "deterministic"
DECISION_SOURCE_HUMAN = "human"


def load_settings(session: Session) -> SessionSettings | None:
    """Lee la session_settings activa (singleton). None si no hay."""
    return (
        session.execute(
            select(SessionSettings).where(SessionSettings.is_active.is_(True))
        )
        .scalars()
        .first()
    )


def call_with_retries(
    session: Session,
    *,
    prompt: str,
    system: str,
    model_size: str,
    response_format: dict,
    retries: int,
    fallback_model: str | None = None,
    thinking: dict | bool | None = None,
) -> tuple[str, str, bool]:
    """Llama a complete() con fallback. Devuelve
    (texto, modelo_usado, usó_fallback). Lanza la última excepción si todo
    falla (el caller decide cómo degradar).

    Los reintentos del modelo primario los hace tenacity DENTRO de
    complete() (src/llm/together.py::_post_with_retry) según
    session_settings.llm_retries — este wrapper NO repite el bucle
    (antes: retries × retries = 9 llamadas con el default 3).

    `thinking` se pasa a complete(). Default: thinking=None → default del
    modelo (razonamiento ACTIVO). Las tareas que NO quieran razonar deben
    pasar `thinking=False` explícitamente (p. ej. summarize_document).
    """
    from src.llm.together import complete

    last_error: Exception | None = None
    try:
        return (
            complete(
                session,
                prompt,
                model_size=model_size,
                system=system,
                response_format=response_format,
                thinking=thinking,
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
                    thinking=thinking,
                ),
                fallback_model,
                True,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    raise last_error  # type: ignore[misc]


def parse_llm_output(text: str) -> dict:
    """Parsea el JSON del LLM. Devuelve dict vacío si no es JSON válido.

    Delega en src.llm.sanitize (fences, texto alrededor, trailing commas y
    detección de truncamiento). Compatibilidad: devuelve {} en vez de lanzar;
    el truncamiento se LOGUEA (el reintento ya ocurrió en _extract_content).
    """
    if not isinstance(text, str):
        return {}
    try:
        return sanitize_llm_json(text)
    except LLMTruncatedError as exc:
        logger.warning("parse_llm_output: output LLM truncado: %s", exc)
        return {}
    except LLMJSONError:
        return {}


def ok_result(**extra) -> dict:
    """Resultado de una decisión tomada por el LLM (camino por defecto).

    El usuario puede revisarla, pero NO requiere aceptación obligatoria.
    """
    return {
        "status": "ok",
        "decision_source": DECISION_SOURCE_LLM,
        "requires_user_acceptance": False,
        **extra,
    }


def degraded_result(reason: str, **extra) -> dict:
    """Resultado degradado: el LLM no estuvo disponible y se usó el default
    determinista. SIEMPRE requiere aceptación del usuario."""
    return {
        "status": "degraded",
        "decision_source": DECISION_SOURCE_DETERMINISTIC,
        "requires_user_acceptance": True,
        "reason": reason,
        **extra,
    }


def skipped_result(reason: str, **extra) -> dict:
    """Resultado omitido: no hay config LLM o artefacto compilado. La
    decisión es determinista por construcción — requiere aceptación."""
    return {
        "status": "skipped",
        "decision_source": DECISION_SOURCE_DETERMINISTIC,
        "requires_user_acceptance": True,
        "reason": reason,
        **extra,
    }
