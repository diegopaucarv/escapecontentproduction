"""Generación del snapshot JSON por proyecto (Fase 5, §20.3) — 0007.

Filosofía compilador: el LLM escribe datos (el snapshot JSON editable),
nunca edita archivos. Prompt-as-code: el prompt del generador vive en
prompt_templates/prompt_artifacts (task `project_generator`), nunca
hardcodeado. Si el LLM no está disponible o devuelve salida inválida, se
degrada a un snapshot determinista (esqueleto con placeholders) y
`requires_user_acceptance=True` — el pipeline se pausa para que el humano
decida (puede editar y aprobar, o reintentar).
"""

from __future__ import annotations

import json

from src.llm.base import (
    call_with_retries,
    degraded_result,
    load_settings,
    ok_result,
    parse_llm_output,
    skipped_result,
)

# Task key del artefacto compilado (prompt-as-code) del generador.
GENERATOR_TASK_KEY = "project_generator"


def _deterministic_snapshot(project, format_spec) -> dict:
    """Esqueleto determinista de emergencia: placeholders por campo del
    formato. Marca explícitamente que requiere aceptación del usuario."""
    snapshot: dict = {}
    if format_spec is not None:
        for key in format_spec.structure or {}:
            snapshot[key] = f"[{key} pendiente de edición]"
    if not snapshot:
        snapshot["contenido"] = "[contenido pendiente de edición]"
    return snapshot


def _user_payload(project, template, format_spec) -> dict:
    """El payload JSON que se pasa como mensaje de usuario al LLM. El
    system (reglas) viene del artefacto compilado; aquí van SOLO los datos."""
    payload: dict = {"topic": project.topic}
    if template is not None:
        payload["template"] = {
            "name": template.name,
            "content": template.content or {},
        }
    if format_spec is not None:
        payload["format_spec"] = {
            "structure": format_spec.structure or {},
            "constraints": format_spec.constraints or {},
            "visual_requirements": format_spec.visual_requirements or {},
        }
    return payload


def generate_snapshot(session, project, template, format_spec) -> dict:
    """Genera el snapshot JSON del proyecto vía LLM (compilador, 0007).

    Devuelve el contrato de decisión de src/llm/base.py con el campo
    `snapshot`. Nunca lanza: ante cualquier fallo degrada (status
    'degraded'/'skipped') y el snapshot determinista SIEMPRE requiere
    aceptación del usuario.
    """
    settings = load_settings(session)
    if settings is None:
        return skipped_result(
            "no_settings",
            snapshot=_deterministic_snapshot(project, format_spec),
            reasoning=(
                "No hay session_settings activa; se usó el esqueleto determinista."
            ),
        )

    small_model = settings.small_model
    fallback = getattr(settings, "fallback_model", None) or None
    retries = int(getattr(settings, "llm_retries", 3) or 3)

    # Artefacto compilado (import lazy: el compilador puede no existir aún).
    try:
        from src.llm.compiler import get_active_prompt

        artifact = get_active_prompt(session, small_model, GENERATOR_TASK_KEY)
    except Exception:  # noqa: BLE001
        artifact = None
    if artifact is None:
        return skipped_result(
            "not_compiled",
            snapshot=_deterministic_snapshot(project, format_spec),
            reasoning=(
                "No hay artefacto compilado para project_generator; se usó el "
                "esqueleto determinista."
            ),
        )

    user_message = json.dumps(
        _user_payload(project, template, format_spec), ensure_ascii=False
    )

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
    except Exception:  # noqa: BLE001 — LLM no disponible: esqueleto determinista
        return degraded_result(
            "llm_unavailable",
            snapshot=_deterministic_snapshot(project, format_spec),
            reasoning="LLM no disponible; se usó el esqueleto determinista.",
            model_used=small_model,
            fallback_used=False,
        )

    parsed = parse_llm_output(text)
    if not parsed:
        return degraded_result(
            "invalid_output",
            snapshot=_deterministic_snapshot(project, format_spec),
            reasoning="Salida del LLM inválida; se usó el esqueleto determinista.",
            model_used=model_used,
            fallback_used=used_fallback,
        )

    return ok_result(
        snapshot=parsed,
        reasoning="Snapshot generado por el compilador LLM.",
        model_used=model_used,
        fallback_used=used_fallback,
    )
