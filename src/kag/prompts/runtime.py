"""Runtime de resolución de prompts KAG (prompt-as-code).

Resuelve el par (system, user) para una task_key: primero intenta el
artefacto compilado (src/llm/compiler.py::get_active_prompt); si no hay
artefacto (tests sin DB, modelos sin compilar), degrada a los fallbacks
de src/kag/prompts/fallbacks.py y al user_template de la spec
(src/kag/prompts/specs.py::USER_TEMPLATES).
"""

from __future__ import annotations

from src.kag.prompts.specs import USER_TEMPLATES


def get_user_template(task_key: str) -> str:
    """user_template de la spec para `task_key` ('' si no existe)."""
    return USER_TEMPLATES.get(task_key, "")


def _get_system_prompt(session, model_name, task_key, fallback: str) -> str:
    """System prompt desde el artefacto compilado, o el fallback actual.

    Import perezoso + try/except: si no hay DB/artefacto (tests con sesiones
    falsas), degrada a la constante de hoy sin romper nada.
    """
    try:
        from src.llm.compiler import get_active_prompt

        artifact = get_active_prompt(session, model_name, task_key)
        if artifact is not None and artifact.prompt_text:
            return artifact.prompt_text
    except Exception:  # noqa: BLE001 — degradación natural
        pass
    return fallback


def _get_prompt_pair(
    session,
    model_name,
    task_key,
    system_fallback: str,
    user_fallback: str | None = None,
) -> tuple[str, str]:
    """(system, user) desde el artefacto compilado, o los fallbacks actuales.

    El artefacto (0021) congela el SYSTEM renderizado en `prompt_text` y el
    USER template parametrizable en `user_template`. Sin artefacto (tests sin
    DB) devuelve los fallbacks actuales — comportamiento EXACTO de hoy.

    Si `user_fallback` es None, se resuelve desde la spec con
    get_user_template(task_key): la spec es la fuente única del user template.
    """
    if user_fallback is None:
        user_fallback = get_user_template(task_key)
    try:
        from src.llm.compiler import get_active_prompt

        artifact = get_active_prompt(session, model_name, task_key)
        if artifact is not None and artifact.prompt_text and artifact.user_template:
            return artifact.prompt_text, artifact.user_template
    except Exception:  # noqa: BLE001 — degradación natural
        pass
    return system_fallback, user_fallback
