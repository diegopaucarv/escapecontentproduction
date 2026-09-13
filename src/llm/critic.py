"""
Consumidor del spec `critic_checklist` (0004).

Diseño (aprobado por el usuario):
- El LLM es la opción SIEMPRE presente: la decisión se orquesta por LLM y
  se resuelve con el SLM (modelo pequeño). El crítico LLM verifica el
  borrador contra el checklist de publicación recibido COMO DATO (array),
  usando las interpretaciones del spec. El checklist es variable por bucket
  (pipeline_templates.checklist), así que nunca se hardcodea en el prompt.
- Regla defensiva EN CÓDIGO: todo ítem del checklist SIN interpretación
  en rules -> status "no_evaluado", NUNCA "ok". El prompt también lo
  pide, pero el código lo garantiza (no depende de que el LLM obedezca).
- Reintentos: session_settings.llm_retries (default 3). Agotados ->
  fallback_model -> agotados -> degradación elegante. La degradación es
  determinista y SIEMPRE requiere aceptación del usuario
  (requires_user_acceptance=True).
- JSON schema SIEMPRE forzado (response_format json_object) en la llamada
  al modelo pequeño.
"""

from __future__ import annotations

import json
from typing import Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.models import PromptTemplate
from src.llm.base import (
    call_with_retries,
    degraded_result,
    load_settings,
    ok_result,
    parse_llm_output,
    skipped_result,
)

# Veredictos válidos del output_schema de critic_checklist.
VALID_VERDICTS = {"auto_pass", "needs_human_review", "fail"}

CRITIC_TASK_KEY = "critic_checklist"


def _interpreted_items(spec_rules: list[str], checklist: list[str]) -> set[str]:
    """Ítems del checklist que tienen interpretación en rules.

    Convención (misma que el compilador): una regla cubre un ítem si
    empieza con '<item>:' o '<item> '.
    """
    covered: set[str] = set()
    for rule in spec_rules or []:
        rule = rule.strip()
        for item in checklist or []:
            if rule.startswith(f"{item}:") or rule.startswith(f"{item} "):
                covered.add(item)
    return covered


def _normalize(
    parsed: dict,
    checklist: list[str],
    covered: set[str],
    model_used: str,
    used_fallback: bool,
) -> dict:
    """Normaliza la salida del LLM.

    Regla defensiva: un ítem sin interpretación en rules, o sin resultado
    del LLM, queda "no_evaluado" — nunca "ok".
    """
    raw_results = parsed.get("checklist_results", [])
    if not isinstance(raw_results, list):
        raw_results = []

    llm_map: dict[str, str] = {}
    for entry in raw_results:
        if isinstance(entry, dict):
            item = entry.get("item")
            status = entry.get("status")
            if isinstance(item, str) and status in ("ok", "fail"):
                llm_map[item] = status

    results: dict[str, str] = {}
    for item in checklist:
        if item not in covered:
            results[item] = "no_evaluado"
        else:
            results[item] = llm_map.get(item, "no_evaluado")

    verdict = parsed.get("verdict")
    if verdict not in VALID_VERDICTS:
        verdict = None

    return {
        "status": "ok",
        "checklist_results": results,
        "verdict": verdict,
        "reasoning": parsed.get("reasoning", ""),
        "model_used": model_used,
        "fallback_used": used_fallback,
    }


def run_critic_checklist(
    session: Session,
    draft: str,
    checklist: list[str],
    brand_objective: str,
) -> dict:
    """Ejecuta el crítico LLM sobre el borrador.

    Devuelve:
    {
      "status": "ok" | "degraded" | "skipped",
      "checklist_results": {item: "ok"|"fail"|"no_evaluado", ...},
      "verdict": "auto_pass"|"needs_human_review"|"fail"|None,
      "reasoning": str,
      "model_used": str,
      "fallback_used": bool,
    }
    Nunca lanza: ante cualquier fallo degrada (status 'degraded'/'skipped').
    """
    settings = load_settings(session)
    if settings is None:
        return skipped_result("no_settings")

    small_model = settings.small_model
    fallback = getattr(settings, "fallback_model", None) or None
    retries = int(getattr(settings, "llm_retries", 3) or 3)

    # Artefacto compilado (import lazy: el compilador puede no existir aún).
    try:
        from src.llm.compiler import get_active_prompt

        artifact = get_active_prompt(session, small_model, CRITIC_TASK_KEY)
    except Exception:  # noqa: BLE001
        artifact = None
    if artifact is None:
        return skipped_result("not_compiled")

    # Spec activa: sus rules definen qué ítems tienen interpretación.
    template = (
        session.execute(
            select(PromptTemplate).where(
                PromptTemplate.task_key == CRITIC_TASK_KEY,
                PromptTemplate.is_active.is_(True),
            )
        )
        .scalars()
        .first()
    )
    covered = _interpreted_items(template.rules if template else [], checklist)

    user_payload = {
        "draft": draft,
        "checklist": checklist,
        "brand_objective": brand_objective,
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
    except Exception:  # noqa: BLE001 — LLM no disponible: degradación elegante
        return degraded_result(
            "llm_unavailable",
            checklist_results={item: "no_evaluado" for item in checklist},
            verdict=None,
            model_used=small_model,
            fallback_used=False,
        )

    parsed = parse_llm_output(text)
    return ok_result(
        **_normalize(parsed, checklist, covered, model_used, used_fallback)
    )


def make_critic_checklist_fn(session: Session) -> Callable[[str, list, str], dict]:
    """Vincula la sesión a run_critic_checklist para inyectarla en el grafo
    Producer-Critic (el nodo espera Callable[[str, list, str], dict])."""

    def _fn(draft: str, checklist: list, brand_objective: str) -> dict:
        return run_critic_checklist(session, draft, checklist, brand_objective)

    return _fn
