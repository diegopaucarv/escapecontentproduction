"""Refinamiento del snapshot (Fase 5, §20.5) — 0007.

Reglas de negocio en código, no en prompts: se valida/coerce el snapshot
contra `format_spec.qa_checks` y `format_spec.constraints`. El LLM
(opcional, prompt-as-code task `project_refine`) solo puede corregir los
warnings; si falla, degrada al resultado determinista con
`requires_user_acceptance=True` (nunca lanza).
"""

from __future__ import annotations

import json

from sqlalchemy import select

from src.db.models import FormatSpec
from src.llm.base import (
    call_with_retries,
    degraded_result,
    load_settings,
    ok_result,
    parse_llm_output,
    skipped_result,
)

# Task key del artefacto compilado (prompt-as-code) del refinador.
REFINE_TASK_KEY = "project_refine"

# Coerciones mecánicas conocidas: qa_check -> clave del snapshot que asegura.
_QA_KEYS = {
    "cta_unico": "cta",
    "gancho_15s_ok": "hook_15s",
    "subtitulos_presentes": "subtitulos",
}


def resolve_format_spec(session, brand_objective, artifact_type) -> FormatSpec | None:
    """Resuelve la FormatSpec por (brand_objective, artifact_type)."""
    if brand_objective is None:
        return None
    return (
        session.execute(
            select(FormatSpec).where(
                FormatSpec.brand_objective == brand_objective,
                FormatSpec.artifact_type == artifact_type,
            )
        )
        .scalars()
        .first()
    )


def _required_keys(format_spec) -> list[str]:
    """Claves requeridas: las de `structure` + 'cta' si constraints lo pide."""
    if format_spec is None:
        return []
    keys = list((format_spec.structure or {}).keys())
    constraints = format_spec.constraints or {}
    if constraints.get("cta_unico") and "cta" not in keys:
        keys.append("cta")
    return keys


def _placeholder(key: str) -> str:
    if key == "cta":
        return "[CTA pendiente de edición]"
    return f"[{key} pendiente de edición]"


def _apply_deterministic_rules(
    session, project, snapshot, rag_context=None
) -> tuple[dict, dict]:
    """Valida/coerce el snapshot contra la spec de formato (reglas en código).

    Devuelve (snapshot_refinado, report) con
    report = {"applied": [...], "warnings": [...], "llm_used": False}.
    `rag_context` es un hook FUTURO (RAG diferido): solo se registra su
    longitud en el report — no se llama a embeddings.
    """
    applied: list[str] = []
    warnings: list[str] = []

    format_spec = resolve_format_spec(
        session, project.brand_objective, project.artifact_type
    )
    refined = dict(snapshot or {})

    # 1. claves requeridas presentes y no vacías
    for key in _required_keys(format_spec):
        if key not in refined:
            refined[key] = _placeholder(key)
            applied.append(f"clave requerida '{key}' añadida")
            warnings.append(f"faltaba la clave requerida '{key}'")
        elif isinstance(refined[key], str) and not refined[key].strip():
            refined[key] = _placeholder(key)
            applied.append(f"'{key}' vacío reemplazado por placeholder")
            warnings.append(f"'{key}' estaba vacío")

    # 2. coerciones mecánicas de qa_checks conocidos
    for check in (format_spec.qa_checks or []) if format_spec is not None else []:
        key = _QA_KEYS.get(check)
        if key is None:
            continue
        value = refined.get(key)
        if key not in refined or (isinstance(value, str) and not value.strip()):
            refined[key] = _placeholder(key)
            applied.append(f"qa_check '{check}' → clave '{key}' asegurada")
            warnings.append(f"qa_check '{check}' requería '{key}'")

    report: dict = {"applied": applied, "warnings": warnings, "llm_used": False}
    if rag_context is not None:
        # Hook futuro: RAG diferido — solo se registra la longitud.
        report["rag_context_len"] = len(rag_context)
    return refined, report


def refine_snapshot(
    session,
    project,
    snapshot,
    rag_context=None,
    use_llm=False,
) -> dict:
    """Refina el snapshot: reglas deterministas SIEMPRE + LLM opcional.

    Devuelve el contrato de decisión de src/llm/base.py con los campos
    `snapshot` y `report`. Nunca lanza: si el LLM falla, degrada con
    `requires_user_acceptance=True` y se conserva el resultado determinista.
    """
    refined, report = _apply_deterministic_rules(
        session, project, snapshot, rag_context
    )

    if not use_llm:
        # Camino determinista por elección explícita del usuario: no es una
        # degradación, no requiere aceptación forzada.
        return ok_result(
            decision_source="deterministic",
            snapshot=refined,
            report=report,
            reasoning="Refinamiento determinista (reglas en código).",
            model_used=None,
            fallback_used=False,
        )

    settings = load_settings(session)
    if settings is None:
        return skipped_result(
            "no_settings",
            snapshot=refined,
            report=report,
            reasoning=(
                "No hay session_settings activa; se mantiene el resultado determinista."
            ),
        )

    small_model = settings.small_model
    fallback = getattr(settings, "fallback_model", None) or None
    retries = int(getattr(settings, "llm_retries", 3) or 3)

    try:
        from src.llm.compiler import get_active_prompt

        artifact = get_active_prompt(session, small_model, REFINE_TASK_KEY)
    except Exception:  # noqa: BLE001
        artifact = None
    if artifact is None:
        return skipped_result(
            "not_compiled",
            snapshot=refined,
            report=report,
            reasoning=(
                "No hay artefacto compilado para project_refine; se mantiene el "
                "resultado determinista."
            ),
        )

    user_payload = {
        "topic": project.topic,
        "snapshot": refined,
        "warnings": report["warnings"],
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
    except Exception:  # noqa: BLE001 — LLM no disponible: resultado determinista
        report["warnings"].append(
            "refinamiento LLM falló; se mantiene el resultado determinista"
        )
        return degraded_result(
            "llm_unavailable",
            snapshot=refined,
            report=report,
            reasoning="LLM no disponible; se mantiene el resultado determinista.",
            model_used=small_model,
            fallback_used=False,
        )

    llm_output = parse_llm_output(text)
    if not llm_output:
        report["warnings"].append(
            "refinamiento LLM devolvió salida inválida; se mantiene el "
            "resultado determinista"
        )
        return degraded_result(
            "invalid_output",
            snapshot=refined,
            report=report,
            reasoning="Salida del LLM inválida; se mantiene el resultado determinista.",
            model_used=model_used,
            fallback_used=used_fallback,
        )

    # Fusión conservadora: solo claves no vacías del output del LLM.
    for key, value in llm_output.items():
        if value not in (None, "", [], {}):
            refined[key] = value
    report["llm_used"] = True
    report["applied"].append("refinamiento LLM aplicado (fusión conservadora)")

    return ok_result(
        snapshot=refined,
        report=report,
        reasoning="Refinamiento LLM aplicado sobre los warnings.",
        model_used=model_used,
        fallback_used=used_fallback,
    )
