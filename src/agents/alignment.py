"""
Alineamiento estratégico — etapa 'alignment' del pipeline (ver
sql/001_init.sql: pipeline_stages).

Revisa un Content Brief contra el plan de marca (pipeline_templates:
checklist de publicación + novelty_weights) y produce el semáforo:

  🟢 auto_pass          — checklist OK, riesgo bajo, ruta repetitiva
  🟡 needs_human_review — riesgo medio/alto, segmento sensible (S5/S6),
                          ruta nueva o producción completa
  🔴 fail               — el brief no cumple el checklist de la marca

Diseño deliberado (consistente con gatekeeper.py): el alineamiento
automático NUNCA aprueba en solitario lo que el canon exige revisar por
un humano. Solo puede pasar en verde lo objetivamente verificable y de
bajo riesgo. Los ítems del checklist que no se pueden verificar contra
los campos del brief se marcan como needs_human: los decide el líder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class AlignmentVerdict(str, Enum):
    auto_pass = "auto_pass"  # 🟢
    needs_human_review = "needs_human_review"  # 🟡
    fail = "fail"  # 🔴


class ChecklistStatus(str, Enum):
    ok = "ok"
    fail = "fail"
    needs_human = "needs_human"


@dataclass
class ChecklistItemResult:
    item: str
    status: ChecklistStatus
    reason: str = ""


@dataclass
class AlignmentInput:
    """Lo mínimo que hace falta del brief + su plan de marca para alinear."""

    brand_objective: str
    content_bucket: str
    checklist: list[str]
    novelty_weights: dict
    # Campos del brief que el checklist puede verificar objetivamente
    evidence_source: str | None
    pitch_15s: str | None
    cta: str | None
    risk_level: str
    segment_client: str | None
    validation_required: list | None
    # Contexto de enrutamiento (lo setea el novelty_router, async)
    route_decision: str | None = None
    production_route: str | None = None
    novelty_score: int | None = None


@dataclass
class AlignmentResult:
    verdict: AlignmentVerdict
    checklist_results: list[ChecklistItemResult]
    failed_items: list[str] = field(default_factory=list)
    human_review_items: list[str] = field(default_factory=list)
    reasoning: str = ""


# ---------------------------------------------------------------------
# Verificaciones automáticas del checklist contra los campos del brief.
# Cada check devuelve (ok: bool, reason: str). Los ítems sin check
# registrado se marcan como needs_human (los decide el líder).
# ---------------------------------------------------------------------


def _check_fuente_verificable(b: AlignmentInput) -> tuple[bool, str]:
    ok = bool(b.evidence_source)
    return ok, "evidence_source presente" if ok else "falta evidence_source"


def _check_gancho_15s_ok(b: AlignmentInput) -> tuple[bool, str]:
    ok = bool(b.pitch_15s)
    return ok, "pitch_15s presente" if ok else "falta pitch_15s"


def _check_cta_unico(b: AlignmentInput) -> tuple[bool, str]:
    ok = bool(b.cta)
    return ok, "cta presente" if ok else "falta cta"


def _check_anonimizacion_no_aplica(b: AlignmentInput) -> tuple[bool, str]:
    # Aplica solo si no hay datos de terceros sensibles (S5/S6) ni riesgo alto
    ok = b.segment_client not in ("S5", "S6") and b.risk_level != "alto"
    return (
        ok,
        "sin datos de terceros sensibles"
        if ok
        else "segmento/riesgo exige anonimización",
    )


def _check_anonimizacion_verificada(b: AlignmentInput) -> tuple[bool, str]:
    reqs = b.validation_required or []
    ok = "anonimizacion" in reqs and bool(b.evidence_source)
    return ok, "anonimización verificada" if ok else "falta validación de anonimización"


def _check_revision_legal(b: AlignmentInput) -> tuple[bool, str]:
    reqs = b.validation_required or []
    ok = "legal" in reqs
    return (
        ok,
        "revisión legal requerida"
        if ok
        else "falta revisión legal en validation_required",
    )


CHECKLIST_CHECKS: dict[str, callable] = {
    "fuente_verificable": _check_fuente_verificable,
    "gancho_15s_ok": _check_gancho_15s_ok,
    "cta_unico": _check_cta_unico,
    "anonimizacion_no_aplica": _check_anonimizacion_no_aplica,
    "anonimizacion_verificada": _check_anonimizacion_verificada,
    "revision_legal": _check_revision_legal,
}

# Checks "blandos": si fallan, NO es un rechazo del brief, sino una señal
# de que se necesita el camino alternativo (p. ej. anonimización) y un
# humano debe confirmarlo. Se marcan como needs_human, no como fail.
SOFT_CHECKS = {"anonimizacion_no_aplica"}


def run_alignment(inputs: AlignmentInput) -> AlignmentResult:
    """Punto de entrada único del alineamiento estratégico."""
    results: list[ChecklistItemResult] = []
    failed: list[str] = []
    human: list[str] = []

    for item in inputs.checklist:
        check = CHECKLIST_CHECKS.get(item)
        if check is None:
            results.append(
                ChecklistItemResult(
                    item,
                    ChecklistStatus.needs_human,
                    "Ítem no verificable automáticamente; lo decide el líder.",
                )
            )
            human.append(item)
            continue
        ok, reason = check(inputs)
        if ok:
            results.append(ChecklistItemResult(item, ChecklistStatus.ok, reason))
        elif item in SOFT_CHECKS:
            # El check falló pero es blando: p. ej. "anonimizacion_no_aplica"
            # falla porque la anonimización SÍ aplica -> lo decide un humano.
            results.append(
                ChecklistItemResult(
                    item,
                    ChecklistStatus.needs_human,
                    f"{reason} — requiere confirmación del líder.",
                )
            )
            human.append(item)
        else:
            results.append(ChecklistItemResult(item, ChecklistStatus.fail, reason))
            failed.append(item)

    if failed:
        return AlignmentResult(
            verdict=AlignmentVerdict.fail,
            checklist_results=results,
            failed_items=failed,
            reasoning=f"El brief no cumple el checklist de la marca: {', '.join(failed)}.",
        )

    must_review = (
        inputs.risk_level in ("medio", "alto")
        or inputs.segment_client in ("S5", "S6")
        or inputs.route_decision == "nueva_solucion"
        or inputs.production_route == "complete"
        or bool(human)
    )

    if must_review:
        return AlignmentResult(
            verdict=AlignmentVerdict.needs_human_review,
            checklist_results=results,
            human_review_items=human,
            reasoning=(
                "Checklist objetivo OK, pero el riesgo/segmento/ruta exige "
                "aprobación de un 🟨 líder antes de continuar (OWNER_APPROVAL)."
            ),
        )

    return AlignmentResult(
        verdict=AlignmentVerdict.auto_pass,
        checklist_results=results,
        reasoning="Riesgo bajo, ruta repetitiva, checklist de marca completo.",
    )
