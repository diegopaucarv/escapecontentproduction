"""
Gatekeeper del Pre-Deploy Gate.

Límite de diseño deliberado (ver evaluación crítica): este agente NO
reemplaza a OWNER_APPROVAL. Solo puede fallar (bloquear) una pieza; nunca
puede aprobarla en solitario si el riesgo es medio/alto o si la marca
es Ergalia en un segmento sensible (S5 ONG, S6 Gobierno) o si el bucket
exige anonimización. En esos casos, el resultado siempre es
`needs_human_review`, incluso si todos los checks pasan.

Esto es intencional: automatizar el "go/no-go" completo para contenido
que involucra datos de terceros, gobierno o ONGs sería una regresión de
gobernanza frente a lo que ya exige el canon (risk_level, validation_required).
El Gatekeeper filtra lo objetivamente verificable para no hacerle perder
tiempo al líder revisando checklists mecánicos, no decide por él.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class GateVerdict(str, Enum):
    auto_pass = "auto_pass"          # solo posible en bajo riesgo + ruta repetitiva
    needs_human_review = "needs_human_review"
    fail = "fail"


# Segmentos que SIEMPRE exigen revisión humana, sin importar el resultado
# del checklist automático (canon §3.1: S5 ONG, S6 Gobierno).
ALWAYS_HUMAN_SEGMENTS = {"S5", "S6"}


@dataclass
class GateInput:
    risk_level: str  # "bajo" | "medio" | "alto"
    production_route: str | None  # "fast" | "complete" | None (None = ruta repetitiva)
    route_decision: str  # "repetitivo" | "solucion_previa" | "nueva_solucion"
    segment_client: str | None
    checklist_items: dict[str, bool]  # ej. {"fuente_verificable": True, "cta_unico": True, ...}
    requires_anonymization: bool = False


@dataclass
class GateResult:
    verdict: GateVerdict
    failed_items: list[str] = field(default_factory=list)
    reason: str = ""


def evaluate_gate(gate_input: GateInput) -> GateResult:
    failed = [item for item, ok in gate_input.checklist_items.items() if not ok]

    if failed:
        return GateResult(
            verdict=GateVerdict.fail,
            failed_items=failed,
            reason="Uno o más ítems del checklist de publicación no pasaron.",
        )

    must_review = (
        gate_input.risk_level in ("medio", "alto")
        or gate_input.requires_anonymization
        or (gate_input.segment_client in ALWAYS_HUMAN_SEGMENTS)
        or gate_input.route_decision == "nueva_solucion"
        or gate_input.production_route == "complete"
    )

    if must_review:
        return GateResult(
            verdict=GateVerdict.needs_human_review,
            reason=(
                "Checklist objetivo OK, pero el riesgo/segmento/ruta exige "
                "aprobación de un 🟨 líder antes de publicar (OWNER_APPROVAL)."
            ),
        )

    return GateResult(
        verdict=GateVerdict.auto_pass,
        reason="Riesgo bajo, ruta repetitiva, checklist objetivo completo.",
    )
