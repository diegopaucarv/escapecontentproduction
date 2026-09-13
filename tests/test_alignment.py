"""Tests del alineamiento estratégico (src/agents/alignment.py).

Cubre el semáforo 🟢🟡🔴 y la verificación del checklist de marca.
"""

from src.agents.alignment import (
    AlignmentInput,
    AlignmentVerdict,
    ChecklistStatus,
    run_alignment,
)


def _base_inputs(**overrides) -> AlignmentInput:
    defaults = dict(
        brand_objective="ESCAPE_SOCIAL",
        content_bucket="difusion_cientifica",
        checklist=[
            "fuente_verificable",
            "gancho_15s_ok",
            "cta_unico",
            "anonimizacion_no_aplica",
        ],
        novelty_weights={
            "bucket_nuevo": 3,
            "formato_nuevo": 2,
            "canal_nuevo": 2,
            "angulo_nuevo": 3,
        },
        evidence_source="Estudio 2026 (fuente primaria).",
        pitch_15s="¿Crees que ya no necesitas el refuerzo?",
        cta="Descarga la guía",
        risk_level="bajo",
        segment_client="S1",
        validation_required=[],
        route_decision="repetitivo",
        production_route="fast",
    )
    defaults.update(overrides)
    return AlignmentInput(**defaults)


def test_green_light_when_checklist_ok_low_risk_repetitive():
    result = run_alignment(_base_inputs())
    assert result.verdict == AlignmentVerdict.auto_pass
    assert all(r.status == ChecklistStatus.ok for r in result.checklist_results)


def test_yellow_when_medium_risk_even_if_checklist_passes():
    result = run_alignment(_base_inputs(risk_level="medio"))
    assert result.verdict == AlignmentVerdict.needs_human_review


def test_yellow_when_sensitive_segment_s5():
    result = run_alignment(_base_inputs(segment_client="S5"))
    assert result.verdict == AlignmentVerdict.needs_human_review


def test_yellow_when_new_solution_route():
    result = run_alignment(_base_inputs(route_decision="nueva_solucion"))
    assert result.verdict == AlignmentVerdict.needs_human_review


def test_red_when_checklist_item_fails():
    result = run_alignment(_base_inputs(evidence_source=None))
    assert result.verdict == AlignmentVerdict.fail
    assert "fuente_verificable" in result.failed_items


def test_unknown_checklist_item_requires_human():
    result = run_alignment(_base_inputs(checklist=["item_personalizado_del_lider"]))
    assert result.verdict == AlignmentVerdict.needs_human_review
    assert "item_personalizado_del_lider" in result.human_review_items


def test_anonymization_verified_check():
    result = run_alignment(
        _base_inputs(
            brand_objective="ERGALIA_COMERCIAL",
            content_bucket="caso_autoridad",
            checklist=[
                "anonimizacion_verificada",
                "fuente_verificable",
                "cta_unico",
                "revision_legal",
            ],
            validation_required=["anonimizacion", "legal"],
            risk_level="alto",
            segment_client="S5",
        )
    )
    assert result.verdict == AlignmentVerdict.needs_human_review  # riesgo alto + S5
    assert all(r.status == ChecklistStatus.ok for r in result.checklist_results)
