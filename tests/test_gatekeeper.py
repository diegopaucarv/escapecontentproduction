from src.agents.gatekeeper import GateInput, GateVerdict, evaluate_gate


def test_auto_pass_only_for_low_risk_repetitive():
    result = evaluate_gate(
        GateInput(
            risk_level="bajo",
            production_route=None,
            route_decision="repetitivo",
            segment_client="S2",
            checklist_items={"fuente_verificable": True, "cta_unico": True},
        )
    )
    assert result.verdict == GateVerdict.auto_pass


def test_ngo_segment_always_needs_human_review_even_if_checklist_passes():
    result = evaluate_gate(
        GateInput(
            risk_level="bajo",
            production_route=None,
            route_decision="repetitivo",
            segment_client="S5",  # ONG — canon §3.1
            checklist_items={"fuente_verificable": True},
        )
    )
    assert result.verdict == GateVerdict.needs_human_review


def test_failed_checklist_item_blocks_regardless_of_risk():
    result = evaluate_gate(
        GateInput(
            risk_level="bajo",
            production_route=None,
            route_decision="repetitivo",
            segment_client="S1",
            checklist_items={"fuente_verificable": False},
        )
    )
    assert result.verdict == GateVerdict.fail
    assert "fuente_verificable" in result.failed_items


def test_new_solution_route_always_needs_human_review():
    result = evaluate_gate(
        GateInput(
            risk_level="bajo",
            production_route="fast",
            route_decision="nueva_solucion",
            segment_client="S2",
            checklist_items={"fuente_verificable": True},
        )
    )
    assert result.verdict == GateVerdict.needs_human_review
