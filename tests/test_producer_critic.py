from langgraph.types import Command

from src.agents.producer_critic import ProducerCriticState, build_graph


def _base_state(**overrides) -> ProducerCriticState:
    state: ProducerCriticState = {
        "brief_id": "brief-test",
        "brand_objective": "ESCAPE_SOCIAL",
        "content_bucket": "difusion_cientifica",
        "risk_level": "bajo",
        "route_decision": "repetitivo",
        "production_route": None,
        "segment_client": "S1",
        "checklist_template": ["fuente_verificable", "cta_unico"],
        "draft": "",
        "iteration": 0,
        "checklist_results": {},
        "verdict": None,
        "human_decision": None,
    }
    state.update(overrides)
    return state


def test_low_risk_repetitive_auto_passes_without_pausing():
    graph = build_graph()
    result = graph.invoke(_base_state(), config={"configurable": {"thread_id": "t1"}})
    assert result["verdict"] == "auto_pass"
    assert "__interrupt__" not in result


def test_high_risk_pauses_for_human_and_resumes():
    graph = build_graph()
    config = {"configurable": {"thread_id": "t2"}}
    result = graph.invoke(
        _base_state(risk_level="alto", route_decision="nueva_solucion"), config=config
    )
    assert "__interrupt__" in result

    resumed = graph.invoke(Command(resume={"decision": "approve"}), config=config)
    assert resumed["human_decision"] == "approve"


def test_ngo_segment_always_pauses_even_at_low_risk():
    graph = build_graph()
    result = graph.invoke(
        _base_state(segment_client="S5"), config={"configurable": {"thread_id": "t3"}}
    )
    assert "__interrupt__" in result
