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
        # 0007: contexto de generación y estado de aceptación.
        "insight_core": "insight de prueba",
        "context_pack": {"contexto": "pack de prueba"},
        "artifact_type": "video_corto",
        "channel": "tiktok",
        "requires_user_acceptance": False,
        "critic_feedback": None,
    }
    state.update(overrides)
    return state


def _fake_producer_ok(brief_data, context_pack, critic_feedback=None):
    return {"status": "ok", "draft": "borrador LLM"}


def _fake_critic_all_ok(draft, checklist, brand_objective):
    return {
        "status": "ok",
        "checklist_results": {item: "ok" for item in checklist},
        "reasoning": "todo ok",
    }


def test_low_risk_repetitive_auto_passes_without_pausing():
    """0007: con productor y crítico LLM ok, el grafo auto-pasa sin pausar."""
    graph = build_graph(
        producer_fn=_fake_producer_ok, critic_checklist_fn=_fake_critic_all_ok
    )
    result = graph.invoke(_base_state(), config={"configurable": {"thread_id": "t1"}})
    assert result["verdict"] == "auto_pass"
    assert "__interrupt__" not in result
    assert result["requires_user_acceptance"] is False
    assert result["draft"] == "borrador LLM"


def test_low_risk_repetitive_without_producer_pauses_for_acceptance():
    """0007: sin productor LLM -> borrador determinista + pausa para que el
    humano acepte la degradación (nunca auto_pass silencioso)."""
    graph = build_graph()
    result = graph.invoke(_base_state(), config={"configurable": {"thread_id": "t1"}})
    assert "__interrupt__" in result
    interrupt_value = result["__interrupt__"][0].value
    assert interrupt_value["requires_user_acceptance"] is True
    assert "determinista" in interrupt_value["pregunta"]
    assert result["requires_user_acceptance"] is True


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


def test_degraded_producer_pauses_for_acceptance():
    """0007: productor degradado -> borrador determinista + pausa con la
    pregunta de aceptación de la degradación."""

    def fake_producer(brief_data, context_pack, critic_feedback=None):
        return {"status": "degraded", "reason": "llm_unavailable"}

    graph = build_graph(producer_fn=fake_producer)
    result = graph.invoke(_base_state(), config={"configurable": {"thread_id": "t4"}})
    assert "__interrupt__" in result
    interrupt_value = result["__interrupt__"][0].value
    assert interrupt_value["requires_user_acceptance"] is True
    assert "determinista" in interrupt_value["pregunta"]
    assert result["requires_user_acceptance"] is True
    assert "determinista" in result["draft"]


def test_raising_producer_pauses_for_acceptance():
    """0007: productor con excepción -> mismo tratamiento: determinista +
    pausa para aceptación humana."""

    def boom(brief_data, context_pack, critic_feedback=None):
        raise RuntimeError("productor caído")

    graph = build_graph(producer_fn=boom)
    result = graph.invoke(_base_state(), config={"configurable": {"thread_id": "t5"}})
    assert "__interrupt__" in result
    interrupt_value = result["__interrupt__"][0].value
    assert interrupt_value["requires_user_acceptance"] is True
    assert result["requires_user_acceptance"] is True
    assert "determinista" in result["draft"]


def test_producer_ok_critic_degraded_pauses():
    """0007: productor ok + crítico degradado -> la degradación del crítico
    gana: nada evaluado, needs_human_review y pausa para aceptación."""

    def fake_critic(draft, checklist, brand_objective):
        return {"status": "degraded", "reason": "llm_unavailable"}

    graph = build_graph(producer_fn=_fake_producer_ok, critic_checklist_fn=fake_critic)
    result = graph.invoke(_base_state(), config={"configurable": {"thread_id": "t6"}})
    assert "__interrupt__" in result
    assert result["verdict"] == "needs_human_review"
    assert result["requires_user_acceptance"] is True
    assert result["checklist_results"] == {
        "fuente_verificable": False,
        "cta_unico": False,
    }


def test_producer_degraded_critic_ok_still_pauses():
    """0007 (regresión línea ~155 de producer_critic.py): productor degradado
    + crítico ok -> el camino ok CONSERVA la bandera del productor y el
    grafo igual se pausa para aceptación humana."""

    def fake_producer(brief_data, context_pack, critic_feedback=None):
        return {"status": "degraded", "reason": "llm_unavailable"}

    graph = build_graph(
        producer_fn=fake_producer, critic_checklist_fn=_fake_critic_all_ok
    )
    result = graph.invoke(_base_state(), config={"configurable": {"thread_id": "t7"}})
    assert "__interrupt__" in result
    interrupt_value = result["__interrupt__"][0].value
    assert interrupt_value["requires_user_acceptance"] is True
    assert result["requires_user_acceptance"] is True
    assert result["verdict"] == "auto_pass"  # el gate ve el checklist ok
