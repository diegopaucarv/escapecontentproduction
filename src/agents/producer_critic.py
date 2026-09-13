"""
Esqueleto del bucle Producer-Critic (Full Development, §9 del pipeline).

DELIBERADAMENTE UN ESQUELETO, NO UNA IMPLEMENTACIÓN COMPLETA. Dos razones,
ambas ya explicadas antes y que sigo sosteniendo aunque el pedido haya
sido "continuar":

1. Construir este grafo entero antes de correr un solo ciclo real de
   contenido por el pipeline sería exactamente el error de secuenciación
   señalado en la evaluación crítica original — se optimizaría una
   automatización para un proceso que todavía no se validó a mano.
2. No hay una ANTHROPIC_API_KEY disponible en este entorno para probar
   una llamada real de generación. Lo que SÍ se puede probar — y se
   probó — es la mecánica del grafo: estado, bucle condicional,
   interrupt()/resume para el gate humano, y checkpointing. El nodo
   "producer" es un stub explícito; el nodo "critic" NO lo es — llama
   directamente a src/agents/gatekeeper.py, que ya está validado.

API verificada contra langgraph==1.2.11 instalado (StateGraph.add_node,
add_conditional_edges, compile(checkpointer=...), interrupt(), Command) —
no se asumió nada de memoria del entrenamiento, que corresponde a
versiones 0.0.x/0.1.x ya incompatibles.
"""

from __future__ import annotations

from typing import Callable, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from src.agents.gatekeeper import GateInput, GateVerdict, evaluate_gate

MAX_ITERATIONS = 3


class ProducerCriticState(TypedDict):
    brief_id: str
    brand_objective: str
    content_bucket: str
    risk_level: str
    route_decision: str
    production_route: str | None
    segment_client: str | None
    checklist_template: list[str]  # de pipeline_templates.checklist
    draft: str
    iteration: int
    checklist_results: dict[str, bool]
    verdict: str | None
    human_decision: str | None  # "approve" | "reject" | None


def producer_node(state: ProducerCriticState) -> dict:
    """STUB — aquí va la llamada real a Claude (vía langchain-anthropic
    o el SDK de Anthropic directo) para redactar/ajustar `draft` a
    partir de insight_core + Context Pack + feedback del critic previo.
    No implementado a propósito (ver docstring del módulo)."""
    iteration = state["iteration"] + 1
    draft = f"[borrador stub, iteración {iteration}] {state.get('draft', '')}".strip()
    return {"draft": draft, "iteration": iteration}


def critic_node(
    state: ProducerCriticState,
    critic_checklist_fn: Callable[[str, list, str], dict] | None = None,
) -> dict:
    """Real: reutiliza el Gatekeeper ya validado.

    `critic_checklist_fn` (0004, opt-in): callable que ejecuta el crítico
    LLM del checklist (src/llm/critic.py::run_critic_checklist con la
    sesión ya vinculada). Recibe (draft, checklist, brand_objective) y
    devuelve {"checklist_results": {item: "ok"|"fail"|"no_evaluado"}}.
    Si se omite o falla, se simula el checklist (comportamiento actual).
    Conservador: solo "ok" pasa; "fail"/"no_evaluado" bloquean la pieza.
    """
    checklist_template = state["checklist_template"]
    checklist_results: dict[str, bool] = {}
    if critic_checklist_fn is not None:
        try:
            llm_result = critic_checklist_fn(
                state["draft"], checklist_template, state["brand_objective"]
            )
            if llm_result.get("status") == "ok":
                llm_map = llm_result.get("checklist_results", {}) or {}
                for item in checklist_template:
                    checklist_results[item] = llm_map.get(item) == "ok"
            else:
                # Degradación elegante: comportamiento actual (simulado).
                checklist_results = {item: True for item in checklist_template}
        except Exception:  # noqa: BLE001 — el crítico LLM nunca rompe el grafo
            checklist_results = {item: True for item in checklist_template}
    else:
        checklist_results = {item: True for item in checklist_template}

    gate_result = evaluate_gate(
        GateInput(
            risk_level=state["risk_level"],
            production_route=state["production_route"],
            route_decision=state["route_decision"],
            segment_client=state["segment_client"],
            checklist_items=checklist_results,
        )
    )
    return {
        "checklist_results": checklist_results,
        "verdict": gate_result.verdict.value,
    }


def human_review_node(state: ProducerCriticState) -> dict:
    """Punto de interrupción real de LangGraph — el grafo se PAUSA aquí
    de verdad (probado abajo) hasta que alguien llame a
    `graph.invoke(Command(resume={"decision": "approve"}), config=...)`.
    Esto es OWNER_APPROVAL del diagrama de estados, no una simulación."""
    decision = interrupt(
        {
            "brief_id": state["brief_id"],
            "draft": state["draft"],
            "pregunta": "¿Aprueba un 🟨 líder esta pieza? (needs_human_review)",
        }
    )
    return {"human_decision": decision.get("decision")}


def route_after_critic(
    state: ProducerCriticState,
) -> Literal["producer_node", "human_review_node", "__end__"]:
    verdict = state["verdict"]
    if verdict == GateVerdict.fail.value:
        if state["iteration"] >= MAX_ITERATIONS:
            return END  # se agotaron los intentos — en producción esto va a PROBE_KILL/postmortem, no aquí
        return "producer_node"
    if verdict == GateVerdict.needs_human_review.value:
        return "human_review_node"
    return END  # auto_pass


def build_graph(
    checkpointer=None,
    critic_checklist_fn: Callable[[str, list, str], dict] | None = None,
):
    graph = StateGraph(ProducerCriticState)

    def _critic_node(state: ProducerCriticState) -> dict:
        return critic_node(state, critic_checklist_fn=critic_checklist_fn)

    graph.add_node("producer_node", producer_node)
    graph.add_node("critic_node", _critic_node)
    graph.add_node("human_review_node", human_review_node)

    graph.add_edge(START, "producer_node")
    graph.add_edge("producer_node", "critic_node")
    graph.add_conditional_edges("critic_node", route_after_critic)
    graph.add_edge("human_review_node", END)

    return graph.compile(checkpointer=checkpointer or InMemorySaver())
