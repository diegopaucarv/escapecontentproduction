"""
Bucle Producer-Critic (Full Development, §9 del pipeline) — 0007.

Filosofía (aprobada por el usuario):
- El productor es guiado por LLM (prompt-as-code vía el artefacto
  `producer_draft`, src/llm/producer.py): genera/ajusta `draft` a partir
  de insight_core + Context Pack + feedback del crítico previo. El SLM
  (modelo pequeño) resuelve la generación; el LLM orquesta la decisión.
- Degradación a lo determinista SIEMPRE requiere aceptación del usuario:
  si el productor o el crítico LLM no están disponibles, el grafo se
  pausa en human_review_node (`requires_user_acceptance=True`) — nunca
  continúa sin el visto bueno del humano.
- El crítico degrada de forma CONSERVADORA: sin LLM, todo el checklist
  queda "no evaluado" y el veredicto es needs_human_review — NUNCA
  auto_pass simulado.
- Los roles pueden ser humanos o guiados por LLM, salvo los pasos
  cruciales (OWNER_APPROVAL: S5/S6, riesgo medio/alto, ruta nueva,
  producción completa) que siguen siendo humanos.

El grafo es testeable sin sesión: `producer_fn` y `critic_checklist_fn`
son callables inyectados (ver make_producer_fn / make_critic_checklist_fn).
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
    # 0007: contexto de generación y estado de aceptación.
    insight_core: str
    context_pack: dict
    artifact_type: str | None
    channel: str | None
    requires_user_acceptance: bool
    critic_feedback: str | None


def producer_node(
    state: ProducerCriticState,
    producer_fn: Callable[[dict, dict, str | None], dict] | None = None,
) -> dict:
    """Nodo productor — guiado por LLM (0007).

    `producer_fn` (opt-in): callable que ejecuta la generación LLM
    (src/llm/producer.py::run_producer_generation con la sesión ya
    vinculada). Recibe (brief_data, context_pack, critic_feedback) y
    devuelve el contrato de decisión de src/llm/base.py con `draft`.
    Si se omite, o degrada/omite/falla, se usa el borrador determinista
    y SIEMPRE se marca `requires_user_acceptance=True`: el pipeline se
    pausa para que el humano acepte la degradación.
    """
    iteration = state["iteration"] + 1

    def _deterministic_draft() -> str:
        return (
            f"[borrador determinista, iteración {iteration}] {state.get('draft', '')}"
        ).strip()

    if producer_fn is None:
        return {
            "draft": _deterministic_draft(),
            "iteration": iteration,
            "requires_user_acceptance": True,
        }

    brief_data = {
        "insight_core": state.get("insight_core", ""),
        "brand_objective": state.get("brand_objective", ""),
        "content_bucket": state.get("content_bucket", ""),
        "artifact_type": state.get("artifact_type"),
        "channel": state.get("channel"),
    }
    try:
        result = producer_fn(
            brief_data=brief_data,
            context_pack=state.get("context_pack", {}),
            critic_feedback=state.get("critic_feedback"),
        )
    except Exception:  # noqa: BLE001 — el productor LLM nunca rompe el grafo
        result = None

    if result is not None and result.get("status") == "ok":
        return {
            "draft": result["draft"],
            "iteration": iteration,
            "requires_user_acceptance": False,
        }

    # Degradado/omitido/excepción: borrador determinista + aceptación humana.
    return {
        "draft": _deterministic_draft(),
        "iteration": iteration,
        "requires_user_acceptance": True,
    }


def critic_node(
    state: ProducerCriticState,
    critic_checklist_fn: Callable[[str, list, str], dict] | None = None,
) -> dict:
    """Crítico — degradación CONSERVADORA (0007).

    `critic_checklist_fn` (0004, opt-in): callable que ejecuta el crítico
    LLM del checklist (src/llm/critic.py::run_critic_checklist con la
    sesión ya vinculada). Recibe (draft, checklist, brand_objective) y
    devuelve {"checklist_results": {item: "ok"|"fail"|"no_evaluado"}}.
    Solo "ok" pasa; "fail"/"no_evaluado" bloquean la pieza.

    Si el crítico LLM no está disponible (fn omitida, degradada, omitida
    o con excepción), TODO el checklist queda "no evaluado" y el veredicto
    es needs_human_review con `requires_user_acceptance=True` — NUNCA se
    simula auto_pass (comportamiento permisivo anterior, corregido).
    """
    checklist_template = state["checklist_template"]
    checklist_results: dict[str, bool] = {}
    critic_feedback: str | None = None
    requires_user_acceptance = False
    verdict: str | None = None

    if critic_checklist_fn is not None:
        try:
            llm_result = critic_checklist_fn(
                state["draft"], checklist_template, state["brand_objective"]
            )
            if llm_result.get("status") == "ok":
                llm_map = llm_result.get("checklist_results", {}) or {}
                for item in checklist_template:
                    checklist_results[item] = llm_map.get(item) == "ok"
                critic_feedback = llm_result.get("reasoning", "")
                # El crítico no degradó, pero conserva la bandera del
                # productor: si el borrador es determinista (degradación
                # previa), el pipeline igual debe pausar para aceptación.
                requires_user_acceptance = state.get("requires_user_acceptance", False)
                gate_result = evaluate_gate(
                    GateInput(
                        risk_level=state["risk_level"],
                        production_route=state["production_route"],
                        route_decision=state["route_decision"],
                        segment_client=state["segment_client"],
                        checklist_items=checklist_results,
                    )
                )
                verdict = gate_result.verdict.value
            else:
                # Degradado/omitido: nada evaluado, revisión humana obligatoria.
                checklist_results = {item: False for item in checklist_template}
                requires_user_acceptance = True
                verdict = GateVerdict.needs_human_review.value
                critic_feedback = llm_result.get("reasoning") or (
                    "Crítico LLM no disponible; checklist simulado como no evaluado."
                )
        except Exception:  # noqa: BLE001 — el crítico LLM nunca rompe el grafo
            checklist_results = {item: False for item in checklist_template}
            requires_user_acceptance = True
            verdict = GateVerdict.needs_human_review.value
            critic_feedback = (
                "Crítico LLM no disponible; checklist simulado como no evaluado."
            )
    else:
        checklist_results = {item: False for item in checklist_template}
        requires_user_acceptance = True
        verdict = GateVerdict.needs_human_review.value
        critic_feedback = (
            "Crítico LLM no disponible; checklist simulado como no evaluado."
        )

    return {
        "checklist_results": checklist_results,
        "verdict": verdict,
        "requires_user_acceptance": requires_user_acceptance,
        "critic_feedback": critic_feedback,
    }


def human_review_node(state: ProducerCriticState) -> dict:
    """Punto de interrupción real de LangGraph — el grafo se PAUSA aquí
    de verdad hasta que alguien llame a
    `graph.invoke(Command(resume={"decision": "approve"}), config=...)`.
    Esto es OWNER_APPROVAL del diagrama de estados, no una simulación.

    Si `requires_user_acceptance` es True, la pausa es por DEGRADACIÓN
    determinista (productor/crítico LLM no disponible) y la pregunta lo
    dice explícitamente: el humano debe aceptar la decisión determinista.
    """
    requires_acceptance = state.get("requires_user_acceptance", False)
    if requires_acceptance:
        pregunta = (
            "El LLM no estuvo disponible; la decisión es determinista. ¿La acepta?"
        )
    else:
        pregunta = "¿Aprueba un 🟨 líder esta pieza? (needs_human_review)"
    decision = interrupt(
        {
            "brief_id": state["brief_id"],
            "draft": state["draft"],
            "pregunta": pregunta,
            "requires_user_acceptance": requires_acceptance,
        }
    )
    return {"human_decision": decision.get("decision")}


def route_after_critic(
    state: ProducerCriticState,
) -> Literal["producer_node", "human_review_node", "__end__"]:
    # Una decisión determinista (degradación) SIEMPRE pausa para aceptación
    # humana, sin importar el veredicto del gate.
    if state.get("requires_user_acceptance"):
        return "human_review_node"
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
    producer_fn: Callable[[dict, dict, str | None], dict] | None = None,
    critic_checklist_fn: Callable[[str, list, str], dict] | None = None,
):
    graph = StateGraph(ProducerCriticState)

    def _producer_node(state: ProducerCriticState) -> dict:
        return producer_node(state, producer_fn=producer_fn)

    def _critic_node(state: ProducerCriticState) -> dict:
        return critic_node(state, critic_checklist_fn=critic_checklist_fn)

    graph.add_node("producer_node", _producer_node)
    graph.add_node("critic_node", _critic_node)
    graph.add_node("human_review_node", human_review_node)

    graph.add_edge(START, "producer_node")
    graph.add_edge("producer_node", "critic_node")
    graph.add_conditional_edges("critic_node", route_after_critic)
    graph.add_edge("human_review_node", END)

    return graph.compile(checkpointer=checkpointer or InMemorySaver())
