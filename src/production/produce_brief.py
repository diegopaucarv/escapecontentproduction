"""Wiring del grafo Producer-Critic a sesiones reales (0007).

Conecta el grafo (src/agents/producer_critic.py) con las funciones vinculadas a
sesión (make_producer_fn / make_critic_checklist_fn) y con el checkpointer. El
grafo se invoca con thread_id = str(brief.id) para que interrupt/resume funcione
entre requests del mismo proceso.

Checkpointer: InMemorySaver a nivel de módulo (singleton). Limitación documentada:
no sobrevive reinicios ni multi-proceso; cuando haya persistencia real se activa
langgraph-checkpoint-postgres (ver requirements.txt, comentado a propósito).
"""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from sqlalchemy.orm import Session

from src.agents.producer_critic import ProducerCriticState, build_graph
from src.db.models import ContentBrief, PipelineTemplate
from src.llm.critic import make_critic_checklist_fn
from src.llm.producer import make_producer_fn

# Singleton a nivel de módulo: el interrupt/resume necesita el MISMO
# checkpointer entre requests (mismo proceso).
_checkpointer = InMemorySaver()


def _val(v):
    """Convierte un enum de SQLAlchemy a su valor string; pasa el resto tal cual."""
    return v.value if hasattr(v, "value") else v


def build_context_pack(brief: ContentBrief) -> dict:
    """Context Pack mínimo desde el brief (sin RAG todavía)."""
    return {
        "resumen": brief.resumen,
        "evidence_source": brief.evidence_source,
        "prior_attempts": brief.prior_attempts,
        "repurpose_plan": brief.repurpose_plan,
    }


def build_state(brief: ContentBrief, template: PipelineTemplate) -> ProducerCriticState:
    """Construye el estado inicial del grafo desde el brief + su plan de marca."""
    return {
        "brief_id": str(brief.id),
        "brand_objective": _val(brief.brand_objective),
        "content_bucket": _val(brief.content_bucket),
        "risk_level": _val(brief.risk_level),
        "route_decision": _val(brief.route_decision),
        "production_route": _val(brief.production_route),
        "segment_client": _val(brief.segment_client),
        "checklist_template": template.checklist or [],
        "draft": "",
        "iteration": 0,
        "checklist_results": {},
        "verdict": None,
        "human_decision": None,
        "insight_core": brief.insight_core or "",
        "context_pack": build_context_pack(brief),
        "artifact_type": _val(brief.artifact_type),
        "channel": brief.channel,
        "requires_user_acceptance": False,
        "critic_feedback": None,
    }


def run_produce(
    session: Session, brief: ContentBrief, template: PipelineTemplate
) -> dict:
    """Ejecuta el grafo Producer-Critic con funciones vinculadas a la sesión real.

    thread_id = str(brief.id): el interrupt/resume del grafo queda asociado al
    brief y puede reanudarse desde otro request (mismo proceso). Devuelve el
    resultado crudo del grafo (puede contener "__interrupt__").
    """
    graph = build_graph(
        checkpointer=_checkpointer,
        producer_fn=make_producer_fn(session),
        critic_checklist_fn=make_critic_checklist_fn(session),
    )
    config = {"configurable": {"thread_id": str(brief.id)}}
    return graph.invoke(build_state(brief, template), config=config)


def resume_produce(session: Session, brief_id: str, decision: str) -> dict:
    """Reanuda un grafo pausado en human_review_node con la decisión humana.

    Solo tiene sentido tras una pausa (interrupt) del mismo brief; si no hay
    pausa, LangGraph continúa desde el último checkpoint (documentado, no es
    un error del endpoint).
    """
    graph = build_graph(
        checkpointer=_checkpointer,
        producer_fn=make_producer_fn(session),
        critic_checklist_fn=make_critic_checklist_fn(session),
    )
    config = {"configurable": {"thread_id": brief_id}}
    return graph.invoke(Command(resume={"decision": decision}), config=config)
