"""
Enrutamiento por novedad (ver pipeline_unificado_produccion_contenidos.md §4.4).

Diseño deliberado:
- No depende de LangChain/LangGraph. Es lógica de negocio determinista con
  UNA sola llamada externa (embeddings), inyectada como callable. Esto la
  hace trivial de testear sin mockear un framework de agentes completo, y
  la deja intercambiable entre Voyage AI / OpenAI / lo que venga después.
- Implementa el Paso 1 (búsqueda de duplicado por similitud semántica) y
  el Paso 2 (puntaje ponderado) tal como quedaron definidos en el pipeline.
  No inventa un "novelty_score" mágico de una sola fórmula: primero busca,
  y solo puntúa si no encontró nada comparable.

Implementación concreta de `EmbedFn`: src/embeddings.py::embed_text
(Voyage AI). Se pasa por parámetro para no acoplar esta lógica de
enrutamiento a un proveedor específico ni a la disponibilidad de red.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.models import ArtifactLibraryRow  # ver nota al final del archivo

EmbedFn = Callable[[str], Sequence[float]]

# Similitud coseno mínima para considerar dos piezas "el mismo núcleo temático".
# Empírico — calibrar con datos reales de cada marca antes de confiar en el default.
DUPLICATE_SIMILARITY_THRESHOLD = 0.86

DEFAULT_WEIGHTS = {
    "bucket_nuevo": 3,
    "formato_nuevo": 2,
    "canal_nuevo": 2,
    "angulo_nuevo": 3,
}


@dataclass
class NoveltyInputs:
    """Lo mínimo que hace falta del Content Brief para enrutar."""
    brand_objective: str
    content_bucket: str
    artifact_type: str | None
    channel: str | None
    insight_core: str
    bucket_ever_used_by_brand: bool
    artifact_type_ever_used_by_brand: bool
    channel_ever_used_by_brand: bool
    weights: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))


@dataclass
class NoveltyResult:
    route_decision: str  # "repetitivo" | "solucion_previa" | "nueva_solucion"
    novelty_score: int | None
    matched_artifact_id: str | None
    reasoning: str


def find_closest_prior_artifact(
    session: Session,
    embed_fn: EmbedFn,
    insight_core: str,
    brand_objective: str,
) -> tuple[ArtifactLibraryRow | None, float]:
    """Paso 1 — busca en la Artifact Library la pieza más parecida
    (misma marca) por similitud coseno. Devuelve (fila, similitud)."""
    query_vector = embed_fn(insight_core)

    # pgvector: el operador <=> es distancia coseno (0 = idéntico).
    # similitud = 1 - distancia.
    stmt = (
        select(
            ArtifactLibraryRow,
            (1 - ArtifactLibraryRow.embedding.cosine_distance(query_vector)).label("similarity"),
        )
        .order_by(ArtifactLibraryRow.embedding.cosine_distance(query_vector))
        .limit(1)
    )
    row = session.execute(stmt).first()
    if row is None:
        return None, 0.0
    artifact_row, similarity = row
    return artifact_row, float(similarity)


def score_novelty(inputs: NoveltyInputs) -> int:
    """Paso 2 — solo se llama si Paso 1 no encontró duplicado."""
    w = inputs.weights
    score = 0
    if not inputs.bucket_ever_used_by_brand:
        score += w["bucket_nuevo"]
    if inputs.artifact_type and not inputs.artifact_type_ever_used_by_brand:
        score += w["formato_nuevo"]
    if inputs.channel and not inputs.channel_ever_used_by_brand:
        score += w["canal_nuevo"]
    # "angulo_nuevo" no se puede inferir de flags booleanos: si llegamos
    # aquí es porque el Paso 1 ya confirmó que no hay pieza comparable,
    # así que el ángulo es nuevo por definición.
    score += w["angulo_nuevo"]
    return score


def route_brief(
    session: Session,
    embed_fn: EmbedFn,
    inputs: NoveltyInputs,
    evidence_is_current: Callable[[ArtifactLibraryRow], bool] | None = None,
    score_threshold: int = 4,
) -> NoveltyResult:
    """Punto de entrada único. Reemplaza la idea de un `novelty_score`
    calculado a ciegas: primero busca duplicado, luego decide."""
    match, similarity = find_closest_prior_artifact(
        session, embed_fn, inputs.insight_core, inputs.brand_objective
    )

    if match is not None and similarity >= DUPLICATE_SIMILARITY_THRESHOLD:
        is_current = evidence_is_current(match) if evidence_is_current else True
        if is_current:
            decision = "repetitivo" if inputs.bucket_ever_used_by_brand else "solucion_previa"
            reasoning = (
                f"Coincidencia con pieza previa (similitud={similarity:.2f}); "
                f"evidencia vigente -> {decision}."
            )
        else:
            decision = "solucion_previa"
            reasoning = (
                f"Coincidencia con pieza previa (similitud={similarity:.2f}) "
                "pero evidencia desactualizada -> requiere OBSOLETE_CHECK."
            )
        return NoveltyResult(decision, None, str(match.id), reasoning)

    score = score_novelty(inputs)
    decision = "nueva_solucion" if score >= score_threshold else "repetitivo"
    reasoning = f"Sin duplicado (mejor similitud={similarity:.2f}); score={score} -> {decision}."
    return NoveltyResult(decision, score, None, reasoning)


# NOTA DE INTEGRACIÓN: este módulo referencia `ArtifactLibraryRow`, un
# modelo ORM para la tabla `artifact_library` que se añade en models.py
# junto al resto. Se mantiene fuera de este archivo para no acoplar la
# lógica de enrutamiento a los detalles de columnas de la tabla vectorial.
