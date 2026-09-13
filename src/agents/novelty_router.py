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
(Jina jina-embeddings-v5-text-nano, inferencia local). Se pasa por
parámetro para no acoplar esta lógica de enrutamiento a un proveedor
específico ni a la disponibilidad de red.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.models import ArtifactLibraryRow  # ver nota al final del archivo

EmbedFn = Callable[[str], Sequence[float]]

# Similitud cosena mínima para considerar dos piezas "el mismo núcleo temático".
# Empírico — calibrar con datos reales de cada marca antes de confiar en el default.
DUPLICATE_SIMILARITY_THRESHOLD = 0.86

# Zona gris (0004): si la mejor similitud cae entre este umbral y
# DUPLICATE_SIMILARITY_THRESHOLD, el Paso 1 es inconcluso y se puede
# consultar al LLM (refine_angle_novelty) para juzgar el ángulo.
GRAY_ZONE_LOW_THRESHOLD = 0.70

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
            (1 - ArtifactLibraryRow.embedding.cosine_distance(query_vector)).label(
                "similarity"
            ),
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


def _artifact_to_dict(row: ArtifactLibraryRow) -> dict:
    """Serializa la fila previa para pasarla al refinador LLM (JSON-safe)."""
    return {
        "id": str(row.id),
        "content_summary": row.content_summary,
        "retention_24h": row.retention_24h,
        "conversion_30d": row.conversion_30d,
    }


def route_brief(
    session: Session,
    embed_fn: EmbedFn,
    inputs: NoveltyInputs,
    evidence_is_current: Callable[[ArtifactLibraryRow], bool] | None = None,
    score_threshold: int = 4,
    refine_angle: Callable[[dict, list], dict] | None = None,
) -> NoveltyResult:
    """Punto de entrada único. Reemplaza la idea de un `novelty_score`
    calculado a ciegas: primero busca duplicado, luego decide.

    `refine_angle` (0004, opt-in): callable que juzga si el ángulo está
    cubierto por piezas previas. SOLO se consulta en la zona gris
    (GRAY_ZONE_LOW_THRESHOLD <= similitud < DUPLICATE_SIMILARITY_THRESHOLD):
    ni duplicado claro ni claramente nuevo. Si se omite, se usa el
    comportamiento determinista actual (ángulo nuevo por definición).
    El callable recibe (brief_data, prior_artifacts) y devuelve un dict
    con `angulo_nuevo`; si falla o no está disponible, se degrada al
    default determinista.
    """
    match, similarity = find_closest_prior_artifact(
        session, embed_fn, inputs.insight_core, inputs.brand_objective
    )

    if match is not None and similarity >= DUPLICATE_SIMILARITY_THRESHOLD:
        is_current = evidence_is_current(match) if evidence_is_current else True
        if is_current:
            decision = (
                "repetitivo" if inputs.bucket_ever_used_by_brand else "solucion_previa"
            )
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

    # Zona gris (0004): sin duplicado claro, el ángulo es el único
    # componente que el código asume fijo. Si hay un refinador LLM
    # inyectado, se le consulta SOLO el ángulo y SOLO en la banda gris
    # (ni duplicado claro ni claramente nuevo); si falla, se degrada
    # al default determinista (ángulo nuevo).
    if (
        refine_angle is not None
        and GRAY_ZONE_LOW_THRESHOLD <= similarity < DUPLICATE_SIMILARITY_THRESHOLD
    ):
        try:
            brief_data = {
                "insight_core": inputs.insight_core,
                "content_bucket": inputs.content_bucket,
                "artifact_type": inputs.artifact_type,
                "channel": inputs.channel,
            }
            prior_artifacts = [_artifact_to_dict(match)] if match is not None else []
            refinement = refine_angle(brief_data, prior_artifacts)
            if (
                refinement.get("status") == "ok"
                and refinement.get("angulo_nuevo") is False
            ):
                # El ángulo ya está cubierto: se resta el peso del ángulo.
                score -= inputs.weights.get("angulo_nuevo", 3)
        except Exception:  # noqa: BLE001 — degradación elegante
            pass

    decision = "nueva_solucion" if score >= score_threshold else "repetitivo"
    reasoning = f"Sin duplicado (mejor similitud={similarity:.2f}); score={score} -> {decision}."
    return NoveltyResult(decision, score, None, reasoning)


def route_brief_with_refinement(
    session: Session,
    embed_fn: EmbedFn,
    inputs: NoveltyInputs,
    evidence_is_current: Callable[[ArtifactLibraryRow], bool] | None = None,
    score_threshold: int = 4,
) -> NoveltyResult:
    """Coordinador (0004): enruta con refinamiento LLM de la zona gris.

    Vincula la sesión a refine_angle_novelty (el hook de route_brief
    espera Callable[[dict, list], dict]) y la pasa como `refine_angle`.
    Si el LLM no está disponible, refine_angle_novelty degrada al default
    determinista (ángulo nuevo) y el enrutamiento no cambia.
    """
    from src.llm.novelty_refinement import refine_angle_novelty

    def _refine(brief_data: dict, prior_artifacts: list) -> dict:
        return refine_angle_novelty(session, brief_data, prior_artifacts)

    return route_brief(
        session,
        embed_fn,
        inputs,
        evidence_is_current=evidence_is_current,
        score_threshold=score_threshold,
        refine_angle=_refine,
    )


# NOTA DE INTEGRACIÓN: este módulo referencia `ArtifactLibraryRow`, un
# modelo ORM para la tabla `artifact_library` que se añade en models.py
# junto al resto. Se mantiene fuera de este archivo para no acoplar la
# lógica de enrutamiento a los detalles de columnas de la tabla vectorial.
