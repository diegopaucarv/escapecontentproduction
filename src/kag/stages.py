"""Máquina de estados de ingesta KAG — atomicidad por etapa.

El pipeline clásico (`kag_ingest.py`) tiene su propia secuencia de etapas.
Este módulo centraliza:

  - La definición de las secuencias (orden canónico de etapas).
  - La lectura/escritura de `stage` en la tabla maestra.
  - El cálculo de la etapa desde la que reanudar (resume point).
  - La limpieza idempotente de datos parciales de una etapa.

Contrato de una etapa:
  1. `cleanup(session, doc_id)` — borra SOLO los datos parciales de esa etapa
     (idempotencia: re-ejecutar la etapa no duplica).
  2. `run(session, ...)` — hace el trabajo y commitea.
  3. `set_stage(session, doc_id, stage)` — registra la etapa completada.

Si el proceso se interrumpe a mitad de una etapa, la fila queda con el stage
 de la ÚLTIMA etapa completada; al reanudar se salta todo lo ya hecho y se
re-ejecuta solo la etapa interrumpida (con su cleanup previo).
"""

from __future__ import annotations

from sqlalchemy import text

# ---------------------------------------------------------------------
# Secuencias canónicas de etapas
# ---------------------------------------------------------------------

# Pipeline clásico (src/kag_ingest.py) — tabla kag_documents.
KAG_INGEST_STAGES = [
    "pending",  # fila insertada, nada persistido
    "analysis",  # ficha documental + capítulos detectados por LLM (Fase 2)
    "segmented",  # chunks insertados (embedding NULL) — segmentación persistida
    "paraphrased",  # paráfrasis de chunks por capítulo (Fase 4)
    "chunked",  # embeddings + entidades + relaciones
    "figures",  # figuras indexadas
    "ready",  # resumen + status='ready'
]

# ---------------------------------------------------------------------
# Helpers genéricos
# ---------------------------------------------------------------------


def stage_index(stages: list[str], stage: str) -> int:
    """Índice de una etapa en la secuencia; -1 si no pertenece."""
    try:
        return stages.index(stage)
    except ValueError:
        return -1


def next_stage(stages: list[str], stage: str) -> str | None:
    """Etapa siguiente a `stage` en la secuencia (None si es la última)."""
    idx = stage_index(stages, stage)
    if idx < 0 or idx + 1 >= len(stages):
        return None
    return stages[idx + 1]


def resume_from(stages: list[str], stage: str) -> str:
    """Etapa desde la que reanudar.

    - Si `stage` no pertenece a la secuencia (None/vacío/desconocido), se
      reanuda desde la primera etapa.
    - Si `stage` es la última ('ready'), no hay nada que reanudar (None).
    - Si `stage` es intermedia, se reanuda desde la SIGUIENTE (la actual ya
      está completada).
    """
    if stage == "ready":
        return "ready"
    idx = stage_index(stages, stage)
    if idx < 0:
        return stages[0]
    return stages[idx + 1] if idx + 1 < len(stages) else "ready"


def get_stage(session, table: str, doc_id: int) -> str:
    """Lee la columna `stage` de la fila maestra (default: primera etapa)."""
    row = session.execute(
        text(f"SELECT stage FROM {table} WHERE id = :id"), {"id": doc_id}
    ).first()
    return row.stage if row and row.stage else ""


def set_stage(session, table: str, doc_id: int, stage: str) -> None:
    """Registra la etapa completada (commit incluido)."""
    session.execute(
        text(f"UPDATE {table} SET stage = :stage, updated_at = now() WHERE id = :id"),
        {"stage": stage, "id": doc_id},
    )
    session.commit()


def cleanup_stage(session, table: str, doc_id: int, stage: str) -> None:
    """Borra los datos parciales de una etapa (idempotencia).

    Cada pipeline registra sus cleanups por etapa en CLEANUP_SQL; este helper
    los ejecuta en orden. Si una etapa no tiene cleanup, no hace nada.
    """
    for sql in CLEANUP_SQL.get(table, {}).get(stage, []):
        session.execute(text(sql), {"doc_id": doc_id})
    session.commit()


# ---------------------------------------------------------------------
# Cleanups por tabla y etapa (idempotencia de re-ejecución)
# ---------------------------------------------------------------------

CLEANUP_SQL: dict[str, dict[str, list[str]]] = {
    # Pipeline clásico — kag_documents
    "kag_documents": {
        "analysis": [
            # Re-análisis (Fase 2): borra capítulos y la ficha/esqueleto del doc.
            "DELETE FROM kag_chapters WHERE doc_id = :doc_id",
            "UPDATE kag_documents SET ficha_jsonb = NULL, sections_json = NULL "
            "WHERE id = :doc_id",
        ],
        "segmented": [
            # Re-segmentar: borra chunks (cascada a entidades/relaciones/figuras
            # vía ON DELETE CASCADE de chunk_id; las de doc_id también).
            "DELETE FROM kag_chunks WHERE doc_id = :doc_id",
            # El índice de frecuencia de palabras se reescribe por doc en esta
            # etapa (migración 0023): limpiarlo junto con los chunks.
            "DELETE FROM kag_word_freq WHERE doc_id = :doc_id",
            # Las proposiciones se borran con los chunks (CASCADE); el DELETE
            # explícito es redundante pero inofensivo.
            "DELETE FROM kag_propositions WHERE doc_id = :doc_id",
        ],
        "paraphrased": [
            # Re-ejecutar la extracción FUSIONADA (Fases 4+5): borra las
            # paráfrasis, el content_hash (cache 0026 — fuerza re-extracción),
            # las proposiciones, las entidades/relaciones LLM y el índice
            # temático que la extracción fusionada pudo persistir.
            "UPDATE kag_chunks SET paraphrase = NULL, content_hash = NULL "
            "WHERE doc_id = :doc_id",
            "DELETE FROM kag_relations WHERE doc_id = :doc_id",
            "DELETE FROM kag_entities WHERE doc_id = :doc_id",
            "DELETE FROM kag_propositions WHERE doc_id = :doc_id",
            "DELETE FROM summary_index WHERE doc_id = :doc_id",
        ],
        "chunked": [
            # Re-embeder: los chunks ya están; el UPDATE de embedding es
            # idempotente. Las entidades/relaciones NO se borran: la extracción
            # fusionada (etapa 'paraphrased') ya las persistió y _store_entities_
            # relations dedup por name_norm (re-insertar spaCy no duplica).
        ],
        "figures": [
            "DELETE FROM kag_figures WHERE doc_id = :doc_id",
        ],
    },
}
