"""Task keys del pipeline KAG (prompt-as-code).

Cada task key identifica una spec en `prompt_templates` (ver
src/kag/prompts/specs.py) y su artefacto compilado en `prompt_artifacts`.
El runtime resuelve el prompt con `get_active_prompt(session, model, task_key)`
y degrada a los fallbacks de src/kag/prompts/fallbacks.py si no hay artefacto.
"""

from __future__ import annotations

# --- Ingesta (src/kag_ingest.py) -------------------------------------
TASK_EXTRACT_ENTITIES = "kag_extract_entities"
TASK_QWEN_SUMMARY = "kag_qwen_summary"
TASK_PROPOSITION_CHUNKING = "kag_proposition_chunking"
TASK_DOCUMENT_SEPARATION = "kag_document_separation"
TASK_DOCUMENT_ANALYSIS = "kag_document_analysis"
TASK_CHUNK_PARAPHRASE = "kag_chunk_paraphrase"
TASK_CHAPTER_PROPOSITIONS = "kag_chapter_propositions"
TASK_DOCUMENT_EXTRACT = "kag_document_extract"

# --- Consulta (src/kag_query.py) -------------------------------------
TASK_GROUNDED_ENTITIES = "kag_grounded_entities"
TASK_CRITIC_REGEX = "kag_critic_regex"
TASK_CRITIC_LINKING = "kag_critic_linking"
TASK_QUERY_STRATEGY = "kag_query_strategy"
TASK_QUERY_METADATA = "kag_query_metadata"
TASK_QUERY_SUBQUERIES = "kag_query_subqueries"
TASK_QUERY_ANSWER = "kag_query_answer"
TASK_SYNTHESIS = "kag_synthesis"
TASK_CONTRADICTIONS = "kag_contradictions"
TASK_SUFFICIENCY = "kag_sufficiency"
TASK_AUDIT_FUSED = "kag_audit_fused"
TASK_ANSWER = "kag_answer"

# Todas las task keys (para validación/seed).
ALL_TASK_KEYS = [
    TASK_EXTRACT_ENTITIES,
    TASK_QWEN_SUMMARY,
    TASK_PROPOSITION_CHUNKING,
    TASK_DOCUMENT_SEPARATION,
    TASK_DOCUMENT_ANALYSIS,
    TASK_CHUNK_PARAPHRASE,
    TASK_CHAPTER_PROPOSITIONS,
    TASK_DOCUMENT_EXTRACT,
    TASK_GROUNDED_ENTITIES,
    TASK_CRITIC_REGEX,
    TASK_CRITIC_LINKING,
    TASK_QUERY_STRATEGY,
    TASK_QUERY_METADATA,
    TASK_QUERY_SUBQUERIES,
    TASK_QUERY_ANSWER,
    TASK_SYNTHESIS,
    TASK_CONTRADICTIONS,
    TASK_SUFFICIENCY,
    TASK_AUDIT_FUSED,
    TASK_ANSWER,
]
