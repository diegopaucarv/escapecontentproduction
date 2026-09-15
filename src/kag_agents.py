"""
Agentes query-time del rediseño KAG (Fase 2, Agente 5).

Capa proposicional: recuperación de proposiciones atómicas (pgvector),
síntesis fáctica, resolución de contradicciones, auditoría de suficiencia
(Agente Auditor Epistemológico), expansión Branch B (nuevos documentos,
expansión de grafo y subconsultas ortogonales FTS), verificación de
grounding verbatim y ensamblaje del contexto final.

Módulo LIGERO a propósito: no importa torch/spacy/transformers. El embedding
de la query usa src.embeddings (carga perezosa del modelo Jina) y el LLM
usa src.llm.base.call_with_retries. Toda degradación es natural
(try/except con log, nunca romper).

Uso:
    from src.kag_agents import ask_propositional
    result = ask_propositional(session, "¿Qué fórmula usa la propagación hacia atrás?")
"""

from __future__ import annotations

import json
import re

from rapidfuzz import fuzz
from sqlalchemy import text

from src.kag_ingest import embedding_to_sql
from src.llm.base import call_with_retries, load_settings, parse_llm_output

# ---------------------------------------------------------------------
# Prompts (prompt-as-code, constantes)
# ---------------------------------------------------------------------

# Task keys de prompt-as-code (specs en src/db/seed_kag_prompts.py).
TASK_SYNTHESIS = "kag_synthesis"
TASK_CONTRADICTIONS = "kag_contradictions"
TASK_SUFFICIENCY = "kag_sufficiency"
TASK_ANSWER = "kag_answer"

# System prompts cortos actuales — fallback EXACTO de hoy cuando no hay
# artefacto compilado (tests sin DB: get_active_prompt devuelve None).
SYNTHESIS_SYSTEM_SHORT = "Eres un agente de consolidación fáctica de alta precisión."
CONTRADICTION_SYSTEM_SHORT = "Eres un analista epistemológico."
SUFFICIENCY_SYSTEM_SHORT = (
    "Eres el Agente Auditor Epistemológico de un sistema de recuperación avanzada."
)
ANSWER_SYSTEM_SHORT = (
    "Eres un asistente de conocimiento con estándares epistémicos estrictos."
)

# Prompts combinados SYSTEM+USER — fallback EXACTO de hoy (sin artefacto).
SYNTHESIS_PROMPT = """SYSTEM: Eres un agente de consolidación fáctica de alta precisión. Tu tarea es procesar un conjunto de fragmentos proposicionales recuperados por el motor de búsqueda y extraer exclusivamente los hechos que aportan a la consulta del usuario. Instrucciones: 1. Para cada chunk, evalúa su pertinencia respecto a la consulta formulada. 2. Si el chunk es irrelevante, marca 'relevance_level': 'irrelevant' y deja 'atomic_summary' y 'verbatim_evidence' vacíos. 3. Si el chunk contiene información relevante: 'atomic_summary': Sintetiza el argumento o dato clave eliminando oraciones de relleno y ambigüedades anafóricas. 'verbatim_evidence': Copia una frase textual literal del fragmento que funcione como evidencia irrefutable. Prohibido alterar o parafrasear esta cita. 'academic_citations': Mantén todas las referencias académicas explícitas vinculadas al dato.
USER: Consulta del usuario: "{query}"

Fragmentos recuperados para análisis:
{candidate_chunks_json}

Devuelve el JSON: {{"query": str, "total_chunks_processed": int, "synthesized_facts": [{{"chunk_id": str, "document_id": str, "source_file": str, "relevance_level": "direct_answer|supporting_evidence|contextual_background|irrelevant", "atomic_summary": str, "verbatim_evidence": str, "academic_citations": [str]}}]}}"""

CONTRADICTION_PROMPT = """SYSTEM: Eres un analista epistemológico. En ciencias sociales las contradicciones rara vez son errores fácticos; suelen representar tensiones paradigmáticas, condiciones de contorno empíricas divergentes o evoluciones diacrónicas. No debes forzar una síntesis artificial ni descartar fuentes; debes tipificar y explicitar la divergencia. Tipología: 'paradigmatic_theoretical_divergence' (desacuerdo estructural entre escuelas), 'empirical_contextual_boundary' (resultados opuestos por unidad de análisis/geografía/muestreo), 'temporal_diachronic_shift' (un autor rectifica un postulado previo), 'terminological_homonymy' (ambigüedad conceptual resuelta por la nota de alcance ISO 25964).
USER: Consulta: "{query}"

Hechos sintetizados:
{synthesized_facts_json}

Devuelve el JSON: {{"contradictions_detected": bool, "analysis_cases": [{{"conflict_type": "paradigmatic_theoretical_divergence|empirical_contextual_boundary|temporal_diachronic_shift|terminological_homonymy", "divergence_summary": str, "thesis_a": {{"proposition_id": str, "document_id": str, "claim": str, "author_or_framework": str, "empirical_context": str}}, "thesis_b": {{"proposition_id": str, "document_id": str, "claim": str, "author_or_framework": str, "empirical_context": str}}, "epistemic_reconciliation": str}}]}}"""

SUFFICIENCY_PROMPT = """SYSTEM: Eres el Agente Auditor Epistemológico de un sistema de recuperación avanzada. Tu función es dictaminar con imparcialidad si el conjunto de proposiciones y argumentos recuperados basta para responder con rigor analítico la consulta formulada, o si se debe ejecutar una de dos acciones de control. Directrices de Decisión: 1. 'SUFFICIENT_FOR_SYNTHESIS': Toda premisa necesaria para responder la consulta está respaldada por proposiciones verificables. Si existen contradicciones, han sido tipificadas satisfactoriamente. 2. 'NEGATIVE_REJECTION' (Abstención Temprana Obligatoria): Actívala si la consulta solicita hechos, autores o metodologías que colisionan con el ámbito temático (ISO 25964) y las clasificaciones Library of Congress (LCC/LCSH) de los documentos disponibles. No procedas a iterar búsquedas adicionales si el corpus carece del dominio conceptual de base. Genera una abstención formal fundamentada. 3. 'INSUFFICIENT_TRIGGER_BRANCH_B': Actívala si el corpus sí cubre la temática general, pero la recuperación actual omitió vínculos causales intermedios, evidencia de apoyo o nodos relacionales específicos.
USER: Consulta: "{query}"

Metadatos del Corpus Disponible (Descriptores ISO 25964 y LCC presentes en DB):
{active_corpus_metadata}

Proposiciones recuperadas (Nivel 1):
{synthesized_propositions_json}

Contextos escalados (Nivel 2, si aplicó):
{parent_contexts_json}

Emite tu evaluación formal: {{"verdict": "SUFFICIENT_FOR_SYNTHESIS|INSUFFICIENT_TRIGGER_BRANCH_B|NEGATIVE_REJECTION", "confidence_score": float, "negative_rejection_details": {{"reason": "out_of_thematic_scope_iso25964|classification_mismatch_lcc|total_absence_in_knowledge_graph|unsupported_technical_granularity", "closest_available_topics": [str], "formal_abstention_statement": str}}, "branch_b_instructions": {{"unresolved_subqueries": [str], "target_thesaurus_concepts": [str]}}}}"""

ANSWER_PROMPT = """Eres un asistente de conocimiento con estándares epistémicos estrictos. Responde la consulta del usuario usando SOLO la evidencia verificada que se te proporciona. Reglas: 1. Cada afirmación debe estar respaldada por un ítem de 'grounded_evidence'; cita el document_title y el bibtex_citation_key de su source_metadata. 2. Si existen tensiones epistémicas (epistemic_tensions), explícalas explícitamente SIN forzar una síntesis artificial ni descartar ninguna fuente. 3. Si la evidencia verificada es insuficiente para responder, dilo con claridad. 4. Responde en el idioma de la consulta.

Consulta del usuario: "{query}"

Evidencia verificada (grounded_evidence):
{grounded_evidence_json}

Tensiones epistémicas (epistemic_tensions):
{epistemic_tensions_json}"""

# Plantillas USER (runtime cuando hay artefacto compilado).
SYNTHESIS_USER = """Consulta del usuario: "{query}"

Fragmentos recuperados para análisis:
{candidate_chunks_json}

Devuelve el JSON: {{"query": str, "total_chunks_processed": int, "synthesized_facts": [{{"chunk_id": str, "document_id": str, "source_file": str, "relevance_level": "direct_answer|supporting_evidence|contextual_background|irrelevant", "atomic_summary": str, "verbatim_evidence": str, "academic_citations": [str]}}]}}"""

CONTRADICTION_USER = """Consulta: "{query}"

Hechos sintetizados:
{synthesized_facts_json}

Devuelve el JSON: {{"contradictions_detected": bool, "analysis_cases": [{{"conflict_type": "paradigmatic_theoretical_divergence|empirical_contextual_boundary|temporal_diachronic_shift|terminological_homonymy", "divergence_summary": str, "thesis_a": {{"proposition_id": str, "document_id": str, "claim": str, "author_or_framework": str, "empirical_context": str}}, "thesis_b": {{"proposition_id": str, "document_id": str, "claim": str, "author_or_framework": str, "empirical_context": str}}, "epistemic_reconciliation": str}}]}}"""

SUFFICIENCY_USER = """Consulta: "{query}"

Metadatos del Corpus Disponible (Descriptores ISO 25964 y LCC presentes en DB):
{active_corpus_metadata}

Proposiciones recuperadas (Nivel 1):
{synthesized_propositions_json}

Contextos escalados (Nivel 2, si aplicó):
{parent_contexts_json}

Emite tu evaluación formal: {{"verdict": "SUFFICIENT_FOR_SYNTHESIS|INSUFFICIENT_TRIGGER_BRANCH_B|NEGATIVE_REJECTION", "confidence_score": float, "negative_rejection_details": {{"reason": "out_of_thematic_scope_iso25964|classification_mismatch_lcc|total_absence_in_knowledge_graph|unsupported_technical_granularity", "closest_available_topics": [str], "formal_abstention_statement": str}}, "branch_b_instructions": {{"unresolved_subqueries": [str], "target_thesaurus_concepts": [str]}}}}"""

ANSWER_USER = """Consulta del usuario: "{query}"

Evidencia verificada (grounded_evidence):
{grounded_evidence_json}

Tensiones epistémicas (epistemic_tensions):
{epistemic_tensions_json}"""


# ---------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------


def _settings_retries(session) -> tuple[int, str | None]:
    """Lee llm_retries y fallback_model de session_settings (defaults seguros)."""
    try:
        settings = load_settings(session)
    except Exception:  # noqa: BLE001 — degradación natural
        settings = None
    if settings is None:
        return 3, None
    retries = int(getattr(settings, "llm_retries", 3) or 3)
    fallback = getattr(settings, "fallback_model", None)
    return retries, fallback


def _settings_models(session) -> tuple[str | None, str | None]:
    """Lee small_model y large_model de session_settings (defaults None)."""
    try:
        settings = load_settings(session)
    except Exception:  # noqa: BLE001 — degradación natural
        settings = None
    if settings is None:
        return None, None
    return getattr(settings, "small_model", None), getattr(
        settings, "large_model", None
    )


def _get_prompt_artifact(session, model_name: str, task_key: str):
    """Artefacto compilado activo o None (import lazy, degradación natural)."""
    try:
        from src.llm.compiler import get_active_prompt

        return get_active_prompt(session, model_name, task_key)
    except Exception:  # noqa: BLE001 — sin DB / compilador ausente
        return None


def _get_system_prompt(session, model_name: str, task_key: str, fallback: str) -> str:
    """System prompt desde el artefacto compilado, con fallback a la constante."""
    artifact = _get_prompt_artifact(session, model_name, task_key)
    if artifact is not None and artifact.prompt_text:
        return artifact.prompt_text
    return fallback


def _get_prompt_pair(
    session, model_name: str, task_key: str, system_fallback: str, user_fallback: str
) -> tuple[str, str]:
    """(system, user) desde el artefacto compilado, o los fallbacks actuales.

    El artefacto (0021) congela el SYSTEM renderizado en `prompt_text` y el
    USER template parametrizable en `user_template`. Sin artefacto (tests sin
    DB) devuelve las constantes actuales — comportamiento EXACTO de hoy.
    """
    artifact = _get_prompt_artifact(session, model_name, task_key)
    if artifact is not None and artifact.prompt_text and artifact.user_template:
        return artifact.prompt_text, artifact.user_template
    return system_fallback, user_fallback


def _json_dumps(obj) -> str:
    """json.dumps con ensure_ascii=False y fallback a repr si serializar falla."""
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return repr(obj)


def _fts_tsquery(terms: list[str]) -> str:
    """Construye un tsquery 'simple' con comillas DOBLES siempre.

    El `-` es el operador NOT de tsquery, así que cada término se escapa
    duplicando las comillas dobles y se envuelve en comillas dobles.
    """
    parts = []
    for term in terms:
        t = str(term).strip()
        if not t:
            continue
        t = t.replace('"', '""')
        parts.append(f'"{t}"')
    return " | ".join(parts)


# ---------------------------------------------------------------------
# 1. Recuperación proposicional (Nivel 1)
# ---------------------------------------------------------------------


def query_focused_proposition_extractor(
    session, query_embedding, top_propositions=15
) -> list[dict]:
    """Top-N proposiciones por similitud coseno (pgvector <=>).

    Solo documentos 'ready'. Devuelve lista de dicts con la proposición,
    el contexto de capítulo/documento y la distancia coseno.
    """
    if query_embedding is None:
        return []
    q = embedding_to_sql(query_embedding)
    rows = session.execute(
        text(
            "SELECT p.id AS prop_id, p.document_id, p.chapter_id, p.statement, "
            "p.text_span, p.char_start, p.char_end, p.citation_references, "
            "c.title AS chapter_title, d.title AS doc_title, d.bibtex, "
            "(p.embedding <=> CAST(:query_embedding AS vector)) AS cosine_distance "
            "FROM propositional_chunks p "
            "JOIN document_chapters c ON p.chapter_id = c.id "
            "JOIN documents d ON p.document_id = d.id "
            "WHERE d.status = 'ready' "
            "ORDER BY cosine_distance ASC "
            "LIMIT :top_propositions"
        ),
        {"query_embedding": q, "top_propositions": top_propositions},
    ).fetchall()
    out = []
    for r in rows:
        out.append(
            {
                "chunk_id": r.prop_id,
                "document_id": r.document_id,
                "chapter_id": r.chapter_id,
                "statement": r.statement,
                "text_span": r.text_span,
                "char_start": r.char_start,
                "char_end": r.char_end,
                "citation_references": r.citation_references or [],
                "chapter_title": r.chapter_title,
                "doc_title": r.doc_title,
                "bibtex": r.bibtex,
                "cosine_distance": float(r.cosine_distance),
            }
        )
    return out


# ---------------------------------------------------------------------
# 2. Escalado a contexto de capítulo (Nivel 2)
# ---------------------------------------------------------------------


def escalate_to_parent_context(session, chapter_ids) -> dict:
    """Resúmenes de capítulo para los ids dados. Devuelve {id: summary}."""
    if not chapter_ids:
        return {}
    rows = session.execute(
        text(
            "SELECT id, summary, line_start, line_end "
            "FROM document_chapters WHERE id = ANY(:ids)"
        ),
        {"ids": chapter_ids},
    ).fetchall()
    return {r.id: r.summary for r in rows}


# ---------------------------------------------------------------------
# 3. Agente de Síntesis
# ---------------------------------------------------------------------


def synthesize_chunks(session, query, chunks, verbose=False) -> list[dict]:
    """Sintetiza los chunks recuperados en hechos atómicos (LLM pequeño).

    Degradación: cada chunk con relevance_level='supporting_evidence',
    atomic_summary=statement truncado y verbatim_evidence=text_span.
    """
    if not chunks:
        return []
    retries, fallback = _settings_retries(session)
    small_model, _large_model = _settings_models(session)
    candidate_chunks_json = _json_dumps(chunks)
    system, user_template = _get_prompt_pair(
        session, small_model, TASK_SYNTHESIS, SYNTHESIS_SYSTEM_SHORT, SYNTHESIS_PROMPT
    )
    prompt = user_template.format(
        query=query, candidate_chunks_json=candidate_chunks_json
    )
    try:
        text_out, _model, _used_fallback = call_with_retries(
            session,
            prompt=prompt,
            system=system,
            model_size="small",
            response_format={"type": "json_object"},
            retries=retries,
            fallback_model=fallback,
        )
        data = parse_llm_output(text_out)
        facts = data.get("synthesized_facts") or []
        if isinstance(facts, list) and facts:
            if verbose:
                print(
                    f"[KAG-Agents] Síntesis: {len(facts)} hechos "
                    f"(fallback={_used_fallback})"
                )
            return facts
    except Exception as exc:  # noqa: BLE001 — degradación natural
        session.rollback()
        if verbose:
            print(f"[KAG-Agents] ⚠ Síntesis LLM falló ({exc}); degradando.")
    # Degradación determinista: cada chunk como supporting_evidence.
    degraded = []
    for c in chunks:
        statement = (c.get("statement") or "").strip()
        text_span = (c.get("text_span") or "").strip()
        degraded.append(
            {
                "chunk_id": str(c.get("chunk_id", "")),
                "document_id": str(c.get("document_id", "")),
                "source_file": str(c.get("doc_title", "")),
                "relevance_level": "supporting_evidence",
                "atomic_summary": statement[:500],
                "verbatim_evidence": text_span[:1000],
                "academic_citations": c.get("citation_references") or [],
            }
        )
    if verbose:
        print(f"[KAG-Agents] Síntesis degradada: {len(degraded)} hechos.")
    return degraded


# ---------------------------------------------------------------------
# 4. Resolución de contradicciones
# ---------------------------------------------------------------------


def resolve_contradictions(session, query, synthesized_facts, verbose=False) -> dict:
    """Tipifica contradicciones entre hechos sintetizados (LLM pequeño).

    Degradación: contradictions_detected=False, analysis_cases=[].
    """
    if not synthesized_facts:
        return {"contradictions_detected": False, "analysis_cases": []}
    retries, fallback = _settings_retries(session)
    small_model, _large_model = _settings_models(session)
    system, user_template = _get_prompt_pair(
        session,
        small_model,
        TASK_CONTRADICTIONS,
        CONTRADICTION_SYSTEM_SHORT,
        CONTRADICTION_PROMPT,
    )
    prompt = user_template.format(
        query=query, synthesized_facts_json=_json_dumps(synthesized_facts)
    )
    try:
        text_out, _model, _used_fallback = call_with_retries(
            session,
            prompt=prompt,
            system=system,
            model_size="small",
            response_format={"type": "json_object"},
            retries=retries,
            fallback_model=fallback,
        )
        data = parse_llm_output(text_out)
        report = {
            "contradictions_detected": bool(data.get("contradictions_detected", False)),
            "analysis_cases": data.get("analysis_cases") or [],
        }
        if verbose:
            print(
                f"[KAG-Agents] Contradicciones: "
                f"{report['contradictions_detected']} "
                f"({len(report['analysis_cases'])} casos, fallback={_used_fallback})"
            )
        return report
    except Exception as exc:  # noqa: BLE001 — degradación natural
        session.rollback()
        if verbose:
            print(f"[KAG-Agents] ⚠ Análisis de contradicciones falló ({exc}).")
        return {"contradictions_detected": False, "analysis_cases": []}


# ---------------------------------------------------------------------
# 5. Auditoría de suficiencia (Agente Auditor Epistemológico)
# ---------------------------------------------------------------------


def evaluate_sufficiency(
    session, query, synthesized_facts, corpus_metadata, verbose=False
) -> dict:
    """Dictamina si la recuperación basta para responder (LLM pequeño).

    Degradación: verdict='SUFFICIENT_FOR_SYNTHESIS' si hay ≥1 fact con
    relevance_level direct_answer/supporting_evidence; si no,
    'INSUFFICIENT_TRIGGER_BRANCH_B'.
    """
    retries, fallback = _settings_retries(session)
    small_model, _large_model = _settings_models(session)
    system, user_template = _get_prompt_pair(
        session,
        small_model,
        TASK_SUFFICIENCY,
        SUFFICIENCY_SYSTEM_SHORT,
        SUFFICIENCY_PROMPT,
    )
    prompt = user_template.format(
        query=query,
        active_corpus_metadata=_json_dumps(corpus_metadata),
        synthesized_propositions_json=_json_dumps(synthesized_facts),
        parent_contexts_json="[]",
    )
    try:
        text_out, _model, _used_fallback = call_with_retries(
            session,
            prompt=prompt,
            system=system,
            model_size="small",
            response_format={"type": "json_object"},
            retries=retries,
            fallback_model=fallback,
        )
        data = parse_llm_output(text_out)
        verdict = data.get("verdict")
        if verdict in (
            "SUFFICIENT_FOR_SYNTHESIS",
            "INSUFFICIENT_TRIGGER_BRANCH_B",
            "NEGATIVE_REJECTION",
        ):
            if verbose:
                print(
                    f"[KAG-Agents] Suficiencia: {verdict} "
                    f"(confianza={data.get('confidence_score')}, fallback={_used_fallback})"
                )
            return data
    except Exception as exc:  # noqa: BLE001 — degradación natural
        session.rollback()
        if verbose:
            print(f"[KAG-Agents] ⚠ Auditoría de suficiencia falló ({exc}).")
    # Degradación determinista.
    relevant = [
        f
        for f in synthesized_facts
        if f.get("relevance_level") in ("direct_answer", "supporting_evidence")
    ]
    verdict = (
        "SUFFICIENT_FOR_SYNTHESIS" if relevant else "INSUFFICIENT_TRIGGER_BRANCH_B"
    )
    if verbose:
        print(f"[KAG-Agents] Suficiencia degradada: {verdict}.")
    return {"verdict": verdict, "confidence_score": 0.5}


# ---------------------------------------------------------------------
# 6. Branch B: expansión de la recuperación
# ---------------------------------------------------------------------


class BranchBOrchestrator:
    """Expansión Branch B: nuevos documentos, expansión de grafo y
    subconsultas ortogonales FTS. Devuelve chunks adicionales deduplicados.

    Nota: kag_relations NO tiene columna `weight` (verificado en
    alembic/versions/0013_kag.py: solo id, doc_id, chunk_id,
    source_entity_id, target_entity_id, relation_type, description,
    created_at). Se ordena por r.id DESC (relaciones más recientes).
    """

    def __init__(self, session, max_iterations=2, verbose=False):
        self.session = session
        self.max_iterations = max_iterations
        self._verbose = verbose

    def execute_expansion(self, current_state, evaluation) -> list[dict]:
        """Ejecuta una iteración de expansión. Devuelve chunks nuevos (dedup)."""
        iteration = int(current_state.get("iteration", 0) or 0)
        if iteration > self.max_iterations:
            return []
        visited_docs = current_state.get("visited_docs") or []
        visited_chunks = current_state.get("visited_chunks") or []
        entity_ids = current_state.get("entity_ids") or []
        branch_b = (evaluation or {}).get("branch_b_instructions") or {}
        unresolved = branch_b.get("unresolved_subqueries") or []
        new_chunks: list[dict] = []
        seen: set = set(visited_chunks)

        # (a) Nuevos documentos por ámbito temático (ISO 25964 / scope_thematic).
        terms = branch_b.get("target_thesaurus_concepts") or []
        if terms:
            try:
                terms_jsonb = json.dumps(terms, ensure_ascii=False)
                rows = self.session.execute(
                    text(
                        "SELECT id, title FROM documents "
                        "WHERE id != ALL(:visited_docs) "
                        "AND (thematic_areas_iso25964 @> :terms_jsonb "
                        "OR scope_thematic ILIKE ANY(:terms)) LIMIT 3"
                    ),
                    {
                        "visited_docs": visited_docs,
                        "terms_jsonb": terms_jsonb,
                        "terms": terms,
                    },
                ).fetchall()
                new_doc_ids = [r.id for r in rows]
                if new_doc_ids:
                    visited_docs = list(dict.fromkeys(visited_docs + new_doc_ids))
            except Exception as exc:  # noqa: BLE001 — degradación natural
                self.session.rollback()
                print(f"[KAG-Agents] ⚠ Branch B (a) falló: {exc}")
                new_doc_ids = []

        # (b) Expansión de grafo: vecinos de las entidades semilla.
        if entity_ids:
            try:
                rows = self.session.execute(
                    text(
                        "SELECT r.target_entity_id, e.name "
                        "FROM kag_relations r "
                        "JOIN kag_entities e ON r.target_entity_id = e.id "
                        "WHERE r.source_entity_id = ANY(:entity_ids) "
                        "ORDER BY r.id DESC LIMIT 10"
                    ),
                    {"entity_ids": entity_ids},
                ).fetchall()
                neighbor_names = [r.name for r in rows]
                if neighbor_names:
                    terms = list(dict.fromkeys(terms + neighbor_names))
            except Exception as exc:  # noqa: BLE001 — degradación natural
                self.session.rollback()
                print(f"[KAG-Agents] ⚠ Branch B (b) falló: {exc}")

        # (c) Subconsultas ortogonales: FTS sobre propositional_chunks.
        subqueries = list(unresolved) + terms
        for sub in subqueries:
            if not sub or not str(sub).strip():
                continue
            tokens = [t for t in re.split(r"[\s,;]+", str(sub).strip()) if t]
            if not tokens:
                continue
            tsq = _fts_tsquery(tokens)
            try:
                rows = self.session.execute(
                    text(
                        "SELECT p.id AS chunk_id, p.statement, p.document_id, "
                        "ts_rank_cd(to_tsvector('simple', p.statement), "
                        "to_tsquery('simple', :tsq)) AS score "
                        "FROM propositional_chunks p "
                        "JOIN documents d ON p.document_id = d.id "
                        "WHERE d.status = 'ready' "
                        "AND to_tsvector('simple', p.statement) @@ "
                        "to_tsquery('simple', :tsq) "
                        "ORDER BY score DESC LIMIT 5"
                    ),
                    {"tsq": tsq},
                ).fetchall()
            except Exception as exc:  # noqa: BLE001 — degradación natural
                self.session.rollback()
                if hasattr(self, "_verbose") and self._verbose:
                    print(f"[KAG-Agents] ⚠ Branch B (c) FTS falló: {exc}")
                continue
            for r in rows:
                if r.chunk_id in seen:
                    continue
                seen.add(r.chunk_id)
                new_chunks.append(
                    {
                        "chunk_id": r.chunk_id,
                        "statement": r.statement,
                        "document_id": r.document_id,
                        "score": float(r.score),
                    }
                )
        return new_chunks


# ---------------------------------------------------------------------
# 7. Verificación de grounding verbatim
# ---------------------------------------------------------------------


def verify_claim_grounding(session, claim, claimed_chunk_id, claimed_verbatim) -> dict:
    """Verifica que una cita verbatim exista en el chunk reclamado.

    Usa rapidfuzz.fuzz.partial_ratio (umbral 95.0) si la cita no es
    substring exacto del text_span en DB.
    """
    row = session.execute(
        text(
            "SELECT text_span, char_start, char_end, citation_references, "
            "document_id FROM propositional_chunks WHERE id = :cid"
        ),
        {"cid": claimed_chunk_id},
    ).fetchone()
    if not row:
        return {"status": "UNGROUNDED_INVALID_CHUNK_ID", "verified": False}
    db_span = row.text_span or ""
    claimed_verbatim = claimed_verbatim or ""
    if claimed_verbatim in db_span:
        match_score = 100.0
    else:
        match_score = float(fuzz.partial_ratio(claimed_verbatim, db_span))
    is_valid = match_score >= 95.0
    return {
        "status": "VERIFIED" if is_valid else "FAILED_VERBATIM_MISMATCH",
        "verified": is_valid,
        "match_score": match_score,
        "char_start": row.char_start,
        "char_end": row.char_end,
        "academic_citations": row.citation_references or [],
        "document_id": row.document_id,
    }


# ---------------------------------------------------------------------
# 8. Ensamblaje del contexto final
# ---------------------------------------------------------------------


def assemble_final_context(query, grounded_evidence, epistemic_tensions) -> dict:
    """Ensambla el contexto final con evidencia verificada y tensiones."""
    evidence = []
    for ev in grounded_evidence:
        evidence.append(
            {
                "claim_id": ev.get("claim_id"),
                "verified_fact": ev.get("verified_fact"),
                "verbatim_quote": ev.get("verbatim_quote"),
                "source_metadata": {
                    "document_title": ev.get("document_title"),
                    "chapter_title": ev.get("chapter_title"),
                    "bibtex_citation_key": ev.get("bibtex_citation_key"),
                },
            }
        )
    tensions = []
    for t in epistemic_tensions or []:
        tensions.append(
            {
                "tension_label": t.get("tension_label"),
                "divergence_description": t.get("divergence_description"),
                "framework_a": t.get("framework_a"),
                "framework_b": t.get("framework_b"),
            }
        )
    return {
        "query": query,
        "grounded_evidence": evidence,
        "epistemic_tensions": tensions,
    }


# ---------------------------------------------------------------------
# 9. Orquestador completo
# ---------------------------------------------------------------------


def ask_propositional(
    session, query, top_propositions=15, max_iterations=2, verbose=False
) -> dict:
    """Flujo proposicional completo (§3 del diseño).

    Devuelve dict: {"answer", "verdict", "grounded_evidence",
    "epistemic_tensions", "used_fallback"}.
    """
    if verbose:
        print(f"\n🔎 [KAG-Agents] Pregunta proposicional: {query}")

    # 1. Embedding de la query (input_type='query', igual que kag_query.ask).
    query_embedding = None
    try:
        from src.embeddings import embed_text

        query_embedding = embed_text(query, input_type="query")
    except Exception as exc:  # noqa: BLE001 — degradación natural
        session.rollback()
        if verbose:
            print(f"[KAG-Agents] ⚠ Embeddings no disponibles ({exc}).")

    # 2. Recuperación proposicional (Nivel 1).
    chunks = query_focused_proposition_extractor(
        session, query_embedding, top_propositions=top_propositions
    )
    if verbose:
        print(f"[KAG-Agents] Nivel 1: {len(chunks)} proposiciones recuperadas.")

    # 3. Síntesis fáctica.
    facts = synthesize_chunks(session, query, chunks, verbose=verbose)

    # 4. Resolución de contradicciones.
    contradiction_report = resolve_contradictions(
        session, query, facts, verbose=verbose
    )

    # 5. Auditoría de suficiencia + Branch B (hasta max_iterations).
    corpus_metadata = []
    try:
        rows = session.execute(
            text(
                "SELECT title, thematic_areas_iso25964, library_of_congress "
                "FROM documents WHERE status = 'ready' LIMIT 20"
            )
        ).fetchall()
        corpus_metadata = [
            {
                "title": r.title,
                "thematic_areas_iso25964": r.thematic_areas_iso25964 or [],
                "library_of_congress": r.library_of_congress or {},
            }
            for r in rows
        ]
    except Exception as exc:  # noqa: BLE001 — degradación natural
        session.rollback()
        if verbose:
            print(f"[KAG-Agents] ⚠ Metadatos de corpus fallaron ({exc}).")

    evaluation = evaluate_sufficiency(
        session, query, facts, corpus_metadata, verbose=verbose
    )
    verdict = evaluation.get("verdict", "SUFFICIENT_FOR_SYNTHESIS")

    if verdict == "NEGATIVE_REJECTION":
        details = evaluation.get("negative_rejection_details") or {}
        abstention = details.get("formal_abstention_statement") or (
            "El corpus disponible no cubre el dominio conceptual requerido "
            "por la consulta (abstención temprana)."
        )
        if verbose:
            print(f"[KAG-Agents] ⛔ Abstención formal: {abstention}")
        return {
            "answer": abstention,
            "verdict": verdict,
            "grounded_evidence": [],
            "epistemic_tensions": [],
            "used_fallback": False,
        }

    orchestrator = BranchBOrchestrator(session, max_iterations=max_iterations)
    state = {
        "iteration": 0,
        "visited_docs": list(
            {c.get("document_id") for c in chunks if c.get("document_id")}
        ),
        "visited_chunks": [c.get("chunk_id") for c in chunks if c.get("chunk_id")],
        "entity_ids": [],
    }
    while (
        verdict == "INSUFFICIENT_TRIGGER_BRANCH_B"
        and state["iteration"] <= max_iterations
    ):
        if verbose:
            print(
                f"[KAG-Agents] Branch B iteración {state['iteration']} "
                f"(max {max_iterations})."
            )
        extra = orchestrator.execute_expansion(state, evaluation)
        if not extra:
            break
        # Re-sintetizar los chunks nuevos y re-evaluar.
        extra_facts = synthesize_chunks(session, query, extra, verbose=verbose)
        facts = list(facts) + extra_facts
        contradiction_report = resolve_contradictions(
            session, query, facts, verbose=verbose
        )
        evaluation = evaluate_sufficiency(
            session, query, facts, corpus_metadata, verbose=verbose
        )
        verdict = evaluation.get("verdict", "SUFFICIENT_FOR_SYNTHESIS")
        state["iteration"] += 1
        state["visited_chunks"] = list(
            dict.fromkeys(state["visited_chunks"] + [c.get("chunk_id") for c in extra])
        )

    # 6. Verificación de grounding de cada fact relevante.
    grounded_evidence = []
    for f in facts:
        if f.get("relevance_level") not in ("direct_answer", "supporting_evidence"):
            continue
        chunk_id = f.get("chunk_id")
        verbatim = f.get("verbatim_evidence") or ""
        if not chunk_id or not verbatim:
            continue
        try:
            check = verify_claim_grounding(session, query, chunk_id, verbatim)
        except Exception as exc:  # noqa: BLE001 — degradación natural
            session.rollback()
            if verbose:
                print(f"[KAG-Agents] ⚠ Grounding falló para chunk {chunk_id}: {exc}")
            continue
        if not check.get("verified"):
            if verbose:
                print(
                    f"[KAG-Agents] Grounding rechazado: chunk {chunk_id} "
                    f"(score {check.get('match_score'):.1f})."
                )
            continue
        # Enriquecer con metadatos del chunk (título de doc/capítulo, bibtex).
        meta = {}
        try:
            row = session.execute(
                text(
                    "SELECT d.id AS doc_id, d.title AS doc_title, d.bibtex, "
                    "c.title AS chapter_title "
                    "FROM propositional_chunks p "
                    "JOIN documents d ON p.document_id = d.id "
                    "JOIN document_chapters c ON p.chapter_id = c.id "
                    "WHERE p.id = :cid"
                ),
                {"cid": chunk_id},
            ).fetchone()
            if row:
                meta = {
                    "document_id": row.doc_id,
                    "document_title": row.doc_title,
                    "chapter_title": row.chapter_title,
                    "bibtex_citation_key": row.bibtex,
                }
        except Exception as exc:  # noqa: BLE001 — degradación natural
            session.rollback()
            if verbose:
                print(f"[KAG-Agents] ⚠ Metadatos de chunk {chunk_id} fallaron: {exc}")
        grounded_evidence.append(
            {
                "claim_id": str(chunk_id),
                "verified_fact": f.get("atomic_summary") or f.get("statement", ""),
                "verbatim_quote": verbatim,
                "document_id": meta.get("document_id"),
                "document_title": meta.get("document_title"),
                "chapter_title": meta.get("chapter_title"),
                "bibtex_citation_key": meta.get("bibtex_citation_key"),
            }
        )

    # 7. Ensamblar contexto final.
    tensions = []
    for case in contradiction_report.get("analysis_cases") or []:
        tensions.append(
            {
                "tension_label": case.get(
                    "conflict_type", "paradigmatic_theoretical_divergence"
                ),
                "divergence_description": case.get("divergence_summary", ""),
                "framework_a": (case.get("thesis_a") or {}).get(
                    "author_or_framework", ""
                ),
                "framework_b": (case.get("thesis_b") or {}).get(
                    "author_or_framework", ""
                ),
            }
        )
    final_context = assemble_final_context(query, grounded_evidence, tensions)

    # 8. Respuesta final con el LLM grande.
    retries, fallback = _settings_retries(session)
    _small_model, large_model = _settings_models(session)
    used_fallback = False
    system, user_template = _get_prompt_pair(
        session, large_model, TASK_ANSWER, ANSWER_SYSTEM_SHORT, ANSWER_PROMPT
    )
    prompt = user_template.format(
        query=query,
        grounded_evidence_json=_json_dumps(final_context["grounded_evidence"]),
        epistemic_tensions_json=_json_dumps(final_context["epistemic_tensions"]),
    )
    try:
        text_out, _model, used_fallback = call_with_retries(
            session,
            prompt=prompt,
            system=system,
            model_size="large",
            response_format=None,
            retries=retries,
            fallback_model=fallback,
        )
        answer = text_out
    except Exception as exc:  # noqa: BLE001 — degradación: contexto crudo
        session.rollback()
        if verbose:
            print(f"[KAG-Agents] ⚠ Respuesta LLM falló ({exc}); contexto crudo.")
        answer = _json_dumps(final_context)
        used_fallback = True

    if verbose:
        print(f"[KAG-Agents] Verdict: {verdict} | Evidencia: {len(grounded_evidence)}")
    return {
        "answer": answer,
        "verdict": verdict,
        "grounded_evidence": grounded_evidence,
        "epistemic_tensions": tensions,
        "used_fallback": used_fallback,
    }
