"""
Seed de las specs de prompts del pipeline KAG (prompt-as-code).

Inserta/actualiza en `prompt_templates` las 16 specs agnósticas de los
prompts del sistema KAG (ingesta proposicional, agentes query-time, query
clásica e ingesta clásica). Cada spec captura el rol (intent) y las
instrucciones (rules) del SYSTEM prompt actual; el USER prompt (con sus
placeholders) se construye en runtime y NO vive en la spec.

A diferencia de src/db/seed_ai.py / seed_vision.py, NO requiere variables de
entorno: es datos puros. Tampoco compila artefactos — eso lo hace el
compilador una vez que los modelos y specs existen:

    python -m src.db.seed_kag_prompts   # inserta/actualiza las specs
    python -m src.llm.compile_prompts   # transpila specs -> prompt_artifacts

Uso:
    python -m src.db.seed_kag_prompts
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from sqlalchemy import select

from src.db.models import PromptTemplate

# ---------------------------------------------------------------------
# Specs agnósticas de prompts KAG (tabla prompt_templates)
# ---------------------------------------------------------------------

KAG_TEMPLATES = [
    # --- A. src/kag_propositional.py -----------------------------------
    {
        "task_key": "kag_metadata",
        "version": "1.0",
        "intent": (
            "Eres un indexador bibliografico especializado en Ciencias Sociales "
            "y normas de documentacion formal (ISO 25964 y Library of Congress). "
            "Tu rol es extraer los metadatos estructurales del documento "
            "delimitado por las lineas indicadas."
        ),
        "rules": [
            (
                "Normalizacion de Tesauros (ISO 25964): proporciona un minimo de "
                "3 tematicas principales con preferred_term, non_preferred_terms, "
                "scope_note_disambiguation, broader_term, narrower_term y "
                "related_terms"
            ),
            (
                "Clasificacion de la Biblioteca del Congreso (LCC y LCSH): "
                "proporciona las materias LCSH con su codigo LCC representativo"
            ),
            "Registro BibTeX formal del documento",
            "Salida JSON estricta con el schema indicado",
        ],
        "input_schema": {
            "source_file": "string",
            "document_id": "string",
            "line_start": "integer",
            "line_end": "integer",
            "document_head_snippet": "string",
        },
        "output_schema": {
            "source_file": "string",
            "document_id": "string",
            "title": "string",
            "technical_level": "introductory|intermediate|advanced|research",
            "bibtex": "string",
            "thematic_areas_iso25964": "array",
            "library_of_congress": "object",
            "key_entities": "array",
        },
        "few_shot": [],
    },
    {
        "task_key": "kag_chapters",
        "version": "1.0",
        "intent": (
            "Eres un parser de estructura textual. Analizas el archivo Markdown "
            "provisto y extraes la segmentacion completa de capitulos para el "
            "documento especificado."
        ),
        "rules": [
            "Cada capitulo debe contener el numero exacto de linea de inicio y fin",
            ("La salida debe reflejar la jerarquia: archivo -> documento -> capitulos"),
            "Identificar los titulos de capitulo (#, ## o mayusculas canonicas)",
            (
                "El line_end del capitulo N debe ser la linea inmediatamente "
                "anterior al line_start del capitulo N+1; para el ultimo, el "
                "line_end global del documento"
            ),
            (
                "Si existen subsecciones relevantes, mapear sus lineas sin quebrar "
                "los rangos del capitulo contenedor"
            ),
            "Salida JSON estricta con el schema indicado",
        ],
        "input_schema": {
            "source_file": "string",
            "document_id": "string",
            "line_start": "integer",
            "line_end": "integer",
            "numbered_text_block": "string",
        },
        "output_schema": {
            "source_file": "string",
            "document_id": "string",
            "total_chapters": "integer",
            "chapters": "array",
        },
        "few_shot": [],
    },
    {
        "task_key": "kag_document_analysis",
        "version": "1.0",
        "intent": (
            "Eres un indexador bibliografico especializado en Ciencias Sociales "
            "y un parser estructural de documentos academicos. Tu rol es producir, "
            "en UNA sola pasada, el analisis documental completo del documento "
            "delimitado por las lineas indicadas."
        ),
        "rules": [
            (
                "Ficha bibliografica: title, technical_level "
                "(introductory|intermediate|advanced|research) y bibtex (registro "
                "BibTeX formal)"
            ),
            (
                "Normalizacion de Tesauros (ISO 25964): minimo de 3 tematicas "
                "principales con preferred_term, non_preferred_terms, "
                "scope_note_disambiguation, broader_term, narrower_term y "
                "related_terms"
            ),
            (
                "Clasificacion de la Biblioteca del Congreso (LCC y LCSH): materias "
                "LCSH con su codigo LCC representativo"
            ),
            (
                "Resumen Ejecutivo Global: summary, sintesis de 2-4 oraciones de la "
                "tesis central y la progresion argumental de TODO el documento"
            ),
            (
                "Estructura de capitulos: chapters[] con chapter_index, title, "
                "line_start, line_end, main_theme, summary y subsections[]; el "
                "line_end del capitulo N es la linea anterior al line_start del "
                "N+1 y el ultimo usa el line_end global"
            ),
            (
                "Entidades Rectoras: key_entities, entre 15 y 30 conceptos "
                "ontologicos nucleares del documento"
            ),
            (
                "La salida debe ser un unico objeto JSON atomico que combine ficha, "
                "tematicas, clasificacion, resumen global, capitulos y entidades"
            ),
        ],
        "input_schema": {
            "source_file": "string",
            "document_id": "string",
            "line_start": "integer",
            "line_end": "integer",
            "document_skeleton": "string",
        },
        "output_schema": {
            "source_file": "string",
            "document_id": "string",
            "title": "string",
            "technical_level": "introductory|intermediate|advanced|research",
            "bibtex": "string",
            "thematic_areas_iso25964": "array",
            "library_of_congress": "object",
            "key_entities": "array",
            "summary": "string",
            "chapters": "array",
        },
        "few_shot": [],
    },
    {
        "task_key": "kag_propositional_chunking",
        "version": "1.0",
        "intent": (
            "Eres un analista de epistemologia y analisis del discurso. Tu objetivo "
            "es realizar una extraccion proposicional jerarquica sobre el texto de "
            "un capitulo."
        ),
        "rules": [
            (
                "Nivel 1 - Ideas Centrales (Central Claims): formular las tesis "
                "teoricas de alto nivel defendidas en el texto"
            ),
            (
                "Nivel 2 - Argumentos (Supporting Arguments): identificar las "
                "premisas logicas, pruebas empiricas o deducciones conceptuales "
                "que sustentan cada tesis"
            ),
            (
                "Nivel 3 - Proposiciones Atomicas (Chunks): descomponer cada "
                "argumento en proposiciones elementales gramaticalmente "
                "independientes y autocontenidas, reemplazando anforas por el "
                "sujeto explicito"
            ),
            ("verbatim_span debe contener el fragmento de texto exacto del original"),
            (
                "REGLA DE REFERENCIAS DUPLICADAS: si una frase contiene una "
                "referencia academica, debe preservarse y duplicarse en "
                "citations_references de TODAS las proposiciones atomicas que "
                "deriven de ella; prohibido descartar o separar citas bibliograficas"
            ),
            "Salida JSON estricta con el schema indicado",
        ],
        "input_schema": {
            "source_file": "string",
            "document_id": "string",
            "chapter_index": "integer",
            "chapter_title": "string",
            "line_start": "integer",
            "line_end": "integer",
            "chapter_text_content": "string",
        },
        "output_schema": {
            "source_file": "string",
            "document_id": "string",
            "chapter_index": "integer",
            "chapter_title": "string",
            "core_ideas": "array",
        },
        "few_shot": [],
    },
    {
        "task_key": "kag_topic_label",
        "version": "1.0",
        "intent": (
            "Eres un analista bibliometrico y de modelado de topicos. Se te presenta "
            "una secuencia temporal ordenada de proposiciones atomicas agrupadas "
            "mediante agrupamiento aglomerativo secuencial (c-TF-IDF). Tu funcion "
            "es sintetizar el significado del grupo y asignarle una etiqueta "
            "tematica precisa que preserve la progresion del texto original."
        ),
        "rules": [
            (
                "macro_phase_label: asignar un titulo representativo y conciso que "
                "describa la funcion del bloque tematico dentro de la obra"
            ),
            (
                "epistemic_summary: resumir en dos oraciones la tesis o progresion "
                "conceptual central del rango de proposiciones"
            ),
            (
                "Conservar estrictamente los identificadores de chunk de inicio y "
                "fin provistos"
            ),
            "Salida JSON estricta con el schema indicado",
        ],
        "input_schema": {
            "source_file": "string",
            "document_id": "string",
            "sequential_order": "integer",
            "top_ctfidf_keywords": "string",
            "start_chunk_id": "string",
            "end_chunk_id": "string",
            "cluster_statements_text": "string",
        },
        "output_schema": {
            "source_file": "string",
            "document_id": "string",
            "sequential_order": "integer",
            "macro_phase_label": "string",
            "representative_keywords": "array",
            "start_chunk_id": "string",
            "end_chunk_id": "string",
            "epistemic_summary": "string",
        },
        "few_shot": [],
    },
    {
        "task_key": "kag_vision_analysis",
        "version": "1.0",
        "intent": (
            "Eres un asistente de investigacion visual especializado en analisis de "
            "diagramas cientificos, graficos metodologicos y esquemas teoricos. Tu "
            "objetivo es analizar la imagen suministrada junto al contexto textual "
            "donde fue citada dentro del documento."
        ),
        "rules": [
            (
                "dense_visual_description: transcribir todo texto, etiquetas de "
                "ejes, valores numericos, leyendas y flujos de cajas o flechas "
                "visibles; desglosar los componentes exactos sin generalizaciones"
            ),
            (
                "epistemic_contribution: explicar con rigor que fenomeno o "
                "mecanismo teorico/metodologico formaliza o comprueba la imagen"
            ),
            (
                "faq_indexing: generar entre 3 y 5 preguntas explicitas cuya "
                "respuesta este contenida visualmente en la imagen (Reverse HyDE), "
                "incluyendo las variables y relaciones exactas representadas"
            ),
            "Extraer las entidades clave directamente referenciadas en el grafico",
            "Salida JSON estricta con el schema indicado",
        ],
        "input_schema": {
            "document_id": "string",
            "anchor_line": "integer",
            "markdown_tag": "string",
            "surrounding_text_context": "string",
        },
        "output_schema": {
            "image_id": "string",
            "document_id": "string",
            "file_path": "string",
            "anchor_line": "integer",
            "caption": "string",
            "image_type": "diagram|chart_or_plot|flowchart|conceptual_illustration|screenshot|table_image|photograph",
            "dense_visual_description": "string",
            "epistemic_contribution": "string",
            "faq_indexing": "array",
            "associated_entities": "array",
        },
        "few_shot": [],
    },
    # --- B. src/kag_agents.py -------------------------------------------
    {
        "task_key": "kag_synthesis",
        "version": "1.0",
        "intent": (
            "Eres un agente de consolidacion factual de alta precision. Tu tarea es "
            "procesar un conjunto de fragmentos proposicionales recuperados por el "
            "motor de busqueda y extraer exclusivamente los hechos que aportan a la "
            "consulta del usuario."
        ),
        "rules": [
            "Para cada chunk, evaluar su pertinencia respecto a la consulta formulada",
            (
                "Si el chunk es irrelevante, marcar relevance_level 'irrelevant' y "
                "dejar atomic_summary y verbatim_evidence vacios"
            ),
            (
                "Si el chunk contiene informacion relevante: atomic_summary "
                "sintetiza el argumento o dato clave eliminando relleno y "
                "ambiguedades anafoticas"
            ),
            (
                "verbatim_evidence: copiar una frase textual literal del fragmento "
                "como evidencia irrefutable; prohibido alterar o parafrasear la cita"
            ),
            (
                "academic_citations: mantener todas las referencias academicas "
                "explicitas vinculadas al dato"
            ),
            "Salida JSON estricta con el schema indicado",
        ],
        "input_schema": {
            "query": "string",
            "candidate_chunks_json": "string",
        },
        "output_schema": {
            "query": "string",
            "total_chunks_processed": "integer",
            "synthesized_facts": "array",
        },
        "few_shot": [],
    },
    {
        "task_key": "kag_contradictions",
        "version": "1.0",
        "intent": (
            "Eres un analista epistemologico. En ciencias sociales las "
            "contradicciones rara vez son errores facticos; suelen representar "
            "tensiones paradigmaticas, condiciones de contorno empiricas divergentes "
            "o evoluciones diacronicas. No debes forzar una sintesis artificial ni "
            "descartar fuentes; debes tipificar y explicitar la divergencia."
        ),
        "rules": [
            (
                "Tipologia: paradigmatic_theoretical_divergence (desacuerdo "
                "estructural entre escuelas), empirical_contextual_boundary "
                "(resultados opuestos por unidad de analisis/geografia/muestreo), "
                "temporal_diachronic_shift (un autor rectifica un postulado previo) "
                "y terminological_homonymy (ambiguedad conceptual resuelta por la "
                "nota de alcance ISO 25964)"
            ),
            "No forzar una sintesis artificial ni descartar fuentes",
            "Explicitar la divergencia con su tipologia y reconciliacion epistemica",
            "Salida JSON estricta con el schema indicado",
        ],
        "input_schema": {
            "query": "string",
            "synthesized_facts_json": "string",
        },
        "output_schema": {
            "contradictions_detected": "boolean",
            "analysis_cases": "array",
        },
        "few_shot": [],
    },
    {
        "task_key": "kag_sufficiency",
        "version": "1.0",
        "intent": (
            "Eres el Agente Auditor Epistemologico de un sistema de recuperacion "
            "avanzada. Tu funcion es dictaminar con imparcialidad si el conjunto de "
            "proposiciones y argumentos recuperados basta para responder con rigor "
            "analitico la consulta formulada, o si se debe ejecutar una de dos "
            "acciones de control."
        ),
        "rules": [
            (
                "SUFFICIENT_FOR_SYNTHESIS: toda premisa necesaria para responder la "
                "consulta esta respaldada por proposiciones verificables y las "
                "contradicciones estan tipificadas satisfactoriamente"
            ),
            (
                "NEGATIVE_REJECTION (Abstencion Temprana Obligatoria): activarla si "
                "la consulta colisiona con el ambito tematico (ISO 25964) y las "
                "clasificaciones Library of Congress de los documentos disponibles; "
                "no iterar busquedas adicionales y generar una abstencion formal "
                "fundamentada"
            ),
            (
                "INSUFFICIENT_TRIGGER_BRANCH_B: activarla si el corpus cubre la "
                "tematica general pero la recuperacion actual omitio vinculos "
                "causales intermedios, evidencia de apoyo o nodos relacionales "
                "especificos"
            ),
            "Salida JSON estricta con el schema indicado",
        ],
        "input_schema": {
            "query": "string",
            "active_corpus_metadata": "string",
            "synthesized_propositions_json": "string",
            "parent_contexts_json": "string",
        },
        "output_schema": {
            "verdict": "SUFFICIENT_FOR_SYNTHESIS|INSUFFICIENT_TRIGGER_BRANCH_B|NEGATIVE_REJECTION",
            "confidence_score": "float",
            "negative_rejection_details": "object",
            "branch_b_instructions": "object",
        },
        "few_shot": [],
    },
    {
        "task_key": "kag_answer",
        "version": "1.0",
        "intent": (
            "Eres un asistente de conocimiento con estandares epistemicos estrictos. "
            "Responde la consulta del usuario usando SOLO la evidencia verificada "
            "que se te proporciona."
        ),
        "rules": [
            (
                "Cada afirmacion debe estar respaldada por un item de "
                "grounded_evidence; citar el document_title y el "
                "bibtex_citation_key de su source_metadata"
            ),
            (
                "Si existen tensiones epistemicas (epistemic_tensions), explicarlas "
                "explicitamente SIN forzar una sintesis artificial ni descartar "
                "ninguna fuente"
            ),
            "Si la evidencia verificada es insuficiente para responder, decirlo con claridad",
            "Responder en el idioma de la consulta",
        ],
        "input_schema": {
            "query": "string",
            "grounded_evidence_json": "string",
            "epistemic_tensions_json": "string",
        },
        "output_schema": {
            "answer": "string",
        },
        "few_shot": [],
    },
    # --- C. src/kag_query.py --------------------------------------------
    {
        "task_key": "kag_grounded_entities",
        "version": "1.0",
        "intent": (
            "Eres un selector de entidades. Seleccionas las entidades canonicas del "
            "grafo de conocimiento SOLO entre los candidatos provistos que se "
            "mencionan en la pregunta."
        ),
        "rules": [
            "Elegir SOLO de la lista de candidatas provista",
            "Si ninguna candidata se menciona en la pregunta, devolver entities vacio",
            "Salida JSON estricta con el schema indicado",
        ],
        "input_schema": {
            "candidates": "string",
            "query": "string",
        },
        "output_schema": {
            "entities": "array",
        },
        "few_shot": [],
    },
    {
        "task_key": "kag_critic_regex",
        "version": "1.0",
        "intent": (
            "Eres un critico de busqueda. Dada una pregunta, decides si contiene "
            "terminos EXACTOS que requieren busqueda textual (regex/FTS) en vez de "
            "busqueda semantica."
        ),
        "rules": [
            (
                "Detectar terminos exactos: nombres propios, paises, ciudades, "
                "organizaciones, codigos alfanumericos (CVE-2024-3094, SKU-123), "
                "acronimos, fechas, cifras, identificadores o terminos tecnicos "
                "raros"
            ),
            (
                "needs_regex: true si hay al menos un termino exacto que buscar; "
                "false en caso contrario"
            ),
            "terms: los terminos exactos (max 5), tal como aparecen en la pregunta",
            "Salida JSON estricta con el schema indicado",
        ],
        "input_schema": {
            "query": "string",
        },
        "output_schema": {
            "needs_regex": "boolean",
            "terms": "array",
        },
        "few_shot": [],
    },
    {
        "task_key": "kag_critic_linking",
        "version": "1.0",
        "intent": (
            "Eres un critico de busqueda y selector de entidades. Dada una pregunta, "
            "decides si contiene terminos EXACTOS que requieren busqueda textual "
            "(regex/FTS) y seleccionas las entidades canonicas SOLO entre los "
            "candidatos del grafo que se mencionan en la pregunta."
        ),
        "rules": [
            (
                "Detectar terminos exactos: nombres propios, paises, ciudades, "
                "organizaciones, codigos alfanumericos (CVE-2024-3094, SKU-123), "
                "acronimos, fechas, cifras, identificadores o terminos tecnicos "
                "raros"
            ),
            (
                "Seleccionar las entidades canonicas SOLO entre los candidatos del "
                "grafo que se mencionan en la pregunta; si ninguna, []"
            ),
            (
                "needs_regex: true si hay al menos un termino exacto que buscar; "
                "terms: los terminos exactos (max 5)"
            ),
            "Salida JSON estricta con el schema indicado",
        ],
        "input_schema": {
            "candidates": "string",
            "query": "string",
        },
        "output_schema": {
            "needs_regex": "boolean",
            "terms": "array",
            "entities": "array",
        },
        "few_shot": [],
    },
    {
        "task_key": "kag_query_answer",
        "version": "1.0",
        "intent": (
            "Eres un asistente de conocimiento. Responde la pregunta del usuario "
            "usando SOLO el contexto proporcionado."
        ),
        "rules": [
            "Si el contexto no contiene la respuesta, decirlo claramente",
            "Citar los documentos cuando sea posible",
            "Responder en el idioma de la pregunta",
        ],
        "input_schema": {
            "context": "string",
            "query": "string",
        },
        "output_schema": {
            "answer": "string",
        },
        "few_shot": [],
    },
    # --- D. src/kag_ingest.py --------------------------------------------
    {
        "task_key": "kag_extract_entities",
        "version": "1.0",
        "intent": (
            "Eres un extractor de conocimiento. Extraes las entidades y relaciones "
            "del fragmento de texto provisto."
        ),
        "rules": [
            (
                "Entidades: conceptos, metodos, leyes, personas, organizaciones o "
                "figuras relevantes"
            ),
            "Relaciones: solo entre entidades presentes en el fragmento",
            "Si no hay entidades, devolver entities y relations vacios",
            "Salida JSON estricta con la forma exacta indicada",
        ],
        "input_schema": {
            "chunk": "string",
        },
        "output_schema": {
            "entities": "array",
            "relations": "array",
        },
        "few_shot": [],
    },
    {
        "task_key": "kag_qwen_summary",
        "version": "1.0",
        "intent": (
            "You are a summarization assistant. You produce a single-sentence "
            "summary of the provided text following strict formatting rules."
        ),
        "rules": [
            "Output exactly ONE sentence, maximum 30 words",
            "No preamble, no explanations, no markdown, no bullet points",
            "Only facts present in the text. Do not invent",
            "Do not start with phrases like 'This text...' or 'The text describes...'",
            "Respond in the same language as the text",
        ],
        "input_schema": {
            "text": "string",
        },
        "output_schema": {
            "summary": "string",
        },
        "few_shot": [],
    },
]


def _fix_db_host() -> None:
    """Reemplaza '@db:' por '@localhost:' en DATABASE_URL/DATABASE_URL_ASYNC.

    El host 'db' es la red Docker y no resuelve desde el host; las credenciales
    son las mismas. Debe llamarse ANTES de importar src.db.session.
    """
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    for var in ("DATABASE_URL", "DATABASE_URL_ASYNC"):
        val = os.environ.get(var, "")
        if not val and env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith(f"{var}="):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
        if "@db:" in val:
            os.environ[var] = val.replace("@db:", "@localhost:")


def _upsert_template(session, data: dict) -> PromptTemplate:
    template = (
        session.execute(
            select(PromptTemplate).where(PromptTemplate.task_key == data["task_key"])
        )
        .scalars()
        .first()
    )
    if template is None:
        template = PromptTemplate(**data)
        session.add(template)
        session.flush()
    else:
        for k, v in data.items():
            setattr(template, k, v)
    return template


def seed(session=None) -> dict:
    """Inserta/actualiza las specs de prompts KAG. Devuelve un resumen.

    No requiere variables de entorno. No compila artefactos (eso lo hace
    src/llm/compile_prompts.py).
    """
    own_session = session is None
    if own_session:
        _fix_db_host()
        from src.db.session import SessionLocal

        session = SessionLocal()
    try:
        templates = [_upsert_template(session, data) for data in KAG_TEMPLATES]
        session.commit()
        return {
            "templates": [str(t.id) for t in templates],
            "task_keys": [t.task_key for t in templates],
        }
    finally:
        if own_session:
            session.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed de specs de prompts KAG (prompt-as-code)."
    )
    parser.parse_args()
    _fix_db_host()
    result = seed()
    print("Specs de prompts KAG listas:")
    print(f"  Specs de prompts ({len(result['task_keys'])}):")
    for key in result["task_keys"]:
        print(f"    - {key}")


if __name__ == "__main__":
    main()
