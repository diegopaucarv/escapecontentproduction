"""
Seed de las specs de prompts del pipeline KAG (prompt-as-code).

Inserta/actualiza en `prompt_templates` las 11 specs agnósticas de los
prompts del sistema KAG (modo audited de la consulta, query clásica e
ingesta clásica expandida). Cada spec captura el rol (intent) y las
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
    # --- Modo audited (src/kag_query.py) -----------------------------
    {
        "task_key": "kag_synthesis",
        "user_template": """Consulta del usuario: "{query}"

Fragmentos recuperados para análisis:
{candidate_chunks_json}

Devuelve el JSON: {{"query": str, "total_chunks_processed": int, "synthesized_facts": [{{"chunk_id": str, "document_id": str, "source_file": str, "relevance_level": "direct_answer|supporting_evidence|contextual_background|irrelevant", "atomic_summary": str, "verbatim_evidence": str, "academic_citations": [str]}}]}}""",
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
        "user_template": """Consulta: "{query}"

Hechos sintetizados:
{synthesized_facts_json}

Devuelve el JSON: {{"contradictions_detected": bool, "analysis_cases": [{{"conflict_type": "paradigmatic_theoretical_divergence|empirical_contextual_boundary|temporal_diachronic_shift|terminological_homonymy", "divergence_summary": str, "thesis_a": {{"proposition_id": str, "document_id": str, "claim": str, "author_or_framework": str, "empirical_context": str}}, "thesis_b": {{"proposition_id": str, "document_id": str, "claim": str, "author_or_framework": str, "empirical_context": str}}, "epistemic_reconciliation": str}}]}}""",
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
        "user_template": """Consulta: "{query}"

Metadatos del Corpus Disponible (Descriptores ISO 25964 y LCC presentes en DB):
{active_corpus_metadata}

Proposiciones recuperadas (Nivel 1):
{synthesized_propositions_json}

Contextos escalados (Nivel 2, si aplicó):
{parent_contexts_json}

Emite tu evaluación formal: {{"verdict": "SUFFICIENT_FOR_SYNTHESIS|INSUFFICIENT_TRIGGER_BRANCH_B|NEGATIVE_REJECTION", "confidence_score": float, "negative_rejection_details": {{"reason": "out_of_thematic_scope_iso25964|classification_mismatch_lcc|total_absence_in_knowledge_graph|unsupported_technical_granularity", "closest_available_topics": [str], "formal_abstention_statement": str}}, "branch_b_instructions": {{"unresolved_subqueries": [str], "target_thesaurus_concepts": [str]}}}}""",
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
        "user_template": """Consulta del usuario: "{query}"

Evidencia verificada (grounded_evidence):
{grounded_evidence_json}

Tensiones epistémicas (epistemic_tensions):
{epistemic_tensions_json}""",
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
        "user_template": """Entidades candidatas del grafo de conocimiento:
{candidates}

Pregunta: {query}

Devuelve SOLO JSON:
{{"entities": ["Entidad 1", "Entidad 2"]}}

Elige SOLO de la lista de candidatas. Si ninguna se menciona en la
pregunta, devuelve {{"entities": []}}.
""",
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
        "user_template": """Eres un crítico de búsqueda. Dada una pregunta, decide si
contiene términos EXACTOS que requieren búsqueda textual (regex/FTS) en vez
de búsqueda semántica: nombres propios, países, ciudades, organizaciones,
códigos alfanuméricos (CVE-2024-3094, SKU-123), acrónimos, fechas, cifras,
identificadores o términos técnicos raros.

El corpus es MULTILINGÜE. Para cada término exacto, incluye su traducción a
TODOS los idiomas soportados: {languages}. Los códigos y nombres propios no
se traducen (se repiten igual en todos los idiomas).

Palabras muy frecuentes en el corpus (NO las propongas: matchearían
demasiados chunks y no aportan precisión): {common_words}

Devuelve SOLO JSON:
{{"needs_regex": true/false, "terms": ["término y sus traducciones..."]}}

- needs_regex: true si hay al menos un término exacto que buscar.
- terms: máx 3 términos, cada uno con su traducción a todos los idiomas
  (ej. ["discriminación negativa", "negative discrimination", ...]).
- Si no hay términos exactos, devuelve {{"needs_regex": false, "terms": []}}.

Pregunta: {query}
""",
        "version": "1.1",
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
            (
                "El corpus es multilingue: cada termino exacto se traduce a TODOS "
                "los idiomas soportados; los codigos y nombres propios no se "
                "traducen"
            ),
            (
                "No proponer palabras muy frecuentes del corpus (contexto "
                "common_words): matchearian demasiados chunks"
            ),
            "terms: max 3 terminos, cada uno con su traduccion a todos los idiomas",
            "Salida JSON estricta con el schema indicado",
        ],
        "input_schema": {
            "query": "string",
            "languages": "string",
            "common_words": "string",
        },
        "output_schema": {
            "needs_regex": "boolean",
            "terms": "array",
        },
        "few_shot": [],
    },
    {
        "task_key": "kag_critic_linking",
        "user_template": """Eres un crítico de búsqueda y selector de entidades.
Dada una pregunta:

1. Decide si contiene términos EXACTOS que requieren búsqueda textual
   (regex/FTS): nombres propios, países, ciudades, organizaciones, códigos
   alfanuméricos (CVE-2024-3094, SKU-123), acrónimos, fechas, cifras,
   identificadores o términos técnicos raros.
2. Selecciona las entidades canónicas que se mencionan en la pregunta. Elige
   de la lista de candidatos cuando sea posible; si la lista es insuficiente
   o está vacía, propón entidades adicionales tú mismo (nombres canónicos,
   posiblemente en inglés — el sistema las resolverá por similitud).

El corpus es MULTILINGÜE. Para cada término exacto, incluye su traducción a
TODOS los idiomas soportados: {languages}. Los códigos y nombres propios no
se traducen (se repiten igual en todos los idiomas).

Palabras muy frecuentes en el corpus (NO las propongas: matchearían
demasiados chunks y no aportan precisión): {common_words}

Candidatos del grafo:
{{candidates}}

Devuelve SOLO JSON:
{{"needs_regex": true/false, "terms": ["término y sus traducciones..."], "entities": ["Entidad 1"]}}

- needs_regex: true si hay al menos un término exacto que buscar.
- terms: máx 3 términos, cada uno con su traducción a todos los idiomas
  (ej. ["discriminación negativa", "negative discrimination", ...]).
- entities: 3-5 entidades relevantes. Prefiere las de la lista de candidatos;
  si la lista es insuficiente o está vacía, propón entidades adicionales tú
  mismo (nombres canónicos, posiblemente en inglés). Si ninguna, [].
- Si no hay términos exactos, devuelve {{"needs_regex": false, "terms": []}}.

Pregunta: {query}
""",
        "version": "1.1",
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
                "El corpus es multilingue: cada termino exacto se traduce a TODOS "
                "los idiomas soportados; los codigos y nombres propios no se "
                "traducen"
            ),
            (
                "No proponer palabras muy frecuentes del corpus (contexto "
                "common_words): matchearian demasiados chunks"
            ),
            (
                "needs_regex: true si hay al menos un termino exacto que buscar; "
                "terms: max 3 terminos, cada uno con su traduccion a todos los "
                "idiomas"
            ),
            "Salida JSON estricta con el schema indicado",
        ],
        "input_schema": {
            "candidates": "string",
            "query": "string",
            "languages": "string",
            "common_words": "string",
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
        "user_template": """Contexto:
{context}

Pregunta: {query}

Responde con precisión basándote en el contexto.""",
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
        "user_template": """Extrae las entidades y relaciones del siguiente fragmento de texto.

Devuelve SOLO JSON con esta forma exacta:
{
  "entities": [
    {"name": "Nombre de la entidad", "type": "concept|method|law|person|org|figure", "description": "breve descripción"}
  ],
  "relations": [
    {"source": "Entidad origen", "target": "Entidad destino", "type": "RELACIÓN_EN_MAYÚSCULAS", "description": "breve descripción"}
  ]
}

Reglas:
- Entidades: conceptos, métodos, leyes, personas, organizaciones o figuras relevantes.
- Relaciones: solo entre entidades presentes en el fragmento.
- Si no hay entidades, devuelve {"entities": [], "relations": []}.

Texto:
<text>
{chunk}
</text>
""",
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
        "user_template": """<text>
{text}
</text>

Summary:""",
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
    # --- D. src/kag_ingest.py (expansión proposicional, Agente B) ---------
    {
        "task_key": "kag_proposition_chunking",
        "user_template": """Archivo: {source_file}
Documento: {document_id}
Capítulo: {chapter_id}
Rango: Línea {line_start} a Línea {line_end}

Texto a procesar:
---
{chapter_text_content}
---

Genera el JSON con las proposiciones organizadas por divisiones (capítulos del texto):
{{"divisions": [{{"chapter_id": str, "propositions": [{{"core_idea_id": str, "argument_id": str, "statement": str, "text_span": str, "char_start": int, "char_end": int, "line_start": int, "line_end": int, "citations_references": [str]}}]}}]}}""",
        "version": "2.0",
        "intent": (
            "Eres un analista de epistemologia y analisis del discurso. Tu "
            "objetivo es descomponer el texto en proposiciones atomicas "
            "gramaticalmente independientes y autocontenidas, organizadas por "
            "capitulos (divisiones)."
        ),
        "rules": [
            (
                "Proposiciones Atomicas: descomponer el texto en proposiciones "
                "elementales gramaticalmente independientes y autocontenidas, "
                "reemplazando anforas por el sujeto explicito"
            ),
            "text_span debe contener el fragmento de texto exacto del original",
            (
                "char_start/char_end y line_start/line_end: offsets absolutos "
                "del text_span dentro del texto procesado"
            ),
            (
                "Organizar el output por divisiones: cada division declara su "
                "chapter_id (el capitulo del texto al que pertenecen "
                "sus proposiciones) y su lista de proposiciones"
            ),
            (
                "REGLA DE REFERENCIAS DUPLICADAS: si una frase contiene una "
                "referencia academica, debe preservarse y duplicarse en "
                "citations_references de TODAS las proposiciones atomicas que "
                "deriven de ella; prohibido descartar o separar citas "
                "bibliograficas"
            ),
            "Salida JSON estricta con el schema indicado",
        ],
        "input_schema": {
            "source_file": "string",
            "document_id": "string",
            "chapter_id": "string",
            "line_start": "integer",
            "line_end": "integer",
            "chapter_text_content": "string",
        },
        "output_schema": {
            "divisions": "array",
        },
        "few_shot": [],
    },
    # --- E. Pipeline documental nuevo (Fases 1-5) -------------------------
    {
        "task_key": "kag_document_separation",
        "user_template": """Archivo: {source_file}

Esqueleto del archivo (líneas del texto plano):
---
{skeleton}
---

Documentos detectados determinísticamente (MultibookFinderTool, límites físicos por ISBN/separadores):
---
{deterministic_documents}
---

Identifica los documentos contenidos en el archivo y devuelve el JSON:
{{"documents": [{{"document_id": str, "title": str, "line_start": int, "line_end": int, "language": str}}]}}

REGLAS:
- La lista determinista es la BASE: confirma cada documento detectado (puedes ajustar títulos/idioma).
- AÑADE divisiones adicionales SOLO si encuentras libros/papers/artículos SEPARADOS que la detección física no capturó (p. ej. un libro que empieza sin ISBN ni separador).
- NO dividas un libro en capítulos/secciones: los capítulos se detectan en la Fase 2 (análisis documental), no aquí.
- Cada documento debe tener un document_id unico y estable; line_start/line_end delimitan su rango en el archivo fuente.""",
        "version": "1.1",
        "intent": (
            "Eres un analista documental. Tu tarea es separar un archivo en "
            "los documentos que contiene, partiendo de la deteccion fisica "
            "determinista (MultibookFinderTool) y anadiendo divisiones solo "
            "si hay libros/papers separados no detectados."
        ),
        "rules": [
            "La lista determinista es la BASE: confirmar cada documento detectado",
            "Anadir divisiones SOLO si hay libros/papers separados no detectados fisicamente",
            "NUNCA dividir un libro en capitulos: eso es la Fase 2 (analisis documental)",
            "Cada documento debe tener un document_id unico y estable",
            "line_start/line_end delimitan el rango de lineas del documento en el archivo fuente",
            "language: codigo ISO 639-1 del idioma principal del documento",
            "Salida JSON estricta con el schema indicado",
        ],
        "input_schema": {
            "source_file": "string",
            "skeleton": "string",
            "deterministic_documents": "string",
        },
        "output_schema": {
            "documents": "array",
        },
        "few_shot": [],
    },
    {
        "task_key": "kag_document_analysis",
        "user_template": """Archivo: {source_file}
Documento: {document_id}

Contexto del documento COMPLETO (texto plano, líneas 1..N):
---
{document_context}
---

Analiza el documento y devuelve el JSON:
{{"ficha": {{"title": str, "technical_level": str, "thematic_areas_iso25964": [{{"preferred_term": str, "non_preferred_terms": [str], "scope_note_disambiguation": str, "broader_term": str, "narrower_terms": [str], "related_terms": [str]}}], "library_of_congress": {{"lcsh_terms": [str], "lcc_classification": {{"label": str, "call_number": str}}}}, "bibtex": str, "key_entities": [{{"name": str, "type": str}}]}}, "index": [{{"division": str, "chapters": [{{"chapter_id": str, "title": str, "line_start": int, "line_end": int, "has_images": bool}}]}}]}}

REGLAS:
- El contexto es el documento COMPLETO: úsalo para detectar TODOS los capítulos reales (no solo los que tengan headers de markdown).
- Los capítulos suelen aparecer como líneas de texto plano: títulos numerados ("3. Dimensions of Cultural Variation"), líneas "PART X", títulos en MAYÚSCULAS, o entradas del índice (TOC).
- Reagrupa los capítulos en un índice JERÁRQUICO: cada división (p. ej. "PART I Foundations") agrupa sus capítulos. Si el documento no tiene divisiones, usa UNA división con el título del documento.
- line_start/line_end son RELATIVOS al documento (línea 1 = primera línea del contexto).
- has_images: true si el capítulo contiene imágenes o figuras.""",
        "version": "1.1",
        "intent": (
            "Eres un analista documental y bibliotecario. Tu tarea es producir "
            "la ficha documental (tesauro ISO 25964, clasificacion LCC/LCSH, "
            "cita BibTeX) y detectar los capitulos del documento con su rango "
            "de lineas, reagrupandolos en un indice jerarquico."
        ),
        "rules": [
            "thematic_areas_iso25964: minimo 3 tematicas con termino preferido, no preferidos, nota de alcance y relaciones de tesauro",
            "library_of_congress.lcsh_terms: encabezamientos de materia; lcc_classification: clasificacion de la Biblioteca del Congreso",
            "bibtex: cita completa en formato BibTeX",
            "key_entities: entidades clave del documento con su tipo",
            "index: indice jerarquico de capitulos; cada division agrupa sus capitulos (division unica si no hay partes)",
            "chapters: cada capitulo con chapter_id unico, titulo y rango de lineas RELATIVO al documento (SIN resumen: los resumenes de capitulo se reemplazan por proposiciones atomicas)",
            "has_images: true si el capitulo contiene imagenes o figuras",
            "Usa el contexto COMPLETO del documento para detectar capitulos aunque no tengan headers de markdown",
            "Salida JSON estricta con el schema indicado",
        ],
        "input_schema": {
            "source_file": "string",
            "document_id": "string",
            "document_context": "string",
        },
        "output_schema": {
            "ficha": "object",
            "index": "array",
        },
        "few_shot": [],
    },
    {
        "task_key": "kag_chunk_paraphrase",
        "user_template": """Archivo: {source_file}
Documento: {document_id}
Capítulo: {chapter_id}

Chunks del capítulo:
---
{chunks_json}
---

Parafrasea cada chunk y devuelve el JSON:
{{"paraphrases": [{{"chunk_index": int, "paraphrase": str}}]}}""",
        "version": "1.0",
        "intent": (
            "Eres un parafraseador academico. Tu tarea es reescribir cada chunk "
            "preservando el significado, la estructura argumentativa y las "
            "referencias, sin anadidos ni omisiones."
        ),
        "rules": [
            "Cada chunk_index del input debe tener exactamente una paraphrase",
            "La parafrasis preserva el significado y las referencias del original",
            "No anadir informacion que no este en el chunk",
            "Salida JSON estricta con el schema indicado",
        ],
        "input_schema": {
            "source_file": "string",
            "document_id": "string",
            "chapter_id": "string",
            "chunks_json": "string",
        },
        "output_schema": {
            "paraphrases": "array",
        },
        "few_shot": [],
    },
    {
        "task_key": "kag_chapter_propositions",
        "user_template": """Archivo: {source_file}
Documento: {document_id}
Capítulo: {chapter_id}

Paráfrasis del capítulo:
---
{paraphrases_json}
---

Extrae las proposiciones atomicas y devuelve el JSON:
{{"propositions": [{{"chunk_index": int, "core_idea_id": str, "argument_id": str, "statement": str, "text_span": str, "citations_references": [str]}}], "entities": [{{"name": str, "type": str, "description": str}}], "relations": [{{"source": str, "target": str, "type": str, "description": str}}]}}""",
        "version": "1.1",
        "intent": (
            "Eres un analista de epistemologia y analisis del discurso. Tu "
            "objetivo es descomponer las parafrasis del capitulo en "
            "proposiciones atomicas autocontenidas y extraer las entidades y "
            "relaciones del capitulo."
        ),
        "rules": [
            "Proposiciones atomicas gramaticalmente independientes y autocontenidas",
            "chunk_index: indice del chunk (0-based) al que pertenece la proposicion",
            "text_span debe contener el fragmento de texto exacto de la parafrasis",
            "citations_references: preservar y duplicar las referencias academicas en todas las proposiciones derivadas",
            "entities: entidades del capitulo con su tipo y descripcion",
            "relations: relaciones entre entidades con su tipo y descripcion",
            "Salida JSON estricta con el schema indicado",
        ],
        "input_schema": {
            "source_file": "string",
            "document_id": "string",
            "chapter_id": "string",
            "paraphrases_json": "string",
        },
        "output_schema": {
            "propositions": "array",
            "entities": "array",
            "relations": "array",
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

    No requiere variables de entorno. Compila los artefactos automáticamente
    (src/llm/compiler.py::compile_prompts) — idempotente: solo crea versiones
    nuevas si el contenido cambió.
    """
    own_session = session is None
    if own_session:
        _fix_db_host()
        from src.db.session import SessionLocal

        session = SessionLocal()
    try:
        templates = [_upsert_template(session, data) for data in KAG_TEMPLATES]
        session.commit()
        from src.llm.compiler import compile_prompts

        compiled = compile_prompts(session)
        return {
            "templates": [str(t.id) for t in templates],
            "task_keys": [t.task_key for t in templates],
            "compiled": compiled["compiled"],
            "skipped": compiled["skipped"],
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
