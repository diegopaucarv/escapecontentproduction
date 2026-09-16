"""Tests del seed de prompts KAG (src/db/seed_kag_prompts.py) — sin DB.

Verifica la migración completa a prompt-as-code (0021): las 11 specs
incluyen `user_template` (el USER prompt parametrizable que antes era
constante en código) además del SYSTEM (intent + rules).
"""

from src.db.seed_kag_prompts import KAG_TEMPLATES
from src.kag.prompts import get_user_template

EXPECTED_TASK_KEYS = {
    # A. Modo audited (src/kag_query.py)
    "kag_synthesis",
    "kag_contradictions",
    "kag_sufficiency",
    "kag_answer",
    "kag_audit_fused",
    # B. src/kag_query.py (clásico)
    "kag_grounded_entities",
    "kag_critic_regex",
    "kag_critic_linking",
    "kag_query_answer",
    # C. src/kag_ingest.py
    "kag_extract_entities",
    "kag_qwen_summary",
    # C. src/kag_ingest.py (expansión proposicional, Agente B)
    "kag_proposition_chunking",
    # E. Pipeline documental nuevo (Fases 1-5)
    "kag_document_separation",
    "kag_document_analysis",
    "kag_chunk_paraphrase",
    "kag_chapter_propositions",
    # E. Extracción FUSIONADA por documento (Fases 4+5 en UNA llamada)
    "kag_document_extract",
    # F. src/kag_query.py (clasificación SLM de la estrategia)
    "kag_query_strategy",
    # F. src/kag_query.py (extracción de filtros de metadatos)
    "kag_query_metadata",
    # G. src/kag_query.py (descomposición en subconsultas)
    "kag_query_subqueries",
}


def test_kag_templates_has_11_specs():
    assert len(KAG_TEMPLATES) == 20


def test_kag_templates_task_keys_match_expected():
    keys = {t["task_key"] for t in KAG_TEMPLATES}
    assert keys == EXPECTED_TASK_KEYS


def test_all_kag_templates_have_user_template():
    """Cada spec (0021) debe tener el USER prompt parametrizable no vacío."""
    for t in KAG_TEMPLATES:
        assert t.get("user_template"), f"falta user_template en {t['task_key']}"
        assert "{" in t["user_template"], (
            f"user_template de {t['task_key']} no tiene placeholders"
        )


def test_all_kag_templates_have_intent_and_rules():
    """El SYSTEM sigue vivo en la spec (intent + rules)."""
    for t in KAG_TEMPLATES:
        assert t.get("intent"), f"falta intent en {t['task_key']}"
        assert isinstance(t.get("rules"), list) and t["rules"], (
            f"faltan rules en {t['task_key']}"
        )


def test_kag_ingest_fallbacks_iguales_a_spec_v11():
    """El runtime (get_user_template) resuelve el user_template EXACTO de la
    spec (fuente única): los fallbacks de src/kag_ingest.py ya no viven en
    código, se resuelven desde src/kag/prompts/specs.py.
    """
    by_key = {t["task_key"]: t for t in KAG_TEMPLATES}
    assert (
        get_user_template("kag_document_separation")
        == by_key["kag_document_separation"]["user_template"]
    )
    assert (
        get_user_template("kag_document_analysis")
        == by_key["kag_document_analysis"]["user_template"]
    )


def test_kag_query_strategy_fallback_igual_a_spec():
    """get_user_template('kag_query_strategy') debe ser EXACTO al user_template
    de la spec (mismo placeholder {query})."""
    by_key = {t["task_key"]: t for t in KAG_TEMPLATES}
    assert (
        get_user_template("kag_query_strategy")
        == by_key["kag_query_strategy"]["user_template"]
    )


def test_kag_query_metadata_fallback_igual_a_spec():
    """get_user_template('kag_query_metadata') debe ser EXACTO al user_template
    de la spec (mismos placeholders {query} y {corpus_metadata})."""
    by_key = {t["task_key"]: t for t in KAG_TEMPLATES}
    assert (
        get_user_template("kag_query_metadata")
        == by_key["kag_query_metadata"]["user_template"]
    )


def test_kag_query_subqueries_fallback_igual_a_spec():
    """get_user_template('kag_query_subqueries') debe ser EXACTO al
    user_template de la spec (mismos placeholders {query} y {corpus_metadata})."""
    by_key = {t["task_key"]: t for t in KAG_TEMPLATES}
    assert (
        get_user_template("kag_query_subqueries")
        == by_key["kag_query_subqueries"]["user_template"]
    )


def test_kag_document_extract_fallback_igual_a_spec():
    """get_user_template('kag_document_extract') debe ser EXACTO al
    user_template de la spec (mismos placeholders {source_file}, {document_id},
    {chapters_json}, {chunks_json})."""
    by_key = {t["task_key"]: t for t in KAG_TEMPLATES}
    assert (
        get_user_template("kag_document_extract")
        == by_key["kag_document_extract"]["user_template"]
    )


def test_kag_document_extract_spec_defines_three_layers():
    """La spec kag_document_extract define las TRES capas lingüísticas
    (paráfrasis, proposición, resumen) en intent y rules."""
    by_key = {t["task_key"]: t for t in KAG_TEMPLATES}
    spec = by_key["kag_document_extract"]
    intent = spec["intent"].lower()
    rules_text = " ".join(spec["rules"]).lower()
    assert "parafrasis" in intent and "proposicion" in intent and "resumen" in intent
    assert "parafrasis" in rules_text
    assert "proposicion" in rules_text
    assert "resumen" in rules_text
    # El schema de salida incluye las 6 claves del mapeo jerárquico.
    assert set(spec["output_schema"].keys()) == {
        "paraphrases",
        "propositions",
        "entities",
        "relations",
        "section_summaries",
        "document_summary",
    }
