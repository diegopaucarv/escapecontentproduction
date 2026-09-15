"""Tests del seed de prompts KAG (src/db/seed_kag_prompts.py) — sin DB.

Verifica la migración completa a prompt-as-code (0021): las 11 specs
incluyen `user_template` (el USER prompt parametrizable que antes era
constante en código) además del SYSTEM (intent + rules).
"""

from src.db.seed_kag_prompts import KAG_TEMPLATES

EXPECTED_TASK_KEYS = {
    # A. Modo audited (src/kag_query.py)
    "kag_synthesis",
    "kag_contradictions",
    "kag_sufficiency",
    "kag_answer",
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
}


def test_kag_templates_has_11_specs():
    assert len(KAG_TEMPLATES) == 15


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
    """Los fallbacks de src/kag_ingest.py deben ser EXACTOS a las specs v1.1.

    FIX H1: las constantes DOCUMENT_*_USER_SHORT copian el user_template de la
    spec (mismos placeholders: {deterministic_documents}, {document_context}).
    """
    from src.kag_ingest import (
        DOCUMENT_ANALYSIS_USER_SHORT,
        DOCUMENT_SEPARATION_USER_SHORT,
    )

    by_key = {t["task_key"]: t for t in KAG_TEMPLATES}
    assert (
        DOCUMENT_SEPARATION_USER_SHORT
        == by_key["kag_document_separation"]["user_template"]
    )
    assert (
        DOCUMENT_ANALYSIS_USER_SHORT == by_key["kag_document_analysis"]["user_template"]
    )
