"""Tests del seed de prompts KAG (src/db/seed_kag_prompts.py) — sin DB.

Verifica la migración completa a prompt-as-code (0021): las 16 specs
incluyen `user_template` (el USER prompt parametrizable que antes era
constante en código) además del SYSTEM (intent + rules).
"""

from src.db.seed_kag_prompts import KAG_TEMPLATES

EXPECTED_TASK_KEYS = {
    # A. src/kag_propositional.py
    "kag_metadata",
    "kag_chapters",
    "kag_document_analysis",
    "kag_propositional_chunking",
    "kag_topic_label",
    "kag_vision_analysis",
    # B. src/kag_agents.py
    "kag_synthesis",
    "kag_contradictions",
    "kag_sufficiency",
    "kag_answer",
    # C. src/kag_query.py
    "kag_grounded_entities",
    "kag_critic_regex",
    "kag_critic_linking",
    "kag_query_answer",
    # D. src/kag_ingest.py
    "kag_extract_entities",
    "kag_qwen_summary",
}


def test_kag_templates_has_16_specs():
    assert len(KAG_TEMPLATES) == 16


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
