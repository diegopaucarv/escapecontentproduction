"""Tests de configuración jerárquica del KAG (src/kag/config.py).

Rápidos, sin DB, sin torch/spacy: defaults, env vars, precedencia,
coacción de tipos y valores inválidos.
"""

from __future__ import annotations

import argparse
import os

from src.kag.config import (
    KAG_DEFAULTS,
    apply_cli_overrides,
    get_config_value,
    parse_env_config,
    resolve_config,
)

# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------


def test_defaults_exactos():
    assert KAG_DEFAULTS == {
        "KAG_QUERY_MODE": "fast",
        "KAG_INJECT_PROPOSITIONS": True,
        "KAG_RERANK_ENABLED": False,
        "KAG_QUERY_PARALLEL": True,
        "KAG_BRANCH_B_MAX_ITERS": 2,
        "KAG_GROUNDING_THRESHOLD": 95.0,
        "KAG_GROUNDING_PARALLEL": True,
        "KAG_PPR_ALPHA": 0.15,
        "KAG_AUTO_THRESHOLDS": True,
        "KAG_EXTRACT_PROPOSITIONS": True,
        "KAG_PROPOSITION_BATCH_SIZE": 400000,
        "KAG_PROPOSITION_MODEL": "large",
        "KAG_PROPOSITION_PARALLEL": 3,
        "KAG_LLM_ENTITIES": False,
        "KAG_USE_COREF": "auto",
        "KAG_PROCESS_FIGURES": True,
        "KAG_ENTITY_N_PROCESS": 1,
        "KAG_FIGURE_PARALLEL": 3,
        "KAG_SUMMARY_PARALLEL": 3,
    }


def test_resolve_config_sin_session_usa_defaults():
    config = resolve_config()
    assert config == KAG_DEFAULTS


# ---------------------------------------------------------------------
# Env vars
# ---------------------------------------------------------------------


def test_env_vars_se_aplican(monkeypatch):
    monkeypatch.setenv("KAG_QUERY_MODE", "audited")
    monkeypatch.setenv("KAG_RERANK_ENABLED", "1")
    monkeypatch.setenv("KAG_BRANCH_B_MAX_ITERS", "5")
    monkeypatch.setenv("KAG_GROUNDING_THRESHOLD", "90.5")
    config = resolve_config()
    assert config["KAG_QUERY_MODE"] == "audited"
    assert config["KAG_RERANK_ENABLED"] is True
    assert config["KAG_BRANCH_B_MAX_ITERS"] == 5
    assert config["KAG_GROUNDING_THRESHOLD"] == 90.5


def test_parse_env_config_tipado(monkeypatch):
    monkeypatch.setenv("KAG_QUERY_MODE", "audited")
    monkeypatch.setenv("KAG_INJECT_PROPOSITIONS", "false")
    monkeypatch.setenv("KAG_PROPOSITION_BATCH_SIZE", "1000")
    monkeypatch.setenv("KAG_PPR_ALPHA", "0.3")
    monkeypatch.setenv("KAG_SUMMARY_PARALLEL", "5")
    env = parse_env_config()
    assert env["KAG_QUERY_MODE"] == "audited"
    assert env["KAG_INJECT_PROPOSITIONS"] is False
    assert env["KAG_PROPOSITION_BATCH_SIZE"] == 1000
    assert env["KAG_PPR_ALPHA"] == 0.3
    assert env["KAG_SUMMARY_PARALLEL"] == 5


def test_parse_env_config_ignora_vars_desconocidas(monkeypatch):
    monkeypatch.setenv("KAG_QUERY_MODE", "audited")
    monkeypatch.setenv("KAG_NO_EXISTE", "1")
    env = parse_env_config()
    assert "KAG_NO_EXISTE" not in env
    assert env["KAG_QUERY_MODE"] == "audited"


# ---------------------------------------------------------------------
# Precedencia: overrides > env > defaults
# ---------------------------------------------------------------------


def test_overrides_ganan_sobre_env(monkeypatch):
    monkeypatch.setenv("KAG_QUERY_MODE", "audited")
    monkeypatch.setenv("KAG_BRANCH_B_MAX_ITERS", "5")
    config = resolve_config(overrides={"KAG_QUERY_MODE": "fast"})
    assert config["KAG_QUERY_MODE"] == "fast"
    # Flag sin override sigue viniendo del env.
    assert config["KAG_BRANCH_B_MAX_ITERS"] == 5


def test_overrides_ignoran_flags_desconocidos():
    config = resolve_config(overrides={"KAG_NO_EXISTE": 1, "KAG_QUERY_MODE": "audited"})
    assert "KAG_NO_EXISTE" not in config
    assert config["KAG_QUERY_MODE"] == "audited"


# ---------------------------------------------------------------------
# Coacción de tipos
# ---------------------------------------------------------------------


def test_coercion_bool():
    assert (
        resolve_config(overrides={"KAG_RERANK_ENABLED": "true"})["KAG_RERANK_ENABLED"]
        is True
    )
    assert (
        resolve_config(overrides={"KAG_RERANK_ENABLED": "1"})["KAG_RERANK_ENABLED"]
        is True
    )
    assert (
        resolve_config(overrides={"KAG_RERANK_ENABLED": "FALSE"})["KAG_RERANK_ENABLED"]
        is False
    )
    assert (
        resolve_config(overrides={"KAG_RERANK_ENABLED": "0"})["KAG_RERANK_ENABLED"]
        is False
    )


def test_coercion_int_float():
    assert (
        resolve_config(overrides={"KAG_BRANCH_B_MAX_ITERS": "3"})[
            "KAG_BRANCH_B_MAX_ITERS"
        ]
        == 3
    )
    assert (
        resolve_config(overrides={"KAG_GROUNDING_THRESHOLD": "80"})[
            "KAG_GROUNDING_THRESHOLD"
        ]
        == 80.0
    )
    assert resolve_config(overrides={"KAG_PPR_ALPHA": "0.5"})["KAG_PPR_ALPHA"] == 0.5


def test_coercion_enum():
    assert (
        resolve_config(overrides={"KAG_QUERY_MODE": "AUDITED"})["KAG_QUERY_MODE"]
        == "audited"
    )
    assert (
        resolve_config(overrides={"KAG_USE_COREF": "never"})["KAG_USE_COREF"] == "never"
    )


# ---------------------------------------------------------------------
# Valores inválidos → default
# ---------------------------------------------------------------------


def test_valor_invalido_usa_default(monkeypatch):
    monkeypatch.setenv("KAG_QUERY_MODE", "bogus")
    monkeypatch.setenv("KAG_BRANCH_B_MAX_ITERS", "abc")
    monkeypatch.setenv("KAG_GROUNDING_THRESHOLD", "no-es-numero")
    monkeypatch.setenv("KAG_RERANK_ENABLED", "quizas")
    config = resolve_config()
    assert config["KAG_QUERY_MODE"] == "fast"
    assert config["KAG_BRANCH_B_MAX_ITERS"] == 2
    assert config["KAG_GROUNDING_THRESHOLD"] == 95.0
    assert config["KAG_RERANK_ENABLED"] is False


def test_override_invalido_usa_default():
    config = resolve_config(overrides={"KAG_QUERY_MODE": "bogus", "KAG_PPR_ALPHA": "x"})
    assert config["KAG_QUERY_MODE"] == "fast"
    assert config["KAG_PPR_ALPHA"] == 0.15


# ---------------------------------------------------------------------
# Capa DB opcional y defensiva
# ---------------------------------------------------------------------


def test_session_settings_dict():
    class FakeSession:
        session_settings = {"KAG_QUERY_MODE": "audited", "KAG_RERANK_ENABLED": "1"}

    config = resolve_config(session=FakeSession())
    assert config["KAG_QUERY_MODE"] == "audited"
    assert config["KAG_RERANK_ENABLED"] is True


def test_session_settings_objeto():
    class FakeSettings:
        KAG_QUERY_MODE = "audited"
        KAG_BRANCH_B_MAX_ITERS = "7"

    class FakeSession:
        session_settings = FakeSettings()

    config = resolve_config(session=FakeSession())
    assert config["KAG_QUERY_MODE"] == "audited"
    assert config["KAG_BRANCH_B_MAX_ITERS"] == 7


def test_session_sin_settings_no_rompe():
    class FakeSession:
        pass

    config = resolve_config(session=FakeSession())
    assert config == KAG_DEFAULTS


def test_session_settings_rotas_no_rompen(monkeypatch):
    class FakeSession:
        @property
        def session_settings(self):
            raise RuntimeError("DB caída")

    config = resolve_config(session=FakeSession())
    assert config == KAG_DEFAULTS


def test_db_gana_sobre_env_y_pierde_ante_overrides(monkeypatch):
    monkeypatch.setenv("KAG_QUERY_MODE", "fast")

    class FakeSession:
        session_settings = {"KAG_QUERY_MODE": "audited"}

    config = resolve_config(session=FakeSession())
    assert config["KAG_QUERY_MODE"] == "audited"

    config = resolve_config(session=FakeSession(), overrides={"KAG_QUERY_MODE": "fast"})
    assert config["KAG_QUERY_MODE"] == "fast"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def test_get_config_value():
    config = {"KAG_QUERY_MODE": "audited"}
    assert get_config_value(config, "KAG_QUERY_MODE") == "audited"
    assert get_config_value(config, "KAG_NO_EXISTE") is None
    assert get_config_value(config, "KAG_NO_EXISTE", "x") == "x"


def test_apply_cli_overrides():
    parser = argparse.ArgumentParser()
    apply_cli_overrides(parser)
    args = parser.parse_args(
        ["--kag-query-mode", "audited", "--kag-branch-b-max-iters", "4"]
    )
    ns = vars(args)
    assert ns["KAG_QUERY_MODE"] == "audited"
    assert ns["KAG_BRANCH_B_MAX_ITERS"] == "4"
    assert ns["KAG_RERANK_ENABLED"] is None
