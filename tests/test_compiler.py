"""Tests del compilador de prompts (src/llm/compiler.py) — sin base real."""

import uuid
from types import SimpleNamespace

from src.llm.compiler import (
    GenericPromptAdapter,
    _content_hash,
    validate_critic_spec,
)


def _spec(**overrides):
    defaults = dict(
        task_key="alignment_reinforcement",
        version="1.0",
        intent="Eres el refuerzo del semaforo.",
        rules=["Nunca bajar la severidad"],
        input_schema={"brief": "object"},
        output_schema={"verdict": "string"},
        few_shot=[],
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _model(**overrides):
    defaults = dict(
        model_name="meta-models/Muse-Glimmer-30B",
        temperature_default=0.7,
        max_output_tokens=8192,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _profile(**overrides):
    defaults = dict(
        api_style="openai_chat",
        system_role_name="system",
        instruction_formatting={"style": "markdown"},
        structured_output={"mode": "json_object"},
        tool_calling={"mode": "none", "supports_strict_tools": False},
        prompt_caching={"supports_prefix_caching": True},
        reasoning_mode={"supports_reasoning": False},
    )
    defaults.update(overrides)
    return defaults


def test_build_system_message_markdown():
    adapter = GenericPromptAdapter(_profile())
    msg = adapter.build_system_message(_spec())
    assert msg["role"] == "system"
    assert "# Role" in msg["content"]
    assert "Eres el refuerzo del semaforo." in msg["content"]
    assert "# Rules" in msg["content"]
    assert "- Nunca bajar la severidad" in msg["content"]
    assert "# Output Schema" in msg["content"]


def test_build_system_message_xml_style():
    profile = _profile(
        instruction_formatting={
            "style": "xml",
            "root_tag": "instructions",
            "section_delimiters": {
                "rules": "rules",
                "context": "context",
                "examples": "examples",
            },
        }
    )
    adapter = GenericPromptAdapter(profile)
    msg = adapter.build_system_message(_spec())
    assert msg["content"].startswith("<instructions>")
    assert "<context>" in msg["content"]
    assert "<rules>" in msg["content"]
    assert "</instructions>" in msg["content"]


def test_build_request_params_json_object():
    adapter = GenericPromptAdapter(_profile())
    params = adapter.build_request_params(_spec(), _model())
    assert params["response_format"] == {"type": "json_object"}
    assert params["temperature"] == 0.7
    assert params["max_tokens"] == 8192


def test_build_request_params_none_mode():
    profile = _profile(structured_output={"mode": "none"})
    adapter = GenericPromptAdapter(profile)
    params = adapter.build_request_params(_spec(), _model())
    assert "response_format" not in params


def test_build_request_params_reasoning():
    profile = _profile(
        reasoning_mode={
            "supports_reasoning": True,
            "thinking_parameter": "reasoning_effort",
        }
    )
    adapter = GenericPromptAdapter(profile)
    params = adapter.build_request_params(_spec(), _model())
    assert params["reasoning_effort"] == "medium"


def test_validate_critic_spec_all_covered():
    spec = _spec(
        task_key="critic_checklist",
        rules=[
            "fuente_verificable: cita fuente",
            "cta_unico: un CTA",
            "revision_legal: respaldo legal",
        ],
    )
    missing = validate_critic_spec(
        spec, ["fuente_verificable", "cta_unico", "revision_legal"]
    )
    assert missing == []


def test_validate_critic_spec_missing_items():
    spec = _spec(
        task_key="critic_checklist",
        rules=["fuente_verificable: cita fuente"],
    )
    missing = validate_critic_spec(spec, ["fuente_verificable", "revision_legal"])
    assert missing == ["revision_legal"]


def test_content_hash_is_sha256_hex():
    h = _content_hash("hola")
    assert len(h) == 64
    assert h == _content_hash("hola")
    assert h != _content_hash("hola!")


def test_render_prompt_includes_header_and_params():
    adapter = GenericPromptAdapter(_profile())
    text = adapter.render_prompt(_spec(), _model())
    assert "task: alignment_reinforcement" in text
    assert "model: meta-models/Muse-Glimmer-30B" in text
    assert "## System message" in text
    assert "## Request params" in text
