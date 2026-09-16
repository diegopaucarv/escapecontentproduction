"""Tests de los guardrails universales de respuestas LLM (src/llm/sanitize.py)."""

from __future__ import annotations

import pytest

from src.llm.sanitize import (
    LLMJSONError,
    LLMTruncatedError,
    sanitize_llm_json,
    sanitize_llm_response,
)


def test_sanitize_valid_json_passthrough():
    assert sanitize_llm_response('{"a": 1}') == '{"a": 1}'
    assert sanitize_llm_json('{"a": 1}') == {"a": 1}


def test_sanitize_fences():
    text = '```json\n{"a": 1}\n```'
    assert sanitize_llm_json(text) == {"a": 1}


def test_sanitize_text_around():
    text = 'Aquí está el JSON: {"a": 1} Espero que sirva'
    assert sanitize_llm_json(text) == {"a": 1}


def test_sanitize_trailing_commas():
    assert sanitize_llm_json('{"a": 1, "b": [2, 3,],}') == {"a": 1, "b": [2, 3]}


def test_sanitize_truncated_string_raises():
    # "Unterminated string" — cortado a mitad de un valor string.
    with pytest.raises(LLMTruncatedError):
        sanitize_llm_json('{"a": "citations_references')


def test_sanitize_truncated_mid_object_raises():
    # "Expecting property name" — termina a mitad de estructura.
    with pytest.raises(LLMTruncatedError):
        sanitize_llm_json('{"a": 1, "b":')


def test_sanitize_truncated_open_brace_raises():
    # Termina en "{" sin cerrar.
    with pytest.raises(LLMTruncatedError):
        sanitize_llm_json('{"a": {"b": [1, 2')


def test_sanitize_not_json_raises():
    with pytest.raises(LLMJSONError):
        sanitize_llm_json("esto no es json")


def test_sanitize_non_dict_json_raises():
    with pytest.raises(LLMJSONError):
        sanitize_llm_json("[1, 2, 3]")


def test_sanitize_empty_raises():
    with pytest.raises(LLMJSONError):
        sanitize_llm_json("")
    with pytest.raises(LLMJSONError):
        sanitize_llm_json("   ")
