"""Test rápido de parse_llm_output con fences de markdown."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm.base import parse_llm_output

CASES = [
    # Fences de markdown (el caso que fallaba)
    (
        '```json\n{"divisions": [{"chapter_id": "x", "propositions": [{"statement": "hola"}]}]}\n```',
        "divisions",
    ),
    # Texto antes/después del JSON
    ('Aquí tienes: {"a": 1} espero que sirva', "a"),
    # JSON puro
    ('{"a": 1}', "a"),
    # No-JSON
    ("no hay json aquí", None),
]

for text, expected_key in CASES:
    result = parse_llm_output(text)
    if expected_key is None:
        status = "OK" if result == {} else f"FAIL (esperaba {{}}, got {result})"
    else:
        status = (
            "OK"
            if expected_key in result
            else f"FAIL (esperaba key {expected_key}, got {result})"
        )
    print(
        f"{status}: {text[:60]!r} -> {list(result.keys()) if isinstance(result, dict) else result}"
    )
