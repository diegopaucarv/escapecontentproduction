"""Analiza el output crudo del LLM guardado: ¿por qué falla json.loads?"""

from __future__ import annotations

import json
import sys
from pathlib import Path

path = Path(__file__).resolve().parent / "_raw_llm_output.json"
text = path.read_text(encoding="utf-8")
print(f"Tamaño: {len(text)} chars")

# 1. json.loads directo
try:
    data = json.loads(text)
    print("json.loads directo: OK")
    print(f"  keys: {list(data.keys())}")
    print(f"  divisions: {len(data.get('divisions', []))}")
except json.JSONDecodeError as exc:
    print(
        f"json.loads directo: FALLA en pos {exc.pos} (linea {exc.lineno}, col {exc.colno})"
    )
    print(f"  msg: {exc.msg}")
    # Mostrar contexto alrededor del error
    start = max(0, exc.pos - 120)
    end = min(len(text), exc.pos + 120)
    print(f"  contexto: ...{text[start:end]!r}...")

# 2. Probar el parse_llm_output actual
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.llm.base import parse_llm_output

data2 = parse_llm_output(text)
print(
    f"\nparse_llm_output: keys={list(data2.keys()) if isinstance(data2, dict) else type(data2)}"
)

# 3. Probar limpieza de trailing commas
import re

cleaned = re.sub(r",\s*([}\]])", r"\1", text)
try:
    data3 = json.loads(cleaned)
    print(f"\ncon trailing commas limpiados: OK, keys={list(data3.keys())}")
except json.JSONDecodeError as exc:
    print(
        f"\ncon trailing commas limpiados: FALLA en pos {exc.pos} (linea {exc.lineno})"
    )
    start = max(0, exc.pos - 120)
    end = min(len(cleaned), exc.pos + 120)
    print(f"  contexto: ...{cleaned[start:end]!r}...")
