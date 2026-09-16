"""Corrige el typo (backtick) en src/llm/base.py L140."""

from __future__ import annotations

import io
from pathlib import Path

path = Path(__file__).resolve().parent.parent / "src" / "llm" / "base.py"
content = path.read_text(encoding="utf-8", newline="")

bad = 're.sub(r",\\s*([}\\]])`, r"\\1", cand)'
good = 're.sub(r",\\s*([}\\]])", r"\\1", cand)'

count = content.count(bad)
print(f"ocurrencias del typo: {count}")
if count != 1:
    raise SystemExit(f"ERROR: se esperaba 1, se encontraron {count}")

path.write_text(content.replace(bad, good), encoding="utf-8", newline="")
print("OK: typo corregido")
