"""Consulta la config LLM activa: max_tokens, modelos, y el reasoning_mode."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_env_path = Path(__file__).resolve().parent.parent / ".env"
for _var in ("DATABASE_URL", "DATABASE_URL_ASYNC"):
    _val = os.environ.get(_var, "")
    if not _val and _env_path.exists():
        for _line in _env_path.read_text(encoding="utf-8").splitlines():
            _line = _line.strip()
            if _line.startswith(f"{_var}="):
                _val = _line.split("=", 1)[1].strip().strip('"').strip("'")
                break
    if "@db:" in _val:
        os.environ[_var] = _val.replace("@db:", "@localhost:")

from sqlalchemy import text

from src.db.session import SessionLocal

session = SessionLocal()
try:
    print("=== session_settings activa ===")
    rows = session.execute(
        text(
            "SELECT id, small_model, large_model, max_tokens_small, "
            "max_tokens_large, temperature_small, temperature_large, "
            "llm_retries, fallback_model FROM session_settings "
            "WHERE is_active = TRUE"
        )
    ).fetchall()
    for r in rows:
        print(dict(r._mapping))

    print("\n=== llm_models (large) ===")
    rows = session.execute(
        text(
            "SELECT id, model_name, provider, context_window, max_output_tokens, "
            "syntax_profile FROM llm_models WHERE model_name LIKE '%DeepSeek%' "
            "OR model_name LIKE '%Muse%'"
        )
    ).fetchall()
    for r in rows:
        m = dict(r._mapping)
        sp = m.get("syntax_profile") or {}
        print(
            f"  id={m['id']} name={m['model_name']} provider={m['provider']} "
            f"ctx={m['context_window']} max_out={m['max_output_tokens']}"
        )
        print(f"    reasoning_mode={sp.get('reasoning_mode')}")
finally:
    session.close()
