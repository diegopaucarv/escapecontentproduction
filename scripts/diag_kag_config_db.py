"""Consulta session_settings: ¿hay overrides KAG_* en la DB?"""

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
    rows = session.execute(
        text("SELECT id, kag_config FROM session_settings WHERE is_active = TRUE")
    ).fetchall()
    for r in rows:
        m = dict(r._mapping)
        kag = m.get("kag_config") or {}
        print(f"id={m['id']}")
        if isinstance(kag, dict):
            for k, v in kag.items():
                if "PROPOSITION" in k or "BATCH" in k:
                    print(f"  {k} = {v}")
            if not any("PROPOSITION" in k or "BATCH" in k for k in kag):
                print("  (sin overrides de PROPOSITION/BATCH)")
        else:
            print(f"  kag_config tipo: {type(kag)}")
finally:
    session.close()
