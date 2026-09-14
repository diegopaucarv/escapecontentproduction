"""
CLI del compilador de prompts.

Uso:
    python -m src.llm.compile_prompts

Compila todas las specs activas (prompt_templates) para todos los modelos
de chat activos (llm_models) y congela los artefactos en prompt_artifacts.
Idempotente: si nada cambió, no crea versiones nuevas.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.llm.compiler import compile_prompts


def _fix_db_host() -> None:
    """Reemplaza '@db:' por '@localhost:' en DATABASE_URL/DATABASE_URL_ASYNC.

    El host 'db' es la red Docker y no resuelve desde el host; las credenciales
    son las mismas. Debe llamarse ANTES de importar src.db.session.
    """
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    for var in ("DATABASE_URL", "DATABASE_URL_ASYNC"):
        val = os.environ.get(var, "")
        if not val and env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith(f"{var}="):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
        if "@db:" in val:
            os.environ[var] = val.replace("@db:", "@localhost:")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compila las specs de prompts a artefactos inmutables."
    )
    parser.parse_args()

    _fix_db_host()
    from src.db.session import SessionLocal

    session = SessionLocal()
    try:
        summary = compile_prompts(session)
    finally:
        session.close()

    print("Compilación de prompts:")
    print(f"  Compilados: {summary['compiled']}")
    print(f"  Sin cambios (skip): {summary['skipped']}")
    if summary["warnings"]:
        print("  Warnings:")
        for w in summary["warnings"]:
            print(f"    - {w}")
    for a in summary["artifacts"]:
        print(f"  - {a['model']} / {a['task']} v{a['version']} ({a['hash'][:12]}…)")


if __name__ == "__main__":
    main()
