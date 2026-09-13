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

from src.db.session import SessionLocal
from src.llm.compiler import compile_prompts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compila las specs de prompts a artefactos inmutables."
    )
    parser.parse_args()

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
