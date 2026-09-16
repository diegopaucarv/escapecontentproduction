"""
Seed de las specs de prompts del pipeline KAG (prompt-as-code).

Las specs viven en src/kag/prompts/specs.py (lista KAG_TEMPLATES, 20 specs
agnósticas de los prompts del sistema KAG: modo audited de la consulta, query
clásica e ingesta clásica expandida) y este seed las importa desde ahí como
fuente única. Cada spec captura el rol (intent) y las instrucciones (rules)
del SYSTEM prompt actual; el USER prompt (con sus placeholders) se construye
en runtime y NO vive en la spec.

A diferencia de src/db/seed_ai.py / seed_vision.py, NO requiere variables de
entorno: es datos puros. Tampoco compila artefactos — eso lo hace el
compilador una vez que los modelos y specs existen:

    python -m src.db.seed_kag_prompts   # inserta/actualiza las specs
    python -m src.llm.compile_prompts   # transpila specs -> prompt_artifacts

Uso:
    python -m src.db.seed_kag_prompts
"""

from __future__ import annotations

import argparse

from sqlalchemy import select

from src.db.models import PromptTemplate
from src.kag.prompts.specs import KAG_TEMPLATES


def _upsert_template(session, data: dict) -> PromptTemplate:
    template = (
        session.execute(
            select(PromptTemplate).where(PromptTemplate.task_key == data["task_key"])
        )
        .scalars()
        .first()
    )
    if template is None:
        template = PromptTemplate(**data)
        session.add(template)
        session.flush()
    else:
        for k, v in data.items():
            setattr(template, k, v)
    return template


def seed(session=None) -> dict:
    """Inserta/actualiza las specs de prompts KAG. Devuelve un resumen.

    No requiere variables de entorno. Compila los artefactos automáticamente
    (src/llm/compiler.py::compile_prompts) — idempotente: solo crea versiones
    nuevas si el contenido cambió.
    """
    own_session = session is None
    if own_session:
        from src.db.session import SessionLocal

        session = SessionLocal()
    try:
        templates = [_upsert_template(session, data) for data in KAG_TEMPLATES]
        session.commit()
        from src.llm.compiler import compile_prompts

        compiled = compile_prompts(session)
        return {
            "templates": [str(t.id) for t in templates],
            "task_keys": [t.task_key for t in templates],
            "compiled": compiled["compiled"],
            "skipped": compiled["skipped"],
        }
    finally:
        if own_session:
            session.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed de specs de prompts KAG (prompt-as-code)."
    )
    parser.parse_args()
    result = seed()
    print("Specs de prompts KAG listas:")
    print(f"  Specs de prompts ({len(result['task_keys'])}):")
    for key in result["task_keys"]:
        print(f"    - {key}")


if __name__ == "__main__":
    main()
