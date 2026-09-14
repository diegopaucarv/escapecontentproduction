"""Demo del sistema KAG: indexa (opcional) y responde una pregunta.

Uso:
    python scripts/kag_demo.py --index "¿De qué trata el libro?"
    python scripts/kag_demo.py "¿Qué fórmula usa la propagación hacia atrás?"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Asegura que la raíz del proyecto esté en sys.path al correr como script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _fix_db_host() -> None:
    """Reemplaza '@db:' por '@localhost:' en DATABASE_URL/DATABASE_URL_ASYNC.

    El host 'db' es la red Docker y no resuelve desde el host; las credenciales
    son las mismas. Debe llamarse ANTES de importar src.db.session. Si la
    variable no está en el entorno, la lee del .env (pydantic-settings da
    prioridad a las env vars reales sobre el archivo .env).
    """
    env_path = Path(__file__).resolve().parent.parent / ".env"
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


def _warn_missing_embedding_settings(session) -> None:
    """Avisa si no hay embedding_settings activa (la búsqueda densa degradará).

    No bloquea: la consulta degrada a solo FTS (ver src/kag_query.py::ask).
    """
    from sqlalchemy import select

    from src.db.models import EmbeddingSetting

    setting = (
        session.execute(
            select(EmbeddingSetting).where(EmbeddingSetting.is_active.is_(True))
        )
        .scalars()
        .first()
    )
    if setting is None:
        print(
            "[KAG] ⚠ No hay embedding_settings activa: la búsqueda densa (pgvector) "
            "no estará disponible y la consulta degradará a solo FTS. "
            "Créala con `python -m src.db.seed_kag` (o POST /embedding-settings) "
            "y re-indexa con `python -m src.kag_ingest --force` para poblar "
            "los embeddings de los chunks existentes."
        )


def main() -> None:
    # Windows: la consola usa cp1252 y no imprime emojis — forzar UTF-8.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Demo del sistema KAG.")
    parser.add_argument(
        "query",
        nargs="?",
        help="Pregunta a responder (si no se usa --index).",
    )
    parser.add_argument(
        "--index",
        metavar="PREGUNTA",
        help="Indexa todo el knowledge_repository y luego responde esta pregunta.",
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--global-top-k", type=int, default=20)
    args = parser.parse_args()

    _fix_db_host()

    from src.db.session import SessionLocal
    from src.kag_ingest import index_all
    from src.kag_query import ask

    session = SessionLocal()
    try:
        if args.index:
            print("\n📚 FASE 1 — INGESTA")
            print("=" * 60)
            index_all(session, verbose=True)
            query = args.index
        elif args.query:
            query = args.query
        else:
            parser.error("Proporciona una pregunta o usa --index 'pregunta'.")

        print("\n💬 FASE 2 — CONSULTA")
        print("=" * 60)
        _warn_missing_embedding_settings(session)
        answer = ask(
            session,
            query,
            top_k=args.top_k,
            global_top_k=args.global_top_k,
            verbose=True,
        )
        print("\n" + "=" * 60)
        print("RESPUESTA FINAL")
        print("=" * 60)
        print(answer)
    finally:
        session.close()


if __name__ == "__main__":
    main()
