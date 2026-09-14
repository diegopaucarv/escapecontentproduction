"""Diagnóstico del estado de la DB KAG (sin modificar nada).

Cuenta documentos, chunks, entidades, tripletas, figuras, resúmenes y
comprueba cuántos chunks tienen embedding no nulo. Útil para saber por qué
una consulta devuelve 0 resultados.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _fix_db_host() -> None:
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


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    _fix_db_host()
    from sqlalchemy import text

    from src.db.session import SessionLocal

    session = SessionLocal()
    try:
        tables = [
            "kag_documents",
            "kag_chunks",
            "kag_entities",
            "kag_relations",
            "kag_figures",
            "kag_doc_summaries",
        ]
        print("=== CONTEO POR TABLA ===")
        for t in tables:
            try:
                n = session.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar()
                print(f"  {t}: {n}")
            except Exception as exc:  # noqa: BLE001
                print(f"  {t}: ERROR {type(exc).__name__}: {exc}")

        print("\n=== CHUNKS: embedding NULL vs poblado ===")
        try:
            total = session.execute(text("SELECT COUNT(*) FROM kag_chunks")).scalar()
            null_emb = session.execute(
                text("SELECT COUNT(*) FROM kag_chunks WHERE embedding IS NULL")
            ).scalar()
            print(f"  total chunks: {total}")
            print(f"  embedding NULL: {null_emb}")
            print(f"  embedding poblado: {total - null_emb}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR {type(exc).__name__}: {exc}")

        print("\n=== CHUNKS: content_tsv NULL vs poblado ===")
        try:
            total = session.execute(text("SELECT COUNT(*) FROM kag_chunks")).scalar()
            null_tsv = session.execute(
                text("SELECT COUNT(*) FROM kag_chunks WHERE content_tsv IS NULL")
            ).scalar()
            print(f"  content_tsv NULL: {null_tsv}")
            print(f"  content_tsv poblado: {total - null_tsv}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR {type(exc).__name__}: {exc}")

        print(
            "\n=== MUESTRA DE CHUNKS (id, doc_id, token_estimate, len(embedding)) ==="
        )
        try:
            rows = session.execute(
                text(
                    "SELECT id, doc_id, token_estimate, "
                    "CASE WHEN embedding IS NULL THEN NULL "
                    "ELSE vector_dims(embedding) END AS dims "
                    "FROM kag_chunks LIMIT 5"
                )
            ).fetchall()
            for r in rows:
                print(
                    f"  chunk {r.id} | doc {r.doc_id} | tokens {r.token_estimate} | dims {r.dims}"
                )
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR {type(exc).__name__}: {exc}")

        print("\n=== DOCUMENTOS (id, doc_path) ===")
        try:
            rows = session.execute(
                text("SELECT id, doc_path FROM kag_documents LIMIT 10")
            ).fetchall()
            for r in rows:
                print(f"  {r.id} | {r.doc_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR {type(exc).__name__}: {exc}")
    finally:
        session.close()


if __name__ == "__main__":
    main()
