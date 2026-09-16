"""Extrae SOLO proposiciones de los documentos ya indexados (stage='ready').

Paso aislado de la ingesta clásica: para cada documento en stage='ready',
llama a `_extract_propositions_for_doc` (src/kag_ingest.py), que orquesta
`_extract_propositions_batch` + `_run_proposition_batches_parallel` +
`_store_propositions` sobre TODOS los chunks ya persistidos del doc.

NO corre la ingesta completa (chunking, entidades, resúmenes, figuras).

Uso:
    python scripts/extract_propositions_only.py
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

# Asegura que la raíz del proyecto esté en sys.path (correr como script).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Fix del host de DB ANTES de importar src.db.session (patrón de alembic/env.py):
# el .env apunta a host 'db' (red Docker), que no resuelve desde el host.
# Reemplaza '@db:' por '@localhost:' en DATABASE_URL/DATABASE_URL_ASYNC.
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
from src.kag_ingest import _extract_propositions_for_doc, _proposition_flags


def main() -> None:
    # Windows: la consola usa cp1252 y no imprime emojis — forzar UTF-8.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    session = SessionLocal()
    try:
        # Flags de proposiciones desde config (KAG_EXTRACT_PROPOSITIONS,
        # KAG_PROPOSITION_BATCH_SIZE, KAG_PROPOSITION_MODEL,
        # KAG_PROPOSITION_PARALLEL).
        enabled, batch_size, model_size, max_parallel = _proposition_flags(
            session, True
        )
        if not enabled:
            print(
                "[KAG] ⚠ KAG_EXTRACT_PROPOSITIONS está desactivado; "
                "no se extraerá nada."
            )
            return

        docs = session.execute(
            text(
                "SELECT id, doc_path, title, doc_type, stage FROM kag_documents "
                "WHERE stage = 'ready' ORDER BY id"
            )
        ).fetchall()
        if not docs:
            print("[KAG] No hay documentos en stage='ready'.")
            return

        print(
            f"[KAG] Documentos en stage='ready': {len(docs)} "
            f"(batch_size={batch_size}, model={model_size}, parallel={max_parallel})"
        )

        # Embeddings de los statements (mismo embed_fn que la ingesta; si el
        # modelo local falla, _store_propositions degrada a embedding NULL).
        from src.embeddings import embed_texts

        total_new = 0
        failures = []
        for row in docs:
            doc_id, doc_path, title, doc_type, stage = row
            print(
                f"\n[KAG] 📄 doc_id={doc_id} '{title}' "
                f"({doc_type}, stage={stage}, path={doc_path})"
            )
            before = session.execute(
                text("SELECT count(*) FROM kag_propositions WHERE doc_id = :id"),
                {"id": doc_id},
            ).scalar()
            try:
                _extract_propositions_for_doc(
                    session,
                    doc_id,
                    doc_path,
                    verbose=True,
                    batch_size_tokens=batch_size,
                    embed_fn=embed_texts,
                    model_size=model_size,
                    max_parallel=max_parallel,
                )
                session.commit()
            except Exception as exc:  # noqa: BLE001 — reportar y seguir
                session.rollback()
                failures.append((doc_id, exc))
                print(f"[KAG] ❌ doc_id={doc_id} FALLÓ: {type(exc).__name__}: {exc}")
                traceback.print_exc()
                continue
            after = session.execute(
                text("SELECT count(*) FROM kag_propositions WHERE doc_id = :id"),
                {"id": doc_id},
            ).scalar()
            new = after - before
            total_new += new
            print(
                f"[KAG] ✅ doc_id={doc_id}: {before} → {after} proposiciones (+{new})"
            )

        print(f"\n[KAG] Resumen: {total_new} proposiciones nuevas en total.")
        if failures:
            print(f"[KAG] ⚠ {len(failures)} documento(s) con errores:")
            for doc_id, exc in failures:
                print(f"  - doc_id={doc_id}: {type(exc).__name__}: {exc}")
    finally:
        session.close()


if __name__ == "__main__":
    main()
