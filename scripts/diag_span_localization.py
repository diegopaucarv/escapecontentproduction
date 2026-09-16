"""Diagnóstico: ¿los text_span del LLM son localizables en los chunks?

Carga los chunks de los docs ready y prueba localizar spans de ejemplo
(extraídos del debug output del LLM) con: exacto, whitespace-normalizado,
y fuzzy (partial_ratio_alignment).
"""

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

import re

from sqlalchemy import text

from src.db.session import SessionLocal

SPANS = [
    # doc 4 (del debug output)
    "Una ley no 'toca' el cerebro; un pensamiento no es comunicacion hasta que se emite.",
    # doc 5 (del debug output)
    "El error de las ciencias sociales ha sido colapsar tres fenómenos distintos: el sistema psíquico autopoiético (Maturana/Luhmann), la persona social como constructo comunicativo, y el proceso termodinámico de individualización (Simondon).",
    "Simondon acierta al ver la individualización como dinámica interna de reestructuración somato-psíquica, pero carece de un marco para la clausura comunicativa.",
    "Luhmann acierta en la clausura operativa y en situar al humano fuera de la sociedad, pero a veces supedita la individualización psíquica a un mero epifenómeno de la semántica social.",
]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    try:
        from rapidfuzz import fuzz
        from rapidfuzz.fuzz import partial_ratio_alignment
    except ImportError as exc:
        print(f"rapidfuzz no disponible: {exc}")
        return
    print(
        f"rapidfuzz OK: partial_ratio_alignment={hasattr(fuzz, 'partial_ratio_alignment')}"
    )

    session = SessionLocal()
    try:
        rows = session.execute(
            text(
                "SELECT id, content FROM kag_chunks "
                "WHERE doc_id IN (4, 5) ORDER BY doc_id, chunk_index"
            )
        ).fetchall()
        print(f"{len(rows)} chunks cargados.")
        for span in SPANS:
            print(f"\n--- span ({len(span)} chars): {span[:80]}...")
            found_exact = found_norm = found_fuzzy = None
            best = (0, None)
            for cid, content in rows:
                if span in content:
                    found_exact = cid
                    break
                norm_c = re.sub(r"\s+", " ", content)
                norm_s = re.sub(r"\s+", " ", span).strip()
                if norm_s and norm_s in norm_c:
                    found_norm = cid
                    break
                try:
                    res = partial_ratio_alignment(norm_c, norm_s)
                    if res and res.score > best[0]:
                        best = (res.score, cid)
                except Exception:
                    pass
            if found_exact is not None:
                print(f"  EXACTO en chunk {found_exact}")
            elif found_norm is not None:
                print(f"  NORMALIZADO en chunk {found_norm}")
            else:
                print(f"  FUZZY best: score={best[0]:.1f} chunk={best[1]}")
    finally:
        session.close()


if __name__ == "__main__":
    main()
