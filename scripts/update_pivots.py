"""Agente mantenedor de pivots conversacionales multilingües.

Actualiza src/kag/pivots.json (los pivots que usa conversational_sbd en
src/kag/segmentador.py) cuando se añade un nuevo idioma.

Uso:
    python scripts/update_pivots.py --lang it --pivots "allora,bene,cioè,ma,perché,poi"
    python scripts/update_pivots.py --lang it --pivots "allora" "bene" "cioè"
    python scripts/update_pivots.py --lang it --pivots "allora,bene" --replace

Reglas:
  - Los pivots se guardan en minúsculas y sin duplicados.
  - Por defecto se AÑADEN a los existentes del idioma; con --replace se
    reemplaza la lista completa.
  - Si el idioma no existe, se crea.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PIVOTS_PATH = ROOT / "src" / "kag" / "pivots.json"


def _load() -> dict[str, list[str]]:
    if not PIVOTS_PATH.exists():
        return {}
    return json.loads(PIVOTS_PATH.read_text(encoding="utf-8"))


def _save(data: dict[str, list[str]]) -> None:
    PIVOTS_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", required=True, help="código ISO del idioma (ej. it)")
    parser.add_argument(
        "--pivots",
        nargs="+",
        required=True,
        help="pivots a añadir (separados por coma o como argumentos separados)",
    )
    parser.add_argument(
        "--replace", action="store_true", help="reemplaza la lista del idioma"
    )
    args = parser.parse_args()

    lang = args.lang.lower()
    # Acepta "a,b,c" o ["a", "b", "c"].
    new_pivots: list[str] = []
    for chunk in args.pivots:
        new_pivots.extend(p.strip().lower() for p in chunk.split(",") if p.strip())
    new_pivots = list(dict.fromkeys(new_pivots))  # dedup preservando orden

    data = _load()
    if args.replace or lang not in data:
        data[lang] = new_pivots
        action = "creado/reemplazado"
    else:
        existing = data[lang]
        merged = list(dict.fromkeys(existing + new_pivots))
        data[lang] = merged
        action = "actualizado"

    _save(data)
    print(f"✅ pivots.json {action} para '{lang}': {data[lang]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
