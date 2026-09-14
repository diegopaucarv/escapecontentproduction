"""Descarga paciente de los modelos de idioma que falten (spaCy + Stanza).

Lee config/languages.yaml (la fuente de verdad de idiomas activos) y descarga
solo lo que falta, con reintentos ante fallos de red.

Reglas:
  - spaCy: baja `{lang}_core_news_{size}` para cada idioma activo. Si el tamaño
    pedido no existe para un idioma, baja al más cercano disponible.
  - Stanza: baja con los procesadores EXACTOS que usa el segmentador
    (`tokenize,pos,lemma,depparse,constituency,coref`) SOLO para los idiomas
    que soportan coref. Para el resto, no baja Stanza (el segmentador
    desactiva la correferencia vía try/except en get_stanza).

Uso:
    python scripts/ensure_languages.py
    python scripts/ensure_languages.py --dry-run   # solo reporta qué falta
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml

# Procesadores EXACTOS que pide el segmentador (src/kag/segmentador.py get_stanza).
STANZA_PROCESSORS = "tokenize,pos,lemma,depparse,constituency,coref"

# Idiomas con coref de Stanza (fuente: docs oficiales de Stanza).
STANZA_COREF_LANGS = {
    "ca",
    "cs",
    "de",
    "en",
    "es",
    "fr",
    "he",
    "hi",
    "nb",
    "nn",
    "pl",
    "ru",
    "ta",
}

# Orden de tamaños spaCy para el fallback (de más a menos pesado).
SPACY_SIZES = ["trf", "lg", "md", "sm"]


def _load_config() -> dict:
    cfg_path = ROOT / "config" / "languages.yaml"
    if not cfg_path.exists():
        sys.exit(f"❌ No existe {cfg_path}. Crea la config primero.")
    with cfg_path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _spacy_installed(model: str) -> bool:
    try:
        import spacy

        return spacy.util.is_package(model)
    except Exception:  # noqa: BLE001
        return False


def _download_spacy(model: str, retries: int) -> bool:
    import spacy.cli

    for attempt in range(1, retries + 1):
        try:
            print(
                f"[spacy] intento {attempt}/{retries} — descargando {model}...",
                flush=True,
            )
            spacy.cli.download(model, quiet=True)
            if _spacy_installed(model):
                print(f"[spacy] ✅ {model} instalado", flush=True)
                return True
        except Exception as exc:  # noqa: BLE001
            print(
                f"[spacy] ❌ intento {attempt} falló: {type(exc).__name__}: {exc}",
                flush=True,
            )
        if attempt < retries:
            time.sleep(5)
    return False


def _stanza_downloaded(lang: str) -> bool:
    """True si el modelo de Stanza del idioma ya está en disco.

    No basta con mirar el resources.json (lista TODOS los idiomas disponibles,
    no los descargados). Se comprueba que exista el directorio del idioma en
    la ruta de recursos de Stanza (~/stanza_resources/{lang}).
    """
    try:
        from stanza.resources.common import DEFAULT_MODEL_DIR

        return (Path(DEFAULT_MODEL_DIR) / lang).is_dir()
    except Exception:  # noqa: BLE001
        return False


def _download_stanza(lang: str, retries: int) -> bool:
    import stanza

    for attempt in range(1, retries + 1):
        try:
            print(
                f"[stanza] intento {attempt}/{retries} — descargando {lang} "
                f"({STANZA_PROCESSORS})...",
                flush=True,
            )
            stanza.download(lang, processors=STANZA_PROCESSORS, verbose=False)
            if _stanza_downloaded(lang):
                print(f"[stanza] ✅ {lang} descargado", flush=True)
                return True
        except Exception as exc:  # noqa: BLE001
            print(
                f"[stanza] ❌ intento {attempt} falló: {type(exc).__name__}: {exc}",
                flush=True,
            )
        if attempt < retries:
            time.sleep(5)
    return False


def _resolve_spacy_model(lang: str, size: str) -> str | None:
    """Devuelve el modelo spaCy a usar, con fallback de tamaño si el pedido
    no existe para el idioma. Usa el catálogo si está disponible.

    El nombre real varía por idioma (en_core_web_md vs es_core_news_md), así
    que se busca en el catálogo el modelo que termina en '_{size}' en vez de
    construirlo a ciegas.
    """
    catalog_path = ROOT / "data" / "language_catalog.json"
    available: list[str] = []
    if catalog_path.exists():
        import json

        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        for name in catalog.get("spacy", {}).get(lang, []):
            available.append(name.split(" (")[0])

    # Orden de preferencia: el tamaño pedido primero, luego los demás.
    ordered = [size] + [s for s in SPACY_SIZES if s != size]
    for s in ordered:
        if available:
            # Buscar el modelo real del catálogo que termina en '_{size}'.
            for name in available:
                if name.endswith(f"_{s}"):
                    return name
        else:
            # Sin catálogo: intentar la convención más común.
            candidate = f"{lang}_core_news_{s}"
            if _spacy_installed(candidate) or s == size:
                return candidate
    return None


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="solo reporta qué falta")
    args = parser.parse_args()

    cfg = _load_config()
    langs = cfg.get("languages", [])
    size = cfg.get("spacy_size", "md")
    use_coref = cfg.get("use_stanza_coref", True)
    retries = cfg.get("download_retries", 8)

    if not langs:
        print("⚠  No hay idiomas activos en config/languages.yaml.")
        return 0

    print(f"Idiomas activos: {', '.join(langs)}")
    print(
        f"spaCy size: {size} | Stanza coref: {'sí' if use_coref else 'no'} | retries: {retries}"
    )

    ok = True

    # ── spaCy ────────────────────────────────────────────────────────────────
    print("\n=== spaCy ===")
    for lang in langs:
        model = _resolve_spacy_model(lang, size)
        if model is None:
            print(f"[spacy] ⚠  No hay modelo spaCy para '{lang}' — se omite.")
            continue
        if _spacy_installed(model):
            print(f"[spacy] ✓ {model} ya instalado")
            continue
        if args.dry_run:
            print(f"[spacy] faltaría: {model}")
            continue
        if not _download_spacy(model, retries):
            ok = False

    # ── Stanza ──────────────────────────────────────────────────────────────
    if use_coref:
        print("\n=== Stanza (coref) ===")
        for lang in langs:
            if lang not in STANZA_COREF_LANGS:
                print(
                    f"[stanza] ⚠  '{lang}' no soporta coref — se omite Stanza "
                    "(el segmentador desactiva la correferencia)."
                )
                continue
            if _stanza_downloaded(lang):
                print(f"[stanza] ✓ {lang} ya descargado")
                continue
            if args.dry_run:
                print(f"[stanza] faltaría: {lang} ({STANZA_PROCESSORS})")
                continue
            if not _download_stanza(lang, retries):
                ok = False
    else:
        print("\n=== Stanza (coref) ===")
        print("use_stanza_coref=false → no se descarga Stanza.")

    print("\n" + ("✅ Todo listo." if ok else "⚠  Hubo fallos — revisa los mensajes."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
