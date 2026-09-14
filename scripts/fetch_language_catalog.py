"""Descarga los catálogos oficiales de modelos de spaCy y Stanza.

Genera dos artefactos:
  - data/language_catalog.json  → catálogo machine-readable (lo usa ensure_languages.py).
  - docs/languages.md           → documento humano con todos los idiomas.

Fuentes oficiales:
  - spaCy:  https://raw.githubusercontent.com/explosion/spacy-models/master/compatibility.json
  - Stanza: resources.json del repo stanfordnlp/stanza-resources (se intenta
            cargar vía stanza si está instalado; si no, por HTTP directo).

Uso:
    python scripts/fetch_language_catalog.py
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CATALOG_PATH = ROOT / "data" / "language_catalog.json"
DOC_PATH = ROOT / "docs" / "languages.md"

SPACY_COMPAT_URL = (
    "https://raw.githubusercontent.com/explosion/spacy-models/master/compatibility.json"
)
STANZA_RAW_URL = "https://raw.githubusercontent.com/stanfordnlp/stanza-resources/main/resources_1.8.0.json"

# Idiomas con coref de Stanza (compartido con src/kag/segmentador.py).
from src.kag.langs import STANZA_COREF_LANGS


def _http_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


def fetch_spacy() -> tuple[str, dict[str, list[str]]]:
    """Devuelve (versión_compatible, {lang: [modelo (vX)...]})."""
    compat = _http_json(SPACY_COMPAT_URL)
    versions = sorted(compat.get("spacy", {}).keys())
    latest = versions[-1]
    models = compat["spacy"][latest]
    by_lang: dict[str, list[str]] = {}
    for name, ver in models.items():
        lang = name.split("_")[0]
        by_lang.setdefault(lang, []).append(f"{name} (v{ver})")
    return latest, by_lang


def fetch_stanza() -> dict[str, list[str]]:
    """Devuelve {lang: [procesadores]}. Intenta vía stanza; si no, HTTP."""
    try:
        import stanza  # noqa: F401
        from stanza.resources.common import load_resources_json

        resources = load_resources_json()
    except Exception:  # noqa: BLE001 — stanza no instalado o falla: HTTP directo
        resources = _http_json(STANZA_RAW_URL)

    by_lang: dict[str, list[str]] = {}
    for key, value in resources.items():
        if not isinstance(value, dict) or "alias" in value:
            continue
        processors = [
            p
            for p in value
            if p
            not in (
                "packages",
                "url",
                "default_processors",
                "default_md5",
                "lang_name",
                "pretrain",
                "backward_charlm",
                "forward_charlm",
            )
        ]
        by_lang[key] = processors
    return by_lang


def _write_doc(spacy_ver: str, spacy: dict, stanza: dict) -> None:
    lines = [
        "# Catálogo de idiomas — spaCy y Stanza",
        "",
        (
            f"Generado por `scripts/fetch_language_catalog.py`. "
            f"spaCy compatible: **{spacy_ver}**."
        ),
        "",
        "## spaCy — idiomas con pipeline entrenado",
        "",
        "| Código | Modelos |",
        "| --- | --- |",
    ]
    for lang in sorted(spacy):
        models = ", ".join(spacy[lang])
        lines.append(f"| **{lang}** | {models} |")

    lines += [
        "",
        "## Stanza — idiomas con modelos UD",
        "",
        "| Código | Procesadores | Coref |",
        "| --- | --- | --- |",
    ]
    for lang in sorted(stanza):
        procs = ", ".join(stanza[lang])
        coref = "✅" if lang in STANZA_COREF_LANGS else "—"
        lines.append(f"| **{lang}** | {procs} | {coref} |")

    lines += [
        "",
        "## Notas",
        "",
        (
            "- El segmentador pide a Stanza los procesadores "
            "`tokenize,pos,lemma,depparse,coref` (sin constituency: la "
            "extracción de sujetos NP la hace spaCy)."
        ),
        "- El coref de Stanza **solo** existe para: "
        + ", ".join(sorted(STANZA_COREF_LANGS))
        + ".",
        (
            "- Para idiomas sin coref, el segmentador usa spaCy y desactiva la "
            "correferencia (try/except en `get_stanza`)."
        ),
    ]
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("Descargando catálogo de spaCy...", flush=True)
    spacy_ver, spacy = fetch_spacy()
    print(f"  → {len(spacy)} idiomas con pipeline (compat {spacy_ver})", flush=True)

    print("Descargando catálogo de Stanza...", flush=True)
    stanza = fetch_stanza()
    print(f"  → {len(stanza)} idiomas con modelos UD", flush=True)

    catalog = {
        "spacy_version": spacy_ver,
        "spacy": spacy,
        "stanza": stanza,
        "stanza_coref_langs": sorted(STANZA_COREF_LANGS),
    }
    CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CATALOG_PATH.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"✅ Catálogo JSON → {CATALOG_PATH.relative_to(ROOT)}", flush=True)

    _write_doc(spacy_ver, spacy, stanza)
    print(f"✅ Documento → {DOC_PATH.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
