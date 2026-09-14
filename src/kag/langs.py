"""Constantes y helpers de idiomas para el segmentador KAG.

Centraliza qué idiomas soportan coref de Stanza y qué procesadores pedirle a
Stanza por idioma. Lo usan src/kag/segmentador.py, scripts/ensure_languages.py
y scripts/fetch_language_catalog.py.
"""

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

# Procesadores base que pide el segmentador (get_stanza).
_BASE_PROCESSORS = ["tokenize", "pos", "lemma", "depparse", "coref"]


def stanza_processors(lang: str) -> str:
    """Procesadores de Stanza para el idioma (CSV).

    El pipeline es uniforme: `tokenize,pos,lemma,depparse,coref`. Ya no se usa
    `constituency` (la extracción de sujetos NP la hace spaCy). Para idiomas sin
    coref se quita `coref`.
    """
    procs = list(_BASE_PROCESSORS)
    if lang not in STANZA_COREF_LANGS:
        procs.remove("coref")
    return ",".join(procs)
