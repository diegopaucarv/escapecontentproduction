"""
Extracción determinista de entidades y relaciones con spaCy (sin LLM).

Reemplaza la extracción LLM por chunk (1 llamada a Together por chunk, un
leak de dinero en docs grandes) por un pipeline 100% local y gratuito basado
en dependency parsing enfocado en sustantivos:

  1. Candidatos: noun_chunks + tokens NOUN/PROPN (nsubj, dobj, nmod) — sin NER.
  2. Supresión NER: los spans que spaCy reconoce como entidad nombrada
     (PERSON, ORG, GPE, LOC, ...) NO entran al grafo — se sirven por la capa
     de búsqueda textual (FTS/regex) en el querying, donde el match exacto
     importa más que la vecindad semántica. Los conceptos que mencionan una
     entidad NER ("capital cultural de Bourdieu") sí se conservan.
  3. Normalización: strip de determinantes + lowercase + colapso de espacios.
  4. Filtro de ruido: frases vacías/numéricas; palabras sueltas solo si
     aparecen ≥2 veces en el chunk o son PROPN.
  5. Canonicalización por embeddings (opcional): agrupa variantes
     superficiales por coseno ≥ umbral usando el modelo de embeddings de la
     DB (embed_fn inyectado, p. ej. src.embeddings.embed_texts).
  6. Relaciones: co-ocurrencia dentro de la misma oración → CO_OCURRE.

Cero costo por llamada; solo CPU + el modelo de embeddings ya en memoria.
"""

from __future__ import annotations

import re
from collections import Counter

# Etiquetas NER que NO entran al grafo: nombres propios, países,
# organizaciones, fechas, cifras... se sirven por búsqueda textual.
_NER_EXCLUDED = {
    "PERSON",
    "ORG",
    "GPE",
    "LOC",
    "NORP",
    "FAC",
    "PRODUCT",
    "EVENT",
    "LAW",
    "LANGUAGE",
    "DATE",
    "TIME",
    "PERCENT",
    "MONEY",
    "QUANTITY",
    "ORDINAL",
    "CARDINAL",
}

# Umbral de coseno para canonicalización (variantes superficiales del mismo
# concepto). Alto a propósito: fusionar conceptos distintos es peor que
# dejar variantes separadas.
DEFAULT_SIM_THRESHOLD = 0.92

# Margen mínimo entre la mejor y la segunda mejor similitud para fusionar
# variantes: si dos representantes están igual de cerca, es ambiguo y NO se
# fusiona (mejor dejar variantes separadas que fusionar conceptos distintos).
DEFAULT_GAP_MARGIN = 0.03


def _norm(name: str) -> str:
    """Minúsculas + strip + colapso de espacios (clave de dedup)."""
    return " ".join(name.lower().strip().split())


def _strip_leading_determiners(tokens) -> str:
    """Quita determinantes iniciales de un noun_chunk de spaCy.

    "el capital cultural" -> "capital cultural". Los tokens son los del
    chunk (con .pos_); se conserva el resto tal cual.
    """
    start = 0
    while start < len(tokens) and tokens[start].pos_ == "DET":
        start += 1
    return " ".join(t.text for t in tokens[start:])


def _is_noise(name: str) -> bool:
    """True si la frase es ruido: vacía, demasiado corta o numérica."""
    n = _norm(name)
    if not n or len(n) < 3:
        return True
    if re.fullmatch(r"[\d\W_]+", n):
        return True
    return False


def _candidate_from_chunk(chunk, lang: str) -> str | None:
    """Convierte un noun_chunk de spaCy en candidato, o None si es ruido/NER.

    Reglas:
    - Se descartan los chunks cuyo núcleo es una entidad NER (nombres
      propios, países, organizaciones → capa textual, no grafo).
    - Se descartan los chunks que son íntegramente un span NER.
    - Se conservan los conceptos que mencionan entidades NER ("capital
      cultural de Bourdieu").
    """
    root = chunk.root
    if root.ent_type_ in _NER_EXCLUDED:
        return None
    # Span íntegramente NER (p. ej. "Francia" o "Banco Mundial"): TODOS los
    # tokens son entidades nombradas. Un concepto que solo MENCIONA una
    # entidad ("capital cultural de Bourdieu") tiene tokens sin ent_type_
    # y se conserva.
    if all(t.ent_type_ in _NER_EXCLUDED for t in chunk):
        return None
    name = _strip_leading_determiners(list(chunk))
    if _is_noise(name):
        return None
    return name


def _candidate_from_token(tok, lang: str) -> str | None:
    """Candidato de un token suelto NOUN/PROPN (fuera de noun_chunk)."""
    if tok.ent_type_ in _NER_EXCLUDED:
        return None
    if tok.pos_ not in ("NOUN", "PROPN"):
        return None
    if _is_noise(tok.text):
        return None
    return tok.text


def _canonicalize(
    names: list[str],
    embed_fn,
    threshold: float = DEFAULT_SIM_THRESHOLD,
    gap_margin: float = DEFAULT_GAP_MARGIN,
):
    """Agrupa variantes superficiales por coseno ≥ threshold (greedy).

    Devuelve dict {canonical_name: [variantes]}. El canónico es la variante
    más frecuente del grupo (empate: la más larga). Greedy O(N×R) contra
    representantes — evita la matriz N×N completa.

    `gap_margin`: además de superar el umbral, la mejor similitud debe
    superar a la segunda mejor por al menos este margen. Si dos
    representantes están igual de cerca (empate), la variante es ambigua y
    se queda separada en vez de fusionarse a ciegas.
    """
    if not names:
        return {}
    freqs = Counter(names)
    unique = list(freqs.keys())
    if embed_fn is None or len(unique) == 1:
        return {n: [n] for n in unique}
    try:
        vecs = embed_fn(unique)
    except Exception:  # noqa: BLE001 — embeddings no disponibles: sin agrupar
        return {n: [n] for n in unique}
    if len(vecs) != len(unique):
        return {n: [n] for n in unique}

    import numpy as np

    arr = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    arr = arr / norms

    groups: list[list[int]] = []
    for i in range(len(unique)):
        best = -1
        best_sim = -1.0
        second_sim = -1.0
        for g, members in enumerate(groups):
            sim = float(np.dot(arr[i], arr[members[0]]))
            if sim > best_sim:
                second_sim = best_sim
                best_sim = sim
                best = g
            elif sim > second_sim:
                second_sim = sim
        # Fusiona solo si supera el umbral Y con margen claro sobre la
        # segunda mejor (si dos representantes están igual de cerca, es
        # ambiguo → no fusionar).
        if (
            best >= 0
            and best_sim >= threshold
            and (best_sim - second_sim) >= gap_margin
        ):
            groups[best].append(i)
        else:
            groups.append([i])

    result = {}
    for members in groups:
        variants = [unique[i] for i in members]
        canonical = max(variants, key=lambda v: (freqs[v], len(v)))
        result[canonical] = variants
    return result


def extract_entities_deterministic(
    nlp,
    text: str,
    embed_fn=None,
    sim_threshold: float = DEFAULT_SIM_THRESHOLD,
    min_freq: int = 2,
    lang: str = "es",
) -> dict:
    """Extrae entidades (sustantivos vía depparse) y relaciones (co-ocurrencia).

    Sin LLM. `nlp` es un pipeline spaCy ya cargado (p. ej. el del
    segmentador). `embed_fn` (opcional) embebe una lista de strings → lista
    de vectores (p. ej. src.embeddings.embed_texts) para canonicalizar
    variantes superficiales.

    Devuelve {"entities": [{"name", "type", "description"}],
              "relations": [{"source", "target", "type", "description"}]}
    — el mismo shape que espera _store_entities_relations.
    """
    doc = nlp(text)
    if doc is None:
        return {"entities": [], "relations": []}

    # 1. Candidatos: noun_chunks + tokens NOUN/PROPN sueltos.
    candidates: list[str] = []
    seen_chunks = set()
    for chunk in doc.noun_chunks:
        name = _candidate_from_chunk(chunk, lang)
        if name:
            key = _norm(name)
            if key not in seen_chunks:
                seen_chunks.add(key)
                candidates.append(name)
    for tok in doc:
        if tok.pos_ in ("NOUN", "PROPN") and not tok.is_stop:
            name = _candidate_from_token(tok, lang)
            if name and _norm(name) not in seen_chunks:
                seen_chunks.add(_norm(name))
                candidates.append(name)

    # 2. Filtro de frecuencia: palabras sueltas solo si aparecen ≥ min_freq
    #    veces en el chunk (o son PROPN). Las frases multi-palabra se quedan.
    freq = Counter(_norm(c) for c in candidates)
    kept: list[str] = []
    for c in candidates:
        n = _norm(c)
        n_words = len(n.split())
        if n_words >= 2 or freq[n] >= min_freq or n.split()[-1].istitle():
            kept.append(c)

    # 3. Canonicalización por embeddings (variantes → canónico).
    canon = _canonicalize(kept, embed_fn, sim_threshold)

    # 4. Relaciones: co-ocurrencia dentro de la misma oración.
    #    Mapeamos cada oración a los canónicos presentes (por substring
    #    normalizado) y creamos CO_OCURRE entre pares.
    canon_norm = {_norm(c): c for c in canon}
    sent_entities: list[list[str]] = []
    for sent in doc.sents:
        sent_text = _norm(sent.text)
        present = []
        for nn, cname in canon_norm.items():
            if nn and nn in sent_text:
                present.append(cname)
        if len(present) >= 2:
            sent_entities.append(present)

    entities = [{"name": c, "type": "concept", "description": ""} for c in canon]
    relations = []
    seen_rels = set()
    for present in sent_entities:
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                a, b = present[i], present[j]
                key = (a, b) if a < b else (b, a)
                if key in seen_rels:
                    continue
                seen_rels.add(key)
                relations.append(
                    {
                        "source": a,
                        "target": b,
                        "type": "CO_OCURRE",
                        "description": "",
                    }
                )
    return {"entities": entities, "relations": relations}
