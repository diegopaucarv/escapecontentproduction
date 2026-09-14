"""Tests de la extracción determinista de entidades (src/kag/entities.py).

Sin DB, sin torch/spacy reales: se usa un spaCy fake (patrón de
tests/test_kag.py) y embeddings falsos (numpy). NO se importa
src.kag.segmentador a nivel de módulo.
"""

import numpy as np

from src.kag.entities import (
    _candidate_from_chunk,
    _candidate_from_token,
    _canonicalize,
    _is_noise,
    _norm,
    _strip_leading_determiners,
    extract_entities_deterministic,
)

# ---------------------------------------------------------------------
# Fakes de spaCy (duck-typed: lo que usa entities.py)
# ---------------------------------------------------------------------


class _FakeToken:
    def __init__(self, text, pos, ent_type="", is_stop=False):
        self.text = text
        self.pos_ = pos
        self.ent_type_ = ent_type
        self.is_stop = is_stop


class _FakeChunk:
    def __init__(self, tokens, root_idx=0):
        self._tokens = tokens
        self.root = tokens[root_idx]

    def __iter__(self):
        return iter(self._tokens)

    @property
    def text(self):
        return " ".join(t.text for t in self._tokens)


class _FakeSent:
    def __init__(self, tokens):
        self._tokens = tokens

    @property
    def text(self):
        return " ".join(t.text for t in self._tokens)


class _FakeDoc:
    def __init__(self, tokens, chunks=None, sents=None):
        self._tokens = tokens
        self.noun_chunks = chunks or []
        self.sents = sents or [_FakeSent(tokens)]

    def __iter__(self):
        return iter(self._tokens)

    @property
    def text(self):
        return " ".join(t.text for t in self._tokens)


class _FakeNlp:
    def __init__(self, doc):
        self._doc = doc

    def __call__(self, text):
        return self._doc


def _tok(text, pos="NOUN", ent="", stop=False):
    return _FakeToken(text, pos, ent, stop)


# ---------------------------------------------------------------------
# Helpers puros
# ---------------------------------------------------------------------


def test_norm_collapses_spaces_and_lowercases():
    assert _norm("  Capital   Cultural ") == "capital cultural"
    assert _norm("") == ""


def test_strip_leading_determiners():
    chunk = _FakeChunk(
        [_tok("el", "DET"), _tok("capital", "NOUN"), _tok("cultural", "ADJ")]
    )
    assert _strip_leading_determiners(list(chunk)) == "capital cultural"


def test_is_noise():
    assert _is_noise("")
    assert _is_noise("ab")  # < 3 chars
    assert _is_noise("12345")
    assert _is_noise("---")
    assert not _is_noise("capital cultural")
    assert not _is_noise("red neuronal")


# ---------------------------------------------------------------------
# _candidate_from_chunk — supresión NER
# ---------------------------------------------------------------------


def test_chunk_root_is_ner_suppressed():
    # "Bourdieu" (PERSON) como núcleo → None (capa textual, no grafo).
    chunk = _FakeChunk([_tok("Bourdieu", "PROPN", "PERSON")], root_idx=0)
    assert _candidate_from_chunk(chunk, "es") is None


def test_chunk_entirely_ner_suppressed():
    # "Banco Mundial" (ORG, ORG) → span íntegramente NER → None.
    chunk = _FakeChunk(
        [_tok("Banco", "PROPN", "ORG"), _tok("Mundial", "PROPN", "ORG")], root_idx=0
    )
    assert _candidate_from_chunk(chunk, "es") is None


def test_chunk_mentioning_ner_is_kept():
    # "capital cultural de Bourdieu": solo Bourdieu es NER → se conserva.
    chunk = _FakeChunk(
        [
            _tok("capital", "NOUN"),
            _tok("cultural", "ADJ"),
            _tok("de", "ADP"),
            _tok("Bourdieu", "PROPN", "PERSON"),
        ],
        root_idx=0,
    )
    assert _candidate_from_chunk(chunk, "es") == "capital cultural de Bourdieu"


def test_chunk_with_determiner_stripped():
    chunk = _FakeChunk(
        [_tok("el", "DET"), _tok("capital", "NOUN"), _tok("cultural", "ADJ")],
        root_idx=1,
    )
    assert _candidate_from_chunk(chunk, "es") == "capital cultural"


def test_candidate_from_token():
    assert _candidate_from_token(_tok("capital", "NOUN"), "es") == "capital"
    assert _candidate_from_token(_tok("Bourdieu", "PROPN", "PERSON"), "es") is None
    assert _candidate_from_token(_tok("corre", "VERB"), "es") is None
    assert _candidate_from_token(_tok("ab", "NOUN"), "es") is None  # ruido corto


# ---------------------------------------------------------------------
# _canonicalize — agrupación por embeddings
# ---------------------------------------------------------------------


def _fake_embed_fn(vectors):
    def embed_fn(texts):
        return [vectors[t] for t in texts]

    return embed_fn


def test_canonicalize_groups_similar_variants():
    # "capital cultural" y "capital cultural de Bourdieu" casi idénticos;
    # "capital economico" distinto. El canónico del grupo es la variante
    # más frecuente (empate: la más larga).
    vecs = {
        "capital cultural": [1.0, 0.0, 0.0],
        "capital cultural de Bourdieu": [0.99, 0.1, 0.0],
        "capital economico": [0.0, 1.0, 0.0],
    }
    result = _canonicalize(list(vecs), _fake_embed_fn(vecs), threshold=0.92)
    assert set(result.keys()) == {"capital cultural de Bourdieu", "capital economico"}
    # La variante más corta quedó agrupada bajo el canónico más largo.
    assert "capital cultural" in result["capital cultural de Bourdieu"]


def test_canonicalize_without_embed_fn_keeps_all():
    names = ["red neuronal", "red de petri"]
    result = _canonicalize(names, None, threshold=0.92)
    assert result == {
        "red neuronal": ["red neuronal"],
        "red de petri": ["red de petri"],
    }


def test_canonicalize_embed_fn_failure_keeps_all():
    def broken(texts):
        raise RuntimeError("embeddings caídos")

    names = ["a b", "a c"]
    result = _canonicalize(names, broken, threshold=0.92)
    assert set(result.keys()) == set(names)


def test_canonicalize_ambiguous_keeps_separate():
    # "a d" está a 0.92 de "a b" y a 0.924 de "a c" (empate casi perfecto
    # entre dos grupos) → margen 0 → ambiguo → no se fusiona a ciegas.
    # Vectores unitarios a propósito (la canonicalización normaliza).
    vecs = {
        "a b": [1.0, 0.0],
        "a c": [0.7, 0.714],
        "a d": [0.92, 0.392],
    }
    result = _canonicalize(
        list(vecs), _fake_embed_fn(vecs), threshold=0.92, gap_margin=0.05
    )
    # "a d" no se fusiona con ninguno: los tres quedan separados.
    assert set(result.keys()) == {"a b", "a c", "a d"}


def test_canonicalize_ambiguous_merges_with_zero_margin():
    # El mismo caso con gap_margin=0.0: "a d" se fusiona con el grupo más
    # cercano ("a c", 0.924 > 0.92) — el margen es lo que evita la fusión
    # a ciegas.
    vecs = {
        "a b": [1.0, 0.0],
        "a c": [0.7, 0.714],
        "a d": [0.92, 0.392],
    }
    result = _canonicalize(
        list(vecs), _fake_embed_fn(vecs), threshold=0.92, gap_margin=0.0
    )
    assert "a d" in result["a c"]


def test_canonicalize_clear_winner_merges():
    # "a c d" está claramente más cerca de "a b" (0.99) que de "a e" (0.5)
    # → margen suficiente → se fusiona con "a b" (canónico: "a c d", el más
    # largo). Vectores unitarios a propósito (la canonicalización normaliza).
    vecs = {
        "a b": [1.0, 0.0],
        "a c d": [0.99, 0.141],
        "a e": [0.5, 0.866],
    }
    result = _canonicalize(
        list(vecs), _fake_embed_fn(vecs), threshold=0.92, gap_margin=0.05
    )
    assert "a b" in result["a c d"]
    assert "a e" not in result["a c d"]
    assert "a e" in result


def test_canonicalize_auto_margin_keeps_ambiguous_separate():
    # gap_margin=None (auto): la pasada 1 con margen 0 recolecta los gaps
    # de los candidatos con señal y el margen final es la mediana.
    #   Orden: ["a b", "a c", "a e", "a f", "a d"]
    #   "a e" → gap 0.1963 (0.99 vs 0.794)
    #   "a f" → gap 0.1519 (0.98 vs 0.828)
    #   "a d" → gap 0.0622 (0.95 vs 0.888)
    # Gaps = [0.1963, 0.1519, 0.0622] → mediana 0.1519. Con margen 0.1519:
    # "a d" (gap 0.0622) queda ambiguo → separado. Resultado:
    # {"a b", "a c", "a d"} ("a e" y "a f" se fusionan con "a b").
    vecs = {
        "a b": [1.0, 0.0],
        "a c": [0.7, 0.714],
        "a e": [0.99, 0.141],
        "a f": [0.98, 0.199],
        "a d": [0.95, 0.312],
    }
    result = _canonicalize(list(vecs), _fake_embed_fn(vecs), threshold=0.92)
    assert set(result.keys()) == {"a b", "a c", "a d"}


def test_canonicalize_fixed_margin_merges_ambiguous():
    # El mismo set con gap_margin=0.03 (fijo): "a d" (gap 0.0622) supera el
    # margen → se fusiona con su grupo más cercano. "a f" (gap 0.0622)
    # también. Resultado: {"a b", "a c"} — "a d" y "a f" se fusionan.
    vecs = {
        "a b": [1.0, 0.0],
        "a c": [0.7, 0.714],
        "a e": [0.99, 0.141],
        "a f": [0.98, 0.199],
        "a d": [0.95, 0.312],
    }
    result = _canonicalize(
        list(vecs), _fake_embed_fn(vecs), threshold=0.92, gap_margin=0.03
    )
    assert set(result.keys()) == {"a b", "a c"}


def test_canonicalize_empty():
    assert _canonicalize([], lambda t: []) == {}


# ---------------------------------------------------------------------
# extract_entities_deterministic — flujo completo
# ---------------------------------------------------------------------


def test_extract_entities_deterministic_basic():
    # "El capital cultural de Bourdieu. El capital economico de Marx."
    # - "capital cultural de Bourdieu" → se conserva (menciona NER).
    # - "capital economico de Marx" → se conserva.
    # - "Bourdieu"/"Marx" (PERSON) → suprimidos del grafo.
    tokens = [
        _tok("El", "DET"),
        _tok("capital", "NOUN"),
        _tok("cultural", "ADJ"),
        _tok("de", "ADP"),
        _tok("Bourdieu", "PROPN", "PERSON"),
        _tok(".", "PUNCT"),
        _tok("El", "DET"),
        _tok("capital", "NOUN"),
        _tok("economico", "ADJ"),
        _tok("de", "ADP"),
        _tok("Marx", "PROPN", "PERSON"),
        _tok(".", "PUNCT"),
    ]
    chunks = [
        _FakeChunk(tokens[1:5], root_idx=1),  # capital cultural de Bourdieu
        _FakeChunk(tokens[7:11], root_idx=1),  # capital economico de Marx
    ]
    sents = [_FakeSent(tokens[:6]), _FakeSent(tokens[6:])]
    doc = _FakeDoc(tokens, chunks=chunks, sents=sents)
    nlp = _FakeNlp(doc)

    # min_freq=3: la palabra suelta "capital" (frecuencia 2) se filtra y no
    # genera una relación espuria entre oraciones.
    data = extract_entities_deterministic(
        nlp, "texto", embed_fn=None, lang="es", min_freq=3
    )

    names = {e["name"] for e in data["entities"]}
    assert "capital cultural de Bourdieu" in names
    assert "capital economico de Marx" in names
    # NER suprimido: ni Bourdieu ni Marx como entidades sueltas.
    assert not any(n in ("Bourdieu", "Marx") for n in names)
    # Relación CO_OCURRE entre los dos conceptos (misma oración no, pero
    # aquí están en oraciones distintas → sin relación).
    assert data["relations"] == []


def test_extract_entities_deterministic_relations_same_sentence():
    # "El capital cultural y el capital economico." → CO_OCURRE.
    tokens = [
        _tok("El", "DET"),
        _tok("capital", "NOUN"),
        _tok("cultural", "ADJ"),
        _tok("y", "CCONJ"),
        _tok("el", "DET"),
        _tok("capital", "NOUN"),
        _tok("economico", "ADJ"),
        _tok(".", "PUNCT"),
    ]
    chunks = [
        _FakeChunk(tokens[1:3], root_idx=1),  # capital cultural
        _FakeChunk(tokens[5:7], root_idx=0),  # capital economico
    ]
    doc = _FakeDoc(tokens, chunks=chunks, sents=[_FakeSent(tokens)])
    nlp = _FakeNlp(doc)

    data = extract_entities_deterministic(nlp, "texto", embed_fn=None, lang="es")

    rels = {(r["source"], r["target"]) for r in data["relations"]}
    assert ("capital cultural", "capital economico") in rels
    assert data["relations"][0]["type"] == "CO_OCURRE"


def test_extract_entities_deterministic_embedding_failure_degrades():
    # Si embed_fn falla, la extracción no revienta: entidades sin agrupar.
    tokens = [_tok("capital", "NOUN"), _tok("cultural", "ADJ")]
    doc = _FakeDoc(tokens, chunks=[_FakeChunk(tokens, root_idx=0)])
    nlp = _FakeNlp(doc)

    def broken(texts):
        raise RuntimeError("embeddings caídos")

    data = extract_entities_deterministic(nlp, "texto", embed_fn=broken, lang="es")
    assert any(e["name"] == "capital cultural" for e in data["entities"])


def test_extract_entities_deterministic_none_doc():
    class _NlpNone:
        def __call__(self, text):
            return None

    data = extract_entities_deterministic(_NlpNone(), "texto")
    assert data == {"entities": [], "relations": []}


def test_extract_entities_deterministic_frequency_filter():
    # Palabra suelta que aparece 1 sola vez y no es PROPN → se descarta.
    tokens = [
        _tok("El", "DET"),
        _tok("gato", "NOUN"),
        _tok("duerme", "VERB"),
        _tok(".", "PUNCT"),
    ]
    doc = _FakeDoc(tokens, chunks=[], sents=[_FakeSent(tokens)])
    nlp = _FakeNlp(doc)

    data = extract_entities_deterministic(nlp, "texto", embed_fn=None, lang="es")
    # "gato" aparece 1 vez, no es PROPN → filtrado por frecuencia.
    assert data["entities"] == []
