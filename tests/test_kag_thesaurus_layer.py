"""Tests de la capa tesauro del grafo (0035) — sin DB real.

Cubre:
  - _store_document_thesaurus: extrae términos de la ficha (iso25964 con
    relaciones, lcsh, lcc) y genera los INSERTs correctos + links.
  - build_adjacency: con KAG_GRAPH_THESAURUS_LAYER=True agrega nodos "d:..."
    y "t:..." con las aristas correctas; con False no.
  - ppr_entity_selection: filtra "d:..."/"t:..."/"p:..." (solo entidades INT).
  - Degradación: sin tabla kag_thesaurus_terms (ProgrammingError),
    build_adjacency degrada a entidades + proposiciones sin romper.
"""

import json
from types import SimpleNamespace

from sqlalchemy.exc import ProgrammingError

import src.kag_ingest as ki
import src.kag_query as kq


class _Result:
    def __init__(self, first=None, scalar=None, fetchall=None):
        self._first = first
        self._scalar = scalar
        self._fetchall = fetchall if fetchall is not None else []

    def scalars(self):
        return self

    def first(self):
        return self._first

    def scalar(self):
        return self._scalar

    def fetchall(self):
        return self._fetchall


class _ThesSession:
    """Sesión fake para _store_document_thesaurus: captura los INSERTs de
    kag_thesaurus_terms (devuelve ids por RETURNING) y el INSERT de
    kag_document_thesaurus."""

    def __init__(self, term_ids):
        self._term_ids = term_ids
        self.term_inserts = []  # params de cada INSERT de kag_thesaurus_terms
        self.link_inserts = []  # params del INSERT de kag_document_thesaurus

    def execute(self, stmt, params=None):
        sql = str(stmt)
        if "INSERT INTO kag_thesaurus_terms" in sql:
            self.term_inserts.append(params)
            idx = len(self.term_inserts) - 1
            tid = self._term_ids[idx] if idx < len(self._term_ids) else idx + 1
            return _Result(scalar=tid)
        if "INSERT INTO kag_document_thesaurus" in sql:
            self.link_inserts.append(params)
            return _Result()
        return _Result()


class _AdjSession:
    """Sesión fake para build_adjacency: kag_relations + capas tesauro."""

    def __init__(
        self, relations, doc_entities=None, doc_terms=None, terms=None, thes_error=None
    ):
        self._relations = relations
        self._doc_entities = doc_entities if doc_entities is not None else []
        self._doc_terms = doc_terms if doc_terms is not None else []
        self._terms = terms if terms is not None else []
        self._thes_error = thes_error

    def execute(self, stmt, params=None):
        sql = str(stmt)
        if "FROM kag_relations" in sql:
            return _Result(fetchall=self._relations)
        if "FROM kag_proposition_links" in sql:
            return _Result(fetchall=[])
        if "FROM kag_entities" in sql:
            if self._thes_error is not None:
                raise self._thes_error
            return _Result(fetchall=self._doc_entities)
        if "FROM kag_document_thesaurus" in sql:
            if self._thes_error is not None:
                raise self._thes_error
            return _Result(fetchall=self._doc_terms)
        if "FROM kag_thesaurus_terms" in sql:
            if self._thes_error is not None:
                raise self._thes_error
            return _Result(fetchall=self._terms)
        return _Result()


def _rel(s, t):
    return SimpleNamespace(source_entity_id=s, target_entity_id=t)


def _doc_ent(doc_id, entity_id):
    return SimpleNamespace(doc_id=doc_id, entity_id=entity_id)


def _doc_term(doc_id, term_id):
    return SimpleNamespace(doc_id=doc_id, term_id=term_id)


def _term(tid, term_norm, source, broader, narrower, related):
    return SimpleNamespace(
        id=tid,
        term_norm=term_norm,
        source=source,
        broader=broader,
        narrower=narrower,
        related=related,
    )


# ---------------------------------------------------------------------
# _store_document_thesaurus
# ---------------------------------------------------------------------


def _ficha():
    return {
        "thematic_areas_iso25964": [
            {
                "preferred_term": "Inteligencia Artificial",
                "non_preferred_terms": ["IA"],
                "scope_note_disambiguation": "",
                "broader_term": "Ciencias de la Computación",
                "narrower_terms": [
                    "Machine Learning",
                    "Procesamiento de Lenguaje Natural",
                ],
                "related_terms": ["Robótica"],
            }
        ],
        "library_of_congress": {
            "lcsh_terms": ["Artificial intelligence", "Neural networks"],
            "lcc_classification": {
                "label": "Machine Learning",
                "call_number": "Q325.5",
            },
        },
    }


def test_store_document_thesaurus_extracts_terms_and_links():
    session = _ThesSession(term_ids=[10, 11, 12, 13])
    ki._store_document_thesaurus(session, doc_id=7, ficha=_ficha())
    # 4 términos: 1 iso25964 + 2 lcsh + 1 lcc.
    assert len(session.term_inserts) == 4
    iso = session.term_inserts[0]
    assert iso["term"] == "Inteligencia Artificial"
    assert iso["term_norm"] == "inteligencia artificial"
    assert iso["source"] == "iso25964"
    assert json.loads(iso["broader"]) == ["Ciencias de la Computación"]
    assert json.loads(iso["narrower"]) == [
        "Machine Learning",
        "Procesamiento de Lenguaje Natural",
    ]
    assert json.loads(iso["related"]) == ["Robótica"]
    lcsh = session.term_inserts[1]
    assert lcsh["source"] == "lcsh"
    assert lcsh["term"] == "Artificial intelligence"
    assert json.loads(lcsh["broader"]) == []
    lcc = session.term_inserts[3]
    assert lcc["source"] == "lcc"
    assert lcc["term"] == "Machine Learning"
    # Un solo INSERT multi-VALUES con los 4 pares (doc 7, term ids).
    assert len(session.link_inserts) == 1
    params = session.link_inserts[0]
    assert params["d0"] == 7 and params["t0"] == 10
    assert params["d3"] == 7 and params["t3"] == 13


def test_store_document_thesaurus_empty_ficha_no_ops():
    session = _ThesSession(term_ids=[])
    ki._store_document_thesaurus(session, doc_id=7, ficha={})
    assert session.term_inserts == []
    assert session.link_inserts == []


# ---------------------------------------------------------------------
# build_adjacency
# ---------------------------------------------------------------------


def test_build_adjacency_includes_thesaurus_nodes(monkeypatch):
    session = _AdjSession(
        relations=[_rel(1, 2)],
        doc_entities=[_doc_ent(7, 1), _doc_ent(7, 2)],
        doc_terms=[_doc_term(7, 10), _doc_term(7, 11)],
        terms=[
            _term(
                10,
                "inteligencia artificial",
                "iso25964",
                ["Ciencias de la Computación"],
                ["Machine Learning"],
                ["Robótica"],
            ),
            _term(11, "machine learning", "iso25964", [], [], []),
            _term(12, "robótica", "iso25964", [], [], []),
        ],
    )
    monkeypatch.setattr(kq, "_kag_config_value", lambda s, name, default=None: True)
    adj = kq.build_adjacency(session)
    # Nodos documento conectados a sus entidades.
    assert "d:7" in adj
    assert adj["d:7"][1] == 1
    assert adj["d:7"][2] == 1
    assert adj[1]["d:7"] == 1
    # Nodos documento conectados a sus términos.
    assert adj["d:7"]["t:10"] == 1
    assert adj["d:7"]["t:11"] == 1
    # Nodos tesauro conectados por broader/narrower/related (resueltos por
    # term_norm dentro del mismo source; el broader sin fila se omite).
    assert adj["t:10"]["t:11"] == 1  # narrower "Machine Learning" → t:11
    assert adj["t:11"]["t:10"] == 1
    assert adj["t:10"]["t:12"] == 1  # related "Robótica" → t:12
    assert adj["t:12"]["t:10"] == 1
    # Las aristas de entidades siguen intactas.
    assert adj[1][2] == 1
    assert adj[2][1] == 1


def test_build_adjacency_flag_false_skips_thesaurus_nodes(monkeypatch):
    session = _AdjSession(
        relations=[_rel(1, 2)],
        doc_entities=[_doc_ent(7, 1)],
        doc_terms=[_doc_term(7, 10)],
        terms=[_term(10, "inteligencia artificial", "iso25964", [], [], [])],
    )
    monkeypatch.setattr(kq, "_kag_config_value", lambda s, name, default=None: False)
    adj = kq.build_adjacency(session)
    assert "d:7" not in adj
    assert "t:10" not in adj
    assert adj[1][2] == 1
    assert adj[2][1] == 1


def test_build_adjacency_degrades_without_thesaurus_tables(monkeypatch):
    session = _AdjSession(
        relations=[_rel(1, 2)],
        thes_error=ProgrammingError(
            "SELECT id, term_norm, source, broader, narrower, related "
            "FROM kag_thesaurus_terms",
            {},
            Exception('relation "kag_thesaurus_terms" does not exist'),
        ),
    )
    monkeypatch.setattr(kq, "_kag_config_value", lambda s, name, default=None: True)
    adj = kq.build_adjacency(session)
    assert "d:7" not in adj
    assert "t:10" not in adj
    assert adj[1][2] == 1
    assert adj[2][1] == 1


# ---------------------------------------------------------------------
# ppr_entity_selection
# ---------------------------------------------------------------------


def test_ppr_entity_selection_filters_document_and_thesaurus_nodes():
    scores = {1: 0.5, 2: 0.3, "d:7": 0.9, "t:10": 0.8, "p:10": 0.7}
    selected = kq.ppr_entity_selection(scores, auto=False)
    assert all(isinstance(eid, int) for eid in selected)
    assert 1 in selected
    assert 2 in selected
    assert "d:7" not in selected
    assert "t:10" not in selected
    assert "p:10" not in selected


def test_ppr_entity_selection_only_string_nodes_returns_empty():
    scores = {"d:7": 0.9, "t:10": 0.8, "p:10": 0.7}
    assert kq.ppr_entity_selection(scores, auto=False) == []
