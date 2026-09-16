"""Tests de la capa proposicional del grafo (0034) — sin DB real.

Cubre:
  - build_adjacency: con KAG_GRAPH_PROPOSITION_LAYER=True agrega nodos
    "p:{id}" conectados a sus entidades; con False no.
  - ppr_entity_selection: filtra los nodos "p:..." (solo entidades INT).
  - Ingesta: _extract_document_propositions inserta los links
    proposición↔entidad (INSERT multi-VALUES con los pares correctos).
  - Degradación: sin tabla kag_proposition_links (ProgrammingError),
    build_adjacency degrada al grafo de entidades sin romper.
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


class _AdjSession:
    """Sesión fake para build_adjacency: kag_relations + kag_proposition_links."""

    def __init__(self, relations, links=None, links_error=None):
        self._relations = relations
        self._links = links if links is not None else []
        self._links_error = links_error

    def execute(self, stmt, params=None):
        sql = str(stmt)
        if "FROM kag_relations" in sql:
            return _Result(fetchall=self._relations)
        if "FROM kag_proposition_links" in sql:
            if self._links_error is not None:
                raise self._links_error
            return _Result(fetchall=self._links)
        return _Result()


def _rel(s, t):
    return SimpleNamespace(source_entity_id=s, target_entity_id=t)


def _link(pid, eid):
    return SimpleNamespace(proposition_id=pid, entity_id=eid)


# ---------------------------------------------------------------------
# build_adjacency
# ---------------------------------------------------------------------


def test_build_adjacency_includes_proposition_nodes(monkeypatch):
    session = _AdjSession(
        relations=[_rel(1, 2)],
        links=[_link(10, 1), _link(11, 2)],
    )
    monkeypatch.setattr(kq, "_kag_config_value", lambda s, name, default=None: True)
    adj = kq.build_adjacency(session)
    # Nodos de proposición con namespace STRING, conectados a sus entidades.
    assert "p:10" in adj
    assert "p:11" in adj
    assert adj["p:10"] == {1: 1}
    assert adj["p:11"] == {2: 1}
    assert adj[1]["p:10"] == 1
    assert adj[2]["p:11"] == 1
    # Las aristas de entidades siguen intactas.
    assert adj[1][2] == 1
    assert adj[2][1] == 1


def test_build_adjacency_flag_false_skips_proposition_nodes(monkeypatch):
    session = _AdjSession(relations=[_rel(1, 2)], links=[_link(10, 1)])
    monkeypatch.setattr(kq, "_kag_config_value", lambda s, name, default=None: False)
    adj = kq.build_adjacency(session)
    assert "p:10" not in adj
    assert adj[1][2] == 1
    assert adj[2][1] == 1


def test_build_adjacency_degrades_without_links_table(monkeypatch):
    session = _AdjSession(
        relations=[_rel(1, 2)],
        links_error=ProgrammingError(
            "SELECT proposition_id, entity_id FROM kag_proposition_links",
            {},
            Exception('relation "kag_proposition_links" does not exist'),
        ),
    )
    monkeypatch.setattr(kq, "_kag_config_value", lambda s, name, default=None: True)
    adj = kq.build_adjacency(session)
    assert "p:10" not in adj
    assert adj[1][2] == 1
    assert adj[2][1] == 1


# ---------------------------------------------------------------------
# ppr_entity_selection
# ---------------------------------------------------------------------


def test_ppr_entity_selection_filters_proposition_nodes():
    scores = {1: 0.5, 2: 0.3, "p:10": 0.9, "p:11": 0.7}
    selected = kq.ppr_entity_selection(scores, auto=False)
    assert all(isinstance(eid, int) for eid in selected)
    assert 1 in selected
    assert 2 in selected
    assert "p:10" not in selected
    assert "p:11" not in selected


def test_ppr_entity_selection_only_proposition_nodes_returns_empty():
    scores = {"p:10": 0.9, "p:11": 0.7}
    assert kq.ppr_entity_selection(scores, auto=False) == []


# ---------------------------------------------------------------------
# Ingesta: _extract_document_propositions inserta los links
# ---------------------------------------------------------------------


class _LinksSession:
    """Sesión fake para _extract_document_propositions con capa de links.

    Devuelve ids para los INSERT de proposiciones/entidades (RETURNING) y
    captura los params del INSERT de kag_proposition_links.
    """

    def __init__(self, chunk_rows, prop_ids, entity_rows):
        self._chunk_rows = chunk_rows
        self._prop_ids = prop_ids
        self._entity_rows = entity_rows  # [(id, name_norm), ...]
        self.updates = []
        self.links_inserts = []

    def execute(self, stmt, params=None):
        sql = str(stmt)
        if "FROM kag_chunks" in sql and "ORDER BY chunk_index" in sql:
            return _Result(fetchall=self._chunk_rows)
        if "FROM kag_propositions" in sql:
            return _Result(fetchall=[])
        if "INSERT INTO kag_propositions" in sql:
            return _Result(fetchall=[SimpleNamespace(id=pid) for pid in self._prop_ids])
        if "FROM kag_entities" in sql:
            return _Result(fetchall=[])
        if "INSERT INTO kag_entities" in sql:
            return _Result(
                fetchall=[
                    SimpleNamespace(id=eid, name_norm=norm)
                    for eid, norm in self._entity_rows
                ]
            )
        if "INSERT INTO kag_proposition_links" in sql:
            self.links_inserts.append(params)
            return _Result()
        if "UPDATE kag_chunks" in sql:
            self.updates.append(params)
            return _Result()
        return _Result()


def _chunk_row(
    cid, content, paraphrase, content_hash, chapter_id="s1", chunk_index=None
):
    return SimpleNamespace(
        id=cid,
        content=content,
        chunk_index=chunk_index if chunk_index is not None else cid,
        paraphrase=paraphrase,
        chapter_id=chapter_id,
        content_hash=content_hash,
    )


def _fake_prompt_pair(session, model_name, task_key, system_fallback, user_fallback):
    return system_fallback, user_fallback


def _patch_own_session(monkeypatch, session):
    monkeypatch.setattr(
        ki,
        "_with_own_session",
        lambda fn, *a, **k: fn(session, *a, **k),
    )


def _llm_ok(props, entities=None, relations=None):
    def fake_call(session, prompt, system=None, model_size=None, **kwargs):
        return (
            json.dumps(
                {
                    "propositions": props,
                    "entities": entities or [],
                    "relations": relations or [],
                }
            ),
            "model",
            False,
        )

    return fake_call


def test_extract_document_propositions_inserts_links(monkeypatch):
    """Los links proposición↔entidad se insertan con los pares correctos."""
    h1 = ki._chunk_content_hash("aaa")
    rows = [_chunk_row(1, "aaa", "Paráfrasis A.", h1, chunk_index=0)]
    session = _LinksSession(
        chunk_rows=rows,
        prop_ids=[100],
        entity_rows=[(200, "entidad x")],
    )
    _patch_own_session(monkeypatch, session)
    fake_call = _llm_ok(
        props=[
            {
                "chunk_index": 0,
                "core_idea_id": "CI1",
                "argument_id": "A1",
                "statement": "La entidad X aparece aquí.",
                "text_span": "Paráfrasis A.",
                "citations_references": [],
            }
        ],
        entities=[{"name": "Entidad X", "type": "concept", "description": "Desc"}],
        relations=[],
    )
    monkeypatch.setattr("src.kag_ingest.call_with_retries", fake_call)
    monkeypatch.setattr(ki, "_get_prompt_pair", _fake_prompt_pair)
    ki._extract_document_propositions(
        session, doc_id=1, doc_path="doc.md", verbose=False, batch_size_tokens=250000
    )
    # Un solo INSERT multi-VALUES con el par (proposición 100, entidad 200).
    assert len(session.links_inserts) == 1
    params = session.links_inserts[0]
    assert params["p0"] == 100
    assert params["e0"] == 200
    # content_hash actualizado para el chunk extraído.
    assert {u["id"] for u in session.updates} == {1}


def test_extract_document_propositions_links_skip_non_matching(monkeypatch):
    """Entidad que no aparece en el statement → sin link."""
    h1 = ki._chunk_content_hash("aaa")
    rows = [_chunk_row(1, "aaa", "Paráfrasis A.", h1, chunk_index=0)]
    session = _LinksSession(
        chunk_rows=rows,
        prop_ids=[100],
        entity_rows=[(200, "entidad x")],
    )
    _patch_own_session(monkeypatch, session)
    fake_call = _llm_ok(
        props=[
            {
                "chunk_index": 0,
                "core_idea_id": "CI1",
                "argument_id": "A1",
                "statement": "Sin mención de la entidad.",
                "text_span": "Paráfrasis A.",
                "citations_references": [],
            }
        ],
        entities=[{"name": "Entidad X", "type": "concept", "description": "Desc"}],
        relations=[],
    )
    monkeypatch.setattr("src.kag_ingest.call_with_retries", fake_call)
    monkeypatch.setattr(ki, "_get_prompt_pair", _fake_prompt_pair)
    ki._extract_document_propositions(
        session, doc_id=1, doc_path="doc.md", verbose=False, batch_size_tokens=250000
    )
    assert session.links_inserts == []
