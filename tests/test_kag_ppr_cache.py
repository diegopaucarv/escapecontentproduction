"""Tests del LRU cache de ego-network y PPR (src/kag_query.py).

Sin DB, sin torch/spacy: sesión falsa con versión de kag_graph_state y
filas de kag_relations (patrón de tests/test_kag.py). Verifica que la
segunda llamada con la misma (semilla, versión, alpha) usa el cache
(contador de llamadas al core), y que cambia la clave → cache miss.
"""

from types import SimpleNamespace

import pytest

import src.kag_query as kq


class _Result:
    def __init__(self, scalar=None, rows=None):
        self._scalar = scalar
        self._rows = rows

    def scalar(self):
        return self._scalar

    def fetchall(self):
        return self._rows


class _FakeSession:
    """Sesión falsa: kag_graph_state devuelve `version`, kag_relations
    devuelve `rows` (SimpleNamespace con source/target)."""

    def __init__(self, version, rows):
        self.version = version
        self.rows = rows
        self.session_settings = {}

    def execute(self, stmt):
        sql = str(stmt)
        if "kag_graph_state" in sql:
            return _Result(scalar=self.version)
        return _Result(rows=self.rows)


ROWS = [
    SimpleNamespace(source_entity_id=1, target_entity_id=2),
    SimpleNamespace(source_entity_id=2, target_entity_id=3),
    SimpleNamespace(source_entity_id=3, target_entity_id=1),
]


@pytest.fixture(autouse=True)
def _reset_caches():
    """Resetea los cachés module-level de src.kag_query entre tests."""
    kq._adjacency_cache = None
    kq._adjacency_version = -1
    kq._ego_network_cached.cache_clear()
    kq._ppr_cached.cache_clear()
    yield
    kq._adjacency_cache = None
    kq._adjacency_version = -1
    kq._ego_network_cached.cache_clear()
    kq._ppr_cached.cache_clear()


def _counting(monkeypatch):
    """Envuelve ego_network/personalized_pagerank con contadores de llamadas."""
    calls = {"ego": 0, "ppr": 0}
    orig_ego = kq.ego_network
    orig_ppr = kq.personalized_pagerank

    def counting_ego(adjacency, seed, hops=2):
        calls["ego"] += 1
        return orig_ego(adjacency, seed, hops=hops)

    def counting_ppr(adjacency, seed, alpha=0.15, max_iter=50, tol=1e-6):
        calls["ppr"] += 1
        return orig_ppr(adjacency, seed, alpha=alpha, max_iter=max_iter, tol=tol)

    monkeypatch.setattr(kq, "ego_network", counting_ego)
    monkeypatch.setattr(kq, "personalized_pagerank", counting_ppr)
    return calls


# ---------------------------------------------------------------------
# ego_network_cached
# ---------------------------------------------------------------------


def test_ego_network_cached_hit_same_seed_and_version(monkeypatch):
    calls = _counting(monkeypatch)
    session = _FakeSession(version=1, rows=ROWS)

    r1 = kq.ego_network_cached(session, [1, 2])
    r2 = kq.ego_network_cached(session, [1, 2])

    assert r1 == r2
    assert calls["ego"] == 1  # segunda llamada: cache hit


def test_ego_network_cached_miss_on_version_change(monkeypatch):
    calls = _counting(monkeypatch)
    s1 = _FakeSession(version=1, rows=ROWS)
    s2 = _FakeSession(version=2, rows=ROWS)

    kq.ego_network_cached(s1, [1, 2])
    kq.ego_network_cached(s2, [1, 2])

    assert calls["ego"] == 2  # versión distinta → cache miss


def test_ego_network_cached_miss_on_seed_change(monkeypatch):
    calls = _counting(monkeypatch)
    session = _FakeSession(version=1, rows=ROWS)

    kq.ego_network_cached(session, [1, 2])
    kq.ego_network_cached(session, [1, 3])

    assert calls["ego"] == 2  # semilla distinta → cache miss


# ---------------------------------------------------------------------
# personalized_pagerank_cached
# ---------------------------------------------------------------------


def test_ppr_cached_hit_same_seed_version_alpha(monkeypatch):
    calls = _counting(monkeypatch)
    session = _FakeSession(version=1, rows=ROWS)

    r1 = kq.personalized_pagerank_cached(session, [1, 2])
    r2 = kq.personalized_pagerank_cached(session, [1, 2])

    assert r1 == r2
    assert calls["ppr"] == 1  # segunda llamada: cache hit
    assert calls["ego"] == 1  # el ego-network también se reutiliza


def test_ppr_cached_miss_on_version_change(monkeypatch):
    calls = _counting(monkeypatch)
    s1 = _FakeSession(version=1, rows=ROWS)
    s2 = _FakeSession(version=2, rows=ROWS)

    kq.personalized_pagerank_cached(s1, [1, 2])
    kq.personalized_pagerank_cached(s2, [1, 2])

    assert calls["ppr"] == 2  # versión distinta → cache miss


def test_ppr_cached_miss_on_alpha_change(monkeypatch):
    calls = _counting(monkeypatch)
    session = _FakeSession(version=1, rows=ROWS)

    kq.personalized_pagerank_cached(session, [1, 2])
    kq.personalized_pagerank_cached(session, [1, 2], alpha=0.3)

    assert calls["ppr"] == 2  # alpha distinto → cache miss


def test_ppr_alpha_reads_from_config(monkeypatch):
    monkeypatch.setenv("KAG_PPR_ALPHA", "0.3")
    captured = {}
    orig_ppr = kq.personalized_pagerank

    def spy(adjacency, seed, alpha=0.15, max_iter=50, tol=1e-6):
        captured["alpha"] = alpha
        return orig_ppr(adjacency, seed, alpha=alpha, max_iter=max_iter, tol=tol)

    monkeypatch.setattr(kq, "personalized_pagerank", spy)
    session = _FakeSession(version=1, rows=ROWS)

    kq.personalized_pagerank_cached(session, [1, 2])

    assert captured["alpha"] == 0.3
