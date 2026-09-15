"""Tests de la máquina de estados de ingesta KAG (src/kag/stages.py).

Lógica pura (sin DB): secuencias canónicas, resume_from, next_stage y los
cleanups por etapa. La sesión falsa simula el SELECT/UPDATE de `stage`.
"""

from types import SimpleNamespace

import pytest

from src.kag.stages import (
    CLEANUP_SQL,
    KAG_INGEST_STAGES,
    cleanup_stage,
    get_stage,
    next_stage,
    resume_from,
    set_stage,
)

# ---------------------------------------------------------------------
# Secuencias canónicas
# ---------------------------------------------------------------------


def test_kag_ingest_stages_order():
    assert KAG_INGEST_STAGES == [
        "pending",
        "segmented",
        "chunked",
        "figures",
        "ready",
    ]


# ---------------------------------------------------------------------
# resume_from
# ---------------------------------------------------------------------


def test_resume_from_first_stage():
    # Sin stage previo (fila nueva) -> primera etapa.
    assert resume_from(KAG_INGEST_STAGES, "") == "pending"
    assert resume_from(KAG_INGEST_STAGES, None) == "pending"


def test_resume_from_unknown_stage():
    # Stage desconocido -> primera etapa (defensivo).
    assert resume_from(KAG_INGEST_STAGES, "bogus") == "pending"


def test_resume_from_intermediate_stage():
    # Stage completado -> la SIGUIENTE (la actual ya está hecha).
    assert resume_from(KAG_INGEST_STAGES, "pending") == "segmented"
    assert resume_from(KAG_INGEST_STAGES, "segmented") == "chunked"
    assert resume_from(KAG_INGEST_STAGES, "chunked") == "figures"
    assert resume_from(KAG_INGEST_STAGES, "figures") == "ready"


def test_resume_from_ready():
    # 'ready' es terminal: no hay nada que reanudar.
    assert resume_from(KAG_INGEST_STAGES, "ready") == "ready"


# ---------------------------------------------------------------------
# next_stage
# ---------------------------------------------------------------------


def test_next_stage():
    assert next_stage(KAG_INGEST_STAGES, "pending") == "segmented"
    assert next_stage(KAG_INGEST_STAGES, "figures") == "ready"
    assert next_stage(KAG_INGEST_STAGES, "ready") is None
    assert next_stage(KAG_INGEST_STAGES, "bogus") is None


# ---------------------------------------------------------------------
# Cleanups por etapa (idempotencia)
# ---------------------------------------------------------------------


def test_cleanup_sql_covers_all_stages():
    # Cada etapa intermedia tiene un cleanup registrado (la primera y 'ready'
    # no necesitan: la primera no tiene datos previos, 'ready' es terminal).
    for stage in KAG_INGEST_STAGES[1:-1]:
        assert CLEANUP_SQL["kag_documents"][stage], f"falta cleanup {stage}"


def test_cleanup_stage_executes_sql():
    executed = []

    class _FakeSession:
        def execute(self, stmt, params=None):
            executed.append((str(stmt), params or {}))
            return SimpleNamespace()

        def commit(self):
            pass

    cleanup_stage(_FakeSession(), "kag_documents", 42, "chunked")
    assert len(executed) == 2
    sql, params = executed[0]
    assert "DELETE FROM kag_relations" in sql
    assert params == {"doc_id": 42}


def test_cleanup_stage_unknown_stage_noop():
    class _FakeSession:
        def execute(self, stmt, params=None):
            raise AssertionError("no debería ejecutar SQL")

        def commit(self):
            pass

    # Etapa sin cleanup registrado -> no ejecuta nada.
    cleanup_stage(_FakeSession(), "kag_documents", 1, "pending")


# ---------------------------------------------------------------------
# get_stage / set_stage
# ---------------------------------------------------------------------


def test_get_stage_reads_column():
    class _FakeSession:
        def execute(self, stmt, params=None):
            assert params == {"id": 7}
            return SimpleNamespace(first=lambda: SimpleNamespace(stage="chunks"))

    assert get_stage(_FakeSession(), "kag_documents", 7) == "chunks"


def test_get_stage_missing_row_returns_empty():
    class _FakeSession:
        def execute(self, stmt, params=None):
            return SimpleNamespace(first=lambda: None)

    assert get_stage(_FakeSession(), "kag_documents", 7) == ""


def test_set_stage_updates_and_commits():
    calls = []

    class _FakeSession:
        def execute(self, stmt, params=None):
            calls.append((str(stmt), params or {}))
            return SimpleNamespace()

        def commit(self):
            calls.append(("COMMIT", {}))

    set_stage(_FakeSession(), "kag_documents", 3, "segmented")
    assert len(calls) == 2
    sql, params = calls[0]
    assert "UPDATE kag_documents" in sql
    assert "SET stage = :stage" in sql
    assert params == {"stage": "segmented", "id": 3}
    assert calls[1] == ("COMMIT", {})
