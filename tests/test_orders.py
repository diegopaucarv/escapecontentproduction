"""Tests de la Ruta A — Kaizen/Repetitivo (§5 del pipeline unificado).

Cubre el flujo completo con TestClient + _FakeSession en memoria (sin DB
ni red): ORDER → ORDER_NOTE → EXECUTE_KAIZEN → MEASURE_KPI →
MICRO_KAIZEN/MICRO_RITUAL → KAIZEN_DECISION → PRE_DEPLOY_ENTRY.

Filosofía 0007 verificada aquí:
- Sin settings LLM, el producer/critic degradan a determinista y la orden
  se PAUSA en 'en_gate' con requires_user_acceptance=True (nunca sigue
  sola, nunca se simula auto_pass).
- Con producer/critic LLM ok (monkeypatch), el gate auto-pasa y la pieza
  se materializa como ContentArtifact.
- KAIZEN_DECISION = update_registry actualiza /components/manifest.md y
  escribe /kb/kaizen_<id>.md; archive documenta sin cambiar la plantilla.
"""

import uuid
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import app, get_session
from src.db.models import (
    CalendarSlot,
    ContentArtifact,
    ContentBrief,
    KaizenCycle,
    PipelineTemplate,
    ProductionOrder,
    ProductionTemplate,
)
from src.orders import flow as flow_mod


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return self._rows

    def first(self):
        return self._rows[0] if self._rows else None

    def scalar_one_or_none(self):
        return self._rows[0] if self._rows else None


def _unwrap(v):
    return getattr(v, "value", v)


class _FakeSession:
    """Sesión mínima en memoria para la Ruta A (sin Postgres)."""

    def __init__(self):
        self._store = {}  # class name -> {id: obj}
        self._added = []
        self.commits = 0

    def _rows(self, name):
        return self._store.setdefault(name, {})

    def get(self, model, ident):
        name = getattr(model, "__name__", "")
        return self._rows(name).get(ident)

    def execute(self, stmt):
        entity = stmt.column_descriptions[0]["entity"]
        name = entity.__name__
        pairs = {}
        for c in stmt._where_criteria:
            if hasattr(c.left, "name"):
                v = getattr(c.right, "value", c.right)
                v = getattr(v, "value", v)  # desenvuelve enum si aplica
                pairs[c.left.name] = v
        rows = list(self._rows(name).values())
        for key, val in pairs.items():
            rows = [r for r in rows if _unwrap(getattr(r, key, None)) == val]
        return _Result(rows)

    def add(self, obj):
        self._added.append(obj)

    def flush(self):
        for obj in self._added:
            if getattr(obj, "id", None) is None:
                obj.id = uuid.uuid4()
            self._rows(obj.__class__.__name__)[obj.id] = obj
        self._added = []

    def commit(self):
        self.commits += 1
        self.flush()

    def refresh(self, obj):
        pass

    def close(self):
        pass


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def data_root(tmp_path: Path, monkeypatch):
    """Redirige los artefactos de la Ruta A a un tmp_path (sin tocar data/)."""
    monkeypatch.setattr(flow_mod, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(flow_mod, "ORDERS_DIR", tmp_path / "orders")
    monkeypatch.setattr(flow_mod, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(flow_mod, "KB_DIR", tmp_path / "kb")
    monkeypatch.setattr(
        flow_mod, "COMPONENTS_MANIFEST", tmp_path / "components" / "manifest.md"
    )
    return tmp_path


def _slot(**overrides) -> CalendarSlot:
    defaults = dict(
        id=uuid.uuid4(),
        slot_code="escape_lunes_short",
        brand_objective="ESCAPE_SOCIAL",
        content_bucket="difusion_cientifica",
        artifact_type="video_corto",
        channel="tiktok",
        default_owner_role="Editor",
        cadence="semanal",
        weekday="lunes",
        is_active=True,
    )
    defaults.update(overrides)
    return CalendarSlot(**defaults)


def _plan(**overrides) -> PipelineTemplate:
    defaults = dict(
        id=1,
        brand_objective="ESCAPE_SOCIAL",
        content_bucket="difusion_cientifica",
        checklist=["fuente_verificable", "gancho_15s_ok", "cta_unico"],
        novelty_weights={},
    )
    defaults.update(overrides)
    return PipelineTemplate(**defaults)


def _template(**overrides) -> ProductionTemplate:
    defaults = dict(
        id=uuid.uuid4(),
        name="Script_SSML_Template",
        content_type="audio",
        phase="preproduccion",
        template_format="json",
        content={"tags": ["pausa"]},
        version="1.0",
        is_active=True,
    )
    defaults.update(overrides)
    return ProductionTemplate(**defaults)


def _order(**overrides) -> ProductionOrder:
    defaults = dict(
        id=uuid.uuid4(),
        calendar_slot="escape_lunes_short",
        brand_objective="ESCAPE_SOCIAL",
        content_bucket="difusion_cientifica",
        artifact_type="video_corto",
        channel="tiktok",
        owner_id=None,
        scheduled_date=date(2026, 9, 14),
        insight_core="El refuerzo de vacunas sigue siendo relevante en 2026.",
        status="creada",
        kpis={},
        created_at=datetime.utcnow() - timedelta(hours=2),
    )
    defaults.update(overrides)
    return ProductionOrder(**defaults)


def _patch_llm(monkeypatch, producer=None, critic=None):
    """Monkeypatchea producer/critic en src.orders.flow (single-pass)."""
    if producer is not None:
        monkeypatch.setattr(flow_mod, "run_producer_generation", producer)
    if critic is not None:
        monkeypatch.setattr(flow_mod, "run_critic_checklist", critic)


def _ok_producer(session, brief_data, context_pack, critic_feedback=None):
    return {
        "status": "ok",
        "draft": "Borrador LLM del short de la semana.",
        "requires_user_acceptance": False,
    }


def _ok_critic(session, draft, checklist, brand_objective):
    return {
        "status": "ok",
        "checklist_results": {item: "ok" for item in checklist},
        "reasoning": "todo ok",
        "requires_user_acceptance": False,
    }


# ----------------------------------------------------------------------
# ORDER + ORDER_NOTE
# ----------------------------------------------------------------------


def test_create_order_404_when_slot_missing(client):
    fake = _FakeSession()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        resp = client.post(
            "/orders",
            json={
                "slot_code": "no_existe",
                "scheduled_date": "2026-09-14",
                "insight_core": "dato",
            },
        )
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 404
    assert "no existe" in resp.json()["detail"]


def test_create_order_inherits_slot_fields(client, data_root):
    fake = _FakeSession()
    fake.add(_slot())
    fake.flush()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        resp = client.post(
            "/orders",
            json={
                "slot_code": "escape_lunes_short",
                "scheduled_date": "2026-09-14",
                "insight_core": "El refuerzo sigue siendo relevante.",
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 201
    body = resp.json()
    assert body["calendar_slot"] == "escape_lunes_short"
    assert body["brand_objective"] == "ESCAPE_SOCIAL"
    assert body["content_bucket"] == "difusion_cientifica"
    assert body["artifact_type"] == "video_corto"
    assert body["channel"] == "tiktok"
    assert body["status"] == "creada"
    assert body["insight_core"] == "El refuerzo sigue siendo relevante."
    # ORDER_NOTE: el markdown se escribió en data/orders/order_<id>.md.
    note_path = data_root / "orders" / f"order_{body['id']}.md"
    assert note_path.exists()
    note = note_path.read_text(encoding="utf-8")
    assert "brand_objective: ESCAPE_SOCIAL" in note
    assert "insight_core: El refuerzo sigue siendo relevante." in note


def test_get_order_note(client, data_root):
    fake = _FakeSession()
    order = _order()
    fake.add(order)
    fake.flush()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        resp = client.get(f"/orders/{order.id}/note")
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    body = resp.json()
    assert body["order_id"] == str(order.id)
    assert "Orden de Producción" in body["note"]
    assert "escape_lunes_short" in body["note"]


def test_list_orders_and_get_order(client):
    fake = _FakeSession()
    order = _order()
    fake.add(order)
    fake.flush()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        resp_list = client.get("/orders")
        resp_get = client.get(f"/orders/{order.id}")
    finally:
        app.dependency_overrides.clear()

    assert resp_list.status_code == 200
    assert len(resp_list.json()) == 1
    assert resp_get.status_code == 200
    assert resp_get.json()["id"] == str(order.id)


def test_get_order_404(client):
    fake = _FakeSession()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        resp = client.get(f"/orders/{uuid.uuid4()}")
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 404


# ----------------------------------------------------------------------
# EXECUTE_KAIZEN (§5.3)
# ----------------------------------------------------------------------


def test_produce_auto_pass_materializes_artifact(client, data_root, monkeypatch):
    """0007: con producer/critic LLM ok, el gate auto-pasa (riesgo bajo +
    repetitivo) y la pieza se materializa como ContentArtifact."""
    _patch_llm(monkeypatch, producer=_ok_producer, critic=_ok_critic)
    fake = _FakeSession()
    order = _order()
    fake.add(order)
    fake.add(_plan())
    fake.flush()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        resp = client.post(f"/orders/{order.id}/produce")
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    body = resp.json()
    assert body["verdict"] == "auto_pass"
    assert body["requires_user_acceptance"] is False
    assert body["status"] == "aprobada"
    assert body["draft"] == "Borrador LLM del short de la semana."
    assert body["artifact_id"] is not None
    # La pieza se materializó como artefacto.
    artifact = fake.get(ContentArtifact, uuid.UUID(body["artifact_id"]))
    assert artifact is not None
    assert artifact.brief_id == uuid.UUID(body["brief_id"])
    assert artifact.artifact_type == "video_corto"
    assert artifact.channel == "tiktok"
    # El borrador se escribió en data/artifacts/kaizen_<id>.md.
    artifact_path = data_root / "artifacts" / f"kaizen_{order.id}.md"
    assert artifact_path.exists()
    assert "Borrador LLM" in artifact_path.read_text(encoding="utf-8")
    # El brief compañero quedó aprobado (gate auto_pass).
    brief = fake.get(ContentBrief, uuid.UUID(body["brief_id"]))
    assert brief.status == "aprobado"
    assert brief.route_decision == "repetitivo"


def test_produce_degraded_pauses_for_acceptance(client, data_root):
    """0007: sin settings LLM, el producer/critic degradan a determinista y
    la orden se PAUSA en 'en_gate' con requires_user_acceptance=True — nunca
    sigue sola, nunca se simula auto_pass."""
    fake = _FakeSession()
    order = _order()
    fake.add(order)
    fake.add(_plan())
    fake.flush()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        resp = client.post(f"/orders/{order.id}/produce")
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    body = resp.json()
    assert body["requires_user_acceptance"] is True
    assert body["verdict"] == "needs_human_review"
    assert body["status"] == "en_gate"
    assert body["artifact_id"] is None  # no se materializó nada
    brief = fake.get(ContentBrief, uuid.UUID(body["brief_id"]))
    assert brief.status == "revision"
    assert brief.requires_user_acceptance is True


def test_produce_fail_goes_to_rework(client, data_root, monkeypatch):
    """Veredicto fail → retrabajo: la orden vuelve a 'en_produccion' y el
    brief a 'generando' (§3.9: revision --> generando: fail)."""

    def failing_critic(session, draft, checklist, brand_objective):
        return {
            "status": "ok",
            "checklist_results": {item: "fail" for item in checklist},
            "reasoning": "falta fuente",
            "requires_user_acceptance": False,
        }

    _patch_llm(monkeypatch, producer=_ok_producer, critic=failing_critic)
    fake = _FakeSession()
    order = _order()
    fake.add(order)
    fake.add(_plan())
    fake.flush()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        resp = client.post(f"/orders/{order.id}/produce")
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    body = resp.json()
    assert body["verdict"] == "fail"
    assert body["status"] == "en_produccion"
    brief = fake.get(ContentBrief, uuid.UUID(body["brief_id"]))
    assert brief.status == "generando"
    # Trazabilidad: rework_count se incrementó.
    assert fake.get(ProductionOrder, order.id).kpis["rework_count"] == 1


def test_produce_409_when_order_not_producible(client):
    fake = _FakeSession()
    order = _order(status="archivada")
    fake.add(order)
    fake.flush()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        resp = client.post(f"/orders/{order.id}/produce")
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 409


# ----------------------------------------------------------------------
# Publicación manual + MEASURE_KPI (§5.4)
# ----------------------------------------------------------------------


def test_publish_requires_approved(client):
    fake = _FakeSession()
    order = _order(status="creada")
    fake.add(order)
    fake.flush()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        resp = client.post(f"/orders/{order.id}/publish")
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 409


def test_publish_and_record_kpis(client, data_root):
    fake = _FakeSession()
    order = _order(
        status="aprobada", kpis={"produced_at": datetime.utcnow().isoformat()}
    )
    fake.add(order)
    fake.flush()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        resp_pub = client.post(f"/orders/{order.id}/publish")
        resp_kpis = client.post(
            f"/orders/{order.id}/kpis",
            json={"kpis": {"guardados": 120, "respuestas": 8}},
        )
    finally:
        app.dependency_overrides.clear()

    assert resp_pub.status_code == 200
    assert resp_pub.json()["status"] == "publicada"
    assert resp_kpis.status_code == 200
    body = resp_kpis.json()
    assert body["kpis"]["guardados"] == 120
    assert body["kpis"]["respuestas"] == 8
    # Métricas de proceso calculadas automáticamente.
    assert "lead_time_minutes" in body["kpis"]
    assert "cycle_time_minutes" in body["kpis"]
    assert "time_in_stage" in body["kpis"]
    assert "first_pass_quality" in body["kpis"]
    assert "rework_rate" in body["kpis"]
    # La orden pasó a 'medida'.
    assert fake.get(ProductionOrder, order.id).status == "medida"


# ----------------------------------------------------------------------
# MICRO_KAIZEN (§5.5) + MICRO_RITUAL (§5.6)
# ----------------------------------------------------------------------


def test_record_micro_kaizen(client):
    fake = _FakeSession()
    order = _order()
    fake.add(order)
    fake.flush()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        resp = client.post(
            f"/orders/{order.id}/kaizen",
            json={
                "experiment": {"hook": "otro gancho", "horario": "18:00"},
                "ritual": {"riesgos": ["polarización"], "mitigaciones": ["revisión"]},
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    body = resp.json()
    micro = body["kpis"]["micro_kaizen"]
    assert len(micro) == 1
    assert micro[0]["experiment"]["hook"] == "otro gancho"
    assert micro[0]["ritual"]["riesgos"] == ["polarización"]


# ----------------------------------------------------------------------
# KAIZEN_DECISION (§5.7) → UPDATE_REGISTRY | ARCHIVE_KAIZEN
# ----------------------------------------------------------------------


def test_kaizen_decision_update_registry(client, data_root):
    fake = _FakeSession()
    order = _order(status="medida", kpis={"first_pass_quality": True})
    fake.add(order)
    fake.flush()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        resp = client.post(
            f"/orders/{order.id}/kaizen-decision",
            json={
                "decision": "update_registry",
                "improvement_summary": "El hook de pregunta incómoda subió retención.",
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    body = resp.json()
    assert body["decision"] == "update_registry"
    assert body["converges_to"] == "PRE_DEPLOY_ENTRY"
    # Se creó el ciclo Kaizen.
    cycle = fake.get(KaizenCycle, uuid.UUID(body["kaizen_cycle_id"]))
    assert cycle is not None
    assert cycle.decision == "update_registry"
    # La orden quedó con la decisión registrada.
    assert fake.get(ProductionOrder, order.id).kaizen_decision == "update_registry"
    # UPDATE_REGISTRY: el manifest se actualizó con la primera fila real.
    manifest = data_root / "components" / "manifest.md"
    assert manifest.exists()
    text = manifest.read_text(encoding="utf-8")
    assert "escape_lunes_short" in text
    assert "_(vacío" not in text
    # /kb/kaizen_<id>.md se escribió.
    kb_doc = data_root / "kb" / f"kaizen_{cycle.id}.md"
    assert kb_doc.exists()
    assert "update_registry" in kb_doc.read_text(encoding="utf-8")


def test_kaizen_decision_archive(client, data_root):
    fake = _FakeSession()
    order = _order(status="medida")
    fake.add(order)
    fake.flush()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        resp = client.post(
            f"/orders/{order.id}/kaizen-decision",
            json={"decision": "archive", "improvement_summary": "Sin mejora medible."},
        )
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    body = resp.json()
    assert body["decision"] == "archive"
    cycle = fake.get(KaizenCycle, uuid.UUID(body["kaizen_cycle_id"]))
    assert cycle.decision == "archive"
    # ARCHIVE_KAIZEN: se documenta en /kb/ sin tocar el manifest.
    kb_doc = data_root / "kb" / f"kaizen_{cycle.id}.md"
    assert kb_doc.exists()
    manifest = data_root / "components" / "manifest.md"
    assert not manifest.exists()  # no se creó el manifest


def test_kaizen_decision_invalid(client):
    fake = _FakeSession()
    order = _order()
    fake.add(order)
    fake.flush()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        resp = client.post(
            f"/orders/{order.id}/kaizen-decision",
            json={"decision": "talvez"},
        )
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 422  # pattern del body


def test_list_kaizen_cycles(client):
    fake = _FakeSession()
    order = _order()
    cycle = KaizenCycle(
        id=uuid.uuid4(),
        order_id=order.id,
        decision="archive",
        improvement_summary="Sin mejora.",
        metrics={},
    )
    fake.add(order)
    fake.add(cycle)
    fake.flush()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        resp = client.get(f"/orders/{order.id}/kaizen-cycles")
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["decision"] == "archive"


# ----------------------------------------------------------------------
# Calendar slots
# ----------------------------------------------------------------------


def test_list_calendar_slots(client):
    fake = _FakeSession()
    fake.add(_slot())
    fake.add(
        _slot(
            slot_code="ergalia_martes_dato_incomodo",
            brand_objective="ERGALIA_COMERCIAL",
            content_bucket="debate_informado",
            artifact_type="post",
            channel="linkedin",
            default_owner_role="CM+Diseñador",
            weekday="martes",
        )
    )
    fake.flush()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        resp = client.get("/calendar-slots")
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 2
    codes = {s["slot_code"] for s in body}
    assert codes == {"escape_lunes_short", "ergalia_martes_dato_incomodo"}
