"""Tests de la Fase 3 — API de producción (src/api/main.py).

Cubre el CRUD de /tool-adapters y /production-templates, los manifiestos y
jobs por artefacto, el endpoint disparador /produce y /tools/validate.
Usa TestClient + _FakeSession en memoria (sin DB ni red): el orquestador
corre en modo directo (McpClient sin base_url no hace red).
"""

import uuid

import pytest
from fastapi.testclient import TestClient

from src.api.main import app, get_session
from src.db.models import (
    AssetJob,
    BrandObjective,
    ContentArtifact,
    FormatSpec,
    ProductionTemplate,
    ToolAdapter,
)
from src.tools import create_manifest


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


class _FakeSession:
    """Sesión mínima en memoria para la API de producción (Fase 3)."""

    def __init__(self):
        self.adapters = {}  # id -> ToolAdapter
        self.templates = {}  # id -> ProductionTemplate
        self.specs = {}  # id -> FormatSpec
        self.artifacts = {}  # id -> ContentArtifact
        self.manifests = []  # list[ProductionManifest]
        self.jobs = []  # list[AssetJob]
        self._added = []

    # ------------------------------------------------------------------
    # consultas
    # ------------------------------------------------------------------

    def _by_name(self, store, name):
        for obj in store.values():
            if obj.name == name:
                return obj
        return None

    def get(self, model, ident):
        if model is ContentArtifact:
            return self.artifacts.get(ident)
        if model is ToolAdapter:
            return self.adapters.get(ident)
        if model is ProductionTemplate:
            return self.templates.get(ident)
        return None

    def execute(self, stmt):
        entity = stmt.column_descriptions[0]["entity"]
        pairs = {}
        for c in stmt._where_criteria:
            if hasattr(c.left, "name"):
                v = getattr(c.right, "value", c.right)
                v = getattr(v, "value", v)  # desenvuelve enum si aplica
                pairs[c.left.name] = v
        name = entity.__name__

        if name == "ToolAdapter":
            if "name" in pairs:
                return _Result([self._by_name(self.adapters, pairs["name"])])
            rows = list(self.adapters.values())
            if "is_active" in pairs:
                rows = [r for r in rows if r.is_active == pairs["is_active"]]
            return _Result(rows)

        if name == "ProductionTemplate":
            if "name" in pairs:
                t = self._by_name(self.templates, pairs["name"])
                if (
                    t is not None
                    and "is_active" in pairs
                    and t.is_active != pairs["is_active"]
                ):
                    t = None
                return _Result([t])
            rows = list(self.templates.values())
            if "content_type" in pairs:
                rows = [r for r in rows if r.content_type == pairs["content_type"]]
            if "phase" in pairs:
                rows = [r for r in rows if r.phase == pairs["phase"]]
            if "is_active" in pairs:
                rows = [r for r in rows if r.is_active == pairs["is_active"]]
            return _Result(rows)

        if name == "ProductionManifest":
            rows = [
                m for m in self.manifests if m.artifact_id == pairs.get("artifact_id")
            ]
            if "version" in pairs:
                rows = [m for m in rows if m.version == pairs["version"]]
            rows = sorted(rows, key=lambda m: m.version, reverse=True)
            return _Result(rows)

        if name == "ContentArtifact":
            return _Result([self.artifacts.get(pairs.get("id"))])

        if name == "AssetJob":
            rows = [j for j in self.jobs if j.artifact_id == pairs.get("artifact_id")]
            rows = sorted(rows, key=lambda j: j.sequence_order)
            return _Result(rows)

        if name == "FormatSpec":
            if "id" in pairs:
                return _Result([self.specs.get(pairs["id"])])
            rows = list(self.specs.values())
            if "artifact_type" in pairs:
                rows = [r for r in rows if r.artifact_type == pairs["artifact_type"]]
            if "is_active" in pairs:
                rows = [r for r in rows if r.is_active == pairs["is_active"]]
            return _Result(rows)

        return _Result([])

    # ------------------------------------------------------------------
    # ciclo de vida
    # ------------------------------------------------------------------

    def add(self, obj):
        self._added.append(obj)

    def flush(self):
        for obj in self._added:
            if getattr(obj, "id", None) is None:
                obj.id = uuid.uuid4()
            cls = obj.__class__.__name__
            if cls == "ToolAdapter":
                self.adapters[obj.id] = obj
            elif cls == "ProductionTemplate":
                self.templates[obj.id] = obj
            elif cls == "FormatSpec":
                self.specs[obj.id] = obj
            elif cls == "ContentArtifact":
                self.artifacts[obj.id] = obj
            elif cls == "ProductionManifest" and obj not in self.manifests:
                self.manifests.append(obj)
            elif cls == "AssetJob" and obj not in self.jobs:
                self.jobs.append(obj)
        self._added = []

    def commit(self):
        self.flush()

    def refresh(self, obj):
        pass

    def delete(self, obj):
        cls = obj.__class__.__name__
        if cls == "ToolAdapter":
            self.adapters.pop(obj.id, None)
        elif cls == "ProductionTemplate":
            self.templates.pop(obj.id, None)
        elif cls == "FormatSpec":
            self.specs.pop(obj.id, None)
        elif cls == "ContentArtifact":
            self.artifacts.pop(obj.id, None)
        elif cls == "ProductionManifest":
            self.manifests = [m for m in self.manifests if m.id != obj.id]
        elif cls == "AssetJob":
            self.jobs = [j for j in self.jobs if j.id != obj.id]

    def close(self):
        pass


# ----------------------------------------------------------------------
# helpers de datos
# ----------------------------------------------------------------------


def _adapter(name="elevenlabs", **overrides) -> ToolAdapter:
    defaults = dict(
        id=uuid.uuid4(),
        name=name,
        mcp_server_name=name,
        execution_mode="local",
        requires_license=None,
        is_active=True,
    )
    defaults.update(overrides)
    return ToolAdapter(**defaults)


def _template(name="Storyboard_Spec", **overrides) -> ProductionTemplate:
    defaults = dict(
        id=uuid.uuid4(),
        name=name,
        content_type="video",
        phase="preproduccion",
        template_format="json",
        content={"escenas": []},
        version="1.0",
        is_active=True,
    )
    defaults.update(overrides)
    return ProductionTemplate(**defaults)


def _spec(**overrides) -> FormatSpec:
    defaults = dict(
        id=uuid.uuid4(),
        brand_objective=BrandObjective.ESCAPE_SOCIAL,
        artifact_type="video_corto",
        structure={},
        constraints={},
        derivation_rules=[],
        visual_requirements={},
        qa_checks=[],
        tool_chain=["producer", "elevenlabs", "comfyui", "resolve"],
        phases={},
        is_active=True,
    )
    defaults.update(overrides)
    return FormatSpec(**defaults)


def _artifact(**overrides) -> ContentArtifact:
    defaults = dict(
        id=uuid.uuid4(),
        brief_id=uuid.uuid4(),
        artifact_type="video_corto",
        channel="instagram",
        status="borrador",
    )
    defaults.update(overrides)
    return ContentArtifact(**defaults)


def _build_produce_scenario():
    """Artefacto video_corto con spec, adapters, templates y manifiesto v1."""
    fake = _FakeSession()
    aid = uuid.uuid4()
    artifact = _artifact(id=aid)
    fake.add(artifact)
    fake.flush()

    for a in [_adapter("elevenlabs"), _adapter("comfyui"), _adapter("resolve")]:
        fake.add(a)
    fake.flush()

    for t in [
        _template("Storyboard_Spec", content_type="video", phase="preproduccion"),
        _template("TTS_Voice_Preset", content_type="audio", phase="produccion"),
        _template("Render_Export_Preset", content_type="video", phase="postproduccion"),
    ]:
        fake.add(t)
    fake.flush()

    phases = {
        "preproduccion": {"steps": ["desglose"], "templates": ["Storyboard_Spec"]},
        "produccion": {"steps": ["voz"], "templates": ["TTS_Voice_Preset"]},
        "postproduccion": {"steps": ["render"], "templates": ["Render_Export_Preset"]},
    }
    spec = _spec(
        tool_chain=["producer", "elevenlabs", "comfyui", "resolve"], phases=phases
    )
    fake.add(spec)
    fake.flush()
    artifact.format_spec_id = spec.id

    create_manifest(
        fake,
        aid,
        {
            "project": {"artifact_id": str(aid), "artifact_type": "video_corto"},
            "audio": {"voice_id": "v1"},
        },
    )
    return fake, aid


@pytest.fixture
def client():
    return TestClient(app)


# ----------------------------------------------------------------------
# CRUD /tool-adapters
# ----------------------------------------------------------------------


def test_tool_adapters_crud(client):
    fake = _FakeSession()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        # list vacío
        resp = client.get("/tool-adapters")
        assert resp.status_code == 200
        assert resp.json() == []

        # create
        resp = client.post(
            "/tool-adapters",
            json={
                "name": "inkscape",
                "mcp_server_name": "inkscape",
                "execution_mode": "local",
            },
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["name"] == "inkscape"
        assert body["mcp_server_name"] == "inkscape"
        assert body["execution_mode"] == "local"
        assert body["is_active"] is True
        assert body["requires_license"] is None
        adapter_id = body["id"]

        # 409 por name duplicado
        resp = client.post(
            "/tool-adapters",
            json={
                "name": "inkscape",
                "mcp_server_name": "inkscape",
                "execution_mode": "local",
            },
        )
        assert resp.status_code == 409

        # list ya no vacío
        resp = client.get("/tool-adapters")
        assert resp.status_code == 200
        assert len(resp.json()) == 1

        # get
        resp = client.get(f"/tool-adapters/{adapter_id}")
        assert resp.status_code == 200
        assert resp.json()["name"] == "inkscape"

        # patch parcial
        resp = client.patch(
            f"/tool-adapters/{adapter_id}",
            json={"is_active": False, "requires_license": "GPL"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["is_active"] is False
        assert body["requires_license"] == "GPL"
        assert body["name"] == "inkscape"  # el resto no cambia

        # delete
        resp = client.delete(f"/tool-adapters/{adapter_id}")
        assert resp.status_code == 204
        assert client.get(f"/tool-adapters/{adapter_id}").status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_tool_adapters_404s(client):
    fake = _FakeSession()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        missing = str(uuid.uuid4())
        assert client.get(f"/tool-adapters/{missing}").status_code == 404
        assert (
            client.patch(
                f"/tool-adapters/{missing}", json={"is_active": False}
            ).status_code
            == 404
        )
        assert client.delete(f"/tool-adapters/{missing}").status_code == 404
    finally:
        app.dependency_overrides.clear()


# ----------------------------------------------------------------------
# CRUD /production-templates
# ----------------------------------------------------------------------


def test_production_templates_crud_and_filters(client):
    fake = _FakeSession()
    fake.add(_template("Storyboard_Spec", content_type="video", phase="preproduccion"))
    fake.add(_template("TTS_Voice_Preset", content_type="audio", phase="produccion"))
    fake.flush()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        # list con filtro content_type
        resp = client.get("/production-templates", params={"content_type": "video"})
        assert resp.status_code == 200
        assert [t["name"] for t in resp.json()] == ["Storyboard_Spec"]

        # list con filtro phase
        resp = client.get("/production-templates", params={"phase": "produccion"})
        assert resp.status_code == 200
        assert [t["name"] for t in resp.json()] == ["TTS_Voice_Preset"]

        # list sin filtros
        resp = client.get("/production-templates")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

        # create
        resp = client.post(
            "/production-templates",
            json={
                "name": "Render_Export_Preset",
                "content_type": "video",
                "phase": "postproduccion",
                "template_format": "json",
                "content": {"codec_video": ["H.264"]},
            },
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["name"] == "Render_Export_Preset"
        assert body["version"] == "1.0"
        assert body["is_active"] is True
        assert body["content"] == {"codec_video": ["H.264"]}
        template_id = body["id"]

        # 409 por name duplicado
        resp = client.post(
            "/production-templates",
            json={
                "name": "Render_Export_Preset",
                "content_type": "video",
                "phase": "postproduccion",
                "template_format": "json",
            },
        )
        assert resp.status_code == 409

        # get
        resp = client.get(f"/production-templates/{template_id}")
        assert resp.status_code == 200
        assert resp.json()["content"]["codec_video"] == ["H.264"]

        # patch parcial
        resp = client.patch(
            f"/production-templates/{template_id}",
            json={"version": "2.0", "is_active": False},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["version"] == "2.0"
        assert body["is_active"] is False
        assert body["name"] == "Render_Export_Preset"

        # delete
        resp = client.delete(f"/production-templates/{template_id}")
        assert resp.status_code == 204
        assert client.get(f"/production-templates/{template_id}").status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_production_templates_404s(client):
    fake = _FakeSession()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        missing = str(uuid.uuid4())
        assert client.get(f"/production-templates/{missing}").status_code == 404
        assert (
            client.patch(
                f"/production-templates/{missing}", json={"version": "2.0"}
            ).status_code
            == 404
        )
        assert client.delete(f"/production-templates/{missing}").status_code == 404
    finally:
        app.dependency_overrides.clear()


# ----------------------------------------------------------------------
# Manifiestos por artefacto
# ----------------------------------------------------------------------


def test_manifests_versioning_and_listing(client):
    fake = _FakeSession()
    aid = uuid.uuid4()
    fake.add(_artifact(id=aid))
    fake.flush()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        # versiona 1 → 2 → 3
        for i in range(1, 4):
            resp = client.post(
                f"/artifacts/{aid}/manifests", json={"manifest": {"v": i}}
            )
            assert resp.status_code == 201
            assert resp.json()["version"] == i

        # list desc
        resp = client.get(f"/artifacts/{aid}/manifests")
        assert resp.status_code == 200
        assert [m["version"] for m in resp.json()] == [3, 2, 1]

        # get por versión
        resp = client.get(f"/artifacts/{aid}/manifests/2")
        assert resp.status_code == 200
        assert resp.json()["manifest"] == {"v": 2}

        # 404 versión inexistente
        resp = client.get(f"/artifacts/{aid}/manifests/99")
        assert resp.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_manifests_404_when_artifact_missing(client):
    fake = _FakeSession()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        missing = str(uuid.uuid4())
        assert client.get(f"/artifacts/{missing}/manifests").status_code == 404
        assert (
            client.post(
                f"/artifacts/{missing}/manifests", json={"manifest": {}}
            ).status_code
            == 404
        )
        assert client.get(f"/artifacts/{missing}/manifests/1").status_code == 404
    finally:
        app.dependency_overrides.clear()


# ----------------------------------------------------------------------
# Jobs por artefacto
# ----------------------------------------------------------------------


def test_artifact_jobs_listing(client):
    fake = _FakeSession()
    aid = uuid.uuid4()
    adapter = _adapter("elevenlabs")
    fake.add(_artifact(id=aid))
    fake.add(adapter)
    fake.flush()
    fake.add(
        AssetJob(
            id=uuid.uuid4(),
            artifact_id=aid,
            tool_adapter_id=adapter.id,
            sequence_order=2,
            status="done",
            phase="produccion",
        )
    )
    fake.add(
        AssetJob(
            id=uuid.uuid4(),
            artifact_id=aid,
            tool_adapter_id=adapter.id,
            sequence_order=1,
            status="done",
            phase="preproduccion",
        )
    )
    fake.flush()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        resp = client.get(f"/artifacts/{aid}/jobs")
        assert resp.status_code == 200
        body = resp.json()
        assert [j["sequence_order"] for j in body] == [1, 2]
        assert all(j["status"] == "done" for j in body)
        assert all(j["artifact_id"] == str(aid) for j in body)
    finally:
        app.dependency_overrides.clear()


def test_artifact_jobs_404_when_artifact_missing(client):
    fake = _FakeSession()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        resp = client.get(f"/artifacts/{uuid.uuid4()}/jobs")
        assert resp.status_code == 404
    finally:
        app.dependency_overrides.clear()


# ----------------------------------------------------------------------
# Endpoint disparador /produce
# ----------------------------------------------------------------------


def test_produce_success(client):
    fake, aid = _build_produce_scenario()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        resp = client.post(f"/artifacts/{aid}/produce")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["artifact_id"] == str(aid)
        assert body["manifest_version"] == 1
        jobs = body["jobs"]
        assert len(jobs) == 3
        assert all(j["status"] == "done" for j in jobs)
        assert [j["phase"] for j in jobs] == [
            "preproduccion",
            "produccion",
            "postproduccion",
        ]
        # jobs persistidos con trazabilidad
        assert len(fake.jobs) == 3
        assert all(j.status == "done" for j in fake.jobs)
        # artifact actualizado
        assert fake.artifacts[aid].status == "listo"
        assert fake.artifacts[aid].storage_path is not None
    finally:
        app.dependency_overrides.clear()


def test_produce_with_manifest_creates_new_version(client):
    fake, aid = _build_produce_scenario()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        resp = client.post(
            f"/artifacts/{aid}/produce",
            json={
                "manifest": {
                    "project": {
                        "artifact_id": str(aid),
                        "artifact_type": "video_corto",
                    },
                    "audio": {"voice_id": "v2"},
                }
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["manifest_version"] == 2
        assert len(fake.manifests) == 2
    finally:
        app.dependency_overrides.clear()


def test_produce_409_without_manifest(client):
    fake = _FakeSession()
    aid = uuid.uuid4()
    artifact = _artifact(id=aid)
    spec = _spec()
    fake.add(artifact)
    fake.add(spec)
    fake.flush()
    artifact.format_spec_id = spec.id
    app.dependency_overrides[get_session] = lambda: fake
    try:
        resp = client.post(f"/artifacts/{aid}/produce")
        assert resp.status_code == 409
        assert "manifiesto" in resp.json()["detail"]
    finally:
        app.dependency_overrides.clear()


def test_produce_409_without_format_spec(client):
    fake = _FakeSession()
    aid = uuid.uuid4()
    fake.add(_artifact(id=aid))
    fake.flush()
    create_manifest(fake, aid, {"project": {}})
    app.dependency_overrides[get_session] = lambda: fake
    try:
        resp = client.post(f"/artifacts/{aid}/produce")
        assert resp.status_code == 409
        assert "format_spec" in resp.json()["detail"]
    finally:
        app.dependency_overrides.clear()


def test_produce_404_when_artifact_missing(client):
    fake = _FakeSession()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        resp = client.post(f"/artifacts/{uuid.uuid4()}/produce")
        assert resp.status_code == 404
    finally:
        app.dependency_overrides.clear()


# ----------------------------------------------------------------------
# Validación de la cadena de herramientas
# ----------------------------------------------------------------------


def test_validate_tool_chain_ok(client):
    fake = _FakeSession()
    fake.add(_adapter("elevenlabs"))
    fake.add(_template("Storyboard_Spec"))
    fake.add(
        _spec(
            tool_chain=["producer", "elevenlabs"],
            phases={"preproduccion": {"templates": ["Storyboard_Spec"]}},
        )
    )
    fake.flush()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        resp = client.get("/tools/validate")
        assert resp.status_code == 200
        assert resp.json() == {"errors": []}
    finally:
        app.dependency_overrides.clear()


def test_validate_tool_chain_reports_errors(client):
    fake = _FakeSession()
    fake.add(
        _spec(
            tool_chain=["producer", "ghost_tool"],
            phases={"produccion": {"templates": ["Ghost_Template"]}},
        )
    )
    fake.flush()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        resp = client.get("/tools/validate")
        assert resp.status_code == 200
        errors = resp.json()["errors"]
        assert any("ghost_tool" in e for e in errors)
        assert any("Ghost_Template" in e for e in errors)
    finally:
        app.dependency_overrides.clear()
