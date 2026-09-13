"""Tests de la Fase 4 — Proyectos (0007): projects + project_versions.

Cubre el CRUD de /projects, el versionado de snapshots (approve/rollback
con historial inmutable) y los 404s. Usa TestClient + _FakeSession en
memoria (sin DB ni red).
"""

import uuid

import pytest
from fastapi.testclient import TestClient

from src.api.main import app, get_session
from src.db.models import ProductionTemplate, Project, ProjectVersion


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
    """Sesión mínima en memoria para la API de proyectos (Fase 4)."""

    def __init__(self):
        self.projects = {}  # id -> Project
        self.versions = {}  # id -> ProjectVersion
        self.templates = {}  # id -> ProductionTemplate
        self._added = []

    # ------------------------------------------------------------------
    # consultas
    # ------------------------------------------------------------------

    def get(self, model, ident):
        if model is Project:
            return self.projects.get(ident)
        if model is ProjectVersion:
            return self.versions.get(ident)
        if model is ProductionTemplate:
            return self.templates.get(ident)
        return None

    def execute(self, stmt):
        entity = stmt.column_descriptions[0]["entity"]
        col_name = stmt.column_descriptions[0].get("name")
        pairs = {}
        for c in stmt._where_criteria:
            if hasattr(c.left, "name"):
                v = getattr(c.right, "value", c.right)
                v = getattr(v, "value", v)  # desenvuelve enum si aplica
                pairs[c.left.name] = v
        name = entity.__name__

        if name == "Project":
            if "id" in pairs:
                return _Result([self.projects.get(pairs["id"])])
            rows = list(self.projects.values())
            if "status" in pairs:
                rows = [r for r in rows if r.status == pairs["status"]]
            if "artifact_type" in pairs:
                rows = [r for r in rows if r.artifact_type == pairs["artifact_type"]]
            return _Result(rows)

        if name == "ProjectVersion":
            rows = [
                v
                for v in self.versions.values()
                if v.project_id == pairs.get("project_id")
            ]
            if "version" in pairs:
                rows = [v for v in rows if v.version == pairs["version"]]
            rows = sorted(rows, key=lambda v: v.version, reverse=True)
            if col_name == "version":  # select(ProjectVersion.version)
                return _Result([v.version for v in rows])
            return _Result(rows)

        if name == "ProductionTemplate":
            return _Result([self.templates.get(pairs.get("id"))])

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
            if cls == "Project":
                self.projects[obj.id] = obj
            elif cls == "ProjectVersion":
                self.versions[obj.id] = obj
            elif cls == "ProductionTemplate":
                self.templates[obj.id] = obj
        self._added = []

    def commit(self):
        self.flush()

    def refresh(self, obj):
        pass

    def delete(self, obj):
        cls = obj.__class__.__name__
        if cls == "Project":
            self.projects.pop(obj.id, None)
            # CASCADE: borra las versiones del proyecto
            self.versions = {
                k: v for k, v in self.versions.items() if v.project_id != obj.id
            }
        elif cls == "ProjectVersion":
            self.versions.pop(obj.id, None)
        elif cls == "ProductionTemplate":
            self.templates.pop(obj.id, None)

    def close(self):
        pass


# ----------------------------------------------------------------------
# helpers de datos
# ----------------------------------------------------------------------


def _template(**overrides) -> ProductionTemplate:
    defaults = dict(
        id=uuid.uuid4(),
        name="Storyboard_Spec",
        content_type="video",
        phase="preproduccion",
        template_format="json",
        content={},
        version="1.0",
        is_active=True,
    )
    defaults.update(overrides)
    return ProductionTemplate(**defaults)


@pytest.fixture
def client():
    return TestClient(app)


def _new_project(client, **overrides) -> dict:
    """Crea un proyecto vía API y devuelve el body de la respuesta."""
    payload = dict(name="Corto Escape", topic="IA local vs nube")
    payload.update(overrides)
    resp = client.post("/project/new", json=payload)
    assert resp.status_code == 201
    return resp.json()


# ----------------------------------------------------------------------
# POST /project/new
# ----------------------------------------------------------------------


def test_project_new_creates_borrador_with_version_1(client):
    fake = _FakeSession()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        body = _new_project(
            client,
            brand_objective="ESCAPE_SOCIAL",
            artifact_type="video_corto",
        )
        project = body["project"]
        version = body["version"]

        assert project["name"] == "Corto Escape"
        assert project["topic"] == "IA local vs nube"
        assert project["brand_objective"] == "ESCAPE_SOCIAL"
        assert project["artifact_type"] == "video_corto"
        assert project["status"] == "borrador"
        assert project["current_version"] == 1
        assert project["template_id"] is None
        assert project["storage_path"] is None

        assert version["project_id"] == project["id"]
        assert version["version"] == 1
        assert version["snapshot"] == {}

        # la versión 1 queda persistida en la sesión fake
        assert len(fake.versions) == 1
    finally:
        app.dependency_overrides.clear()


def test_project_new_404_when_template_missing(client):
    fake = _FakeSession()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        missing = str(uuid.uuid4())
        resp = client.post(
            "/project/new",
            json={"name": "X", "topic": "Y", "template_id": missing},
        )
        assert resp.status_code == 404
        assert fake.projects == {}  # no se creó nada
    finally:
        app.dependency_overrides.clear()


def test_project_new_with_template(client):
    fake = _FakeSession()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        tpl = _template()
        fake.add(tpl)
        fake.flush()
        body = _new_project(client, template_id=str(tpl.id))
        assert body["project"]["template_id"] == str(tpl.id)
    finally:
        app.dependency_overrides.clear()


# ----------------------------------------------------------------------
# GET /projects
# ----------------------------------------------------------------------


def test_projects_list_and_filters(client):
    fake = _FakeSession()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        _new_project(client, name="A", artifact_type="video_corto")
        _new_project(client, name="B", artifact_type="audio")
        _new_project(client, name="C", artifact_type="video_corto")

        resp = client.get("/projects")
        assert resp.status_code == 200
        assert len(resp.json()) == 3

        resp = client.get("/projects", params={"artifact_type": "video_corto"})
        assert resp.status_code == 200
        assert {p["name"] for p in resp.json()} == {"A", "C"}

        resp = client.get("/projects", params={"status": "borrador"})
        assert resp.status_code == 200
        assert len(resp.json()) == 3

        # filtro combinado sin resultados
        resp = client.get(
            "/projects",
            params={"status": "aprobado", "artifact_type": "video_corto"},
        )
        assert resp.status_code == 200
        assert resp.json() == []
    finally:
        app.dependency_overrides.clear()


# ----------------------------------------------------------------------
# GET /projects/{id}
# ----------------------------------------------------------------------


def test_project_detail_and_404(client):
    fake = _FakeSession()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        body = _new_project(client)
        project_id = body["project"]["id"]

        resp = client.get(f"/projects/{project_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == project_id
        assert resp.json()["name"] == "Corto Escape"

        missing = str(uuid.uuid4())
        assert client.get(f"/projects/{missing}").status_code == 404
    finally:
        app.dependency_overrides.clear()


# ----------------------------------------------------------------------
# PATCH /projects/{id}
# ----------------------------------------------------------------------


def test_project_patch_updates_fields(client):
    fake = _FakeSession()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        body = _new_project(client)
        project_id = body["project"]["id"]

        resp = client.patch(
            f"/projects/{project_id}",
            json={"name": "Nuevo nombre", "status": "en_edicion"},
        )
        assert resp.status_code == 200
        updated = resp.json()
        assert updated["name"] == "Nuevo nombre"
        assert updated["status"] == "en_edicion"
        assert updated["topic"] == "IA local vs nube"  # el resto no cambia

        missing = str(uuid.uuid4())
        assert (
            client.patch(f"/projects/{missing}", json={"name": "X"}).status_code == 404
        )
    finally:
        app.dependency_overrides.clear()


# ----------------------------------------------------------------------
# DELETE /projects/{id}
# ----------------------------------------------------------------------


def test_project_delete_cascades_versions(client):
    fake = _FakeSession()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        body = _new_project(client)
        project_id = body["project"]["id"]
        assert len(fake.versions) == 1

        resp = client.delete(f"/projects/{project_id}")
        assert resp.status_code == 204
        assert client.get(f"/projects/{project_id}").status_code == 404
        assert fake.projects == {}
        assert fake.versions == {}  # CASCADE borró las versiones

        missing = str(uuid.uuid4())
        assert client.delete(f"/projects/{missing}").status_code == 404
    finally:
        app.dependency_overrides.clear()


# ----------------------------------------------------------------------
# GET /projects/{id}/versions
# ----------------------------------------------------------------------


def test_project_versions_list_desc(client):
    fake = _FakeSession()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        body = _new_project(client)
        project_id = body["project"]["id"]

        # approve dos veces -> versiones 2 y 3
        client.post(f"/projects/{project_id}/approve", json={"json_editado": {"a": 1}})
        client.post(f"/projects/{project_id}/approve", json={"json_editado": {"b": 2}})

        resp = client.get(f"/projects/{project_id}/versions")
        assert resp.status_code == 200
        versions = resp.json()
        assert [v["version"] for v in versions] == [3, 2, 1]

        missing = str(uuid.uuid4())
        assert client.get(f"/projects/{missing}/versions").status_code == 404
    finally:
        app.dependency_overrides.clear()


# ----------------------------------------------------------------------
# GET /projects/{id}/versions/{version}
# ----------------------------------------------------------------------


def test_project_version_detail_and_404s(client):
    fake = _FakeSession()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        body = _new_project(client)
        project_id = body["project"]["id"]

        resp = client.get(f"/projects/{project_id}/versions/1")
        assert resp.status_code == 200
        assert resp.json()["version"] == 1
        assert resp.json()["snapshot"] == {}

        # versión inexistente
        assert client.get(f"/projects/{project_id}/versions/99").status_code == 404
        # proyecto inexistente
        missing = str(uuid.uuid4())
        assert client.get(f"/projects/{missing}/versions/1").status_code == 404
    finally:
        app.dependency_overrides.clear()


# ----------------------------------------------------------------------
# POST /projects/{id}/approve
# ----------------------------------------------------------------------


def test_project_approve_creates_version_and_marks_aprobado(client):
    fake = _FakeSession()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        body = _new_project(client)
        project_id = body["project"]["id"]

        resp = client.post(
            f"/projects/{project_id}/approve",
            json={"json_editado": {"guion_vocal": "hola", "prompts_img": []}},
        )
        assert resp.status_code == 200
        result = resp.json()
        project = result["project"]
        version = result["version"]

        assert project["current_version"] == 2
        assert project["status"] == "aprobado"
        assert project["storage_path"] is not None
        assert project["storage_path"].endswith(f"/projects/{project_id}")

        assert version["version"] == 2
        assert version["project_id"] == project_id
        assert version["snapshot"] == {"guion_vocal": "hola", "prompts_img": []}

        # la versión 1 sigue intacta (historial inmutable)
        resp = client.get(f"/projects/{project_id}/versions/1")
        assert resp.status_code == 200
        assert resp.json()["snapshot"] == {}

        missing = str(uuid.uuid4())
        assert (
            client.post(
                f"/projects/{missing}/approve", json={"json_editado": {}}
            ).status_code
            == 404
        )
    finally:
        app.dependency_overrides.clear()


def test_project_approve_twice_is_idempotent(client):
    fake = _FakeSession()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        body = _new_project(client)
        project_id = body["project"]["id"]

        r1 = client.post(
            f"/projects/{project_id}/approve", json={"json_editado": {"v": 1}}
        )
        r2 = client.post(
            f"/projects/{project_id}/approve", json={"json_editado": {"v": 2}}
        )
        assert r1.status_code == 200
        assert r2.status_code == 200

        assert r1.json()["version"]["version"] == 2
        assert r2.json()["version"]["version"] == 3
        assert r2.json()["project"]["current_version"] == 3

        resp = client.get(f"/projects/{project_id}/versions")
        versions = resp.json()
        assert [v["version"] for v in versions] == [3, 2, 1]
        assert len(versions) == 3  # sin duplicados
    finally:
        app.dependency_overrides.clear()


# ----------------------------------------------------------------------
# POST /projects/{id}/rollback
# ----------------------------------------------------------------------


def test_project_rollback_restores_as_new_version(client):
    fake = _FakeSession()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        body = _new_project(client)
        project_id = body["project"]["id"]

        # v2 aprobada con contenido
        client.post(
            f"/projects/{project_id}/approve",
            json={"json_editado": {"guion_vocal": "aprobado"}},
        )

        # rollback a la v1 (snapshot {}) -> v3 nueva
        resp = client.post(f"/projects/{project_id}/rollback", json={"version": 1})
        assert resp.status_code == 200
        result = resp.json()
        project = result["project"]
        version = result["version"]

        assert project["current_version"] == 3
        assert project["status"] == "en_edicion"
        assert version["version"] == 3
        assert version["snapshot"] == {}

        # el historial completo sigue: 1, 2, 3
        resp = client.get(f"/projects/{project_id}/versions")
        assert [v["version"] for v in resp.json()] == [3, 2, 1]

        # rollback a la v2 -> v4 con el snapshot de la v2
        resp = client.post(f"/projects/{project_id}/rollback", json={"version": 2})
        assert resp.status_code == 200
        assert resp.json()["version"]["version"] == 4
        assert resp.json()["version"]["snapshot"] == {"guion_vocal": "aprobado"}
        assert resp.json()["project"]["current_version"] == 4
    finally:
        app.dependency_overrides.clear()


def test_project_rollback_404s(client):
    fake = _FakeSession()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        body = _new_project(client)
        project_id = body["project"]["id"]

        # versión inexistente
        assert (
            client.post(
                f"/projects/{project_id}/rollback", json={"version": 99}
            ).status_code
            == 404
        )
        # proyecto inexistente
        missing = str(uuid.uuid4())
        assert (
            client.post(
                f"/projects/{missing}/rollback", json={"version": 1}
            ).status_code
            == 404
        )
    finally:
        app.dependency_overrides.clear()
