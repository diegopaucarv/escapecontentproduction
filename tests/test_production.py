"""Tests de la Fase 5 — Flujo agéntico de proyectos.

Cubre /projects/{id}/generate, /refine, /produce y /postproduction con
TestClient + _FakeSession en memoria (sin DB ni red). El LLM se
monkeypatchea con un stub; el orquestador corre en modo directo
(McpClient sin base_url no hace red).
"""

import uuid
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.api.main import app, get_session
from src.auth import hash_password
from src.db.models import (
    AppUser,
    BrandObjective,
    ContentArtifact,
    FormatSpec,
    ProductionTemplate,
    Project,
    ProjectVersion,
    SessionSettings,
    ToolAdapter,
)

FIXED_JSON = '{"guion_vocal": "Hola", "prompts_img": []}'


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
    """Sesión mínima en memoria para el flujo agéntico (Fase 5)."""

    def __init__(self):
        self._store = {}  # class name -> {id: obj}
        self._added = []
        self.commits = 0

    def _rows(self, name):
        return self._store.setdefault(name, {})

    def get(self, model, ident):
        name = getattr(model, "__name__", "")
        return self._rows(name).get(ident)

    def scalar(self, stmt):
        # select(AppUser).where(AppUser.email == email, AppUser.is_active.is_(True))
        email = None
        for crit in getattr(stmt, "_where_criteria", []):
            if getattr(crit.left, "name", None) == "email":
                email = crit.right.value
        if email is None:
            return None
        return next(
            (
                u
                for u in self._rows("AppUser").values()
                if u.email == email and u.is_active
            ),
            None,
        )

    def execute(self, stmt):
        entity = stmt.column_descriptions[0]["entity"]
        col_name = stmt.column_descriptions[0].get("name")
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
        if name == "ProjectVersion":
            rows = sorted(rows, key=lambda r: r.version, reverse=True)
            if col_name == "version":  # select(ProjectVersion.version)
                return _Result([r.version for r in rows])
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

    def delete(self, obj):
        self._rows(obj.__class__.__name__).pop(obj.id, None)

    def close(self):
        pass


# ----------------------------------------------------------------------
# helpers de datos
# ----------------------------------------------------------------------


@pytest.fixture
def client():
    return TestClient(app)


def _new_project(client, **overrides) -> dict:
    """Crea un proyecto vía API y devuelve el body de la respuesta."""
    payload = dict(
        name="Corto Escape",
        topic="IA local vs nube",
        segment_client="S1",
        risk_level="bajo",
    )
    payload.update(overrides)
    resp = client.post("/project/new", json=payload)
    assert resp.status_code == 201
    return resp.json()


def _template(**overrides) -> ProductionTemplate:
    defaults = dict(
        id=uuid.uuid4(),
        name="Render_Export_Preset",
        content_type="video",
        phase="postproduccion",
        template_format="json",
        content={"codec_video": ["H.264"]},
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
        structure={"guion_vocal": "string", "prompts_img": "array"},
        constraints={"cta_unico": True},
        derivation_rules=[],
        visual_requirements={},
        qa_checks=["cta_unico"],
        tool_chain=["producer", "elevenlabs"],
        phases={
            "postproduccion": {
                "steps": ["render"],
                "templates": ["Render_Export_Preset"],
            },
        },
        is_active=True,
    )
    defaults.update(overrides)
    return FormatSpec(**defaults)


def _adapter(**overrides) -> ToolAdapter:
    defaults = dict(
        id=uuid.uuid4(),
        name="elevenlabs",
        mcp_server_name="elevenlabs",
        execution_mode="local",
        requires_license=None,
        is_active=True,
    )
    defaults.update(overrides)
    return ToolAdapter(**defaults)


def _settings(**overrides) -> SessionSettings:
    """SessionSettings activa para el flujo LLM (0007)."""
    defaults = dict(
        id=uuid.uuid4(),
        api_key_id=uuid.uuid4(),
        small_model="meta-models/Muse-Glimmer-30B",
        large_model="deepseek-ai/DeepSeek-V4-Flash-0731",
        fallback_model=None,
        llm_retries=3,
        temperature_small=0.7,
        temperature_large=0.7,
        max_tokens_small=2048,
        max_tokens_large=4096,
        is_active=True,
    )
    defaults.update(overrides)
    return SessionSettings(**defaults)


def _artifact(prompt_text="Eres el compilador de contenido de Pipeline OS."):
    return SimpleNamespace(prompt_text=prompt_text)


def _patch_compiler(monkeypatch, artifact):
    import src.llm.compiler as compiler_mod

    monkeypatch.setattr(
        compiler_mod,
        "get_active_prompt",
        lambda session, model, task: artifact,
    )


def _patch_complete(monkeypatch, text):
    import src.llm.together as together_mod

    def fake_complete(
        session, prompt, model_size="small", system=None, response_format=None, **kwargs
    ):
        return text

    monkeypatch.setattr(together_mod, "complete", fake_complete)


def _approve(client, project_id: str, fake) -> None:
    """Crea un 🟨 líder y aprueba el proyecto con su token."""
    lider = AppUser(
        id=uuid.uuid4(),
        full_name="Líder de Prueba",
        email="lider@ergalia.com",
        role="lider",
        hashed_password=hash_password("clave-segura"),
        is_active=True,
    )
    fake.add(lider)
    fake.flush()
    resp = client.post(
        "/auth/token",
        data={"username": "lider@ergalia.com", "password": "clave-segura"},
    )
    assert resp.status_code == 200
    token = resp.json()["access_token"]
    resp = client.post(
        f"/projects/{project_id}/approve",
        json={"json_editado": {"guion_vocal": "Hola"}},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200


def _produce_and_approve_brief(client, project_id: str, fake) -> None:
    """Dispara /produce (el gate pide revisión humana), aprueba el brief
    compañero como 🟨 líder y re-dispara /produce hasta que la cadena corre.

    Gobernanza 0009/0010: el brief compañero nace en 'revision' y el
    Gatekeeper exige OWNER_APPROVAL (nueva_solucion + complete siempre
    requieren revisión humana). Este helper simula ese paso humano.
    """
    lider = AppUser(
        id=uuid.uuid4(),
        full_name="Líder de Prueba",
        email="lider@ergalia.com",
        role="lider",
        hashed_password=hash_password("clave-segura"),
        is_active=True,
    )
    fake.add(lider)
    fake.flush()
    resp = client.post(
        "/auth/token",
        data={"username": "lider@ergalia.com", "password": "clave-segura"},
    )
    assert resp.status_code == 200
    token = resp.json()["access_token"]

    # 1er produce: el gate pide revisión humana -> 409 con brief_id
    resp = client.post(f"/projects/{project_id}/produce")
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert "brief" in detail
    # "El brief compañero {id} exige..." -> el id está en la posición 3
    brief_id = detail.split(" ")[3]

    # aprobar el brief compañero (OWNER_APPROVAL)
    resp = client.post(
        f"/briefs/{brief_id}/approve",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200

    # 2do produce: el brief ya está aprobado -> corre la cadena
    resp = client.post(f"/projects/{project_id}/produce")
    assert resp.status_code == 200
    return resp.json()


# ----------------------------------------------------------------------
# POST /projects/{id}/generate
# ----------------------------------------------------------------------


def test_generate_creates_new_version(client, monkeypatch):
    _patch_compiler(monkeypatch, _artifact())
    _patch_complete(monkeypatch, FIXED_JSON)
    fake = _FakeSession()
    fake.add(_settings())
    fake.flush()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        body = _new_project(
            client, brand_objective="ESCAPE_SOCIAL", artifact_type="video_corto"
        )
        project_id = body["project"]["id"]

        resp = client.post(f"/projects/{project_id}/generate", json={})
        assert resp.status_code == 200
        result = resp.json()

        # 0007: decisión del LLM — ok, sin aceptación obligatoria.
        assert result["decision"]["status"] == "ok"
        assert result["decision"]["decision_source"] == "llm"
        assert result["decision"]["requires_user_acceptance"] is False

        assert result["project"]["status"] == "en_edicion"
        assert result["project"]["current_version"] == 2
        assert result["version"]["version"] == 2
        assert result["version"]["snapshot"]["guion_vocal"] == "Hola"
        assert result["version"]["snapshot"]["prompts_img"] == []

        # la versión 1 (esqueleto) sigue intacta
        versions = client.get(f"/projects/{project_id}/versions").json()
        assert [v["version"] for v in versions] == [2, 1]
    finally:
        app.dependency_overrides.clear()


def test_generate_404_when_project_missing(client):
    fake = _FakeSession()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        missing = str(uuid.uuid4())
        resp = client.post(f"/projects/{missing}/generate", json={})
        assert resp.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_generate_degraded_when_llm_fails(client, monkeypatch):
    _patch_compiler(monkeypatch, _artifact())

    import src.llm.together as together_mod

    def _boom(
        session, prompt, model_size="small", system=None, response_format=None, **kwargs
    ):
        raise RuntimeError("llm caído")

    monkeypatch.setattr(together_mod, "complete", _boom)
    fake = _FakeSession()
    fake.add(_settings())
    fake.flush()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        body = _new_project(client)
        project_id = body["project"]["id"]

        resp = client.post(f"/projects/{project_id}/generate", json={})
        # 0007: degradación a determinista NO es error — 200 con contrato.
        assert resp.status_code == 200
        result = resp.json()

        assert result["decision"]["status"] == "degraded"
        assert result["decision"]["reason"] == "llm_unavailable"
        assert result["decision"]["decision_source"] == "deterministic"
        assert result["decision"]["requires_user_acceptance"] is True
        # esqueleto determinista visible para el humano
        assert result["decision"]["snapshot"]["contenido"]
        # no se creó versión nueva (el pipeline se pausa)
        assert fake.get(Project, uuid.UUID(project_id)).current_version == 1
    finally:
        app.dependency_overrides.clear()


def test_generate_uses_template_and_format_spec(client, monkeypatch):
    _patch_compiler(monkeypatch, _artifact())
    captured = {}

    import src.llm.together as together_mod

    def _capture(
        session, prompt, model_size="small", system=None, response_format=None, **kwargs
    ):
        captured["prompt"] = prompt
        captured["system"] = system
        return FIXED_JSON

    monkeypatch.setattr(together_mod, "complete", _capture)
    fake = _FakeSession()
    fake.add(_settings())
    tpl = _template(
        name="Storyboard_Spec",
        content_type="video",
        phase="preproduccion",
        content={"escenas": []},
    )
    fake.add(tpl)
    fake.add(_spec())
    fake.flush()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        body = _new_project(
            client,
            brand_objective="ESCAPE_SOCIAL",
            artifact_type="video_corto",
            template_id=str(tpl.id),
        )
        project_id = body["project"]["id"]

        resp = client.post(f"/projects/{project_id}/generate", json={})
        assert resp.status_code == 200

        # el mensaje de usuario lleva SOLO los datos (topic/template/spec)
        prompt = captured["prompt"]
        assert "IA local vs nube" in prompt  # topic del proyecto
        assert "Storyboard_Spec" in prompt  # template
        assert "guion_vocal" in prompt  # campos derivados de structure
        # el system es el artefacto compilado (prompt-as-code, 0007)
        assert captured["system"] == _artifact().prompt_text
    finally:
        app.dependency_overrides.clear()


# ----------------------------------------------------------------------
# POST /projects/{id}/refine
# ----------------------------------------------------------------------


def test_refine_applies_deterministic_rules(client):
    fake = _FakeSession()
    fake.add(_spec())  # constraints.cta_unico -> cta requerido
    fake.flush()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        body = _new_project(
            client, brand_objective="ESCAPE_SOCIAL", artifact_type="video_corto"
        )
        project_id = body["project"]["id"]
        _approve(client, project_id, fake)  # snapshot sin cta

        resp = client.post(f"/projects/{project_id}/refine", json={})
        assert resp.status_code == 200
        result = resp.json()
        decision = result["decision"]
        report = decision["report"]

        # determinista por elección explícita (use_llm=False): ok, sin aceptación
        assert decision["status"] == "ok"
        assert decision["decision_source"] == "deterministic"
        assert decision["requires_user_acceptance"] is False
        assert report["llm_used"] is False
        assert any("cta" in w for w in report["warnings"])
        assert result["version"]["snapshot"]["cta"]  # cta añadido
        assert result["version"]["snapshot"]["guion_vocal"] == "Hola"
    finally:
        app.dependency_overrides.clear()


def test_refine_with_rag_context_hook(client):
    fake = _FakeSession()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        body = _new_project(client)
        project_id = body["project"]["id"]
        _approve(client, project_id, fake)

        resp = client.post(
            f"/projects/{project_id}/refine",
            json={"rag_context": "contexto de prueba"},
        )
        assert resp.status_code == 200
        report = resp.json()["decision"]["report"]
        assert report["rag_context_len"] == len("contexto de prueba")
        assert report["llm_used"] is False  # sin red
    finally:
        app.dependency_overrides.clear()


def test_refine_llm_degrades_gracefully(client, monkeypatch):
    _patch_compiler(monkeypatch, _artifact())

    import src.llm.together as together_mod

    def _boom(
        session, prompt, model_size="small", system=None, response_format=None, **kwargs
    ):
        raise RuntimeError("llm caído")

    monkeypatch.setattr(together_mod, "complete", _boom)
    fake = _FakeSession()
    fake.add(_settings())
    fake.flush()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        body = _new_project(client)
        project_id = body["project"]["id"]
        _approve(client, project_id, fake)  # v2

        resp = client.post(f"/projects/{project_id}/refine", json={"use_llm": True})
        assert resp.status_code == 200
        result = resp.json()
        decision = result["decision"]

        # 0007: degradación a determinista — requiere aceptación, no guarda versión
        assert decision["status"] == "degraded"
        assert decision["reason"] == "llm_unavailable"
        assert decision["requires_user_acceptance"] is True
        assert decision["report"]["llm_used"] is False
        assert any("LLM" in w for w in decision["report"]["warnings"])
        # el snapshot determinista se conserva (visible para el humano)
        assert decision["snapshot"]["guion_vocal"] == "Hola"
        # no se guardó versión nueva (el pipeline se pausa)
        assert fake.get(Project, uuid.UUID(project_id)).current_version == 2
    finally:
        app.dependency_overrides.clear()


def test_refine_saves_new_version(client):
    fake = _FakeSession()
    fake.add(_spec())
    fake.flush()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        body = _new_project(
            client, brand_objective="ESCAPE_SOCIAL", artifact_type="video_corto"
        )
        project_id = body["project"]["id"]
        _approve(client, project_id, fake)  # v2

        resp = client.post(f"/projects/{project_id}/refine", json={})
        assert resp.status_code == 200
        result = resp.json()

        assert result["project"]["current_version"] == 3
        assert result["version"]["version"] == 3
        # historial inmutable intacto: 1, 2, 3
        versions = client.get(f"/projects/{project_id}/versions").json()
        assert [v["version"] for v in versions] == [3, 2, 1]
    finally:
        app.dependency_overrides.clear()


# ----------------------------------------------------------------------
# POST /projects/{id}/produce
# ----------------------------------------------------------------------


def test_produce_409_when_not_approved(client):
    fake = _FakeSession()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        body = _new_project(client)
        project_id = body["project"]["id"]

        resp = client.post(f"/projects/{project_id}/produce")
        assert resp.status_code == 409
        assert "aprobado" in resp.json()["detail"]
    finally:
        app.dependency_overrides.clear()


def test_produce_runs_tool_chain(client):
    fake = _FakeSession()
    fake.add(_adapter(name="elevenlabs"))
    fake.add(_template())
    fake.add(_spec())
    fake.flush()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        body = _new_project(
            client, brand_objective="ESCAPE_SOCIAL", artifact_type="video_corto"
        )
        project_id = body["project"]["id"]
        _approve(client, project_id, fake)

        result = _produce_and_approve_brief(client, project_id, fake)

        assert result["artifact_id"]
        assert result["manifest_version"] == 1
        assert result["result"]["success"] is True
        assert result["project"]["status"] == "en_produccion"

        # estado persistido en la sesión fake
        project = fake.get(Project, uuid.UUID(project_id))
        assert project.status == "en_produccion"
        assert len(fake._rows("ContentBrief")) == 1
        assert len(fake._rows("ContentArtifact")) == 1
        assert len(fake._rows("ProductionManifest")) == 1
        assert len(fake._rows("AssetJob")) == 1
    finally:
        app.dependency_overrides.clear()


def test_produce_requires_brief_approval_before_tool_chain(client):
    """Gobernanza 0009/0010: el gate NUNCA auto-aprueba un proyecto
    (nueva_solucion + complete siempre exigen revisión humana). El primer
    /produce devuelve 409 con el brief_id; la cadena NO corre."""
    fake = _FakeSession()
    fake.add(_adapter(name="elevenlabs"))
    fake.add(_template())
    fake.add(_spec())
    fake.flush()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        body = _new_project(
            client, brand_objective="ESCAPE_SOCIAL", artifact_type="video_corto"
        )
        project_id = body["project"]["id"]
        _approve(client, project_id, fake)

        resp = client.post(f"/projects/{project_id}/produce")
        assert resp.status_code == 409
        detail = resp.json()["detail"]
        assert "brief" in detail
        assert "approve" in detail

        # el brief compañero existe pero NO se produjo nada
        assert len(fake._rows("ContentBrief")) == 1
        assert len(fake._rows("ContentArtifact")) == 0
        assert len(fake._rows("AssetJob")) == 0
        brief = next(iter(fake._rows("ContentBrief").values()))
        assert brief.status == "revision"
        assert brief.requires_user_acceptance is True
        # el segmento/riesgo reales del proyecto, no S1/bajo hardcodeados
        assert brief.segment_client == "S1"
        assert brief.risk_level == "bajo"
    finally:
        app.dependency_overrides.clear()


def test_produce_409_on_orchestration_error(client):
    fake = _FakeSession()  # sin format_spec -> OrchestrationError
    app.dependency_overrides[get_session] = lambda: fake
    try:
        body = _new_project(
            client, brand_objective="ESCAPE_SOCIAL", artifact_type="video_corto"
        )
        project_id = body["project"]["id"]
        _approve(client, project_id, fake)

        resp = client.post(f"/projects/{project_id}/produce")
        assert resp.status_code == 409
        assert "brief" in resp.json()["detail"]  # pausa por gobernanza, no por spec
    finally:
        app.dependency_overrides.clear()


def test_produce_commits(client):
    fake = _FakeSession()
    fake.add(_adapter(name="elevenlabs"))
    fake.add(_template())
    fake.add(_spec())
    fake.flush()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        body = _new_project(
            client, brand_objective="ESCAPE_SOCIAL", artifact_type="video_corto"
        )
        project_id = body["project"]["id"]
        _approve(client, project_id, fake)

        before = fake.commits
        result = _produce_and_approve_brief(client, project_id, fake)
        assert fake.commits > before  # el endpoint commitea

        # estado persistido
        project = fake.get(Project, uuid.UUID(project_id))
        assert project.status == "en_produccion"
        assert project.storage_path is not None
        artifact = next(iter(fake._rows("ContentArtifact").values()))
        assert artifact.project_id == uuid.UUID(project_id)
        assert artifact.brief_id is not None
    finally:
        app.dependency_overrides.clear()


# ----------------------------------------------------------------------
# GET /projects/{id}/postproduction
# ----------------------------------------------------------------------


def test_postproduction_recommendations(client):
    fake = _FakeSession()
    fake.add(
        _template(
            name="Voice_FX_Chain_Preset",
            content_type="audio",
            phase="postproduccion",
            content={"hpf": 80},
        )
    )
    fake.add(
        _template(
            name="Render_Export_Preset",
            content_type="video",
            phase="postproduccion",
            content={"codec": "H.264"},
        )
    )
    fake.add(
        _template(
            name="Image_Composite_FX_Spec",
            content_type="grafico",
            phase="postproduccion",
            content={"grano": 0.1},
        )
    )
    fake.flush()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        body = _new_project(client)
        project_id = body["project"]["id"]
        _approve(client, project_id, fake)

        resp = client.get(f"/projects/{project_id}/postproduction")
        assert resp.status_code == 200
        result = resp.json()

        assert result["project_id"] == project_id
        assert set(result["recomendaciones"].keys()) == {"audio", "video", "grafico"}
        assert (
            result["recomendaciones"]["audio"][0]["template"] == "Voice_FX_Chain_Preset"
        )
        assert (
            result["recomendaciones"]["video"][0]["template"] == "Render_Export_Preset"
        )
        assert (
            result["recomendaciones"]["grafico"][0]["template"]
            == "Image_Composite_FX_Spec"
        )
        # params = content del template fusionado con el snapshot
        assert result["recomendaciones"]["video"][0]["params"]["codec"] == "H.264"
        assert result["recomendaciones"]["video"][0]["params"]["guion_vocal"] == "Hola"
    finally:
        app.dependency_overrides.clear()


def test_postproduction_empty_when_no_templates(client):
    fake = _FakeSession()
    app.dependency_overrides[get_session] = lambda: fake
    try:
        body = _new_project(client)
        project_id = body["project"]["id"]
        _approve(client, project_id, fake)

        resp = client.get(f"/projects/{project_id}/postproduction")
        assert resp.status_code == 200
        result = resp.json()
        assert result["recomendaciones"] == {}
    finally:
        app.dependency_overrides.clear()
