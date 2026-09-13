"""Tests de la Fase 2 — capa de herramientas (src/tools/) — sin DB ni red.

Usa un _FakeSession en memoria (mismo patrón que test_seed_tools.py) y
mocks del cliente MCP. No requiere base de datos ni red viva.
"""

import urllib.error
import urllib.request
import uuid

import pytest

from src.db.models import (
    BrandObjective,
    ContentArtifact,
    FormatSpec,
    ProductionTemplate,
    ToolAdapter,
)
from src.tools import (
    DirectToolExecutor,
    McpClient,
    McpClientError,
    OrchestrationError,
    Orchestrator,
    build_command,
    create_manifest,
    latest_version,
    list_adapters,
    list_manifests,
    list_templates,
    load_manifest,
    merge_deep,
    render,
    render_for_tool,
    resolve_phases,
    resolve_template,
    resolve_tool_chain,
    validate_tool_chain,
)


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def first(self):
        return self._rows[0] if self._rows else None

    def all(self):
        return self._rows


class _FakeSession:
    """Sesión mínima en memoria para src/tools: adapters, specs, templates,
    manifests, artifacts y jobs."""

    def __init__(self):
        self.adapters = {}  # name -> ToolAdapter
        self.specs = {}  # (brand_value, artifact_type) -> FormatSpec
        self.templates = {}  # name -> ProductionTemplate
        self.manifests = []  # list[ProductionManifest]
        self.artifacts = {}  # id -> ContentArtifact
        self.jobs = []  # list[AssetJob]
        self.added = []

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
                return _Result([self.adapters.get(pairs["name"])])
            rows = list(self.adapters.values())
            if "is_active" in pairs:
                rows = [r for r in rows if r.is_active == pairs["is_active"]]
            return _Result(rows)

        if name == "FormatSpec":
            if "brand_objective" in pairs and "artifact_type" in pairs:
                return _Result(
                    [self.specs.get((pairs["brand_objective"], pairs["artifact_type"]))]
                )
            rows = list(self.specs.values())
            if "artifact_type" in pairs:
                rows = [r for r in rows if r.artifact_type == pairs["artifact_type"]]
            if "id" in pairs:
                rows = [r for r in rows if r.id == pairs["id"]]
            if "is_active" in pairs:
                rows = [r for r in rows if r.is_active == pairs["is_active"]]
            return _Result(rows)

        if name == "ProductionTemplate":
            if "name" in pairs:
                t = self.templates.get(pairs["name"])
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
            return _Result(rows)

        return _Result([])

    def add(self, obj):
        self.added.append(obj)

    def flush(self):
        for obj in self.added:
            if getattr(obj, "id", None) is None:
                obj.id = uuid.uuid4()
            cls = obj.__class__.__name__
            if cls == "ToolAdapter":
                self.adapters[obj.name] = obj
            elif cls == "FormatSpec":
                brand = getattr(obj.brand_objective, "value", obj.brand_objective)
                self.specs[(brand, obj.artifact_type)] = obj
            elif cls == "ProductionTemplate":
                self.templates[obj.name] = obj
            elif cls == "ProductionManifest":
                if obj not in self.manifests:
                    self.manifests.append(obj)
            elif cls == "ContentArtifact":
                self.artifacts[obj.id] = obj
            elif cls == "AssetJob" and obj not in self.jobs:
                self.jobs.append(obj)
        self.added = []

    def commit(self):
        self.flush()


# ----------------------------------------------------------------------
# registry
# ----------------------------------------------------------------------


def _make_spec(
    artifact_type="video_corto",
    brand=BrandObjective.ESCAPE_SOCIAL,
    tool_chain=None,
    phases=None,
    is_active=True,
):
    return FormatSpec(
        brand_objective=brand,
        artifact_type=artifact_type,
        structure={},
        constraints={},
        derivation_rules=[],
        visual_requirements={},
        qa_checks=[],
        tool_chain=tool_chain or ["producer", "elevenlabs", "comfyui", "resolve"],
        phases=phases or {},
        is_active=is_active,
    )


def test_resolve_phases_returns_spec_phases():
    fake = _FakeSession()
    phases = {
        "preproduccion": {"steps": ["desglose"], "templates": ["Storyboard_Spec"]},
        "produccion": {"steps": ["voz"], "templates": ["TTS_Voice_Preset"]},
        "postproduccion": {"steps": ["render"], "templates": ["Render_Export_Preset"]},
    }
    fake.add(_make_spec(phases=phases))
    fake.flush()
    assert resolve_phases(fake, BrandObjective.ESCAPE_SOCIAL, "video_corto") == phases


def test_resolve_phases_returns_empty_when_spec_missing():
    fake = _FakeSession()
    assert resolve_phases(fake, BrandObjective.ESCAPE_SOCIAL, "no_existe") == {}


def test_resolve_tool_chain_returns_chain():
    fake = _FakeSession()
    chain = ["producer", "elevenlabs", "comfyui", "resolve"]
    fake.add(_make_spec(tool_chain=chain))
    fake.flush()
    assert (
        resolve_tool_chain(fake, BrandObjective.ESCAPE_SOCIAL, "video_corto") == chain
    )


def test_resolve_template_by_name():
    fake = _FakeSession()
    tpl = ProductionTemplate(
        name="TTS_Voice_Preset",
        content_type="audio",
        phase="produccion",
        template_format="json",
        content={"stability": 0.5},
        version="1.0",
        is_active=True,
    )
    fake.add(tpl)
    fake.flush()
    assert resolve_template(fake, "TTS_Voice_Preset") is tpl
    assert resolve_template(fake, "no_existe") is None


def test_list_templates_filters():
    fake = _FakeSession()
    t1 = ProductionTemplate(
        name="A",
        content_type="audio",
        phase="produccion",
        template_format="json",
        content={},
        is_active=True,
    )
    t2 = ProductionTemplate(
        name="B",
        content_type="video",
        phase="produccion",
        template_format="json",
        content={},
        is_active=True,
    )
    fake.add(t1)
    fake.add(t2)
    fake.flush()
    assert list_templates(fake) == [t1, t2]
    assert list_templates(fake, content_type="audio") == [t1]
    assert list_templates(fake, phase="produccion") == [t1, t2]
    assert list_templates(fake, content_type="video", phase="produccion") == [t2]


def test_list_adapters_filters_active():
    fake = _FakeSession()
    a1 = ToolAdapter(
        name="inkscape",
        mcp_server_name="inkscape",
        execution_mode="local",
        is_active=True,
    )
    a2 = ToolAdapter(
        name="krita", mcp_server_name="krita", execution_mode="local", is_active=False
    )
    fake.add(a1)
    fake.add(a2)
    fake.flush()
    assert list_adapters(fake) == [a1]
    assert list_adapters(fake, is_active=None) == [a1, a2]


def test_validate_tool_chain_ok_with_valid_data():
    fake = _FakeSession()
    fake.add(
        ToolAdapter(
            name="elevenlabs",
            mcp_server_name="elevenlabs",
            execution_mode="local",
            is_active=True,
        )
    )
    fake.add(
        ToolAdapter(
            name="comfyui",
            mcp_server_name="comfyui",
            execution_mode="local",
            is_active=True,
        )
    )
    fake.add(
        ProductionTemplate(
            name="TTS_Voice_Preset",
            content_type="audio",
            phase="produccion",
            template_format="json",
            content={},
            is_active=True,
        )
    )
    fake.add(
        _make_spec(
            tool_chain=["producer", "elevenlabs", "comfyui"],
            phases={"produccion": {"steps": [], "templates": ["TTS_Voice_Preset"]}},
        )
    )
    fake.flush()
    assert validate_tool_chain(fake) == []


def test_validate_tool_chain_reports_missing_adapter_and_template():
    fake = _FakeSession()
    fake.add(
        ToolAdapter(
            name="elevenlabs",
            mcp_server_name="elevenlabs",
            execution_mode="local",
            is_active=True,
        )
    )
    fake.add(
        _make_spec(
            tool_chain=["producer", "elevenlabs", "comfyui"],
            phases={"produccion": {"steps": [], "templates": ["TTS_Voice_Preset"]}},
        )
    )
    fake.flush()
    errors = validate_tool_chain(fake)
    assert len(errors) == 2
    assert any("comfyui" in e for e in errors)
    assert any("TTS_Voice_Preset" in e for e in errors)


# ----------------------------------------------------------------------
# manifest
# ----------------------------------------------------------------------


def test_create_manifest_versions_1_2_3():
    fake = _FakeSession()
    aid = uuid.uuid4()
    m1 = create_manifest(fake, aid, {"project": {"v": 1}})
    m2 = create_manifest(fake, aid, {"project": {"v": 2}})
    m3 = create_manifest(fake, aid, {"project": {"v": 3}})
    assert (m1.version, m2.version, m3.version) == (1, 2, 3)
    assert latest_version(fake, aid) == 3


def test_load_manifest_latest_without_version():
    fake = _FakeSession()
    aid = uuid.uuid4()
    create_manifest(fake, aid, {"v": 1})
    create_manifest(fake, aid, {"v": 2})
    loaded = load_manifest(fake, aid)
    assert loaded.version == 2
    assert loaded.manifest == {"v": 2}
    assert load_manifest(fake, aid, version=1).manifest == {"v": 1}
    assert load_manifest(fake, aid, version=99) is None


def test_list_manifests_sorted_desc():
    fake = _FakeSession()
    aid = uuid.uuid4()
    create_manifest(fake, aid, {"v": 1})
    create_manifest(fake, aid, {"v": 2})
    create_manifest(fake, aid, {"v": 3})
    versions = [m.version for m in list_manifests(fake, aid)]
    assert versions == [3, 2, 1]


def test_latest_version_zero_without_manifests():
    fake = _FakeSession()
    assert latest_version(fake, uuid.uuid4()) == 0


# ----------------------------------------------------------------------
# template_engine
# ----------------------------------------------------------------------


def test_merge_deep_merges_nested_dicts_and_replaces_lists():
    base = {"a": {"b": 1, "c": [1, 2]}, "d": [1]}
    override = {"a": {"c": [3], "e": 2}, "d": [9]}
    result = merge_deep(base, override)
    assert result == {"a": {"b": 1, "c": [3], "e": 2}, "d": [9]}


def test_render_combines_template_content_and_variables():
    tpl = ProductionTemplate(
        name="TTS_Voice_Preset",
        content_type="audio",
        phase="produccion",
        template_format="json",
        content={
            "stability": 0.5,
            "voice_id": "string",
            "audio": {"sample_rate": 48000, "nested": {"a": 1}},
        },
        version="2.0",
        is_active=True,
    )
    manifest = {
        "audio": {"voice_id": "v123", "nested": {"b": 2}},
        "project": {"artifact_type": "video_corto"},
    }
    contract = render(tpl, manifest, variables={"stability": 0.9})
    assert contract["template_name"] == "TTS_Voice_Preset"
    assert contract["template_version"] == "2.0"
    params = contract["params"]
    assert params["stability"] == 0.9  # variables extra ganan
    assert params["voice_id"] == "string"  # sin override en esa ruta
    # merge profundo en la misma ruta: template + manifiesto + variables
    assert params["audio"] == {
        "sample_rate": 48000,
        "voice_id": "v123",
        "nested": {"a": 1, "b": 2},
    }
    assert params["project"]["artifact_type"] == "video_corto"


def test_render_for_tool_filters_by_tool_subdict():
    tpl = ProductionTemplate(
        name="TTS_Voice_Preset",
        content_type="audio",
        phase="produccion",
        template_format="json",
        content={
            "stability": 0.5,
            "by_tool": {"elevenlabs": {"voice_id": "v1", "stability": 0.7}},
        },
        version="1.0",
        is_active=True,
    )
    contract = render_for_tool(tpl, {}, "elevenlabs")
    assert contract["params"] == {"voice_id": "v1", "stability": 0.7}
    # sin by_tool para otro tool → devuelve todo
    contract2 = render_for_tool(tpl, {}, "inkscape")
    assert contract2["params"]["stability"] == 0.5
    assert "by_tool" in contract2["params"]


def test_render_for_tool_uses_tool_named_key():
    tpl = ProductionTemplate(
        name="Inkscape_Layer_Template",
        content_type="grafico",
        phase="produccion",
        template_format="svg",
        content={"inkscape": {"id": "#title_text"}, "otra": 1},
        version="1.0",
        is_active=True,
    )
    contract = render_for_tool(tpl, {}, "inkscape")
    assert contract["params"] == {"id": "#title_text"}


# ----------------------------------------------------------------------
# mcp_client
# ----------------------------------------------------------------------


def _adapter(name="elevenlabs"):
    return ToolAdapter(
        name=name, mcp_server_name=name, execution_mode="local", is_active=True
    )


def test_build_command_inkscape():
    cmd = build_command("inkscape", {"id": "#title_text", "output": "out.svg"})
    assert (
        cmd == 'inkscape --actions="select-by-id:#title_text; export-filename:out.svg"'
    )


def test_build_command_comfyui():
    cmd = build_command("comfyui", {"workflow": "wf.json", "seed": 7})
    assert cmd == "comfy run --workflow wf.json --seed 7"


def test_build_command_resolve():
    cmd = build_command("resolve", {"project": "MiProyecto"})
    assert cmd == "resolve.LoadProject('MiProyecto')"


def test_build_command_reaper():
    cmd = build_command("reaper", {"input": "voz.wav", "output": "mix.wav"})
    assert cmd == "reaper -i voz.wav mix.wav"


def test_build_command_default():
    arguments = {"foo": "bar"}
    cmd = build_command("krita", arguments)
    assert cmd == f"krita {arguments}"


def test_call_tool_direct_mode_returns_ok_and_output_path():
    client = McpClient(_adapter("elevenlabs"))
    result = client.call_tool(
        "elevenlabs", {"params": {"project": {"artifact_id": "abc"}}}
    )
    assert result["ok"] is True
    assert result["tool"] == "elevenlabs"
    assert result["command"].startswith("elevenlabs ")
    assert result["output_path"].startswith("data/artifacts/abc/elevenlabs_")
    assert result["output_path"].endswith(".wav")


def test_direct_tool_executor_output_path():
    adapter = _adapter("resolve")
    executor = DirectToolExecutor(adapter)
    result = executor.execute("resolve", {"params": {"project": {"artifact_id": "x1"}}})
    assert result["ok"] is True
    assert result["output_path"].startswith("data/artifacts/x1/resolve_")
    assert result["output_path"].endswith(".mp4")


def test_call_tool_http_success(monkeypatch):
    client = McpClient(_adapter("comfyui"), base_url="http://localhost:9999")
    captured = {}

    class _FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"ok": true, "output_path": "x.png"}'

    def _fake_urlopen(request, **kwargs):
        captured["url"] = request.full_url
        captured["data"] = request.data
        return _FakeResp()

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    result = client.call_tool("comfyui", {"workflow": "wf.json"})
    assert result == {"ok": True, "output_path": "x.png"}
    assert captured["url"] == "http://localhost:9999/tools/comfyui"
    assert b"workflow" in captured["data"]


def test_call_tool_http_error_raises_mcp_client_error(monkeypatch):
    client = McpClient(_adapter("comfyui"), base_url="http://localhost:9999")

    def _boom(*args, **kwargs):
        raise urllib.error.URLError("conexión rechazada")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    with pytest.raises(McpClientError, match="conexión rechazada"):
        client.call_tool("comfyui", {"workflow": "wf.json"})


# ----------------------------------------------------------------------
# orchestrator
# ----------------------------------------------------------------------


class _FakeMcpClient:
    def __init__(self, adapter, fail_on=None):
        self.adapter = adapter
        self.fail_on = fail_on
        self.calls = []

    def call_tool(self, tool_name, arguments):
        self.calls.append((tool_name, arguments))
        if self.fail_on == tool_name:
            raise McpClientError(f"fallo simulado en {tool_name}")
        return {
            "ok": True,
            "output_path": f"data/artifacts/{tool_name}.mp4",
            "tool": tool_name,
        }


def _build_scenario(fail_on=None):
    """Artefacto video_corto con spec, templates, adapters y manifiesto."""
    fake = _FakeSession()
    aid = uuid.uuid4()
    artifact = ContentArtifact(
        id=aid,
        brief_id=uuid.uuid4(),
        artifact_type="video_corto",
        channel="instagram",
        status="borrador",
    )
    fake.add(artifact)
    fake.flush()

    adapters = {
        "elevenlabs": _adapter("elevenlabs"),
        "comfyui": _adapter("comfyui"),
        "resolve": _adapter("resolve"),
    }
    for a in adapters.values():
        fake.add(a)
    fake.flush()

    templates = {
        "Storyboard_Spec": ProductionTemplate(
            name="Storyboard_Spec",
            content_type="video",
            phase="preproduccion",
            template_format="json",
            content={"escenas": []},
            is_active=True,
        ),
        "TTS_Voice_Preset": ProductionTemplate(
            name="TTS_Voice_Preset",
            content_type="audio",
            phase="produccion",
            template_format="json",
            content={"stability": 0.5},
            is_active=True,
        ),
        "Render_Export_Preset": ProductionTemplate(
            name="Render_Export_Preset",
            content_type="video",
            phase="postproduccion",
            template_format="json",
            content={"codec_video": ["H.264"]},
            is_active=True,
        ),
    }
    for t in templates.values():
        fake.add(t)
    fake.flush()

    phases = {
        "preproduccion": {"steps": ["desglose"], "templates": ["Storyboard_Spec"]},
        "produccion": {"steps": ["voz"], "templates": ["TTS_Voice_Preset"]},
        "postproduccion": {"steps": ["render"], "templates": ["Render_Export_Preset"]},
    }
    spec = _make_spec(
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

    factory = lambda adapter: _FakeMcpClient(adapter, fail_on=fail_on)
    orch = Orchestrator(fake, mcp_client_factory=factory)
    return fake, aid, orch


def test_run_tool_chain_success():
    fake, aid, orch = _build_scenario()
    result = orch.run_tool_chain(aid)

    assert result["success"] is True
    assert result["artifact_id"] == str(aid)
    assert result["manifest_version"] == 1

    jobs = result["jobs"]
    assert [j["phase"] for j in jobs] == [
        "preproduccion",
        "produccion",
        "postproduccion",
    ]
    assert [j["template"] for j in jobs] == [
        "Storyboard_Spec",
        "TTS_Voice_Preset",
        "Render_Export_Preset",
    ]
    assert [j["tool"] for j in jobs] == ["elevenlabs", "comfyui", "resolve"]
    assert all(j["status"] == "done" for j in jobs)
    assert [j["output_path"] for j in jobs] == [
        "data/artifacts/elevenlabs.mp4",
        "data/artifacts/comfyui.mp4",
        "data/artifacts/resolve.mp4",
    ]

    # jobs persistidos con trazabilidad
    assert len(fake.jobs) == 3
    assert [j.sequence_order for j in fake.jobs] == [1, 2, 3]
    assert all(j.status == "done" for j in fake.jobs)
    assert all(j.manifest_version == 1 for j in fake.jobs)
    assert [j.phase for j in fake.jobs] == [
        "preproduccion",
        "produccion",
        "postproduccion",
    ]

    # artifact actualizado
    artifact = fake.artifacts[aid]
    assert artifact.status == "listo"
    assert artifact.storage_path == "data/artifacts/resolve.mp4"


def test_run_tool_chain_stops_on_failure():
    fake, aid, orch = _build_scenario(fail_on="comfyui")
    result = orch.run_tool_chain(aid)

    assert result["success"] is False
    jobs = result["jobs"]
    assert len(jobs) == 2
    assert jobs[0]["status"] == "done"
    assert jobs[0]["tool"] == "elevenlabs"
    assert jobs[1]["status"] == "failed"
    assert jobs[1]["tool"] == "comfyui"
    assert "fallo simulado" in jobs[1]["error"]

    # la cadena se detuvo: no se creó el job de resolve
    assert len(fake.jobs) == 2
    artifact = fake.artifacts[aid]
    assert artifact.status == "borrador"
    assert artifact.storage_path is None


def test_run_tool_chain_without_manifest_raises():
    fake, aid, orch = _build_scenario()
    fake.manifests.clear()
    with pytest.raises(OrchestrationError, match="manifiesto"):
        orch.run_tool_chain(aid)


def test_run_tool_chain_without_format_spec_raises():
    fake = _FakeSession()
    aid = uuid.uuid4()
    artifact = ContentArtifact(
        id=aid,
        brief_id=uuid.uuid4(),
        artifact_type="video_corto",
        channel="instagram",
        status="borrador",
    )
    fake.add(artifact)
    fake.flush()
    create_manifest(fake, aid, {"project": {}})
    orch = Orchestrator(fake)
    with pytest.raises(OrchestrationError, match="format_spec"):
        orch.run_tool_chain(aid)


def test_run_tool_chain_unknown_artifact_raises():
    fake = _FakeSession()
    orch = Orchestrator(fake)
    with pytest.raises(OrchestrationError, match="no existe"):
        orch.run_tool_chain(uuid.uuid4())


def test_run_tool_chain_falls_back_to_spec_by_artifact_type():
    fake, aid, orch = _build_scenario()
    artifact = fake.artifacts[aid]
    artifact.format_spec_id = None  # sin FK directa → fallback por artifact_type
    result = orch.run_tool_chain(aid)
    assert result["success"] is True
    assert len(result["jobs"]) == 3


def test_create_job_helper():
    fake = _FakeSession()
    aid = uuid.uuid4()
    adapter = _adapter("inkscape")
    fake.add(adapter)
    fake.flush()
    orch = Orchestrator(fake)
    job = orch.create_job(aid, adapter.id, 1, phase="produccion")
    assert job.status == "pending"
    assert job.sequence_order == 1
    assert job.phase == "produccion"
    assert job in fake.jobs
