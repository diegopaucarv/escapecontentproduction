"""Cliente MCP genérico — modo directo primero (0005).

Modo directo (sin MCP real): construye un comando CLI determinista a
partir del adapter y devuelve un resultado simulado. Si se setea
`base_url`, hace un POST HTTP a `{base_url}/tools/{tool_name}` con
`{"arguments": arguments}` usando solo la stdlib (urllib).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import datetime

from src.db.models import ToolAdapter

# Extensión de salida simulada por herramienta (modo directo).
_EXTENSIONS = {
    "inkscape": "svg",
    "krita": "png",
    "comfyui": "png",
    "elevenlabs": "wav",
    "reaper": "wav",
    "resolve": "mp4",
    "veo3": "mp4",
    "canva": "png",
    "gdrive": "bin",
    "filesystem": "bin",
}


class McpClientError(Exception):
    """Error de comunicación/ejecución con una herramienta MCP."""


def _arg(arguments: dict, key: str, default):
    """Lee un argumento del contrato, plano o anidado en `params`."""
    if isinstance(arguments, dict):
        if key in arguments:
            return arguments[key]
        params = arguments.get("params")
        if isinstance(params, dict) and key in params:
            return params[key]
    return default


def build_command(tool_name: str, arguments: dict) -> str:
    """Genera el comando CLI determinista a partir del adapter y los
    arguments. No necesita ser perfecto — es el modo directo/simulado."""
    tool = (tool_name or "").lower()
    if tool == "inkscape":
        obj_id = _arg(arguments, "id", "title_text")
        output = _arg(arguments, "output", "output.svg")
        return f'inkscape --actions="select-by-id:{obj_id}; export-filename:{output}"'
    if tool in ("ffmpeg", "reaper"):
        input_ref = _arg(arguments, "input", "input.wav")
        output = _arg(arguments, "output", "output.wav")
        return f"{tool} -i {input_ref} {output}"
    if tool == "comfyui":
        workflow = _arg(arguments, "workflow", "workflow.json")
        seed = _arg(arguments, "seed", 42)
        return f"comfy run --workflow {workflow} --seed {seed}"
    if tool == "resolve":
        project = _arg(arguments, "project", "Project")
        return f"resolve.LoadProject('{project}')"
    return f"{tool} {arguments}"


def _extract_artifact_id(arguments: dict) -> str:
    """Busca el artifact_id en el contrato (plano o en params.project)."""
    if not isinstance(arguments, dict):
        return "unknown"
    if "artifact_id" in arguments:
        return str(arguments["artifact_id"])
    params = arguments.get("params")
    if isinstance(params, dict):
        if "artifact_id" in params:
            return str(params["artifact_id"])
        project = params.get("project")
        if isinstance(project, dict) and "artifact_id" in project:
            return str(project["artifact_id"])
    return "unknown"


class DirectToolExecutor:
    """Ejecuta build_command y devuelve el resultado simulado (modo sin MCP)."""

    def __init__(self, adapter: ToolAdapter, base_dir: str = "data/artifacts"):
        self.adapter = adapter
        self.base_dir = base_dir

    def execute(self, tool_name: str, arguments: dict) -> dict:
        command = build_command(tool_name, arguments)
        output_path = self._output_path(tool_name, arguments)
        return {
            "ok": True,
            "output_path": output_path,
            "command": command,
            "tool": tool_name,
        }

    def _output_path(self, tool_name: str, arguments: dict) -> str:
        artifact_id = _extract_artifact_id(arguments)
        ext = _EXTENSIONS.get((tool_name or "").lower(), "bin")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.base_dir}/{artifact_id}/{tool_name}_{timestamp}.{ext}"


class McpClient:
    """Cliente MCP genérico: modo directo (simulado) o HTTP a un gateway."""

    def __init__(
        self,
        adapter: ToolAdapter,
        base_url: str | None = None,
        timeout: float = 60.0,
    ):
        self.adapter = adapter
        self.base_url = base_url
        self.timeout = timeout

    def build_command(self, tool_name: str, arguments: dict) -> str:
        return build_command(tool_name, arguments)

    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        if self.base_url:
            return self._call_http(tool_name, arguments)
        return self._call_direct(tool_name, arguments)

    def _call_direct(self, tool_name: str, arguments: dict) -> dict:
        executor = DirectToolExecutor(self.adapter)
        return executor.execute(tool_name, arguments)

    def _call_http(self, tool_name: str, arguments: dict) -> dict:
        url = f"{self.base_url.rstrip('/')}/tools/{tool_name}"
        payload = json.dumps({"arguments": arguments}).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise McpClientError(f"HTTP {exc.code} desde {url}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise McpClientError(f"error de red hacia {url}: {exc.reason}") from exc
        except TimeoutError as exc:
            raise McpClientError(f"timeout hacia {url}") from exc
        try:
            return json.loads(body)
        except ValueError as exc:
            raise McpClientError(f"respuesta no-JSON desde {url}: {exc}") from exc
