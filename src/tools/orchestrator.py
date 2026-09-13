"""Ejecución de AssetJobs + trazabilidad (0005/0006).

El orquestador NUNCA decide qué producir: ejecuta la cadena mecánica que
`format_spec.phases` + `tool_chain` declaran, con los parámetros del
manifiesto (regla del diseño §4). Cada template de cada fase se materializa
en un AssetJob trazable (pending → running → done/failed).

El paso LLM de generación (`producer`, Producer-Critic) se ejecuta ANTES
que la cadena mecánica y NO es una herramienta MCP: el orquestador lo
filtra explícitamente (LLM_STEP) y solo corre los adapters mecánicos.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select

from src.db.models import AssetJob, ContentArtifact, FormatSpec, ToolAdapter
from src.tools.manifest import load_manifest
from src.tools.mcp_client import McpClient
from src.tools.registry import resolve_template
from src.tools.template_engine import render_for_tool

# Fases del marco universal (§1) en orden de ejecución.
PHASE_ORDER = ("preproduccion", "produccion", "postproduccion")

# Paso LLM de generación (Producer-Critic) que precede a la cadena
# mecánica. NO es una herramienta MCP: el orquestador lo excluye de la
# cadena y solo ejecuta los adapters mecánicos.
LLM_STEP = "producer"


class OrchestrationError(Exception):
    """Error de orquestación (artefacto/spec/manifiesto ausente, config inválida)."""


class Orchestrator:
    """Ejecuta la cadena de herramientas de un ContentArtifact."""

    def __init__(self, session, mcp_client_factory=None):
        self.session = session
        # Callable (adapter) -> McpClient; inyectable para mockear en tests.
        self.mcp_client_factory = mcp_client_factory or (
            lambda adapter: McpClient(adapter)
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def create_job(
        self,
        artifact_id,
        tool_adapter_id,
        sequence_order: int,
        phase: str | None = None,
        template_id=None,
        manifest_version: int | None = None,
    ) -> AssetJob:
        """Crea un AssetJob en status pending."""
        job = AssetJob(
            artifact_id=artifact_id,
            tool_adapter_id=tool_adapter_id,
            sequence_order=sequence_order,
            status="pending",
            phase=phase,
            template_id=template_id,
            manifest_version=manifest_version,
        )
        self.session.add(job)
        self.session.flush()
        return job

    def _load_artifact(self, artifact_id) -> ContentArtifact | None:
        return (
            self.session.execute(
                select(ContentArtifact).where(ContentArtifact.id == artifact_id)
            )
            .scalars()
            .first()
        )

    def _resolve_format_spec(self, artifact) -> FormatSpec | None:
        """Resuelve la FormatSpec del artefacto.

        ContentArtifact no tiene brand_objective directo: se usa
        `format_spec_id` si está asignada; si no, se busca la spec por
        artifact_type con cualquier marca (fallback).
        """
        if artifact.format_spec_id is not None:
            return (
                self.session.execute(
                    select(FormatSpec).where(FormatSpec.id == artifact.format_spec_id)
                )
                .scalars()
                .first()
            )
        specs = (
            self.session.execute(
                select(FormatSpec).where(
                    FormatSpec.artifact_type == artifact.artifact_type
                )
            )
            .scalars()
            .all()
        )
        return specs[0] if specs else None

    def _resolve_adapter(self, name) -> ToolAdapter | None:
        return (
            self.session.execute(select(ToolAdapter).where(ToolAdapter.name == name))
            .scalars()
            .first()
        )

    @staticmethod
    def _job_summary(job: AssetJob, phase, template, tool) -> dict:
        return {
            "id": str(job.id),
            "phase": phase,
            "template": template,
            "tool": tool,
            "status": job.status,
            "output_path": job.output_path,
            "error": job.error,
        }

    @staticmethod
    def _summary(artifact_id, manifest_version, jobs, success: bool) -> dict:
        return {
            "artifact_id": str(artifact_id),
            "manifest_version": manifest_version,
            "jobs": jobs,
            "success": success,
        }

    # ------------------------------------------------------------------
    # Ejecución
    # ------------------------------------------------------------------

    def run_tool_chain(
        self, artifact_id: uuid.UUID, manifest_version: int | None = None
    ) -> dict:
        """Ejecuta la cadena de herramientas del artefacto.

        Crea un AssetJob por template de cada fase (en orden), lo ejecuta
        vía el cliente MCP y registra el output. Si un job falla, detiene
        la cadena y devuelve success=False.
        """
        artifact = self._load_artifact(artifact_id)
        if artifact is None:
            raise OrchestrationError(f"artefacto {artifact_id} no existe")

        spec = self._resolve_format_spec(artifact)
        if spec is None:
            raise OrchestrationError("sin format_spec asignada")

        manifest_row = load_manifest(self.session, artifact_id, manifest_version)
        if manifest_row is None:
            raise OrchestrationError("sin manifiesto")

        manifest_data = manifest_row.manifest or {}
        phases = spec.phases or {}
        # El paso 'producer' (generación LLM vía prompt-as-code, artefacto
        # producer_draft) lo ejecuta el grafo Producer-Critic ANTES de la
        # cadena; el orquestador solo corre las herramientas MCP mecánicas.
        chain = [t for t in (spec.tool_chain or []) if t != LLM_STEP]

        jobs: list[dict] = []
        sequence = 0
        tool_index = 0
        last_output = None

        for phase in PHASE_ORDER:
            phase_spec = phases.get(phase)
            if not phase_spec:
                continue
            for template_name in phase_spec.get("templates", []) or []:
                sequence += 1
                template = resolve_template(self.session, template_name)
                if template is None:
                    raise OrchestrationError(
                        f"template inexistente '{template_name}' en fase {phase}"
                    )
                if not chain:
                    raise OrchestrationError(
                        f"sin tool_chain para {spec.artifact_type}: "
                        "el paso LLM 'producer' no puede ser la única "
                        "herramienta de la cadena"
                    )
                # Asignación mecánica: un tool de la cadena por template, en
                # orden, ciclando si hay más templates que herramientas.
                adapter_name = chain[tool_index % len(chain)]
                tool_index += 1
                adapter = self._resolve_adapter(adapter_name)
                if adapter is None:
                    raise OrchestrationError(
                        f"tool_adapter inexistente '{adapter_name}' en tool_chain"
                    )

                job = self.create_job(
                    artifact_id,
                    adapter.id,
                    sequence,
                    phase=phase,
                    template_id=template.id,
                    manifest_version=manifest_row.version,
                )
                job.status = "running"

                contract = render_for_tool(template, manifest_data, adapter.name)
                try:
                    client = self.mcp_client_factory(adapter)
                    result = client.call_tool(adapter.name, contract)
                except Exception as exc:  # noqa: BLE001 — el fallo de la herramienta detiene la cadena
                    job.status = "failed"
                    job.error = str(exc)
                    jobs.append(
                        self._job_summary(job, phase, template.name, adapter.name)
                    )
                    return self._summary(artifact_id, manifest_row.version, jobs, False)

                output_path = (result or {}).get("output_path")
                job.output_path = output_path
                job.status = "done"
                last_output = output_path
                jobs.append(self._job_summary(job, phase, template.name, adapter.name))

        if last_output:
            artifact.storage_path = last_output
        artifact.status = "listo"
        self.session.add(artifact)
        return self._summary(artifact_id, manifest_row.version, jobs, True)
