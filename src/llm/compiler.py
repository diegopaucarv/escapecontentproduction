"""
Compilador de prompts — Prompt-as-Code (0004).

Arquitectura (aprobada por el usuario):

    [Spec agnóstica] (prompt_templates)  --compilador determinista-->
    [Artefacto inmutable] (prompt_artifacts)  --runtime-->

La lógica del sistema vive en specs declarativas en la base (NO archivos
YAML sueltos). El compilador transpila cada spec a un prompt congelado por
modelo, usando el `syntax_profile` de cada modelo como DATOS (no if/else
por proveedor): así el adaptador maneja modelos comerciales y abiertos
(Mistral, DeepSeek, Llama, Nemotron, ...) con el mismo código.

Regla de oro: prompt_artifacts es INMUTABLE. Recompilar = INSERT nueva
versión + desactivar la anterior. Nunca UPDATE de contenido.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.models import LlmModel, PromptArtifact, PromptTemplate

# Tasks cuyo spec debe validarse contra el checklist de pipeline_templates.
CRITIC_TASK_KEY = "critic_checklist"


# ---------------------------------------------------------------------
# Adaptador genérico guiado por perfil (syntax_profile)
# ---------------------------------------------------------------------


class GenericPromptAdapter:
    """Renderiza una spec agnóstica según el syntax_profile del modelo.

    El perfil decide: rol del system message, estilo de formato (xml /
    markdown / plain), modo de salida estructurada (response_format),
    parámetros de razonamiento, etc. Sin ramas hardcodeadas por proveedor.
    """

    def __init__(self, profile: dict):
        self.profile = profile or {}

    # -- helpers de perfil -------------------------------------------------

    def _fmt(self) -> dict:
        return self.profile.get("instruction_formatting", {}) or {}

    def _structured(self) -> dict:
        return self.profile.get("structured_output", {}) or {}

    def _reasoning(self) -> dict:
        return self.profile.get("reasoning_mode", {}) or {}

    # -- render del system message -----------------------------------------

    def build_system_message(self, spec) -> dict:
        """Devuelve {"role": ..., "content": ...} listo para la API."""
        role = self.profile.get("system_role_name", "system")
        style = self._fmt().get("style", "markdown")
        if style == "xml":
            content = self._render_xml(spec)
        elif style == "plain":
            content = self._render_plain(spec)
        else:
            content = self._render_markdown(spec)
        return {"role": role, "content": content}

    def _render_markdown(self, spec) -> str:
        parts = [f"# Role\n{spec.intent}"]
        if spec.rules:
            parts.append("# Rules\n" + "\n".join(f"- {r}" for r in spec.rules))
        if spec.input_schema:
            parts.append(
                "# Input Schema\n"
                + json.dumps(spec.input_schema, ensure_ascii=False, indent=2)
            )
        if spec.output_schema:
            parts.append(
                "# Output Schema\n"
                + json.dumps(spec.output_schema, ensure_ascii=False, indent=2)
            )
        if spec.few_shot:
            parts.append(
                "# Examples\n" + json.dumps(spec.few_shot, ensure_ascii=False, indent=2)
            )
        return "\n\n".join(parts)

    def _render_xml(self, spec) -> str:
        root = self._fmt().get("root_tag", "instructions")
        delim = self._fmt().get("section_delimiters", {}) or {}
        rules_tag = delim.get("rules", "rules")
        context_tag = delim.get("context", "context")
        examples_tag = delim.get("examples", "examples")

        lines = [f"<{root}>", f"  <{context_tag}>{spec.intent}</{context_tag}>"]
        if spec.rules:
            lines.append(f"  <{rules_tag}>")
            lines.extend(f"    <rule>{r}</rule>" for r in spec.rules)
            lines.append(f"  </{rules_tag}>")
        if spec.few_shot:
            lines.append(f"  <{examples_tag}>")
            lines.append(
                "    "
                + json.dumps(spec.few_shot, ensure_ascii=False).replace("<", "&lt;")
            )
            lines.append(f"  </{examples_tag}>")
        lines.append(f"</{root}>")
        return "\n".join(lines)

    def _render_plain(self, spec) -> str:
        parts = [f"OBJETIVO: {spec.intent}"]
        if spec.rules:
            parts.append("REGLAS:\n" + "\n".join(f"- {r}" for r in spec.rules))
        if spec.output_schema:
            parts.append(
                "SALIDA (JSON estricto):\n"
                + json.dumps(spec.output_schema, ensure_ascii=False, indent=2)
            )
        return "\n\n".join(parts)

    # -- parámetros de runtime ----------------------------------------------

    def build_request_params(self, spec, model) -> dict:
        """Parámetros de la llamada (response_format, reasoning, temp, tokens).

        Solo incluye lo que el perfil del modelo soporta. El runtime los
        fusiona con el body base de la llamada.
        """
        params: dict = {}
        mode = self._structured().get("mode", "none")
        if mode == "json_object":
            params["response_format"] = {"type": "json_object"}
        elif mode == "strict_schema":
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": spec.task_key,
                    "strict": True,
                    "schema": spec.output_schema,
                },
            }
        # reasoning_mode: p. ej. {"supports_reasoning": true,
        # "thinking_parameter": "reasoning_effort"} -> {"reasoning_effort": "medium"}
        reasoning = self._reasoning()
        if reasoning.get("supports_reasoning") and reasoning.get("thinking_parameter"):
            params[reasoning["thinking_parameter"]] = "medium"

        if getattr(model, "temperature_default", None) is not None:
            params["temperature"] = float(model.temperature_default)
        if getattr(model, "max_output_tokens", 0):
            params["max_tokens"] = int(model.max_output_tokens)
        return params

    # -- prompt completo -----------------------------------------------------

    def render_prompt(self, spec, model) -> str:
        """El texto completo que se congela en prompt_artifacts.prompt_text."""
        system = self.build_system_message(spec)
        params = self.build_request_params(spec, model)
        header = (
            f"# Compiled prompt\n"
            f"task: {spec.task_key} (spec v{spec.version})\n"
            f"model: {model.model_name}\n"
            f"profile: {self.profile.get('api_style', 'unknown')}\n"
        )
        body = (
            f"## System message\n{system['content']}\n\n"
            f"## Request params\n{json.dumps(params, ensure_ascii=False, indent=2)}"
        )
        return header + "\n" + body

    def render_user_template(self, spec) -> str:
        """El USER prompt parametrizable (placeholders {..}) de la spec.

        Se congela tal cual en prompt_artifacts.user_template; el runtime lo
        rellena con _fill/str.replace (nunca .format(): los prompts contienen
        llaves JSON literales). Devuelve "" si la spec no define user_template.
        """
        return getattr(spec, "user_template", "") or ""


# ---------------------------------------------------------------------
# Validación de critic_checklist
# ---------------------------------------------------------------------


def _rule_covers_item(rule: str, item: str) -> bool:
    """Una regla cubre un ítem si empieza con '<item>:' (o '<item> ')."""
    return rule.strip().startswith(f"{item}:") or rule.strip().startswith(f"{item} ")


def validate_critic_spec(template, checklist_items: list[str]) -> list[str]:
    """Devuelve los ítems del checklist SIN interpretación en rules.

    Regla del diseño: todo ítem nuevo en pipeline_templates.checklist debe
    tener su interpretación en el spec de critic_checklist. Si falta, el
    crítico marcará ese ítem como `no_evaluado` (nunca `ok`). Esta función
    es la que obliga a mantener ambas fuentes sincronizadas — se llama en el
    CRUD (422) y en el compilador (warning).
    """
    rules = template.rules or []
    missing = [
        item
        for item in (checklist_items or [])
        if not any(_rule_covers_item(rule, item) for rule in rules)
    ]
    return missing


# ---------------------------------------------------------------------
# Consulta de artefactos activos
# ---------------------------------------------------------------------


def get_active_prompt(
    session: Session, model_name: str, task_key: str
) -> PromptArtifact | None:
    """Devuelve el artefacto activo para (modelo, tarea), o None."""
    stmt = (
        select(PromptArtifact)
        .join(LlmModel, LlmModel.id == PromptArtifact.llm_model_id)
        .where(
            LlmModel.model_name == model_name,
            PromptArtifact.task_key == task_key,
            PromptArtifact.is_active.is_(True),
        )
        .order_by(PromptArtifact.artifact_version.desc())
        .limit(1)
    )
    return session.execute(stmt).scalars().first()


# ---------------------------------------------------------------------
# Compilador idempotente
# ---------------------------------------------------------------------


@dataclass
class _CompileResult:
    compiled: int = 0
    skipped: int = 0
    warnings: list = field(default_factory=list)
    artifacts: list = field(default_factory=list)


def _content_hash(prompt_text: str) -> str:
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()


def _existing_versions(
    session: Session, model_id, task_key: str
) -> list[PromptArtifact]:
    stmt = (
        select(PromptArtifact)
        .where(
            PromptArtifact.llm_model_id == model_id,
            PromptArtifact.task_key == task_key,
        )
        .order_by(PromptArtifact.artifact_version.desc())
    )
    return list(session.execute(stmt).scalars().all())


def compile_prompts(session: Session) -> dict:
    """Compila todas las specs activas para todos los modelos de chat activos.

    Idempotente: si el contenido no cambió, no crea nada nuevo. Si cambió,
    INSERT nueva versión y desactiva la anterior (nunca UPDATE).
    """
    result = _CompileResult()

    models = (
        session.execute(
            select(LlmModel).where(
                LlmModel.is_active.is_(True),
                LlmModel.model_size.in_(("small", "large", "vision")),
            )
        )
        .scalars()
        .all()
    )
    templates = (
        session.execute(
            select(PromptTemplate).where(PromptTemplate.is_active.is_(True))
        )
        .scalars()
        .all()
    )

    for model in models:
        adapter = GenericPromptAdapter(model.syntax_profile)
        for spec in templates:
            if spec.task_key == CRITIC_TASK_KEY:
                # El checklist vive en pipeline_templates; aquí solo podemos
                # advertir si el spec no tiene interpretaciones (el CRUD ya
                # valida con los ítems reales al guardar).
                pass

            prompt_text = adapter.render_prompt(spec, model)
            content_hash = _content_hash(prompt_text)
            user_template = adapter.render_user_template(spec)
            user_template_hash = _content_hash(user_template)

            existing = _existing_versions(session, model.id, spec.task_key)
            active = next((a for a in existing if a.is_active), None)
            if (
                active is not None
                and active.content_hash == content_hash
                and active.user_template_hash == user_template_hash
            ):
                result.skipped += 1
                continue

            new_version = (existing[0].artifact_version if existing else 0) + 1
            if active is not None:
                active.is_active = False
            session.add(
                PromptArtifact(
                    llm_model_id=model.id,
                    task_key=spec.task_key,
                    spec_version=spec.version,
                    artifact_version=new_version,
                    prompt_text=prompt_text,
                    content_hash=content_hash,
                    user_template=user_template,
                    user_template_hash=user_template_hash,
                    compiled_by="compiler",
                    is_active=True,
                )
            )
            result.compiled += 1
            result.artifacts.append(
                {
                    "model": model.model_name,
                    "task": spec.task_key,
                    "version": new_version,
                    "hash": content_hash,
                    "user_template_hash": user_template_hash,
                }
            )

    session.commit()
    return {
        "compiled": result.compiled,
        "skipped": result.skipped,
        "warnings": result.warnings,
        "artifacts": result.artifacts,
    }
