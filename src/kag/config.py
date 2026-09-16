"""
Configuración jerárquica del KAG — loader unificado de flags.

Precedencia (de mayor a menor):
  CLI Argument > Request Body API > DB Settings (session_settings)
  > Env Vars (KAG_*) > Defaults en código.

Módulo puro: sin imports pesados (kag_ingest, kag_query, torch, spacy,
sentence_transformers). Importable en cualquier contexto.

Los valores de env/DB/overrides se coaccionan al tipo del default
(bool: "0"/"1"/"true"/"false"; int/float: cast; enum: validación contra
valores permitidos). Si un valor es inválido se ignora y se usa el default
— nunca romper.
"""

from __future__ import annotations

import os
from typing import Any

# ---------------------------------------------------------------------
# Defaults (tabla autoritativa del usuario)
# ---------------------------------------------------------------------

KAG_DEFAULTS: dict[str, Any] = {
    # --- Consulta ---
    "KAG_QUERY_MODE": "fast",  # enum: "fast" | "audited"
    "KAG_INJECT_PROPOSITIONS": True,
    "KAG_RERANK_ENABLED": False,
    "KAG_QUERY_PARALLEL": True,  # fase de consulta: hilos para búsqueda/lecturas
    "KAG_BRANCH_B_MAX_ITERS": 2,
    "KAG_GROUNDING_THRESHOLD": 95.0,
    "KAG_GROUNDING_PARALLEL": True,  # auditoría de grounding: hilos para partial_ratio
    "KAG_PPR_ALPHA": 0.15,
    "KAG_AUTO_THRESHOLDS": True,
    # --- Ingesta ---
    "KAG_EXTRACT_PROPOSITIONS": True,
    "KAG_PROPOSITION_BATCH_SIZE": 400000,
    "KAG_PROPOSITION_MODEL": "large",  # enum: "small" | "large"
    "KAG_PROPOSITION_PARALLEL": 3,  # llamadas LLM concurrentes (semáforo)
    "KAG_PARAPHRASE_PARALLEL": 3,  # llamadas LLM de paráfrasis concurrentes
    "KAG_LLM_ENTITIES": False,
    "KAG_USE_COREF": "auto",  # enum: "auto" | "never"
    "KAG_PROCESS_FIGURES": True,
    # Paralelismo de ingesta: nlp.pipe multi-core (spaCy) y VLM de figuras.
    # Default 1 (seguro en Windows/macOS: multiprocessing necesita guard
    # if __name__ == "__main__"); >1 solo si el entorno lo soporta.
    "KAG_ENTITY_N_PROCESS": 1,
    "KAG_FIGURE_PARALLEL": 3,  # hilos para describe_figure (VLM)
    # Fase map de resúmenes jerárquicos (long docs): llamadas complete
    # (TogetherAI, modelo grande) de las secciones H1/H2 en paralelo.
    # 1 = secuencial (comportamiento clásico). El reduce final siempre es
    # secuencial.
    "KAG_SUMMARY_PARALLEL": 3,
}

# Valores permitidos para flags tipo enum.
_ENUM_VALUES: dict[str, tuple[str, ...]] = {
    "KAG_QUERY_MODE": ("fast", "audited"),
    "KAG_USE_COREF": ("auto", "never"),
    "KAG_PROPOSITION_MODEL": ("small", "large"),
}

# ---------------------------------------------------------------------
# Coacción de tipos
# ---------------------------------------------------------------------


def _coerce(name: str, value: Any) -> Any:
    """Coacciona `value` al tipo del default de `name`.

    Devuelve el default si el valor es inválido (nunca romper).
    """
    default = KAG_DEFAULTS[name]
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            v = value.strip().lower()
            if v in ("1", "true", "yes", "on"):
                return True
            if v in ("0", "false", "no", "off"):
                return False
        return default
    if isinstance(default, int):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    if isinstance(default, float):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    if isinstance(default, str):
        allowed = _ENUM_VALUES.get(name)
        if allowed is not None:
            if isinstance(value, str) and value.strip().lower() in allowed:
                return value.strip().lower()
            return default
        return str(value) if value is not None else default
    return default


# ---------------------------------------------------------------------
# Capas de configuración
# ---------------------------------------------------------------------


def parse_env_config() -> dict:
    """Lee las env vars KAG_* (nombre exacto de la flag) y devuelve un
    dict con valores tipados. Valores inválidos se ignoran (el default se
    aplica en la resolución).
    """
    out: dict[str, Any] = {}
    for name in KAG_DEFAULTS:
        raw = os.environ.get(name)
        if raw is None:
            continue
        coerced = _coerce(name, raw)
        if coerced != KAG_DEFAULTS[name]:
            out[name] = coerced
    return out


def _read_session_settings(session) -> dict:
    """Lee session_settings de la sesión (tabla existente).

    Defensivo: soporta dict, objeto mapeable o atributos. Cualquier fallo
    se propaga al caller, que lo ignora y continúa con la siguiente capa.
    """
    out: dict[str, Any] = {}
    raw = getattr(session, "session_settings", None)
    if raw is None:
        return out
    if isinstance(raw, dict):
        items = raw.items()
    else:
        try:
            items = dict(raw).items()
        except Exception:
            merged: dict[str, Any] = {}
            for source in (vars(raw), vars(type(raw))):
                try:
                    merged.update(source)
                except Exception:
                    pass
            items = merged.items()
    for key, value in items:
        name = str(key).upper()
        if name in KAG_DEFAULTS:
            out[name] = _coerce(name, value)
    return out


def resolve_config(session=None, overrides: dict | None = None) -> dict:
    """Resuelve la config efectiva aplicando la precedencia.

    CLI Argument > Request Body API > DB Settings (session_settings)
    > Env Vars (KAG_*) > Defaults en código.

    `overrides` es un dict plano (CLI/API, máxima prioridad). `session`
    es opcional: si es None o la consulta falla, se salta la capa DB —
    testeable sin DB.
    """
    config = dict(KAG_DEFAULTS)

    # Capa 1: env vars (KAG_*).
    config.update(parse_env_config())

    # Capa 2: DB settings (session_settings) — opcional y defensiva.
    if session is not None:
        try:
            config.update(_read_session_settings(session))
        except Exception:
            pass  # nunca romper: continuar con la siguiente capa

    # Capa 3: overrides (CLI/API) — máxima prioridad.
    if overrides:
        for name, value in overrides.items():
            if name in KAG_DEFAULTS:
                config[name] = _coerce(name, value)

    return config


def get_config_value(config: dict, name: str, default=None):
    """Helper de lectura con default."""
    if name in config:
        return config[name]
    return default


def apply_cli_overrides(parser):
    """Añade args `--kag-<name>` al argparse (simple).

    Cada flag se registra con dest=<NAME> y default=None; el caller
    recoge los no-None de vars(parser.parse_args()) como overrides.
    """
    for name, default in KAG_DEFAULTS.items():
        flag = "--" + name.lower().replace("_", "-")
        parser.add_argument(
            flag,
            dest=name,
            default=None,
            help=f"KAG flag {name} (default: {default!r})",
        )
    return parser
