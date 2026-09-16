"""
Configuración jerárquica del KAG — loader unificado de flags.

Precedencia (de mayor a menor):
  CLI Argument > Request Body API > DB Settings (session_settings)
  > Env Vars (KAG_*) > Estimación dinámica (paralelismo) > Defaults.

Módulo puro: sin imports pesados (kag_ingest, kag_query, torch, spacy,
sentence_transformers). Importable en cualquier contexto.

Los valores de env/DB/overrides se coaccionan al tipo del default
(bool: "0"/"1"/"true"/"false"; int/float: cast; enum: validación contra
valores permitidos). Si un valor es inválido se ignora y se usa el default
efectivo (estimación dinámica o default estático) — nunca romper.
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
    # Lote de proposiciones en tokens de ENTRADA. El output (text_span
    # verbatim + statements) es ~2-3x el input y debe caber en
    # max_tokens del modelo grande (8192) incluso con thinking activo
    # (budget 2048): ~1500 tokens de entrada ≈ 2-3 chunks → output ~3-4.5k
    # tokens. 400k tokens de entrada desbordaban el output (truncado).
    "KAG_PROPOSITION_BATCH_SIZE": 1500,
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

# Centinela: valor inválido para una flag. El caller lo ignora y usa el
# default EFECTIVO (que puede ser la estimación dinámica de paralelismo,
# no el default estático) — nunca romper.
_INVALID = object()


def _coerce(name: str, value: Any) -> Any:
    """Coacciona `value` al tipo del default de `name`.

    Devuelve _INVALID si el valor es inválido (el caller lo ignora y usa el
    default efectivo — nunca romper).
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
        return _INVALID
    if isinstance(default, int):
        try:
            return int(value)
        except (TypeError, ValueError):
            return _INVALID
    if isinstance(default, float):
        try:
            return float(value)
        except (TypeError, ValueError):
            return _INVALID
    if isinstance(default, str):
        allowed = _ENUM_VALUES.get(name)
        if allowed is not None:
            if isinstance(value, str) and value.strip().lower() in allowed:
                return value.strip().lower()
            return _INVALID
        return str(value) if value is not None else _INVALID
    return _INVALID


# ---------------------------------------------------------------------
# Capas de configuración
# ---------------------------------------------------------------------


def parse_env_config() -> dict:
    """Lee las env vars KAG_* (nombre exacto de la flag) y devuelve un
    dict con valores tipados. Valores inválidos se ignoran (el default
    efectivo — estimación o estático — se aplica en la resolución).
    """
    out: dict[str, Any] = {}
    for name in KAG_DEFAULTS:
        raw = os.environ.get(name)
        if raw is None:
            continue
        coerced = _coerce(name, raw)
        if coerced is not _INVALID:
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
            coerced = _coerce(name, value)
            if coerced is not _INVALID:
                out[name] = coerced
    return out


# ---------------------------------------------------------------------
# Auto-estimación de paralelismo (capa 0)
# ---------------------------------------------------------------------

# Lazy cache: se calcula una vez por proceso (los flags no cambian a mitad
# de ejecución).
_estimate_cache: dict[str, int] | None = None


def _detect_cpu_count() -> int:
    """Núcleos lógicos del sistema (fallback 4 si no se puede detectar)."""
    try:
        return os.cpu_count() or 4
    except Exception:  # noqa: BLE001 — degradación natural
        return 4


def _detect_available_memory_gb() -> float:
    """Memoria disponible en GB (fallback 8.0).

    Sin psutil (no está en requirements): ctypes en Windows
    (GlobalMemoryStatusEx), os.sysconf en POSIX. Cualquier fallo → 8.0.
    """
    try:
        if os.name == "nt":
            import ctypes

            class _MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return stat.ullAvailPhys / (1024**3)
            return 8.0
        # POSIX
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return (pages * page_size) / (1024**3)
        return 8.0
    except Exception:  # noqa: BLE001 — degradación natural
        return 8.0


def estimate_parallelism() -> dict[str, int]:
    """Estima los flags de paralelismo desde CPU/memoria del sistema.

    - Flags I/O (llamadas LLM concurrentes: proposiciones, paráfrasis,
      figuras, resúmenes): min(max(2, cores*2), 8, max(2, mem_gb//2)).
    - Flag CPU (nlp.pipe multi-core): min(max(1, cores-1), 4).

    Lazy cache: se calcula una vez por proceso. Los valores son un piso
    razonable para no saturar la máquina; env/DB/CLI pueden sobreescribirlos.
    """
    global _estimate_cache
    if _estimate_cache is not None:
        return _estimate_cache
    cores = _detect_cpu_count()
    mem_gb = _detect_available_memory_gb()
    io = min(max(2, cores * 2), 8, max(2, int(mem_gb) // 2))
    cpu = min(max(1, cores - 1), 4)
    _estimate_cache = {
        "KAG_PROPOSITION_PARALLEL": io,
        "KAG_PARAPHRASE_PARALLEL": io,
        "KAG_FIGURE_PARALLEL": io,
        "KAG_SUMMARY_PARALLEL": io,
        "KAG_ENTITY_N_PROCESS": cpu,
    }
    return _estimate_cache


# ---------------------------------------------------------------------
# Capas de configuración
# ---------------------------------------------------------------------


def resolve_config(session=None, overrides: dict | None = None) -> dict:
    """Resuelve la config efectiva aplicando la precedencia.

    CLI Argument > Request Body API > DB Settings (session_settings)
    > Env Vars (KAG_*) > Estimación dinámica (paralelismo) > Defaults.

    `overrides` es un dict plano (CLI/API, máxima prioridad). `session`
    es opcional: si es None o la consulta falla, se salta la capa DB —
    testeable sin DB.
    """
    config = dict(KAG_DEFAULTS)

    # Capa 0: estimación dinámica de paralelismo (CPU/memoria del sistema).
    config.update(estimate_parallelism())

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
                coerced = _coerce(name, value)
                if coerced is not _INVALID:
                    config[name] = coerced

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
