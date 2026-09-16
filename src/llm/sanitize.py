"""
Guardrails universales para respuestas LLM.

Unifica la limpieza de output del LLM: fences de markdown, texto alrededor,
trailing commas y DETECCIÓN de truncamiento (JSON cortado a mitad por
finish_reason='length').

Uso:
    from src.llm.sanitize import sanitize_llm_json, LLMTruncatedError, LLMJSONError

    try:
        data = sanitize_llm_json(text)
    except LLMTruncatedError:
        # reintentar con thinking disabled o degradar
    except LLMJSONError:
        # no es JSON, degradar
"""

from __future__ import annotations

import json
import re

__all__ = [
    "LLMJSONError",
    "LLMTruncatedError",
    "sanitize_llm_response",
    "sanitize_llm_json",
]


class LLMJSONError(ValueError):
    """El texto no contiene un JSON válido (ni siquiera truncado)."""


class LLMTruncatedError(LLMJSONError):
    """El JSON está cortado a mitad de estructura (finish_reason='length')."""


# Errores de json.loads que delatan un corte a mitad de estructura.
_ANYWHERE_TRUNCATION = (
    "Unterminated string",
    "Expecting property name",
    "Invalid \\escape",
    "Unterminated \\uXXXX surrogate",
)
# Solo cuentan como truncamiento si ocurren al final del texto.
_END_TRUNCATION = (
    "Expecting ',' delimiter",
    "Expecting ':' delimiter",
    "Expecting value",
)

# El texto termina a mitad de estructura (sin cerrar).
_OPEN_END_RE = re.compile(r"[{\[,:]\s*$")


def _strip_fences(text: str) -> str:
    """Extrae el JSON de fences ```json ... ``` o el primer {...} del texto."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return m.group(0)
    return text


def _looks_truncated(text: str, exc: json.JSONDecodeError) -> bool:
    """Heurística: ¿el error de parseo delata un corte a mitad de JSON?

    `exc.msg` es el mensaje COMPLETO (p. ej. "Unterminated string starting
    at"), así que se compara con startswith contra los prefijos conocidos.
    """
    msg = exc.msg or ""
    if any(msg.startswith(t) for t in _ANYWHERE_TRUNCATION):
        return True
    if any(msg.startswith(t) for t in _END_TRUNCATION) and exc.pos >= len(text) - 1:
        return True
    if _OPEN_END_RE.search(text):
        return True
    return False


def sanitize_llm_response(text: str, *, expect_json: bool = True) -> str:
    r"""Limpia fences, texto alrededor y trailing commas; devuelve el JSON (str).

    - Idempotente: si `text` ya es JSON válido, se devuelve tal cual.
    - Lanza LLMTruncatedError si el texto está cortado a mitad de estructura.
    - Lanza LLMJSONError si no hay JSON parseable.
    - Riesgo documentado: `re.sub(r",\s*([}\]])", ...)` puede romper strings
      legítimos que contengan `,}` o `,]` (p. ej. `"texto,}"`). Aceptado: el
      LLM produce trailing commas con mucha más frecuencia que ese patrón.
    - Riesgo documentado: `\{.*\}` greedy puede sobre-capturar si un string
      contiene `}`. Aceptado (mismo comportamiento que el parse actual).
    """
    if not isinstance(text, str) or not text.strip():
        raise LLMJSONError("output vacío o no-string")

    candidates: list[str] = [text.strip()]
    extracted = _strip_fences(text)
    if extracted != text.strip():
        candidates.append(extracted)

    if not expect_json:
        return candidates[0]

    last_exc: json.JSONDecodeError | None = None
    last_cleaned = candidates[0]
    for cand in candidates:
        for cleaned in (cand, re.sub(r",\s*([}\]])", r"\1", cand)):
            try:
                json.loads(cleaned)
                return cleaned
            except json.JSONDecodeError as exc:
                last_exc = exc
                last_cleaned = cleaned
            except TypeError:
                continue

    if last_exc is not None and _looks_truncated(last_cleaned, last_exc):
        raise LLMTruncatedError(
            f"JSON truncado (pos {last_exc.pos}): "
            f"...{last_cleaned[max(0, last_exc.pos - 80) : last_exc.pos + 40]!r}..."
        )
    raise LLMJSONError(
        f"no-JSON: {last_exc.msg} en pos {last_exc.pos}" if last_exc else "no-JSON"
    )


def sanitize_llm_json(text: str) -> dict:
    """sanitize_llm_response + json.loads → dict.

    Lanza LLMTruncatedError (JSON cortado) o LLMJSONError (no-JSON o no-dict).
    """
    cleaned = sanitize_llm_response(text)
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise LLMJSONError(f"JSON válido pero no es objeto: {type(data).__name__}")
    return data
