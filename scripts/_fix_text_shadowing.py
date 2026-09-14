"""
Corrige el sombreado de `text` en src/kag_ingest.py (bug "'str' object is not callable").

El bug: en `index_document`, la variable que guarda el markdown se llamaba `text`,
sombreando la función text() de SQLAlchemy. Cada `session.execute(text(...))`
dentro de la función intentaba llamar al string del markdown → TypeError.

El fix: renombrar la variable `text` → `md_text` SOLO dentro de `index_document`,
dejando intactas las llamadas `text(...)` (SQLAlchemy). Es exactamente el fix que
ya está aplicado en el entorno local (commit 27d9470).

Usa tokenize (no regex sobre el texto crudo): solo renombra tokens NAME reales,
NUNCA palabras dentro de comentarios o strings.

Uso (en la máquina con el bug):
    python scripts/_fix_text_shadowing.py

Hace backup a src/kag_ingest.py.bak antes de tocar nada y verifica con AST
que no queda ningún sombreado en src/.
"""

import ast
import io
import shutil
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "src" / "kag_ingest.py"
FUNC_NAME = "index_document"


def _rename_text_in_function(src: str) -> str:
    """Renombra el identificador `text` → `md_text` dentro de FUNC_NAME,
    salvo cuando es el callee de una llamada text(...)."""
    lines = src.splitlines(keepends=True)

    # Localizar la función: línea de `def index_document(` hasta el siguiente `def `.
    start_line = None
    end_line = None
    for i, line in enumerate(lines, 1):
        if line.startswith(f"def {FUNC_NAME}("):
            start_line = i
        elif start_line is not None and line.startswith("def "):
            end_line = i
            break
    if start_line is None:
        raise ValueError(f"No se encontró 'def {FUNC_NAME}(' en el archivo.")
    if end_line is None:
        end_line = len(lines) + 1

    data = src.encode("utf-8")
    tokens = list(tokenize.tokenize(io.BytesIO(data).readline))

    # Tokens NAME 'text' dentro de la función que NO son callee de una llamada.
    targets = []
    for idx, tok in enumerate(tokens):
        if tok.type != tokenize.NAME or tok.string != "text":
            continue
        if not (start_line <= tok.start[0] < end_line):
            continue
        nxt = tokens[idx + 1] if idx + 1 < len(tokens) else None
        if nxt is not None and nxt.type == tokenize.OP and nxt.string == "(":
            continue  # text(...) — llamada SQLAlchemy, no renombrar
        targets.append(tok)

    if not targets:
        return src

    # Aplicar de atrás hacia adelante para no invalidar offsets.
    out = src
    for tok in sorted(targets, key=lambda t: t.start, reverse=True):
        line_start = sum(len(l) for l in lines[: tok.start[0] - 1])
        pos = line_start + tok.start[1]
        out = out[:pos] + "md_text" + out[pos + len("text") :]
    return out


def _bound_names_in_scope(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Nombres ligados en el scope de la función (parámetros + asignaciones)."""
    bound: set[str] = set()
    for a in list(fn.args.args) + list(fn.args.kwonlyargs):
        bound.add(a.arg)
    if fn.args.vararg:
        bound.add(fn.args.vararg.arg)
    if fn.args.kwarg:
        bound.add(fn.args.kwarg.arg)
    for sub in ast.walk(fn):
        if isinstance(sub, ast.Assign):
            for t in sub.targets:
                if isinstance(t, ast.Name):
                    bound.add(t.id)
        elif isinstance(sub, (ast.AnnAssign, ast.AugAssign)):
            if isinstance(sub.target, ast.Name):
                bound.add(sub.target.id)
        elif isinstance(sub, (ast.For, ast.AsyncFor)):
            if isinstance(sub.target, ast.Name):
                bound.add(sub.target.id)
        elif isinstance(sub, ast.With):
            for item in sub.items:
                if item.optional_vars and isinstance(item.optional_vars, ast.Name):
                    bound.add(item.optional_vars.id)
        elif isinstance(sub, ast.ExceptHandler):
            if sub.name:
                bound.add(sub.name)
    return bound


def _find_shadowing() -> list[tuple[str, str, int]]:
    """Re-escanea src/ buscando el patrón del bug (text en scope + text(...))."""
    hits: list[tuple[str, str, int]] = []
    for py in sorted((ROOT / "src").rglob("*.py")):
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if "text" not in _bound_names_in_scope(node):
                continue
            for sub in ast.walk(node):
                if (
                    isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Name)
                    and sub.func.id == "text"
                ):
                    hits.append((str(py.relative_to(ROOT)), node.name, sub.lineno))
    return hits


def main() -> int:
    if not TARGET.exists():
        print(f"❌ No existe {TARGET}")
        return 1

    src = TARGET.read_text(encoding="utf-8")
    fixed = _rename_text_in_function(src)

    if fixed == src:
        print(f"✓ No había nada que corregir: {FUNC_NAME} ya usa md_text.")
    else:
        bak = TARGET.with_suffix(".py.bak")
        shutil.copy2(TARGET, bak)
        print(f"💾 Backup en {bak.relative_to(ROOT)}")
        TARGET.write_text(fixed, encoding="utf-8")
        print(f"✅ Renombrado text → md_text dentro de {FUNC_NAME}.")

    # Verificación final con AST.
    hits = _find_shadowing()
    if hits:
        print("⚠  AÚN quedan sombreados:")
        for rel, fn, line in hits:
            print(f"    {rel}: '{fn}' llama text() en L{line} con 'text' en scope")
        return 1
    print("✓ Verificación AST: no queda ningún sombreado de text() en src/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
