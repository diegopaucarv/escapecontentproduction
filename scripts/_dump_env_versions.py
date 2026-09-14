"""
Diagnóstico del entorno KAG remoto (la máquina donde falla la ingesta).

Uso (en la máquina con el error):
    python scripts/_dump_env_versions.py

Imprime:
  1. Versión de Python y de los paquetes clave (torch, transformers, spacy,
     stanza, sentence-transformers, sqlalchemy, pgvector, psycopg2, httpx,
     tenacity, rapidfuzz, scipy, scikit-learn, hnswlib, numpy, ...).
  2. Commit de git (si el repo es git) y estado — para comparar con este repo.
  3. Escaneo AST de src/ buscando el patrón del bug:
     una función donde `text` es parámetro o variable local Y se llama como
     función (text(...)) — el sombreado que causa "'str' object is not callable".

El escaneo AST es la parte importante: si el otro entorno corre una versión
vieja del código, este script localiza la línea exacta del sombreado sin
tener que leer archivos a mano.
"""

import ast
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

KEY_PACKAGES = [
    "torch",
    "transformers",
    "spacy",
    "stanza",
    "sentence-transformers",
    "sqlalchemy",
    "pgvector",
    "psycopg2",
    "psycopg2-binary",
    "httpx",
    "tenacity",
    "rapidfuzz",
    "scipy",
    "scikit-learn",
    "hnswlib",
    "numpy",
    "pydantic",
    "pydantic-settings",
    "alembic",
    "fastapi",
    "uvicorn",
]


def dump_versions() -> None:
    print("=" * 70)
    print("1) VERSIONES")
    print("=" * 70)
    print(f"python: {platform.python_version()} ({platform.platform()})")
    try:
        import importlib.metadata as md

        for pkg in KEY_PACKAGES:
            try:
                print(f"  {pkg}: {md.version(pkg)}")
            except md.PackageNotFoundError:
                print(f"  {pkg}: NO INSTALADO")
    except Exception as exc:  # noqa: BLE001
        print(f"  (no se pudo leer versiones: {exc})")


def dump_git() -> None:
    print("=" * 70)
    print("2) GIT")
    print("=" * 70)
    for cmd in (
        ["git", "rev-parse", "HEAD"],
        ["git", "log", "-1", "--oneline"],
        ["git", "status", "--short"],
    ):
        try:
            out = subprocess.run(  # noqa: S603
                cmd, cwd=ROOT, capture_output=True, text=True, timeout=10
            )
            print(f"$ {' '.join(cmd)}")
            print((out.stdout or out.stderr).strip() or "(vacío)")
        except Exception as exc:  # noqa: BLE001
            print(f"  (git {cmd[1]}: {exc})")


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


def find_text_shadowing() -> None:
    print("=" * 70)
    print("3) ESCANEO AST: sombreado de `text` (el bug)")
    print("=" * 70)
    hits: list[tuple[str, str, int]] = []
    for py in sorted((ROOT / "src").rglob("*.py")):
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"  {py.relative_to(ROOT)}: parse error {exc}")
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            bound = _bound_names_in_scope(node)
            if "text" not in bound:
                continue
            for sub in ast.walk(node):
                if (
                    isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Name)
                    and sub.func.id == "text"
                ):
                    hits.append((str(py.relative_to(ROOT)), node.name, sub.lineno))
    if hits:
        print("  ⚠  SOMBREADO ENCONTRADO — esto causa 'str' object is not callable:")
        for rel, fn, line in hits:
            print(
                f"      {rel}: función '{fn}' llama text() en L{line} con 'text' en scope"
            )
    else:
        print("  ✓ No se encontró ningún sombreado de text() en src/.")
        print(
            "    (Si el otro entorno falla, su código NO es este — comparar git/versiones.)"
        )


if __name__ == "__main__":
    dump_versions()
    dump_git()
    find_text_shadowing()
