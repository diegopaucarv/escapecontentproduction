"""Setup reproducible del entorno KAG en Google Colab (o cualquier máquina).

Reemplaza el script manual de Colab: en vez de descargar los modelos de spaCy
uno a uno y correr ensure_stanza.py, usa scripts/ensure_languages.py que lee
config/languages.yaml y descarga SOLO lo que falta (spaCy + Stanza) con
reintentos. También instala PostgreSQL + pgvector, migra y siembra la DB.

Pasos:
    1. Desinstala torchvision y torchao (precargados por Colab, incompatibles).
    2. Instala requirements.txt y torch/torchvision (install_torch.py).
    3. Descarga modelos de idioma (ensure_languages.py).
    4. Aplica scripts/_fix_text_shadowing.py (bug 'str' object is not callable).
    5. Instala PostgreSQL + pgvector, crea la DB y el extension vector.
    6. Prepara .env con DATABASE_URL apuntando a localhost.
    7. Migra (alembic upgrade head) y siembra (seed_ai, seed_llm, seed_kag).

Uso:
    python scripts/setup_colab.py
    python scripts/setup_colab.py --skip-deps    # no instala requirements/torch
    python scripts/setup_colab.py --skip-models  # no descarga modelos
    python scripts/setup_colab.py --skip-db      # no toca PostgreSQL
    python scripts/setup_colab.py --together-key tgp_v1_...  # para seed_llm
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Paquetes que Colab precarga con versiones que rompen el stack KAG.
# - torchvision: desajustado contra torch → "operator torchvision::nms does not exist".
# - torchao: peft 0.20.0 exige >=0.16.0, Colab trae 0.10.0 → ImportError.
# Ambos son OPCIONALES para el stack; desinstalarlos es más limpio que fijarlos.
INCOMPATIBLE_PRELOADED = ["torchvision", "torchao"]

# Versión de pgvector a compilar (rama estable).
PGVECTOR_TAG = "v0.7.0"


def _run(cmd: list[str], env: dict | None = None) -> int:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    return subprocess.call(cmd, env=full_env)


def _shell(cmd: str) -> int:
    """Ejecuta un comando de shell (para los pasos de apt/service/psql)."""
    print(f"\n$ {cmd}", flush=True)
    return subprocess.call(cmd, shell=True)


def _setup_db(db_name: str, db_user: str, db_password: str) -> None:
    """Instala PostgreSQL + pgvector, arranca el servicio y crea la DB."""
    print("\n=== 5. Instalando PostgreSQL + pgvector ===")
    _shell("apt-get update -qq > /dev/null")
    _shell(
        "apt-get install -y postgresql postgresql-contrib "
        "postgresql-server-dev-all > /dev/null"
    )
    _shell(
        f"git clone --branch {PGVECTOR_TAG} "
        "https://github.com/pgvector/pgvector.git /tmp/pgvector"
    )
    _shell("cd /tmp/pgvector && make > /dev/null && make install > /dev/null")
    _shell("rm -rf /tmp/pgvector")

    print("\n=== 5b. Arrancando servicio y creando DB ===")
    _shell("service postgresql start")
    _shell(
        f'su - postgres -c "psql -c \\"ALTER USER postgres PASSWORD '
        f"'{db_password}';\\\"\""
    )
    _shell(f'su - postgres -c "createdb {db_name}" || true')
    _shell(
        f'su - postgres -c "psql -d {db_name} -c '
        "'CREATE EXTENSION IF NOT EXISTS vector;'\""
    )
    print(f"✅ DB '{db_name}' lista con extension vector.")


def _prepare_env(db_name: str, db_user: str, db_password: str) -> None:
    """Copia .env.example a .env y apunta DATABASE_URL a localhost."""
    print("\n=== 6. Preparando .env ===")
    env_path = ROOT / ".env"
    if not env_path.exists():
        _shell(f"cp {ROOT / '.env.example'} {env_path}")
    _shell(
        f"sed -i 's|^DATABASE_URL=.*|DATABASE_URL="
        f"postgresql+psycopg2://{db_user}:{db_password}@localhost:5432/{db_name}|' "
        f"{env_path}"
    )
    _shell(
        f"sed -i 's|^DATABASE_URL_ASYNC=.*|DATABASE_URL_ASYNC="
        f"postgresql+asyncpg://{db_user}:{db_password}@localhost:5432/{db_name}|' "
        f"{env_path}"
    )
    print(f"✅ .env apuntando a {db_user}@localhost:5432/{db_name}")


def _migrate_and_seed(together_key: str) -> None:
    """Migra la DB y corre los seeds (ai, llm, kag)."""
    py = sys.executable
    print("\n=== 7. Migraciones y seeds ===")
    _run([py, "-m", "alembic", "upgrade", "head"])
    _run([py, "-m", "src.db.seed_ai"])
    if together_key:
        _run([py, "-m", "src.db.seed_llm"], env={"TOGETHER_API_KEY": together_key})
    else:
        print(
            "⚠  Sin --together-key: se omite seed_llm (la session_settings "
            "quedará sin api_key de Together)."
        )
    _run([py, "-m", "src.db.seed_kag"])


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-deps", action="store_true", help="no instala requirements/torch"
    )
    parser.add_argument(
        "--skip-models", action="store_true", help="no descarga modelos"
    )
    parser.add_argument("--skip-db", action="store_true", help="no toca PostgreSQL")
    parser.add_argument(
        "--db-name", default="escape_ergalia_os", help="nombre de la DB"
    )
    parser.add_argument("--db-user", default="postgres", help="usuario de la DB")
    parser.add_argument("--db-password", default="postgres", help="password de la DB")
    parser.add_argument(
        "--together-key", default="", help="TOGETHER_API_KEY para seed_llm"
    )
    args = parser.parse_args()

    py = sys.executable

    # 1. Limpiar paquetes incompatibles precargados por Colab.
    print("\n=== 1. Limpiando paquetes incompatibles precargados ===")
    for pkg in INCOMPATIBLE_PRELOADED:
        _run([py, "-m", "pip", "uninstall", "-y", pkg])

    # 2. Instalar requirements.
    if not args.skip_deps:
        print("\n=== 2. Instalando requirements.txt ===")
        _run(
            [
                py,
                "-m",
                "pip",
                "install",
                "--quiet",
                "-r",
                str(ROOT / "requirements.txt"),
            ]
        )

    # 2b. Instalar torch/torchvision matcheando el hardware.
    if not args.skip_deps:
        print("\n=== 2b. Instalando torch/torchvision (install_torch.py) ===")
        _run([py, str(ROOT / "scripts" / "install_torch.py")])

    # 3. Descargar modelos de idioma (spaCy + Stanza) según config.
    if not args.skip_models:
        print("\n=== 3. Descargando modelos de idioma (ensure_languages.py) ===")
        _run([py, str(ROOT / "scripts" / "ensure_languages.py")])

    # 4. Aplicar fix del sombreado de `text`.
    print("\n=== 4. Aplicando fix del sombreado de `text` ===")
    _run([py, str(ROOT / "scripts" / "_fix_text_shadowing.py")])

    # 5-7. DB: instalar postgres, preparar .env, migrar y sembrar.
    if not args.skip_db:
        _setup_db(args.db_name, args.db_user, args.db_password)
        _prepare_env(args.db_name, args.db_user, args.db_password)
        _migrate_and_seed(args.together_key)
    else:
        print("\n=== 5-7. DB ===")
        print("--skip-db → no se toca PostgreSQL.")

    print(
        "\n✅ Setup completado. Reinicia el runtime de Colab si acabas de "
        "desinstalar torchvision/torchao."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
