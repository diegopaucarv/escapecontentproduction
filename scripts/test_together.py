"""Test independiente: envía un prompt a Together AI y visualiza el resultado.

Uso:
    python scripts/test_together.py "¿Qué es el backpropagation?"
    python scripts/test_together.py --small "Hola, responde en una línea."
    python scripts/test_together.py --large "Explica el teorema de Bayes."

Sin argumentos, usa un prompt de ejemplo. Muestra con prints detallados qué
config se lee de la DB (api_key, modelos) y qué se envía a Together.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Asegura que la raíz del proyecto esté en sys.path al correr como script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Test de Together AI.")
    parser.add_argument("prompt", nargs="?", help="Prompt a enviar (default: ejemplo).")
    parser.add_argument(
        "--small", action="store_true", help="Usa el modelo pequeño (Muse-Glimmer-30B)."
    )
    parser.add_argument(
        "--large", action="store_true", help="Usa el modelo grande (DeepSeek-V4-Flash)."
    )
    parser.add_argument(
        "--both", action="store_true", help="Envía a ambos modelos y compara."
    )
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    from src.db.session import SessionLocal
    from src.llm.together import complete, get_active_llm_config

    prompt = args.prompt or (
        "Explica en 3 frases qué es un sistema RAG y por qué se usa "
        "recuperación de conocimiento."
    )

    session = SessionLocal()
    try:
        # ── 1. Config activa desde la DB ────────────────────────────────────
        print("\n" + "=" * 60)
        print("CONFIG LLM ACTIVA (desde la DB)")
        print("=" * 60)
        cfg = get_active_llm_config(session)
        print(f"  api_key      : {cfg.api_key[:12]}...{cfg.api_key[-4:]}")
        print(f"  small_model  : {cfg.small_model}")
        print(f"  large_model  : {cfg.large_model}")
        print(f"  temp small   : {cfg.temperature_small}")
        print(f"  temp large   : {cfg.temperature_large}")
        print(f"  max_tok small: {cfg.max_tokens_small}")
        print(f"  max_tok large: {cfg.max_tokens_large}")

        # ── 2. Prompt ───────────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("PROMPT")
        print("=" * 60)
        print(f"  {prompt}")

        sizes = []
        if args.small:
            sizes.append("small")
        if args.large:
            sizes.append("large")
        if args.both or not sizes:
            sizes = ["small", "large"]

        # ── 3. Llamadas a Together ──────────────────────────────────────────
        for size in sizes:
            print("\n" + "=" * 60)
            print(f"LLAMADA A TOGETHER — modelo {size.upper()}")
            print("=" * 60)
            model = cfg.small_model if size == "small" else cfg.large_model
            print(f"  endpoint : https://api.together.xyz/v1/chat/completions")
            print(f"  model    : {model}")
            print(f"  max_tokens: {args.max_tokens}, temperature: {args.temperature}")
            print("  enviando...")
            t0 = time.time()
            try:
                out = complete(
                    session,
                    prompt,
                    model_size=size,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                dt = time.time() - t0
                print(f"  ⏱ {dt:.1f}s")
                print("\n  ── RESPUESTA ──")
                print(f"  {out}")
            except Exception as exc:  # noqa: BLE001
                print(f"  ❌ Error: {type(exc).__name__}: {exc}")

        print("\n" + "=" * 60)
        print("FIN DEL TEST")
        print("=" * 60)
    finally:
        session.close()


if __name__ == "__main__":
    main()
