"""Simula la respuesta del LLM para trazar la ingesta de proposiciones SIN llamar al LLM.

Carga chunks reales de doc_id=4, construye una respuesta canónica (formato
`divisions`, con text_span verbatim reales), la inyecta en
`_extract_propositions_batch` vía monkeypatch de call_with_retries, y traza
cada paso: parse → _propositions_from_llm → _locate_span_in_batch → out.

Variantes probadas:
  A. JSON limpio (debería funcionar)
  B. JSON con fences ```json (debería funcionar con el fix de parse_llm_output)
  C. JSON con texto antes/después (debería funcionar con el fix)
  D. JSON con trailing comma (json.loads lo rechaza — ¿lo maneja el fix?)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_env_path = Path(__file__).resolve().parent.parent / ".env"
for _var in ("DATABASE_URL", "DATABASE_URL_ASYNC"):
    _val = os.environ.get(_var, "")
    if not _val and _env_path.exists():
        for _line in _env_path.read_text(encoding="utf-8").splitlines():
            _line = _line.strip()
            if _line.startswith(f"{_var}="):
                _val = _line.split("=", 1)[1].strip().strip('"').strip("'")
                break
    if "@db:" in _val:
        os.environ[_var] = _val.replace("@db:", "@localhost:")

from sqlalchemy import text

import src.kag_ingest as ki
from src.db.session import SessionLocal


def _first_sentence(content: str, max_len: int = 200) -> str:
    """Primera oración del contenido (verbatim, para que el span localice)."""
    for sep in ("\n", ". ", "! ", "? "):
        idx = content.find(sep)
        if idx > 0:
            return content[: idx + (1 if sep != "\n" else 0)].strip()[:max_len]
    return content[:max_len].strip()


def build_canned(chunks, variant: str) -> str:
    """Construye la respuesta cruda del LLM según la variante."""
    props = []
    for i, ch in enumerate(chunks[:3]):
        span = _first_sentence(ch["content"])
        props.append(
            {
                "core_idea_id": str(i + 1),
                "argument_id": f"1.{i + 1}",
                "statement": f"Proposición simulada {i + 1} sobre: {span[:60]}...",
                "text_span": span,
                "char_start": 0,
                "char_end": 0,
                "line_start": 1,
                "line_end": 1,
                "citations_references": [],
            }
        )
    obj = {"divisions": [{"chapter_id": "(archivo)", "propositions": props}]}
    raw = json.dumps(obj, ensure_ascii=False, indent=2)
    if variant == "A":
        return raw
    if variant == "B":
        return f"```json\n{raw}\n```"
    if variant == "C":
        return f"Aquí está el JSON solicitado:\n{raw}\nEspero que sea útil."
    if variant == "D":
        # Trailing comma: quitar el último '}' del objeto y añadir coma extra
        return raw[:-1] + ",}"
    if variant == "E":
        # Trailing comma REALISTA: dentro del JSON, tras el último elemento
        # de la lista de proposiciones (error típico de LLMs)
        return raw.replace("\n    ]\n  }\n}", "\n    ],\n  }\n}")
    return raw


def run_variant(session, chunks, doc_path, variant: str) -> None:
    canned = build_canned(chunks, variant)
    print(f"\n=== Variante {variant} ({len(canned)} chars) ===")

    # Monkeypatch: call_with_retries devuelve la respuesta simulada.
    def fake_call_with_retries(*args, **kwargs):
        return canned, "large", False

    original = ki.call_with_retries
    ki.call_with_retries = fake_call_with_retries
    try:
        out = ki._extract_propositions_batch(
            session,
            doc_id=4,
            chunks=chunks,
            doc_path=doc_path,
            verbose=True,
            model_size="large",
        )
    finally:
        ki.call_with_retries = original

    print(f"  → proposiciones devueltas: {len(out)}")
    if out:
        print(
            f"  → primera: chunk_id={out[0]['chunk_id']} statement={out[0]['statement'][:70]}..."
        )
    else:
        # Trazar dónde se pierde: parsear manualmente
        data = ki.parse_llm_output(canned)
        print(
            f"  → parse_llm_output: keys={list(data.keys()) if isinstance(data, dict) else type(data)}"
        )
        parsed = list(ki._propositions_from_llm(data, default_chapter_id=""))
        print(f"  → _propositions_from_llm: {len(parsed)}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    session = SessionLocal()
    try:
        rows = session.execute(
            text(
                "SELECT id, content, chunk_index, chapter_id FROM kag_chunks "
                "WHERE doc_id = 4 ORDER BY chunk_index LIMIT 5"
            )
        ).fetchall()
        chunks = [
            {
                "id": r.id,
                "content": r.content,
                "chunk_index": r.chunk_index,
                "chapter_id": r.chapter_id,
            }
            for r in rows
        ]
        print(f"{len(chunks)} chunks cargados de doc_id=4.")
        doc_path = "136-Diego Paucar's Unified Social System.md"
        for variant in ("A", "B", "C", "D", "E"):
            run_variant(session, chunks, doc_path, variant)
    finally:
        session.close()


if __name__ == "__main__":
    main()
