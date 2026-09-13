"""
Seed del canon de marca como Context Pack consultable (0012).

El canon (docs/arquitectura_comercial_y_contenidos_escape_ergalia.md, las
guías de contenido y los _lenguaje_visual.md) vivía como archivos estáticos
que build_context_pack() nunca consultaba. Este seed convierte ese canon en
filas consultables de `brand_knowledge`, etiquetadas por brand_objective +
content_bucket, para que el Producer reciba el tono real de la marca en su
Context Pack en vez de "adivinarlo" por el nombre del bucket.

⚠️ ESTADO ACTUAL: los documentos canónicos NO existen todavía en el repo
(ver docs/pipeline_unificado_produccion_contenidos(1).md §0, que los lista
como fuentes). Por eso este seed inserta MOCKUPS marcados con is_mock=True
y source_doc apuntando al documento real que debe reemplazarlos.

Cómo reemplazar los mockups por el canon real:
  1. Colocar los documentos reales en docs/ (los paths ya están en
     source_doc de cada fila):
       - docs/arquitectura_comercial_y_contenidos_escape_ergalia.md
       - docs/escape/escape_guia_contenidos_y_produccion.md
       - docs/escape/escape_lenguaje_visual.md
       - docs/ergalia/ergalia_guia_contenidos_y_produccion.md
       - docs/ergalia/ergalia_lenguaje_visual.md
  2. Editar las constantes BRAND_SECTIONS / BUCKET_SECTIONS de este archivo
     con el contenido real (sección por sección, como ya están etiquetadas).
  3. Correr `python -m src.db.seed_brand_knowledge` de nuevo: el upsert
     actualiza las filas existentes por clave única (brand_objective,
     content_bucket, section_key). El flag is_mock se mantiene en True
     hasta que el documento real exista; cuando lo reemplaces, edita las
     constantes y pon is_mock=False en _upsert (o borra las filas mock
     y deja que el seed las re-inserte con el contenido real).
  4. Opcional: cuando exista la Artifact Library vectorial completa, migrar
     estas filas a embeddings; mientras tanto, la consulta por
     brand_objective + content_bucket es suficiente.

Uso:
    python -m src.db.seed_brand_knowledge
"""

from __future__ import annotations

import argparse

from sqlalchemy import select

from src.db.models import BrandKnowledge
from src.db.session import SessionLocal

# ---------------------------------------------------------------------
# Secciones transversales por marca (aplican a cualquier bucket)
# ---------------------------------------------------------------------

# brand_objective -> lista de secciones del canon de marca.
BRAND_SECTIONS: dict[str, list[dict]] = {
    "ESCAPE_SOCIAL": [
        {
            "section_key": "tono_y_voz",
            "section_title": "Tono y voz",
            "content": (
                "MOCKUP — reemplazar por docs/escape/escape_guia_contenidos_y_produccion.md. "
                "ESCAPE habla como un divulgador riguroso pero cercano: datos primero, "
                "sin alarmismo, sin tecnicismos innecesarios. Frases cortas. Nunca "
                "condescendiente con la audiencia. El humor es bienvenido solo si no "
                "resta seriedad al dato."
            ),
            "source_doc": "docs/escape/escape_guia_contenidos_y_produccion.md",
        },
        {
            "section_key": "lenguaje_visual",
            "section_title": "Lenguaje visual",
            "content": (
                "MOCKUP — reemplazar por docs/escape/escape_lenguaje_visual.md. "
                "Paleta clara y legible, tipografía sans-serif de alto contraste, "
                "composición limpia con mucho aire. Imágenes auténticas de la marca "
                "o equipo, nunca stock genérico. Gráficos de datos simples y "
                "honestos, sin chartjunk."
            ),
            "source_doc": "docs/escape/escape_lenguaje_visual.md",
        },
        {
            "section_key": "reglas_editoriales",
            "section_title": "Reglas editoriales",
            "content": (
                "MOCKUP — reemplazar por docs/escape/escape_guia_contenidos_y_produccion.md. "
                "Toda afirmación factual cita fuente verificable. Un solo CTA por "
                "pieza. Prioridad de la marca: desmentir mitos de salud pública y "
                "difundir ciencia accesible. Gobernanza editorial: revisión de "
                "riesgo de polarización antes de publicar temas sensibles."
            ),
            "source_doc": "docs/escape/escape_guia_contenidos_y_produccion.md",
        },
    ],
    "ERGALIA_COMERCIAL": [
        {
            "section_key": "tono_y_voz",
            "section_title": "Tono y voz",
            "content": (
                "MOCKUP — reemplazar por docs/ergalia/ergalia_guia_contenidos_y_produccion.md. "
                "Ergalia habla como un consultor B2B: directo, con números "
                "verificables, sin promesas vacías. Tono profesional y sobrio; la "
                "autoridad se demuestra con casos y datos, no con adjetivos. "
                "Anonimización obligatoria de clientes (S5/S6)."
            ),
            "source_doc": "docs/ergalia/ergalia_guia_contenidos_y_produccion.md",
        },
        {
            "section_key": "lenguaje_visual",
            "section_title": "Lenguaje visual",
            "content": (
                "MOCKUP — reemplazar por docs/ergalia/ergalia_lenguaje_visual.md. "
                "Paleta corporativa sobria, tipografía de marca, composición "
                "estructurada tipo informe. Gráficos y diagramas de datos con "
                "jerarquía clara. Assets de la biblioteca de marca, sin stock "
                "genérico."
            ),
            "source_doc": "docs/ergalia/ergalia_lenguaje_visual.md",
        },
        {
            "section_key": "reglas_editoriales",
            "section_title": "Reglas editoriales",
            "content": (
                "MOCKUP — reemplazar por docs/ergalia/ergalia_guia_contenidos_y_produccion.md. "
                "Evidencia verificable antes de decidir una compra B2B: casos con "
                "números, NDA firmado, anonimización. Un solo CTA. Revisión legal "
                "para afirmaciones sobre resultados. Prioridad: autoridad en el "
                "sector logístico."
            ),
            "source_doc": "docs/ergalia/ergalia_guia_contenidos_y_produccion.md",
        },
    ],
}

# ---------------------------------------------------------------------
# Secciones por bucket (aplican a un bucket concreto de una marca)
# ---------------------------------------------------------------------

# (brand_objective, content_bucket) -> lista de secciones.
BUCKET_SECTIONS: dict[tuple[str, str], list[dict]] = {
    ("ESCAPE_SOCIAL", "difusion_cientifica"): [
        {
            "section_key": "bucket_objetivo",
            "section_title": "Objetivo del bucket",
            "content": (
                "MOCKUP — reemplazar por docs/escape/escape_guia_contenidos_y_produccion.md. "
                "Difundir hallazgos científicos en 60 segundos o menos, con el dato "
                "como protagonista. Gancho de 15 segundos que plantea la pregunta "
                "incómoda; el dato la responde. CTA: recurso descargable (guía)."
            ),
            "source_doc": "docs/escape/escape_guia_contenidos_y_produccion.md",
        },
    ],
    ("ESCAPE_SOCIAL", "herramienta_gratuita"): [
        {
            "section_key": "bucket_objetivo",
            "section_title": "Objetivo del bucket",
            "content": (
                "MOCKUP — reemplazar por docs/escape/escape_guia_contenidos_y_produccion.md. "
                "Ofrecer un regalo antes de pedir la venta: plantillas y recursos "
                "listos para usar que ahorran horas. Tono útil y práctico, sin "
                "comercial agresivo. CTA: descarga del recurso."
            ),
            "source_doc": "docs/escape/escape_guia_contenidos_y_produccion.md",
        },
    ],
    ("ERGALIA_COMERCIAL", "caso_autoridad"): [
        {
            "section_key": "bucket_objetivo",
            "section_title": "Objetivo del bucket",
            "content": (
                "MOCKUP — reemplazar por docs/ergalia/ergalia_guia_contenidos_y_produccion.md. "
                "Caso de autoridad con números verificables: cómo un cliente real "
                "redujo costos con la plataforma. Anonimización obligatoria. CTA: "
                "demo personalizada. Riesgo alto por defecto (S5/S6)."
            ),
            "source_doc": "docs/ergalia/ergalia_guia_contenidos_y_produccion.md",
        },
    ],
}

# ---------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------


def _upsert(session, brand: str, bucket: str | None, section: dict) -> BrandKnowledge:
    """Inserta o actualiza una fila de brand_knowledge por clave única
    (brand_objective, content_bucket, section_key)."""
    row = (
        session.execute(
            select(BrandKnowledge).where(
                BrandKnowledge.brand_objective == brand,
                BrandKnowledge.content_bucket == bucket,
                BrandKnowledge.section_key == section["section_key"],
            )
        )
        .scalars()
        .first()
    )
    if row is None:
        row = BrandKnowledge(
            brand_objective=brand,
            content_bucket=bucket,
            section_key=section["section_key"],
            section_title=section["section_title"],
            content=section["content"],
            source_doc=section["source_doc"],
            is_mock=True,
        )
        session.add(row)
        session.flush()
    else:
        row.section_title = section["section_title"]
        row.content = section["content"]
        row.source_doc = section["source_doc"]
        # El contenido de este seed es mockup por definición; cuando se
        # reemplace por el canon real, editar las constantes y correr de
        # nuevo — el flag se mantiene en True hasta que el documento real
        # exista (ver docstring).
        row.is_mock = True
    return row


def seed(session=None) -> dict:
    """Inserta/actualiza el canon de marca en brand_knowledge.

    Devuelve un resumen con las filas insertadas. Idempotente: upsert por
    clave única (brand_objective, content_bucket, section_key).
    """
    own_session = session is None
    if own_session:
        session = SessionLocal()
    try:
        rows: list[BrandKnowledge] = []
        for brand, sections in BRAND_SECTIONS.items():
            for section in sections:
                rows.append(_upsert(session, brand, None, section))
        for (brand, bucket), sections in BUCKET_SECTIONS.items():
            for section in sections:
                rows.append(_upsert(session, brand, bucket, section))
        session.commit()
        return {
            "rows": [str(r.id) for r in rows],
            "count": len(rows),
            "brands": sorted({r.brand_objective for r in rows}),
            "buckets": sorted({r.content_bucket for r in rows if r.content_bucket}),
        }
    finally:
        if own_session:
            session.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed del canon de marca (brand_knowledge) como Context Pack."
    )
    parser.parse_args()
    result = seed()
    print("Canon de marca listo (brand_knowledge):")
    print(f"  Filas: {result['count']}")
    print(f"  Marcas: {', '.join(result['brands'])}")
    print(f"  Buckets: {', '.join(result['buckets'])}")
    print("  ⚠️  Contenido MOCKUP (is_mock=True) — reemplazar por los documentos")
    print("      reales listados en source_doc (ver docstring del seed).")


if __name__ == "__main__":
    main()
