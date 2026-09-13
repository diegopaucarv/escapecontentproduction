"""
Seed de datos de ejemplo (mock) para desarrollo local.

Uso:
    python -m src.db.seed            # inserta los briefs de ejemplo
    python -m src.db.seed --reset    # borra briefs de ejemplo y reinserta

Los briefs de ejemplo cubren los dos brand_objective del canon
(ESCAPE_SOCIAL y ERGALIA_COMERCIAL) y varios buckets, para que el
endpoint GET /briefs/{id} y el alineamiento estratégico tengan datos
reales con los que trabajar sin depender de otro servidor.
"""

from __future__ import annotations

import argparse
import uuid

from sqlalchemy import delete, select

from src.db.models import ContentBrief
from src.db.session import SessionLocal

# Marca de identificación para poder borrar solo los seeds en --reset.
SEED_TAG = "seed-mock-v1"


def _brief(**kwargs) -> ContentBrief:
    """Construye un ContentBrief con los campos mínimos del canon
    (segment_client o segment_community obligatorio, §9.2)."""
    return ContentBrief(**kwargs)


def build_mock_briefs() -> list[ContentBrief]:
    """Los briefs de ejemplo. Cada uno refleja un escenario distinto del
    pipeline: ruta repetitiva, nueva solución, segmento sensible, etc."""
    return [
        _brief(
            brand_objective="ESCAPE_SOCIAL",
            content_bucket="difusion_cientifica",
            resumen=(
                "Explicar en 60 segundos por qué la vacunación de refuerzo "
                "sigue siendo relevante en 2026, con datos del último estudio "
                "de inmunidad poblacional."
            ),
            insight_core=(
                "La gente cree que el refuerzo ya no es necesario; los datos "
                "de inmunidad poblacional muestran lo contrario."
            ),
            pitch_15s="¿Crees que ya no necesitas el refuerzo? Los datos dicen otra cosa.",
            prior_attempts="Un post estático en 2024 con poca interacción.",
            risks="Riesgo de polarización en comentarios sobre vacunas.",
            suggested_product_type="video_corto",
            org_priorities_contrast="Prioridad de la marca: desmentir mitos de salud pública.",
            phase_number=2,
            segment_client="S1",
            audience_tier="tier_1",
            need_id="N1",
            need="Información confiable sobre salud pública.",
            change_hypothesis="Si la audiencia ve el dato de inmunidad, reconsidera el refuerzo.",
            service_category="salud_publica",
            product_anchor="campaña_refuerzo_2026",
            entry_offer="guía_descargable",
            artifact_type="video_corto",
            channel="tiktok",
            channel_role="gancho",
            cta="Descarga la guía completa",
            landing="https://escape.social/refuerzo-2026",
            funnel_stage="top",
            language="es",
            geography_content="Latam",
            geography_sales="Latam",
            evidence_source="Estudio de inmunidad poblacional 2026 (fuente primaria).",
            risk_level="medio",
            validation_required=["factual", "legal"],
            repurpose_plan=[
                {"artifact_type": "post", "channel": "instagram"},
                {"artifact_type": "hilo", "channel": "x"},
            ],
            metric_primary="retention_24h",
            metric_secondary="conversion_30d",
            novelty_score=6,
            route_decision="nueva_solucion",
            production_route="complete",
            status="revision",
        ),
        _brief(
            brand_objective="ERGALIA_COMERCIAL",
            content_bucket="caso_autoridad",
            resumen=(
                "Caso de autoridad: cómo una empresa de logística redujo 30% "
                "sus costos operativos con la plataforma de Ergalia."
            ),
            insight_core=(
                "Los líderes de operaciones no confían en promesas; confían "
                "en casos con números verificables."
            ),
            pitch_15s="30% menos costos operativos. Así lo logró un cliente real.",
            prior_attempts=None,
            risks="Requiere anonimización del cliente (S5/S6).",
            suggested_product_type="white_paper",
            org_priorities_contrast="Prioridad: autoridad en el sector logístico.",
            phase_number=1,
            segment_client="S5",
            audience_tier="tier_2",
            need_id="N3",
            need="Evidencia verificable antes de decidir una compra B2B.",
            change_hypothesis="Un caso con números verificables acelera la decisión de compra.",
            service_category="logistica",
            product_anchor="plataforma_ergalia",
            entry_offer="demo_personalizada",
            artifact_type="white_paper",
            channel="linkedin",
            channel_role="autoridad",
            cta="Agenda una demo",
            landing="https://ergalia.com/demo",
            funnel_stage="middle",
            language="es",
            geography_content="España",
            geography_sales="España",
            evidence_source="Cliente real anonimizado (NDA firmado).",
            risk_level="alto",
            validation_required=["anonimizacion", "legal"],
            repurpose_plan=[
                {"artifact_type": "one_pager", "channel": "linkedin"},
                {"artifact_type": "poster_qr", "channel": "feria"},
            ],
            metric_primary="conversion_30d",
            metric_secondary="retention_24h",
            novelty_score=8,
            route_decision="nueva_solucion",
            production_route="complete",
            status="generando",
        ),
        _brief(
            brand_objective="ESCAPE_SOCIAL",
            content_bucket="herramienta_gratuita",
            resumen=(
                "Plantilla gratuita de planificación de contenido para "
                "organizaciones sin fines de lucro."
            ),
            insight_core=(
                "Las ONGs no tienen tiempo para planificar contenido; una "
                "plantilla lista para usar les ahorra horas."
            ),
            pitch_15s="Planifica tu contenido de un mes en 30 minutos.",
            prior_attempts="Newsletter con baja apertura.",
            risks="Bajo riesgo; contenido educativo.",
            suggested_product_type="pdf_recurso",
            org_priorities_contrast="Prioridad: ser útil a la comunidad sin fines de lucro.",
            phase_number=3,
            segment_community="C2",
            audience_tier="tier_3",
            need_id="N5",
            need="Herramientas prácticas para operar mejor.",
            change_hypothesis="Una plantilla gratuita genera confianza y leads calificados.",
            service_category="sin_fines_de_lucro",
            product_anchor="plantilla_contenido",
            entry_offer="plantilla_gratuita",
            artifact_type="pdf_recurso",
            channel="newsletter",
            channel_role="nutricion",
            cta="Descarga la plantilla",
            landing="https://escape.social/plantilla",
            funnel_stage="top",
            language="es",
            geography_content="Latam",
            geography_sales="Global",
            evidence_source="Encuesta a 50 ONGs (2025).",
            risk_level="bajo",
            validation_required=[],
            repurpose_plan=[
                {"artifact_type": "post", "channel": "instagram"},
                {"artifact_type": "broadcast", "channel": "whatsapp"},
            ],
            metric_primary="conversion_30d",
            metric_secondary="retention_24h",
            novelty_score=4,
            route_decision="repetitivo",
            production_route="fast",
            status="aprobado",
        ),
    ]


def seed(engine=None, reset: bool = False) -> list[uuid.UUID]:
    """Inserta los briefs de ejemplo. Devuelve los ids insertados."""
    session = SessionLocal()
    try:
        if reset:
            # Borra solo los briefs marcados como seed (por el tag en resumen).
            session.execute(
                delete(ContentBrief).where(ContentBrief.resumen.like(f"%{SEED_TAG}%"))
            )
            session.commit()

        ids: list[uuid.UUID] = []
        for brief in build_mock_briefs():
            # Evita duplicados si se corre dos veces sin --reset.
            exists = session.execute(
                select(ContentBrief.id).where(ContentBrief.resumen == brief.resumen)
            ).scalar_one_or_none()
            if exists:
                ids.append(exists)
                continue
            session.add(brief)
            session.flush()
            ids.append(brief.id)
        session.commit()
        return ids
    finally:
        session.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inserta briefs de ejemplo (mock).")
    parser.add_argument(
        "--reset", action="store_true", help="Borra los seeds previos y reinserta."
    )
    args = parser.parse_args()
    ids = seed(reset=args.reset)
    print(f"Briefs de ejemplo listos: {len(ids)}")
    for i in ids:
        print(f"  - {i}")


if __name__ == "__main__":
    main()
