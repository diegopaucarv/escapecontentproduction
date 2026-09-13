"""Carga y versionado del ProjectManifest (0006).

Los manifiestos son inmutables: cada cambio crea una versión nueva
(INSERT), nunca un UPDATE — mismo patrón que `prompt_artifacts`. La versión
activa es la de mayor número para el artefacto.
"""

from __future__ import annotations

from sqlalchemy import select

from src.db.models import ProductionManifest


def latest_version(session, artifact_id) -> int:
    """Versión más reciente del manifiesto del artefacto (0 si no hay)."""
    rows = (
        session.execute(
            select(ProductionManifest).where(
                ProductionManifest.artifact_id == artifact_id
            )
        )
        .scalars()
        .all()
    )
    versions = [m.version for m in rows]
    return max(versions) if versions else 0


def create_manifest(session, artifact_id, manifest_data: dict) -> ProductionManifest:
    """Crea una versión nueva (max+1) del manifiesto y la persiste."""
    version = latest_version(session, artifact_id) + 1
    manifest = ProductionManifest(
        artifact_id=artifact_id, version=version, manifest=manifest_data
    )
    session.add(manifest)
    session.flush()
    return manifest


def load_manifest(
    session, artifact_id, version: int | None = None
) -> ProductionManifest | None:
    """Carga la versión indicada o la más reciente si version es None."""
    stmt = select(ProductionManifest).where(
        ProductionManifest.artifact_id == artifact_id
    )
    if version is not None:
        stmt = stmt.where(ProductionManifest.version == version)
    else:
        stmt = stmt.order_by(ProductionManifest.version.desc())
    return session.execute(stmt).scalars().first()


def list_manifests(session, artifact_id) -> list[ProductionManifest]:
    """Todas las versiones del artefacto, ordenadas por version desc."""
    rows = (
        session.execute(
            select(ProductionManifest).where(
                ProductionManifest.artifact_id == artifact_id
            )
        )
        .scalars()
        .all()
    )
    return sorted(rows, key=lambda m: m.version, reverse=True)
