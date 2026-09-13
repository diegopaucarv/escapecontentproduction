"""Tests de los modelos ORM de producción creativa (0005) — sin base real."""

from sqlalchemy import UniqueConstraint

from src.db.models import (
    AssetJob,
    ComponentLibrary,
    ContentArtifact,
    FormatSpec,
    ToolAdapter,
)


def test_format_spec_table_and_columns():
    assert FormatSpec.__tablename__ == "format_specs"
    cols = {c.name for c in FormatSpec.__table__.columns}
    assert {
        "id",
        "brand_objective",
        "artifact_type",
        "structure",
        "constraints",
        "derivation_rules",
        "visual_requirements",
        "qa_checks",
        "tool_chain",
        "is_active",
        "created_at",
        "updated_at",
    } <= cols


def test_format_spec_unique_constraint():
    uq = [
        c for c in FormatSpec.__table__.constraints if isinstance(c, UniqueConstraint)
    ]
    assert any(
        {col.name for col in c.columns} == {"brand_objective", "artifact_type"}
        for c in uq
    )


def test_component_library_table():
    assert ComponentLibrary.__tablename__ == "component_library"
    cols = {c.name for c in ComponentLibrary.__table__.columns}
    assert {"component_type", "name", "content", "usage_count"} <= cols


def test_tool_adapter_name_unique():
    assert ToolAdapter.__tablename__ == "tool_adapters"
    assert ToolAdapter.__table__.c.name.unique


def test_asset_job_foreign_keys():
    assert AssetJob.__tablename__ == "asset_jobs"
    fks = {fk.target_fullname for fk in AssetJob.__table__.foreign_keys}
    assert "content_artifacts.id" in fks
    assert "tool_adapters.id" in fks


def test_content_artifact_new_columns():
    cols = {c.name for c in ContentArtifact.__table__.columns}
    assert "visual_spec" in cols
    assert "format_spec_id" in cols
    fks = {fk.target_fullname for fk in ContentArtifact.__table__.foreign_keys}
    assert "format_specs.id" in fks
