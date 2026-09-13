"""agrega hashed_password a app_users para RBAC real (login + JWT)

Revision ID: 0002_add_password_hash
Revises: 0001_initial_schema
Create Date: 2026-09-12
"""
from alembic import op
import sqlalchemy as sa

revision = "0002_add_password_hash"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("app_users", sa.Column("hashed_password", sa.String(255), nullable=True))


def downgrade() -> None:
    op.drop_column("app_users", "hashed_password")
