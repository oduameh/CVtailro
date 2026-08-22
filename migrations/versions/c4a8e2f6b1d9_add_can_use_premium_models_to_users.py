"""add can_use_premium_models to users

Revision ID: c4a8e2f6b1d9
Revises:
Create Date: 2026-08-22 23:30:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "c4a8e2f6b1d9"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "users",
        sa.Column("can_use_premium_models", sa.Boolean(), nullable=False, server_default=sa.false()),
    )


def downgrade():
    op.drop_column("users", "can_use_premium_models")
