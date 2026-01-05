"""add nst and sched sync timestamps to leagues

Revision ID: d27df62d3036
Revises: 05a5b2d67380
Create Date: 2026-01-04 15:53:32.519205

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd27df62d3036'
down_revision: Union[str, Sequence[str], None] = '05a5b2d67380'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table('leagues', schema=None) as batch_op:
        batch_op.add_column(sa.Column('last_roster_sync', sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column('last_nst_sync', sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column('last_sched_sync', sa.DateTime(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('leagues', schema=None) as batch_op:
        batch_op.drop_column('last_sched_sync')
        batch_op.drop_column('last_nst_sync')
        batch_op.drop_column('last_roster_sync')
