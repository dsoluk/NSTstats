"""add stat grouping columns and backfill basic ordering

Revision ID: 0002_add_stat_grouping
Revises: 0001_initial
Create Date: 2025-12-06 10:10:00.000000
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '0002_add_stat_grouping'
down_revision = '0001_initial'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add new nullable columns
    op.add_column('stat_categories', sa.Column('group_code', sa.String(length=32), nullable=True))
    op.add_column('stat_categories', sa.Column('group_order', sa.Integer(), nullable=True))
    op.add_column('stat_categories', sa.Column('stat_order', sa.Integer(), nullable=True))

    # Backfill grouping for typical Yahoo NHL leagues using abbr
    conn = op.get_bind()

    # Skater groups
    skater_groups = [
        ('Offensive', ['G', 'A', 'PPP', 'SOG', 'FW']),
        ('Banger', ['HIT', 'BLK', 'PIM']),
        ('Other', ['+/-', 'SHP']),
    ]

    # Assign orders starting from 1
    for gi, (gcode, stats) in enumerate(skater_groups, start=1):
        for si, ab in enumerate(stats, start=1):
            conn.execute(
                sa.text(
                    """
                    UPDATE stat_categories
                    SET group_code = :gcode, group_order = :gord, stat_order = :sord
                    WHERE UPPER(COALESCE(abbr, '')) = :abbr AND position_type = 'P'
                    """
                ),
                dict(gcode=gcode, gord=gi, sord=si, abbr=ab.upper()),
            )

    # Goalie order
    goalie_order = ['W', 'GA', 'GAA', 'SV%', 'SHO']
    for si, ab in enumerate(goalie_order, start=1):
        conn.execute(
            sa.text(
                """
                UPDATE stat_categories
                SET group_code = 'Goalie', group_order = 1, stat_order = :sord
                WHERE UPPER(COALESCE(abbr, '')) = :abbr AND position_type = 'G'
                """
            ),
            dict(sord=si, abbr=ab.upper()),
        )


def downgrade() -> None:
    op.drop_column('stat_categories', 'stat_order')
    op.drop_column('stat_categories', 'group_order')
    op.drop_column('stat_categories', 'group_code')
