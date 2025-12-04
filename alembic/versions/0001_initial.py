"""
Initial tables for Yahoo Fantasy persistence

Revision ID: 0001_initial
Revises: 
Create Date: 2025-12-04
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '0001_initial'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'leagues',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('game_key', sa.String(length=32), nullable=False),
        sa.Column('league_key', sa.String(length=64), nullable=False),
        sa.Column('season', sa.String(length=16), nullable=False),
        sa.UniqueConstraint('league_key', name='uq_league_key'),
    )

    op.create_table(
        'weeks',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('league_id', sa.Integer(), sa.ForeignKey('leagues.id', ondelete='CASCADE'), nullable=False),
        sa.Column('week_num', sa.Integer(), nullable=False),
        sa.Column('start_date', sa.Date(), nullable=True),
        sa.Column('end_date', sa.Date(), nullable=True),
        sa.Column('status', sa.String(length=16), nullable=False, server_default='open'),
        sa.UniqueConstraint('league_id', 'week_num', name='uq_weeks_league_week'),
    )

    op.create_table(
        'teams',
        sa.Column('team_key', sa.String(length=64), primary_key=True),
        sa.Column('league_id', sa.Integer(), sa.ForeignKey('leagues.id', ondelete='CASCADE'), nullable=False),
        sa.Column('team_name', sa.String(length=128), nullable=True),
    )

    op.create_table(
        'players',
        sa.Column('player_key', sa.String(length=64), primary_key=True),
        sa.Column('name', sa.String(length=128), nullable=True),
    )

    op.create_table(
        'player_positions',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('player_key', sa.String(length=64), sa.ForeignKey('players.player_key', ondelete='CASCADE'), nullable=False),
        sa.Column('position', sa.String(length=16), nullable=False),
        sa.UniqueConstraint('player_key', 'position', name='uq_player_position'),
    )

    op.create_table(
        'stat_categories',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('league_id', sa.Integer(), sa.ForeignKey('leagues.id', ondelete='CASCADE'), nullable=False),
        sa.Column('stat_id', sa.Integer(), nullable=False),
        sa.Column('abbr', sa.String(length=32), nullable=True),
        sa.Column('name', sa.String(length=64), nullable=True),
        sa.Column('position_type', sa.String(length=16), nullable=True),
        sa.UniqueConstraint('league_id', 'stat_id', name='uq_stat_cat_league_stat'),
    )

    op.create_table(
        'matchups',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('league_id', sa.Integer(), sa.ForeignKey('leagues.id', ondelete='CASCADE'), nullable=False),
        sa.Column('week_num', sa.Integer(), nullable=False),
        sa.Column('matchup_index', sa.Integer(), nullable=False),
        sa.UniqueConstraint('league_id', 'week_num', 'matchup_index', name='uq_matchup_l_w_m'),
    )

    op.create_table(
        'weekly_totals',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('league_id', sa.Integer(), sa.ForeignKey('leagues.id', ondelete='CASCADE'), nullable=False),
        sa.Column('week_num', sa.Integer(), nullable=False),
        sa.Column('matchup_id', sa.Integer(), sa.ForeignKey('matchups.id', ondelete='SET NULL'), nullable=True),
        sa.Column('team_key', sa.String(length=64), sa.ForeignKey('teams.team_key', ondelete='CASCADE'), nullable=False),
        sa.Column('stat_id', sa.Integer(), nullable=False),
        sa.Column('stat_abbr', sa.String(length=32), nullable=True),
        sa.Column('value', sa.Numeric(18, 6), nullable=True),
        sa.UniqueConstraint('league_id', 'week_num', 'team_key', 'stat_id', name='uq_weekly_total_key'),
    )

    op.create_table(
        'weekly_player_gp',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('league_id', sa.Integer(), sa.ForeignKey('leagues.id', ondelete='CASCADE'), nullable=False),
        sa.Column('week_num', sa.Integer(), nullable=False),
        sa.Column('team_key', sa.String(length=64), sa.ForeignKey('teams.team_key', ondelete='CASCADE'), nullable=False),
        sa.Column('player_key', sa.String(length=64), sa.ForeignKey('players.player_key', ondelete='CASCADE'), nullable=False),
        sa.Column('gp', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('source', sa.String(length=16), nullable=True),
        sa.Column('computed_at', sa.DateTime(), nullable=True),
        sa.UniqueConstraint('league_id', 'week_num', 'team_key', 'player_key', name='uq_weekly_player_gp_key'),
    )

    op.create_table(
        'roster_slots_daily',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('date', sa.Date(), nullable=False),
        sa.Column('league_id', sa.Integer(), sa.ForeignKey('leagues.id', ondelete='CASCADE'), nullable=False),
        sa.Column('team_key', sa.String(length=64), sa.ForeignKey('teams.team_key', ondelete='CASCADE'), nullable=False),
        sa.Column('player_key', sa.String(length=64), sa.ForeignKey('players.player_key', ondelete='CASCADE'), nullable=False),
        sa.Column('selected_position', sa.String(length=16), nullable=True),
        sa.Column('had_game', sa.Boolean(), nullable=True),
        sa.Column('gp', sa.Integer(), nullable=False, server_default='0'),
        sa.UniqueConstraint('date', 'league_id', 'team_key', 'player_key', name='uq_roster_slot_daily_key'),
    )


def downgrade() -> None:
    # Drop in reverse dependency order
    op.drop_table('roster_slots_daily')
    op.drop_table('weekly_player_gp')
    op.drop_table('weekly_totals')
    op.drop_table('matchups')
    op.drop_table('stat_categories')
    op.drop_table('player_positions')
    op.drop_table('players')
    op.drop_table('teams')
    op.drop_table('weeks')
    op.drop_table('leagues')
