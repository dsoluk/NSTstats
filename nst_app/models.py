# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey and OneToOneField has `on_delete` set to the desired behavior
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from django.db import models


class Parameter(models.Model):
    key = models.CharField(max_length=255, unique=True)
    value = models.TextField(blank=True, null=True)
    description = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.key

    class Meta:
        db_table = 'parameters'


class AlembicVersion(models.Model):
    version_num = models.CharField(primary_key=True)

    class Meta:
        managed = False
        db_table = 'alembic_version'


class CurrentRosters(models.Model):
    # pk = models.CompositePrimaryKey('league_id', 'player_key')
    league = models.ForeignKey('Leagues', models.DO_NOTHING)
    player_key = models.ForeignKey('Players', models.DO_NOTHING, db_column='player_key')
    team_key = models.ForeignKey('Teams', models.DO_NOTHING, db_column='team_key')

    class Meta:
        managed = False
        db_table = 'current_rosters'
        unique_together = (('league', 'player_key'),)


class Leagues(models.Model):
    game_key = models.CharField()
    league_key = models.CharField(unique=True)
    season = models.CharField()

    class Meta:
        managed = False
        db_table = 'leagues'


class Matchups(models.Model):
    league = models.ForeignKey(Leagues, models.DO_NOTHING)
    week_num = models.IntegerField()
    matchup_index = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'matchups'
        unique_together = (('league', 'week_num', 'matchup_index'),)


class PlayerPositions(models.Model):
    player_key = models.ForeignKey('Players', models.DO_NOTHING, db_column='player_key')
    position = models.CharField()

    class Meta:
        managed = False
        db_table = 'player_positions'
        unique_together = (('player_key', 'position'),)


class Players(models.Model):
    player_key = models.CharField(primary_key=True)
    name = models.CharField(blank=True, null=True)
    positions = models.CharField(blank=True, null=True)
    last_synced = models.DateTimeField(blank=True, null=True)
    status = models.CharField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'players'


class RosterSlotsDaily(models.Model):
    date = models.DateField()
    league = models.ForeignKey(Leagues, models.DO_NOTHING)
    team_key = models.ForeignKey('Teams', models.DO_NOTHING, db_column='team_key')
    player_key = models.ForeignKey(Players, models.DO_NOTHING, db_column='player_key')
    selected_position = models.CharField(blank=True, null=True)
    had_game = models.BooleanField(blank=True, null=True)
    gp = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'roster_slots_daily'
        unique_together = (('league', 'team_key', 'player_key'),)


class StatCategories(models.Model):
    league = models.ForeignKey(Leagues, models.DO_NOTHING)
    stat_id = models.IntegerField()
    abbr = models.CharField(blank=True, null=True)
    name = models.CharField(blank=True, null=True)
    position_type = models.CharField(blank=True, null=True)
    group_code = models.CharField(blank=True, null=True)
    group_order = models.IntegerField(blank=True, null=True)
    stat_order = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'stat_categories'
        unique_together = (('league', 'stat_id'),)


class Teams(models.Model):
    team_key = models.CharField(primary_key=True)
    league = models.ForeignKey(Leagues, models.DO_NOTHING)
    team_name = models.CharField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'teams'


class WeeklyPlayerGp(models.Model):
    league = models.ForeignKey(Leagues, models.DO_NOTHING)
    week_num = models.IntegerField()
    team_key = models.ForeignKey(Teams, models.DO_NOTHING, db_column='team_key')
    player_key = models.ForeignKey(Players, models.DO_NOTHING, db_column='player_key')
    gp = models.IntegerField()
    source = models.CharField(blank=True, null=True)
    computed_at = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'weekly_player_gp'
        unique_together = (('league', 'week_num', 'team_key', 'player_key'),)


class WeeklyTotals(models.Model):
    league = models.ForeignKey(Leagues, models.DO_NOTHING)
    week_num = models.IntegerField()
    matchup = models.ForeignKey(Matchups, models.DO_NOTHING, blank=True, null=True)
    team_key = models.ForeignKey(Teams, models.DO_NOTHING, db_column='team_key')
    stat_id = models.IntegerField()
    stat_abbr = models.CharField(blank=True, null=True)
    value = models.TextField(blank=True, null=True)  # This field type is a guess.

    class Meta:
        managed = False
        db_table = 'weekly_totals'
        unique_together = (('league', 'week_num', 'team_key', 'stat_id'),)


class Weeks(models.Model):
    league = models.ForeignKey(Leagues, models.DO_NOTHING)
    week_num = models.IntegerField()
    start_date = models.DateField(blank=True, null=True)
    end_date = models.DateField(blank=True, null=True)
    status = models.CharField()

    class Meta:
        managed = False
        db_table = 'weeks'
        unique_together = (('league', 'week_num'),)
