from django.contrib import admin
from .models import Parameter, Leagues, Teams, Players, CurrentRosters, Weeks, Matchups, WeeklyTotals, StatCategories, WeeklyPlayerGp, RosterSlotsDaily

@admin.register(Parameter)
class ParameterAdmin(admin.ModelAdmin):
    list_display = ('key', 'value', 'updated_at')
    search_fields = ('key', 'value')

@admin.register(Leagues)
class LeaguesAdmin(admin.ModelAdmin):
    list_display = ('league_key', 'season', 'game_key')
    search_fields = ('league_key',)

@admin.register(Teams)
class TeamsAdmin(admin.ModelAdmin):
    list_display = ('team_key', 'team_name', 'league')
    list_filter = ('league',)
    search_fields = ('team_name', 'team_key')

@admin.register(Players)
class PlayersAdmin(admin.ModelAdmin):
    list_display = ('player_key', 'name', 'positions', 'status')
    search_fields = ('name', 'player_key')
    list_filter = ('status',)

@admin.register(Weeks)
class WeeksAdmin(admin.ModelAdmin):
    list_display = ('league', 'week_num', 'start_date', 'end_date', 'status')
    list_filter = ('league', 'status')

@admin.register(StatCategories)
class StatCategoriesAdmin(admin.ModelAdmin):
    list_display = ('league', 'stat_id', 'abbr', 'name', 'position_type')
    list_filter = ('league', 'position_type')
