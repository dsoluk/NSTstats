# Temporary shim to reference existing Yahoo client location during reorg
try:
    from yahoo.yfs_client import (
        YahooFantasyClient,
        extract_game_key,
        make_league_key,
        make_team_key,
        parse_roster_xml,
        extract_num_teams,
    )  # type: ignore
except Exception:  # pragma: no cover
    YahooFantasyClient = None  # type: ignore
