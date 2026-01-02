import json
import os
import sys
import csv
from typing import Optional
import datetime as _dt
import xml.etree.ElementTree as _ET

from dotenv import load_dotenv

from yahoo.yahoo_auth import get_oauth
from yahoo.yfs_client import (
    YahooFantasyClient,
    extract_game_key,
    make_league_key,
    make_team_key,
    parse_roster_xml,
    parse_settings_xml,
    parse_scoreboard_xml,
    parse_players_week_stats_xml,
    extract_num_teams,
)

# Optional DB persistence (SQLAlchemy). Guarded by env PERSIST_DB.
try:
    from infrastructure.persistence import (
        get_session,
        upsert_league,
        upsert_week,
        upsert_team,
        upsert_matchup,
        upsert_weekly_total,
        upsert_stat_category,
        upsert_weekly_player_gp,
        is_week_closed,
        mark_week_closed,
    )
except Exception:
    # Defer import errors until flag is enabled
    get_session = None  # type: ignore
    upsert_league = upsert_week = upsert_team = upsert_matchup = None  # type: ignore
    upsert_weekly_total = upsert_stat_category = None  # type: ignore
    upsert_weekly_player_gp = is_week_closed = mark_week_closed = None  # type: ignore


class YahooFantasyApp:
    """
    Object-oriented orchestrator for the Yahoo Fantasy helper.

    Responsibilities:
    - Load configuration from environment (.env supported)
    - Initialize OAuth and API client
    - Execute the main workflow (fetch game, league, roster)
    """

    def __init__(self,
                 sport: Optional[str] = None,
                 season: Optional[str] = None,
                 league_id: Optional[str] = None,
                 status: Optional[str] = None,
                 count: Optional[str] = None,
                 position: Optional[str] = None,
                 team_id: Optional[str] = None):
        # Allow explicit params to override environment
        load_dotenv()
        self.sport = sport or self._env("SPORT", "nhl")
        self.season = season or self._env("SEASON", "2025")
        self.league_id = league_id or self._env("LEAGUE_ID")
        self.status = status or self._env("STATUS", "")
        self.count = count or self._env("COUNT", "")
        self.position = position or self._env("POSITION")
        self.team_id = team_id or self._env("TEAM_ID", "4")

        client_id = self._env("YAHOO_CLIENT_ID")
        client_secret = self._env("YAHOO_CLIENT_SECRET")

        try:
            oauth = get_oauth(client_id, client_secret)
        except Exception as e:
            print(f"OAuth error: {e}")
            sys.exit(1)

        self.client = YahooFantasyClient(oauth)

    @staticmethod
    def _env(name: str, default: Optional[str] = None) -> Optional[str]:
        v = os.environ.get(name)
        return v if v is not None and v != "" else default

    @staticmethod
    def _pretty(obj) -> str:
        return json.dumps(obj, indent=2, ensure_ascii=False)

    def run(self) -> None:
        """
        Main execution flow. Decomposed into logical steps for readability.
        """
        print("Starting Yahoo Fantasy NHL helper (OO mode)...")
        
        game_key = self._get_and_verify_game_key()
        if not game_key:
            sys.exit(2)

        if not self.league_id:
            self._list_user_leagues(game_key)
            return

        league_key = make_league_key(game_key, self.league_id)
        league_json = self.client.get_league(league_key)
        
        session, league_row = self._init_db(game_key, league_key)

        # Initialize league context (stat mapping, current week)
        context = self._get_league_context(league_key, session, league_row)
        
        # Execute exports
        self._export_weekly_totals(league_key, context, session, league_row)
        self._export_player_gp(game_key, league_key, context, session, league_row)
        self._export_player_gp_daily(game_key, league_key, context)
        self._export_player_gp_derived(game_key, league_key, context, session, league_row)
        self._export_current_roster(game_key)
        self._export_league_rosters(game_key, league_json)

        if session:
            session.close()

    def _get_and_verify_game_key(self) -> Optional[str]:
        game_json = self.client.get_game(self.sport, self.season)
        game_key = extract_game_key(game_json)
        if not game_key:
            print("\nCould not extract game_key from response.")
        return game_key

    def _init_db(self, game_key: str, league_key: str):
        persist_db = str(self._env("PERSIST_DB", "0")).lower() in {"1", "true", "yes", "y"}
        if not (persist_db and get_session):
            return None, None
        try:
            session = get_session()
            league_row = upsert_league(session, game_key=str(game_key), league_key=str(league_key), season=str(self.season))
            session.commit()
            return session, league_row
        except Exception as e:
            if 'session' in locals() and session: 
                session.rollback()
            print(f"[Warn] DB initialization failed: {e}")
            return None, None

    def _get_league_context(self, league_key: str, session, league_row) -> dict:
        settings_payload = self.client.get_league_settings(league_key)
        raw_xml = settings_payload.get("_raw_xml")
        context = {'stat_map': {}, 'current_week': 9}
        
        if raw_xml:
            settings = parse_settings_xml(raw_xml)
            if settings.get("current_week"):
                context['current_week'] = settings["current_week"]
                print(f"[Diag] Yahoo current_week: {context['current_week']}")
            else:
                print(f"[Diag] No current_week in Yahoo settings; defaulting to 9")
            
            for s in settings.get("stat_categories", []):
                sid = int(s["id"])
                context['stat_map'][sid] = s.get("display") or s.get("name") or str(sid)
                if session and league_row:
                    try:
                        upsert_stat_category(session, league_id=league_row.id, stat_id=sid, 
                                             abbr=s.get("display"), name=s.get("name"), 
                                             position_type=s.get("position_type"))
                    except Exception: pass
            try:
                if session: session.commit()
            except Exception as e:
                if session: session.rollback()
                print(f"[Warn] Failed to commit league context: {e}")
        return context

    def _export_weekly_totals(self, league_key, context, session, league_row):
        if not str(self._env("YAHOO_EXPORT_WEEKLY", "1")).lower() in {"1", "true", "yes", "y"}:
            return
        
        weeks_in_season = int(self._env("WEEKS_IN_SEASON", "24") or 24)
        # Allow override of current week via env
        current_week = int(self._env("YAHOO_CURRENT_WEEK") or context.get('current_week') or 9)
        current_week = max(1, min(current_week, weeks_in_season))
        stat_map = context['stat_map']

        print(f"\nExporting weekly totals for weeks 1..{current_week}...")
        rows = []
        for wk in range(1, int(current_week) + 1):
            try:
                if session and league_row and is_week_closed(session, league_id=league_row.id, week_num=wk):
                    print(f"[Diag] Week {wk} is closed in DB; skipping.")
                    continue

                print(f"[Diag] Fetching scoreboard for week {wk}...")
                sb_payload = self.client.get_league_scoreboard(league_key, wk)
                sb_xml = sb_payload.get("_raw_xml")
                if not sb_xml:
                    print(f"[Diag] No XML returned for week {wk} scoreboard.")
                    continue
                
                parsed = parse_scoreboard_xml(sb_xml, week=wk)
                matchups = parsed.get("matchups", [])
                print(f"[Diag] Week {wk}: parsed {len(matchups)} matchups.")
                
                # Logic for persisting scoreboard data
                for m in matchups:
                    midx = m.get("matchup_index")
                    matchup_id_db = None
                    if session and league_row and midx is not None:
                        matchup = upsert_matchup(session, league_id=league_row.id, week_num=wk, matchup_index=midx)
                        session.flush()
                        matchup_id_db = matchup.id
                    
                    for t in m.get("teams", []):
                        tkey, tname = t.get("team_key"), t.get("team_name")
                        if session and league_row and tkey:
                            upsert_team(session, league_id=league_row.id, team_key=str(tkey), team_name=tname)
                        
                        stats_count = 0
                        for sid, val in t.get("stats", {}).items():
                            rows.append({
                                "week": wk, "matchup_index": midx, "team_key": tkey,
                                "team_name": tname, "stat_id": sid, "stat": stat_map.get(int(sid), str(sid)), "value": val,
                            })
                            if session and league_row and tkey and sid:
                                upsert_weekly_total(session, league_id=league_row.id, week_num=wk, 
                                                    matchup_id=matchup_id_db, team_key=str(tkey), 
                                                    stat_id=int(sid), stat_abbr=stat_map.get(int(sid)), value=val)
                                stats_count += 1
                        print(f"[Diag] Week {wk}, Team {tkey}: saved {stats_count} stats.")
                if session: session.commit()
            except Exception as e:
                if session: session.rollback()
                print(f"[Warn] Failed week {wk} totals: {e}")

        self._write_csv("yahoo_weekly_totals.csv", ["week", "matchup_index", "team_key", "team_name", "stat_id", "stat", "value"], rows)

    def _export_player_gp(self, game_key, league_key, context, session, league_row):
        if not str(self._env("YAHOO_EXPORT_PLYR_GP", "1")).lower() in {"1", "true", "yes", "y"}:
            return
        """Export weekly per-player GP. Skips API if week is already closed in DB."""
        if not str(self._env("YAHOO_EXPORT_PLYR_GP", "1")).lower() in {"1", "true", "yes", "y"}:
            return
        
        weeks_env = (self._env("YAHOO_GP_WEEKS") or "8").strip()
        # Parse week range (e.g. "1-8") or list ("1,2,3")
        weeks = self._parse_weeks(weeks_env)
        team_ids = [str(self.team_id or "4"), str(self._env("OPP_TEAM_ID", "7"))]
        gp_stat_id = context.get('gp_stat_id') # Assume identified in context
        
        gp_rows = []
        for wk in weeks:
            # REDUNDANCY CHECK: Skip if DB says week is closed
            if session and league_row and is_week_closed(session, league_id=league_row.id, week_num=wk):
                print(f"[Info] GP Export: Week {wk} closed; skipping API.")
                continue
                
            for tid in set(team_ids):
                try:
                    tkey = make_team_key(game_key, self.league_id, tid)
                    payload = self.client.get_team_roster_week(tkey, wk)
                    r_xml = payload.get("_raw_xml")
                    if not r_xml: continue
                    
                    roster = parse_roster_xml(r_xml)
                    team_name = roster.get('team_name') or ''
                    
                    # Batch fetch player stats to respect rate limits (25 per call)
                    pkeys = [f"{game_key}.p.{p['player_id']}" for p in roster['players'] if p.get('player_id')]
                    stats_by_key = self._fetch_player_stats_batch(pkeys, wk)

                    for p in roster['players']:
                        pkey = f"{game_key}.p.{p['player_id']}"
                        stats = stats_by_key.get(pkey, {})
                        gp_num = stats.get(gp_stat_id or 0, 0) # Fallback to 0
                        
                        gp_rows.append({
                            'week': wk, 'player_name': p.get('name'), 'team_name': team_name,
                            'positions': ",".join(p.get('positions', [])), 'GP': int(gp_num)
                        })
                        # PERSIST: Update local tables
                        if session and league_row:
                            upsert_weekly_player_gp(session, league_id=league_row.id, week_num=wk, 
                                                    team_key=tkey, player_key=pkey, gp=int(gp_num),
                                                    player_name=p.get('name'))
                    if session: session.commit()
                except Exception as e:
                    if session: session.rollback()
                    print(f"[Warn] GP Export failed for wk {wk} team {tid}: {e}")

        self._write_csv("yahoo_weekly_player_gp.csv", ["week", "player_name", "team_name", "positions", "GP"], gp_rows)

    def _fetch_player_stats_batch(self, pkeys, week) -> dict:
        """Helper to chunk player keys and fetch stats from Yahoo API."""
        stats_map = {}
        for i in range(0, len(pkeys), 25):
            batch = pkeys[i:i + 25]
            try:
                payload = self.client.get_players_week_stats(batch, week)
                if payload.get("_raw_xml"):
                    parsed = parse_players_week_stats_xml(payload["_raw_xml"])
                    for item in parsed.get('players', []):
                        stats_map[item['player_key']] = item.get('stats', {})
            except Exception: continue
        return stats_map

    def _parse_weeks(self, val: str) -> list[int]:
        if '-' in val:
            try:
                lo, hi = map(int, val.split('-', 1))
                return list(range(lo, hi + 1))
            except: return []
        return [int(w.strip()) for w in val.split(',') if w.strip().isdigit()]

    def _export_player_gp_daily(self, game_key, league_key, context):
        if not str(self._env("YAHOO_EXPORT_PLYR_GP_DAILY", "1")).lower() in {"1", "true", "yes", "y"}:
            return
        # (Implementation of Daily GP logic here...)
        pass

    def _export_player_gp_derived(self, game_key, league_key, context, session, league_row):
        if not str(self._env("YAHOO_EXPORT_PLYR_GP_DERIVED", "1")).lower() in {"1", "true", "yes", "y"}:
            return
        # (Implementation of Derived GP logic here...)
        pass

    def _export_current_roster(self, game_key):
        try:
            team_key = make_team_key(game_key, self.league_id, self.team_id)
            payload = self.client.get_team_roster(team_key)
            if payload.get("_source_format") == "xml":
                roster = parse_roster_xml(payload["_raw_xml"])
                out_path = os.path.join("data", f"roster_team{self.team_id}.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(self._pretty(roster))
        except Exception as e:
            print(f"Error exporting roster: {e}")

    def _export_league_rosters(self, game_key, league_json):
        try:
            num_teams = extract_num_teams(league_json) or int(self._env("NUM_TEAMS", "12"))
            print(f"\nBuilding combined roster CSV for {num_teams} teams...")
            rows = []
            for i in range(1, num_teams + 1):
                tkey = make_team_key(game_key, self.league_id, i)
                payload = self.client.get_team_roster(tkey)
                if payload.get("_raw_xml"):
                    r = parse_roster_xml(payload["_raw_xml"])
                    for p in r.get("players", []):
                        rows.append({
                            "team_id": i, "team_key": r.get("team_key") or tkey, "team_name": r.get("team_name"),
                            "player_id": p.get("player_id"), "name": p.get("name"),
                            "positions": ";".join(p.get("positions", [])), "selected_position": p.get("selected_position")
                        })
            self._write_csv("all_rosters.csv", ["team_id", "team_key", "team_name", "player_id", "name", "positions", "selected_position"], rows)
        except Exception as e:
            print(f"Error building league rosters: {e}")

    def _write_csv(self, filename, fieldnames, rows):
        if not rows: return
        out_path = os.path.join("data", filename)
        os.makedirs("data", exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {filename} with {len(rows)} rows.")

    def _list_user_leagues(self, game_key: str):
        leagues_json = self.client.get_user_leagues_for_game(game_key)
        print("\n=== Your Leagues ===")
        print(self._pretty(leagues_json))

# ... existing code ...
