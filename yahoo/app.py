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
        print("Starting Yahoo Fantasy NHL helper (OO mode)...")
        print(
            f"Config: SPORT={self.sport} SEASON={self.season} LEAGUE_ID={self.league_id} "
            f"STATUS={self.status} COUNT={self.count} POSITION={self.position}"
        )

        # 1) Get game object for sport+season
        print(f"Fetching game for sport={self.sport} season={self.season} ...")
        game_json = self.client.get_game(self.sport, self.season)
        src_fmt = game_json.get("_source_format") if isinstance(game_json, dict) else None
        if src_fmt:
            print(f"\n[Info] Response source format: {src_fmt}")
            if src_fmt == "xml":
                print("[Info] Yahoo returned XML. Using BeautifulSoup fallback to extract values.")
        print("\n=== Game Payload ===")
        print(self._pretty(game_json))

        game_key = extract_game_key(game_json)
        if not game_key:
            print("\nCould not extract game_key from response.")
            if src_fmt == "xml":
                print("Hint: Yahoo responded with XML. Ensure the endpoint and parameters are correct. We'll continue improving XML parsing if needed.")
            sys.exit(2)

        # 2) If LEAGUE_ID set, fetch that league. Otherwise, list user's leagues for the game.
        if self.league_id:
            league_key = make_league_key(game_key, self.league_id)
            print(f"\nFetching league {league_key} ...")
            league_json = self.client.get_league(league_key)
            print("\n=== League Payload ===")
            print(self._pretty(league_json))

            # DB bootstrap (optional)
            persist_db = str(self._env("PERSIST_DB", "0")).lower() in {"1", "true", "yes", "y"}
            session = None
            league_row = None
            if persist_db:
                if get_session is None:
                    print("[Warn] PERSIST_DB=1 but persistence layer not available; skipping DB writes.")
                else:
                    try:
                        session = get_session()
                        league_row = upsert_league(session, game_key=str(game_key), league_key=str(league_key), season=str(self.season))
                        session.commit()
                    except Exception as _db_e:
                        print(f"[Warn] Failed to initialize DB persistence: {_db_e}")
                        session = None

            # Optionally export weekly totals across weeks 1..current_week
            try:
                do_export = str(self._env("YAHOO_EXPORT_WEEKLY", "1")).lower() in {"1", "true", "yes", "y"}
                current_week_env = self._env("CURRENT_WEEK")
                weeks_in_season = int(self._env("WEEKS_IN_SEASON", "24") or 24)
                current_week = int(current_week_env) if current_week_env and str(current_week_env).isdigit() else None

                # Fetch settings to map stat_id->display and discover current_week if not provided
                settings_payload = self.client.get_league_settings(league_key)
                raw_xml = settings_payload.get("_raw_xml") if isinstance(settings_payload, dict) else None
                stat_map = {}
                if raw_xml:
                    settings = parse_settings_xml(raw_xml)
                    for s in settings.get("stat_categories", []) or []:
                        stat_map[int(s.get("id"))] = s.get("display") or s.get("name") or str(s.get("id"))
                    if current_week is None:
                        cw = settings.get("current_week")
                        if isinstance(cw, int) and cw > 0:
                            current_week = cw
                if current_week is None:
                    # Fallback to provided season week
                    current_week = 9
                # Clamp to season bounds
                if weeks_in_season and isinstance(weeks_in_season, int):
                    current_week = max(1, min(current_week, weeks_in_season))

                # Persist stat categories once (from settings)
                if raw_xml and session is not None and league_row is not None:
                    try:
                        settings = parse_settings_xml(raw_xml)
                        for s in settings.get("stat_categories", []) or []:
                            try:
                                sid = int(s.get("id"))
                            except Exception:
                                continue
                            abbr = s.get("display") or None
                            name = s.get("name") or None
                            pos = s.get("position_type") or None
                            upsert_stat_category(session, league_id=league_row.id, stat_id=sid, abbr=abbr, name=name, position_type=pos)
                        session.commit()
                    except Exception as _sc:
                        print(f"[Warn] Failed to persist stat categories: {_sc}")

                if do_export:
                    print(f"\nExporting weekly totals for weeks 1..{current_week} (of {weeks_in_season}) ...")
                    rows = []
                    for wk in range(1, int(current_week) + 1):
                        try:
                            # If DB says week is closed, skip any API calls for that week
                            if session is not None and league_row is not None:
                                try:
                                    if is_week_closed(session, league_id=league_row.id, week_num=int(wk)):
                                        print(f"[Info] Week {wk} is closed in DB; skipping scoreboard fetch.")
                                        continue
                                except Exception:
                                    pass
                            sb_payload = self.client.get_league_scoreboard(league_key, wk)
                            sb_xml = sb_payload.get("_raw_xml") if isinstance(sb_payload, dict) else None
                            if not sb_xml:
                                print(f"[Warn] No XML for scoreboard week {wk}; skipping.")
                                continue
                            parsed = parse_scoreboard_xml(sb_xml, week=wk)
                            # Upsert week start/end and stat categories if persistence enabled
                            if session is not None and league_row is not None:
                                try:
                                    # Resolve dates
                                    try:
                                        root = _ET.fromstring(sb_xml)
                                        ws = we = None
                                        for elem in root.iter():
                                            tag = elem.tag.split('}', 1)[-1]
                                            if tag == 'week_start' and elem.text:
                                                ws = elem.text.strip()
                                            elif tag == 'week_end' and elem.text:
                                                we = elem.text.strip()
                                        d_ws = _dt.date.fromisoformat(ws) if ws else None
                                        d_we = _dt.date.fromisoformat(we) if we else None
                                    except Exception:
                                        d_ws = d_we = None
                                    upsert_week(session, league_id=league_row.id, week_num=int(wk), start_date=d_ws, end_date=d_we)
                                    session.commit()
                                except Exception as _werr:
                                    print(f"[Warn] Failed to upsert week {wk}: {_werr}")
                            for m in parsed.get("matchups", []) or []:
                                midx = m.get("matchup_index")
                                matchup_id_db = None
                                if session is not None and league_row is not None and midx is not None:
                                    try:
                                        matchup = upsert_matchup(session, league_id=league_row.id, week_num=int(wk), matchup_index=int(midx))
                                        session.flush()
                                        matchup_id_db = matchup.id
                                    except Exception as _mu:
                                        print(f"[Warn] Failed to upsert matchup L={league_key} W={wk} I={midx}: {_mu}")
                                for t in m.get("teams", []) or []:
                                    tkey = t.get("team_key")
                                    tname = t.get("team_name")
                                    stats = t.get("stats", {}) or {}
                                    if session is not None and league_row is not None and tkey:
                                        try:
                                            upsert_team(session, league_id=league_row.id, team_key=str(tkey), team_name=tname)
                                            session.flush()
                                        except Exception as _tu:
                                            print(f"[Warn] Failed to upsert team {tkey}: {_tu}")
                                    for sid, val in stats.items():
                                        rows.append({
                                            "week": wk,
                                            "matchup_index": midx,
                                            "team_key": tkey,
                                            "team_name": tname,
                                            "stat_id": sid,
                                            "stat": stat_map.get(int(sid), str(sid)),
                                            "value": val,
                                        })
                                        if session is not None and league_row is not None and tkey is not None and sid is not None:
                                            try:
                                                upsert_weekly_total(
                                                    session,
                                                    league_id=league_row.id,
                                                    week_num=int(wk),
                                                    matchup_id=matchup_id_db,
                                                    team_key=str(tkey),
                                                    stat_id=int(sid),
                                                    stat_abbr=stat_map.get(int(sid), str(sid)),
                                                    value=float(val) if isinstance(val, (int, float, str)) and str(val).strip() != '' else None,
                                                )
                                            except Exception as _wt:
                                                print(f"[Warn] Failed to upsert weekly total for team {tkey} stat {sid}: {_wt}")
                            if session is not None:
                                try:
                                    session.commit()
                                except Exception as _c:
                                    print(f"[Warn] Commit failed for week {wk} totals: {_c}")
                        except Exception as _we:
                            print(f"[Warn] Failed week {wk} scoreboard parse: {_we}")
                            continue
                    # Write CSV
                    out_dir = os.path.join("data")
                    os.makedirs(out_dir, exist_ok=True)
                    out_path = os.path.join(out_dir, "yahoo_weekly_totals.csv")
                    import csv as _csv
                    fieldnames = ["week", "matchup_index", "team_key", "team_name", "stat_id", "stat", "value"]
                    with open(out_path, "w", newline="", encoding="utf-8") as f:
                        writer = _csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        for r in rows:
                            writer.writerow(r)
                    print(f"Saved weekly totals CSV to {out_path} with {len(rows)} rows.")
            except Exception as e:
                print(f"[Warn] Weekly export skipped due to error: {e}")

            # Export weekly per-player GP for selected teams and weeks (minimal API by default)
            try:
                do_gp = str(self._env("YAHOO_EXPORT_PLYR_GP", "1")).lower() in {"1", "true", "yes", "y"}
                if do_gp:
                    # Discover current week and stat map (re-use if already fetched)
                    if 'settings_payload' not in locals():
                        settings_payload = self.client.get_league_settings(league_key)
                    settings_xml = settings_payload.get("_raw_xml") if isinstance(settings_payload, dict) else None
                    stat_map = {}
                    current_week = None
                    if settings_xml:
                        settings = parse_settings_xml(settings_xml)
                        for s in settings.get("stat_categories", []) or []:
                            sid = int(s.get("id"))
                            stat_map[sid] = s.get("display") or s.get("name") or str(sid)
                        cw = settings.get("current_week")
                        if isinstance(cw, int) and cw > 0:
                            current_week = cw
                    if current_week is None:
                        current_week_env = self._env("CURRENT_WEEK")
                        current_week = int(current_week_env) if current_week_env and str(current_week_env).isdigit() else 9

                    # Determine GP stat_id (prefer settings match on display==GP)
                    gp_stat_id = None
                    try:
                        for sid, disp in stat_map.items():
                            d = str(disp or '').strip().lower()
                            if d == 'gp' or 'games played' in d:
                                gp_stat_id = int(sid)
                                break
                    except Exception:
                        gp_stat_id = None

                    # Weeks to pull: env override supports comma or range (e.g., "1-8"), default minimal '8'
                    weeks_env = (self._env("YAHOO_GP_WEEKS") or "8").strip()
                    weeks: list[int] = []
                    if '-' in weeks_env:
                        a, b = weeks_env.split('-', 1)
                        try:
                            lo = int(a.strip()); hi = int(b.strip())
                            if lo <= hi:
                                weeks = list(range(lo, hi + 1))
                        except Exception:
                            pass
                    if not weeks:
                        try:
                            weeks = [int(w.strip()) for w in weeks_env.split(',') if w.strip().isdigit()]
                        except Exception:
                            weeks = [8]
                    # Clamp to current_week - 1 by default
                    weeks_in_season = int(self._env("WEEKS_IN_SEASON", "24") or 24)
                    def _clamp_week(w: int) -> int:
                        return max(1, min(w, max(1, weeks_in_season)))
                    weeks = sorted({_clamp_week(w) for w in weeks if w >= 1})

                    # Teams: self.team_id and optional opponent
                    opp_team_id = self._env("OPP_TEAM_ID", "7")
                    team_ids = [str(self.team_id or "4")]
                    if opp_team_id and str(opp_team_id) not in team_ids:
                        team_ids.append(str(opp_team_id))

                    # Helper to chunk lists
                    def _chunks(lst, n):
                        for i in range(0, len(lst), n):
                            yield lst[i:i + n]

                    # Prepare rows
                    gp_rows = []
                    for wk in weeks:
                        # If DB says week is closed, skip without API calls
                        if session is not None and league_row is not None:
                            try:
                                if is_week_closed(session, league_id=league_row.id, week_num=int(wk)):
                                    print(f"[Info] Week {wk} closed; skipping GP weekly API.")
                                    continue
                            except Exception:
                                pass
                        for tid in team_ids:
                            try:
                                tkey = make_team_key(game_key, self.league_id, tid)
                                roster_payload = self.client.get_team_roster_week(tkey, wk)
                                r_xml = roster_payload.get("_raw_xml") if isinstance(roster_payload, dict) else None
                                if not r_xml:
                                    print(f"[Warn] No roster XML for week {wk} team {tid}; skipping.")
                                    continue
                                roster = parse_roster_xml(r_xml)
                                team_name = roster.get('team_name') or roster.get('name') or ''
                                players = roster.get('players', []) or []
                                # Build player_keys for this game
                                pkey_to_meta = {}
                                pkeys = []
                                for p in players:
                                    pid = str(p.get('player_id') or '').strip()
                                    if not pid.isdigit():
                                        continue
                                    pkey = f"{game_key}.p.{pid}"
                                    pkeys.append(pkey)
                                    positions = p.get('positions') or []
                                    pos_str = ",".join([str(x) for x in positions]) if isinstance(positions, (list, tuple)) else str(positions or '')
                                    pname = p.get('name') or p.get('name_full') or ''
                                    pkey_to_meta[pkey] = {'name': pname, 'positions': pos_str}
                                if not pkeys:
                                    continue
                                # Batch query weekly player stats
                                stats_by_key = {}
                                for batch in _chunks(pkeys, 25):
                                    try:
                                        pw_payload = self.client.get_players_week_stats(batch, wk)
                                        pw_xml = pw_payload.get("_raw_xml") if isinstance(pw_payload, dict) else None
                                        if not pw_xml:
                                            continue
                                        parsed = parse_players_week_stats_xml(pw_xml)
                                        for item in parsed.get('players', []) or []:
                                            stats_by_key[item.get('player_key')] = item.get('stats', {}) or {}
                                    except Exception as _pe:
                                        print(f"[Warn] player stats batch failed wk={wk} team={tid}: {_pe}")
                                        continue
                                # Emit rows with GP
                                for pkey in pkeys:
                                    meta = pkey_to_meta.get(pkey, {})
                                    stats = stats_by_key.get(pkey, {})
                                    gp_val = None
                                    if gp_stat_id is not None and gp_stat_id in stats:
                                        gp_val = stats.get(gp_stat_id)
                                    else:
                                        # Fallback common id 0
                                        gp_val = stats.get(0)
                                    try:
                                        gp_num = int(gp_val) if gp_val is not None and str(gp_val).strip() != '' else 0
                                    except Exception:
                                        # If Yahoo returns floats/strings, coerce safely
                                        try:
                                            gp_num = int(float(str(gp_val)))
                                        except Exception:
                                            gp_num = 0
                                    gp_rows.append({
                                        'week': wk,
                                        'player_name': meta.get('name', ''),
                                        'team_name': team_name,
                                        'positions': meta.get('positions', ''),
                                        'GP': gp_num,
                                    })
                                    # Optionally persist GP per player
                                    if session is not None and league_row is not None:
                                        try:
                                            upsert_team(session, league_id=league_row.id, team_key=str(tkey), team_name=team_name)
                                            # build player_key
                                            pid = str(pkey).split('.p.')[-1]
                                            player_key = f"{game_key}.p.{pid}"
                                            upsert_weekly_player_gp(
                                                session,
                                                league_id=league_row.id,
                                                week_num=int(wk),
                                                team_key=str(tkey),
                                                player_key=player_key,
                                                gp=int(gp_num),
                                                source="weekly_api",
                                                player_name=meta.get('name', ''),
                                                positions=meta.get('positions', ''),
                                            )
                                        except Exception as _pgp:
                                            print(f"[Warn] Failed to upsert weekly GP for {meta.get('name','')}: {_pgp}")
                            except Exception as _ge:
                                print(f"[Warn] GP export failed for week {wk} team {tid}: {_ge}")
                                continue

                    # Write CSV
                    if gp_rows:
                        out_dir = os.path.join("data")
                        os.makedirs(out_dir, exist_ok=True)
                        out_path = os.path.join(out_dir, "yahoo_weekly_player_gp.csv")
                        fieldnames = ["week", "player_name", "team_name", "positions", "GP"]
                        with open(out_path, "w", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()
                            for r in gp_rows:
                                writer.writerow(r)
                        print(f"Saved weekly player GP CSV to {out_path} with {len(gp_rows)} rows.")
                    else:
                        print("[Info] No GP rows produced.")
            except Exception as e:
                print(f"[Warn] Player GP export skipped due to error: {e}")

            # Daily GP verification exporter (date-based, counts only active slots with an actual game)
            try:
                do_daily = str(self._env("YAHOO_EXPORT_PLYR_GP_DAILY", "1")).lower() in {"1", "true", "yes", "y"}
                if do_daily:
                    # Discover stat map and GP stat id from settings
                    if 'settings_payload' not in locals():
                        settings_payload = self.client.get_league_settings(league_key)
                    settings_xml = settings_payload.get("_raw_xml") if isinstance(settings_payload, dict) else None
                    stat_map = {}
                    if settings_xml:
                        settings = parse_settings_xml(settings_xml)
                        for s in settings.get("stat_categories", []) or []:
                            sid = int(s.get("id"))
                            stat_map[sid] = s.get("display") or s.get("name") or str(sid)
                    gp_stat_id = None
                    for sid, disp in stat_map.items():
                        d = str(disp or '').strip().lower()
                        if d == 'gp' or 'games played' in d:
                            gp_stat_id = int(sid)
                            break

                    # Dates to pull (comma-separated). Default to the Saturday of Week 8 in 2025-26: 2025-11-29
                    dates_env = (self._env("YAHOO_DATES") or "2025-11-29").strip()
                    dates = [d.strip() for d in dates_env.split(',') if d.strip()]

                    # Team scope: only your team (TEAM_ID)
                    team_ids = [str(self.team_id or "4")]

                    def _chunks(lst, n):
                        for i in range(0, len(lst), n):
                            yield lst[i:i + n]

                    daily_rows = []
                    inactive_slots = {"BN", "IR", "IR+"}
                    for date_iso in dates:
                        for tid in team_ids:
                            try:
                                tkey = make_team_key(game_key, self.league_id, tid)
                                roster_payload = self.client.get_team_roster_date(tkey, date_iso)
                                r_xml = roster_payload.get("_raw_xml") if isinstance(roster_payload, dict) else None
                                if not r_xml:
                                    print(f"[Warn] No roster XML for date {date_iso} team {tid}; skipping.")
                                    continue
                                roster = parse_roster_xml(r_xml)
                                team_name = roster.get('team_name') or roster.get('name') or ''
                                players = roster.get('players', []) or []
                                # Build player keys
                                pkey_to_meta = {}
                                pkeys = []
                                for p in players:
                                    pid = str(p.get('player_id') or '').strip()
                                    if not pid.isdigit():
                                        continue
                                    pkey = f"{game_key}.p.{pid}"
                                    pkeys.append(pkey)
                                    positions = p.get('positions') or []
                                    pos_str = ",".join([str(x) for x in positions]) if isinstance(positions, (list, tuple)) else str(positions or '')
                                    pname = p.get('name') or p.get('name_full') or ''
                                    sel = p.get('selected_position') or ''
                                    pkey_to_meta[pkey] = {'name': pname, 'positions': pos_str, 'selected_position': sel}
                                if not pkeys:
                                    continue
                                # Fetch per-date player stats in batches
                                stats_by_key = {}
                                for batch in _chunks(pkeys, 25):
                                    try:
                                        pd_payload = self.client.get_players_date_stats(batch, date_iso)
                                        pd_xml = pd_payload.get("_raw_xml") if isinstance(pd_payload, dict) else None
                                        if not pd_xml:
                                            continue
                                        parsed = parse_players_week_stats_xml(pd_xml)
                                        for item in parsed.get('players', []) or []:
                                            stats_by_key[item.get('player_key')] = item.get('stats', {}) or {}
                                    except Exception as _pe:
                                        print(f"[Warn] player date stats batch failed date={date_iso} team={tid}: {_pe}")
                                        continue

                                # Emit rows with daily GP determination
                                for pkey in pkeys:
                                    meta = pkey_to_meta.get(pkey, {})
                                    stats = stats_by_key.get(pkey, {}) or {}
                                    # Determine if had a game
                                    gp_val = None
                                    if gp_stat_id is not None and gp_stat_id in stats:
                                        gp_val = stats.get(gp_stat_id)
                                    elif 0 in stats:
                                        gp_val = stats.get(0)
                                    # Fallback heuristic: consider had_game if any numeric stat > 0
                                    had_game = False
                                    if gp_val is not None:
                                        try:
                                            had_game = int(float(str(gp_val))) > 0
                                        except Exception:
                                            had_game = False
                                    else:
                                        # heuristic: any stat value numeric and >0
                                        for v in stats.values():
                                            try:
                                                if float(str(v)) > 0:
                                                    had_game = True
                                                    break
                                            except Exception:
                                                continue

                                    sel_pos = (meta.get('selected_position') or '').strip().upper()
                                    active_slot = (sel_pos != '' and sel_pos not in inactive_slots)
                                    gp_num = 1 if (active_slot and had_game) else 0

                                    daily_rows.append({
                                        'date': date_iso,
                                        'player_name': meta.get('name', ''),
                                        'team_name': team_name,
                                        'positions': meta.get('positions', ''),
                                        'selected_position': sel_pos,
                                        'had_game': bool(had_game),
                                        'GP': gp_num,
                                    })
                            except Exception as _de:
                                print(f"[Warn] Daily GP export failed for date {date_iso} team {tid}: {_de}")
                                continue

                    # Write daily CSV
                    if daily_rows:
                        out_dir = os.path.join("data")
                        os.makedirs(out_dir, exist_ok=True)
                        out_path = os.path.join(out_dir, "yahoo_daily_player_gp.csv")
                        fieldnames = ["date", "player_name", "team_name", "positions", "selected_position", "had_game", "GP"]
                        with open(out_path, "w", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()
                            for r in daily_rows:
                                writer.writerow(r)
                        print(f"Saved daily player GP CSV to {out_path} with {len(daily_rows)} rows.")
                    else:
                        print("[Info] No daily GP rows produced.")
            except Exception as e:
                print(f"[Warn] Daily player GP export skipped due to error: {e}")

            # Derived weekly GP aggregation (sum of eligible daily GP across all dates in each requested week)
            try:
                do_weekly_derived = str(self._env("YAHOO_EXPORT_PLYR_GP_DERIVED", "1")).lower() in {"1", "true", "yes", "y"}
                if do_weekly_derived:
                    # Determine weeks to compute from env; default to week 8 only
                    weeks_env = (self._env("YAHOO_GP_WEEKS") or "8").strip()
                    weeks: list[int] = []
                    if '-' in weeks_env:
                        try:
                            a, b = weeks_env.split('-', 1)
                            lo = int(a.strip()); hi = int(b.strip())
                            if lo <= hi:
                                weeks = list(range(lo, hi + 1))
                        except Exception:
                            weeks = []
                    if not weeks:
                        try:
                            weeks = [int(w.strip()) for w in weeks_env.split(',') if w.strip().isdigit()]
                        except Exception:
                            weeks = [8]

                    weeks_in_season = int(self._env("WEEKS_IN_SEASON", "24") or 24)
                    weeks = sorted({max(1, min(int(w), weeks_in_season)) for w in weeks})

                    # Scope: only your team per request
                    team_ids = [str(self.team_id or "4")]

                    # Resolve GP stat id from settings if available
                    if 'settings_payload' not in locals():
                        settings_payload = self.client.get_league_settings(league_key)
                    settings_xml = settings_payload.get("_raw_xml") if isinstance(settings_payload, dict) else None
                    gp_stat_id = None
                    if settings_xml:
                        settings = parse_settings_xml(settings_xml)
                        for s in settings.get("stat_categories", []) or []:
                            try:
                                sid = int(s.get("id"))
                                disp = s.get("display") or s.get("name") or str(sid)
                                d = str(disp).strip().lower()
                                if gp_stat_id is None and (d == 'gp' or 'games played' in d):
                                    gp_stat_id = sid
                            except Exception:
                                continue

                    inactive_slots = {"BN", "IR", "IR+"}

                    def _daterange(start: _dt.date, end: _dt.date):
                        cur = start
                        while cur <= end:
                            yield cur
                            cur = cur + _dt.timedelta(days=1)

                    def _find_week_dates(_league_key: str, _week: int) -> tuple[_dt.date, _dt.date] | tuple[None, None]:
                        try:
                            sb = self.client.get_league_scoreboard(_league_key, _week)
                            sb_xml = sb.get("_raw_xml") if isinstance(sb, dict) else None
                            if not sb_xml:
                                return None, None
                            root = _ET.fromstring(sb_xml)
                            ws = None; we = None
                            for elem in root.iter():
                                tag = elem.tag.split('}', 1)[-1] if isinstance(elem.tag, str) else elem.tag
                                if tag == 'week_start' and elem.text:
                                    ws = elem.text.strip()
                                elif tag == 'week_end' and elem.text:
                                    we = elem.text.strip()
                            if ws and we:
                                try:
                                    ds = _dt.date.fromisoformat(ws)
                                    de = _dt.date.fromisoformat(we)
                                    return ds, de
                                except Exception:
                                    return None, None
                            return None, None
                        except Exception:
                            return None, None

                    def _chunks(lst, n):
                        for i in range(0, len(lst), n):
                            yield lst[i:i + n]

                    agg_rows = []
                    for wk in weeks:
                        # Skip if week closed
                        if session is not None and league_row is not None:
                            try:
                                if is_week_closed(session, league_id=league_row.id, week_num=int(wk)):
                                    print(f"[Info] Week {wk} closed; skipping derived GP.")
                                    continue
                            except Exception:
                                pass
                        start_date, end_date = _find_week_dates(league_key, wk)
                        if not start_date or not end_date:
                            print(f"[Warn] Could not resolve date range for week {wk}; skipping derived GP.")
                            continue
                        for tid in team_ids:
                            try:
                                tkey = make_team_key(game_key, self.league_id, tid)
                                per_player = {}
                                team_name_cache = ""
                                positions_cache = {}
                                for d in _daterange(start_date, end_date):
                                    date_iso = d.isoformat()
                                    roster_payload = self.client.get_team_roster_date(tkey, date_iso)
                                    r_xml = roster_payload.get("_raw_xml") if isinstance(roster_payload, dict) else None
                                    if not r_xml:
                                        continue
                                    roster = parse_roster_xml(r_xml)
                                    team_name_cache = roster.get('team_name') or roster.get('name') or team_name_cache
                                    players = roster.get('players', []) or []
                                    pkeys = []
                                    meta_by_key = {}
                                    for p in players:
                                        pid = str(p.get('player_id') or '').strip()
                                        if not pid.isdigit():
                                            continue
                                        pkey = f"{game_key}.p.{pid}"
                                        pkeys.append(pkey)
                                        positions = p.get('positions') or []
                                        pos_str = ",".join([str(x) for x in positions]) if isinstance(positions, (list, tuple)) else str(positions or '')
                                        pname = p.get('name') or p.get('name_full') or ''
                                        sel = (p.get('selected_position') or '').strip().upper()
                                        meta_by_key[pkey] = {'name': pname, 'positions': pos_str, 'selected_position': sel}
                                        if pkey not in positions_cache:
                                            positions_cache[pkey] = pos_str
                                    if not pkeys:
                                        continue
                                    stats_by_key = {}
                                    for batch in _chunks(pkeys, 25):
                                        try:
                                            pd_payload = self.client.get_players_date_stats(batch, date_iso)
                                            pd_xml = pd_payload.get("_raw_xml") if isinstance(pd_payload, dict) else None
                                            if not pd_xml:
                                                continue
                                            parsed = parse_players_week_stats_xml(pd_xml)
                                            for item in parsed.get('players', []) or []:
                                                stats_by_key[item.get('player_key')] = item.get('stats', {}) or {}
                                        except Exception:
                                            continue
                                    for pkey in pkeys:
                                        meta = meta_by_key.get(pkey, {})
                                        sel_pos = meta.get('selected_position', '')
                                        active_slot = (sel_pos != '' and sel_pos not in inactive_slots)
                                        stats = stats_by_key.get(pkey, {}) or {}
                                        gp_val = None
                                        if gp_stat_id is not None and gp_stat_id in stats:
                                            gp_val = stats.get(gp_stat_id)
                                        elif 0 in stats:
                                            gp_val = stats.get(0)
                                        had_game = False
                                        if gp_val is not None:
                                            try:
                                                had_game = int(float(str(gp_val))) > 0
                                            except Exception:
                                                had_game = False
                                        else:
                                            for v in stats.values():
                                                try:
                                                    if float(str(v)) > 0:
                                                        had_game = True
                                                        break
                                                except Exception:
                                                    continue
                                        gp_num = 1 if (active_slot and had_game) else 0
                                        if pkey not in per_player:
                                            per_player[pkey] = {'name': meta.get('name', ''), 'gp': 0}
                                        per_player[pkey]['gp'] += gp_num
                                for pkey, info in per_player.items():
                                    # Persist per player GP
                                    if session is not None and league_row is not None:
                                        try:
                                            upsert_team(session, league_id=league_row.id, team_key=str(tkey), team_name=team_name_cache)
                                            pid = str(pkey).split('.p.')[-1]
                                            player_key = f"{game_key}.p.{pid}"
                                            upsert_weekly_player_gp(
                                                session,
                                                league_id=league_row.id,
                                                week_num=int(wk),
                                                team_key=str(tkey),
                                                player_key=player_key,
                                                gp=int(info.get('gp') or 0),
                                                source="derived_daily",
                                                player_name=info.get('name', ''),
                                                positions=positions_cache.get(pkey, ''),
                                            )
                                        except Exception as _dgp:
                                            print(f"[Warn] Failed to upsert derived GP for {info.get('name','')}: {_dgp}")
                                    agg_rows.append({
                                        'week': wk,
                                        'player_name': info.get('name', ''),
                                        'team_name': team_name_cache,
                                        'positions': positions_cache.get(pkey, ''),
                                        'GP': int(info.get('gp') or 0),
                                    })
                                if session is not None:
                                    try:
                                        session.commit()
                                    except Exception as _c2:
                                        print(f"[Warn] Commit failed for derived GP week {wk}: {_c2}")
                            except Exception as _we:
                                print(f"[Warn] Derived weekly GP aggregation failed for week {wk} team {tid}: {_we}")
                                continue
                        # After persisting both totals and GP for this week, mark closed
                        if session is not None and league_row is not None:
                            try:
                                mark_week_closed(session, league_id=league_row.id, week_num=int(wk))
                                session.commit()
                                print(f"[Info] Marked week {wk} as closed.")
                            except Exception as _mc:
                                print(f"[Warn] Failed to mark week {wk} closed: {_mc}")

                    if agg_rows:
                        out_dir = os.path.join("data")
                        os.makedirs(out_dir, exist_ok=True)
                        out_path = os.path.join(out_dir, "yahoo_weekly_player_gp.csv")
                        fieldnames = ["week", "player_name", "team_name", "positions", "GP"]
                        with open(out_path, "w", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()
                            for r in agg_rows:
                                writer.writerow(r)
                        print(f"Saved derived weekly player GP CSV to {out_path} with {len(agg_rows)} rows.")
                    else:
                        print("[Info] No derived weekly GP rows produced.")
            except Exception as e:
                print(f"[Warn] Derived weekly GP export skipped due to error: {e}")

            # Fetch roster for a specific team (retain existing behavior)
            try:
                team_key = make_team_key(game_key, self.league_id, self.team_id)
                print(f"\nFetching roster for team {team_key} ...")
                roster_payload = self.client.get_team_roster(team_key)
                src_fmt_roster = roster_payload.get("_source_format") if isinstance(roster_payload, dict) else None
                raw_xml = roster_payload.get("_raw_xml") if isinstance(roster_payload, dict) else None
                if src_fmt_roster == "xml" and raw_xml:
                    roster = parse_roster_xml(raw_xml)
                    print(f"Roster players count: {roster.get('count')}\nFirst 3 players: {self._pretty(roster.get('players', [])[:3])}")
                    # Save roster JSON to data/
                    out_dir = os.path.join("data")
                    os.makedirs(out_dir, exist_ok=True)
                    out_path = os.path.join(out_dir, f"roster_team{self.team_id}.json")
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(self._pretty(roster))
                    print(f"Saved roster to {out_path}")
                else:
                    print("Roster response was not XML as expected; printing raw payload for inspection.")
                    print(self._pretty(roster_payload))
            except Exception as e:
                print(f"Error fetching/parsing roster: {e}")

            # Build a single CSV across all teams in the league
            try:
                num_teams = extract_num_teams(league_json)
                # Allow override via env NUM_TEAMS if parsing fails
                if not num_teams:
                    env_nt = self._env("NUM_TEAMS")
                    if env_nt and str(env_nt).isdigit():
                        num_teams = int(env_nt)
                if not num_teams:
                    num_teams = 12  # sensible default per request
                print(f"\nBuilding combined roster CSV for {num_teams} teams...")
                rows = []
                for i in range(1, int(num_teams) + 1):
                    try:
                        tkey = make_team_key(game_key, self.league_id, str(i))
                        print(f"Fetching roster for team_id={i} ({tkey}) ...")
                        payload = self.client.get_team_roster(tkey)
                        if isinstance(payload, dict) and payload.get("_source_format") == "xml" and payload.get("_raw_xml"):
                            r = parse_roster_xml(payload["_raw_xml"])
                            team_name = r.get("team_name")
                            team_key = r.get("team_key") or tkey
                            for p in r.get("players", []) or []:
                                positions = p.get("positions") or []
                                rows.append({
                                    "team_id": i,
                                    "team_key": team_key,
                                    "team_name": team_name,
                                    "player_id": p.get("player_id"),
                                    "name": p.get("name"),
                                    "positions": ";".join(positions) if isinstance(positions, list) else positions,
                                    "selected_position": p.get("selected_position"),
                                })
                        else:
                            print(f"[Warn] Unexpected roster payload for team {i}; skipping or inspect manually.")
                    except Exception as inner_e:
                        print(f"[Warn] Failed to fetch/parse roster for team {i}: {inner_e}")
                        continue
                # Write CSV
                out_dir = os.path.join("data")
                os.makedirs(out_dir, exist_ok=True)
                csv_path = os.path.join(out_dir, "all_rosters.csv")
                fieldnames = ["team_id", "team_key", "team_name", "player_id", "name", "positions", "selected_position"]
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in rows:
                        writer.writerow(row)
                print(f"Saved combined roster CSV to {csv_path} with {len(rows)} rows.")
            except Exception as e:
                print(f"Error building combined roster CSV: {e}")
        else:
            print("\nLEAGUE_ID not set. Listing your leagues for this game so you can pick one...")
            leagues_json = self.client.get_user_leagues_for_game(game_key)
            print("\n=== Your Leagues for this game ===")
            print(self._pretty(leagues_json))
            print("\nTip: Find your league_id in the printed JSON and set LEAGUE_ID in .env, then re-run.")
