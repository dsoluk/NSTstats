import os
import pandas as pd
import numpy as np
from typing import Optional
from config import load_default_params
from pipelines import StatPipelineFactory, PlayerIndexPipeline, GoalieIndexPipeline, FantasyPointsPipeline
from infrastructure.persistence import get_session, League, StatCategory, Player
from yahoo.yfs_client import parse_players_stats_xml, parse_players_xml

# Registry removed: using local normalization; keep placeholders for backward compatibility
RegistryFactory = None
SummaryReporter = None
export_players_csv = None
NSTPlayerAdapter = None

# Import Yahoo app separately so a registry failure doesn't break Yahoo
try:
    from yahoo.app import YahooFantasyApp
except Exception:
    YahooFantasyApp = None

# Import merger lazily to avoid hard dependency
try:
    from app.merger import run_merge as _run_merge_impl
except Exception:
    _run_merge_impl = None


def run_yahoo(league_id: Optional[str] = None):
    # If YahooFantasyApp failed to import, provide a clear message and return
    if YahooFantasyApp is None:
        print("Yahoo data source skipped: YahooFantasyApp could not be imported (check yahoo/app.py and dependencies).")
        return
    try:
        app = YahooFantasyApp()
        app.run(league_id=league_id)
    except Exception as err:
        print(f"Yahoo data source skipped due to error: {err}")


def fetch_goalies_from_yahoo(league_key: str):
    """Fetch all goalies and their season/lastweek stats from Yahoo, replacing NST."""
    if YahooFantasyApp is None:
        print("[Error] YahooFantasyApp not available for goalie fetch.")
        return pd.DataFrame()
    
    from yahoo.yfs_client import parse_players_xml, parse_players_stats_xml
    
    try:
        app = YahooFantasyApp()
        client = app.client
        
        print(f"Fetching all goalies from Yahoo league {league_key}...")
        all_goalies = []
        start_idx = 0
        while True:
            payload = client.get_league_players(league_key, position="G", count=25, start=start_idx)
            raw_xml = payload.get("_raw_xml")
            if not raw_xml:
                break
                
            goalies = parse_players_xml(raw_xml)
            if not goalies:
                break
                
            all_goalies.extend(goalies)
            if len(goalies) < 25:
                break
            start_idx += 25
            if start_idx >= 125: # Safety cap
                break

        if not all_goalies:
            print("[Warn] No goalies found in Yahoo league.")
            return pd.DataFrame()
            
        pkeys = [g['player_key'] for g in all_goalies if g['player_key']]
        print(f"Fetching stats for {len(pkeys)} goalies...")
        
        season_stats = {}
        lastweek_stats = {}
        
        for i in range(0, len(pkeys), 25):
            batch = pkeys[i:i+25]
            # Season
            s_payload = client.get_players_season_stats(batch)
            if s_payload.get("_raw_xml"):
                s_parsed = parse_players_stats_xml(s_payload["_raw_xml"])
                for p in s_parsed.get('players', []):
                    season_stats[p['player_key']] = p.get('stats', {})
            # Last Week
            lw_payload = client.get_players_lastweek_stats(batch)
            if lw_payload.get("_raw_xml"):
                lw_parsed = parse_players_stats_xml(lw_payload["_raw_xml"])
                for p in lw_parsed.get('players', []):
                    lastweek_stats[p['player_key']] = p.get('stats', {})

        # Mapping Yahoo ID -> base column name
        id_map = {
            18: 'GP', 19: 'W', 20: 'L', 21: 'OT', 22: 'GA',
            23: 'GAA', 24: 'SA', 25: 'SV', 26: 'SV%', 27: 'SHO', 28: 'TOI'
        }
        
        rows = []
        for g in all_goalies:
            pkey = g['player_key']
            s_data = season_stats.get(pkey, {})
            lw_data = lastweek_stats.get(pkey, {})
            
            row = {'Player': g['name'], 'Team': g['team'], 'Position': 'G', 'player_key': pkey}
            for sid, col in id_map.items():
                row[f'szn_{col}'] = s_data.get(sid)
            for sid, col in id_map.items():
                row[f'l7_{col}'] = lw_data.get(sid)
            rows.append(row)
            
        df = pd.DataFrame(rows)
        
        def convert_toi(val):
            if not val or not isinstance(val, str): return val
            clean = val.replace(',', '')
            if ':' in clean:
                parts = clean.split(':')
                if len(parts) == 2:
                    try:
                        m = int(parts[0]); s = int(parts[1])
                        return f"{m // 60:02d}:{m % 60:02d}:{s:02d}"
                    except: pass
            return clean

        if not df.empty:
            for prefix in ['szn_', 'l7_']:
                col = f'{prefix}TOI'
                if col in df.columns:
                    df[col] = df[col].apply(convert_toi)
                    
            # Ensure numeric types
            for col in df.columns:
                if col not in ['Player', 'Team', 'Position'] and 'TOI' not in col:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
        return df
        
    except Exception as e:
        print(f"[Error] fetch_goalies_from_yahoo failed: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def run_nst(*, refresh_prior: bool = False, skip_prior: bool = False):
    try:
        from helpers.seasons import previous_season_label
        params, url, weights, std_columns, ppp_columns, goalie_columns = load_default_params()
        base_config = {
            "tgps": [410, 7],
            "base_params": params,
            "base_url": url,
        }
        sk8r_config = {
            "std_columns": std_columns,
            "ppp_columns": ppp_columns,
        }
        goalie_config = {"goalie_columns": goalie_columns}

        # Current season pass
        skater_pipeline = StatPipelineFactory.create("skater", {**base_config, **sk8r_config})
        
        # Skaters still use NST
        skater_pipeline.run()
        skater_df = skater_pipeline.merge()
        print(f"Merge complete for skaters: {skater_df.shape} ")

        # Goalies now use Yahoo exclusively
        default_lk = os.getenv("YAHOO_LEAGUE_KEY")
        if not default_lk:
            session = get_session()
            try:
                l = session.query(League).first()
                if l:
                    default_lk = l.league_key
            finally:
                session.close()

        if default_lk:
            print(f"Fetching goalies from Yahoo using league {default_lk}...")
            goalie_df = fetch_goalies_from_yahoo(default_lk)
        else:
            print("[Warn] No Yahoo league key found for goalie fetch. Skipping goalies.")
            goalie_df = pd.DataFrame()

        if not goalie_df.empty:
            print(f"Yahoo fetch complete for goalies: {goalie_df.shape} ")

        # Optional verbose diagnostics for scoring
        verbose = str(os.getenv("VERBOSE_SCORING", "0")).lower() in {"1", "true", "yes", "y"}
        if verbose:
            try:
                off_stats = ["G", "A", "PPP", "SOG", "FOW"]
                ban_stats = ["HIT", "BLK", "PIM"]
                prefixes = ["szn_", "l7_"]
                target_cols = []
                for p in prefixes:
                    for s in off_stats + ban_stats:
                        c = f"{p}{s}"
                        if c in skater_df.columns:
                            target_cols.append(c)
                if target_cols:
                    print("[Diag] Dtypes for scoring columns:")
                    print(skater_df[target_cols].dtypes.to_string())
                    # Show a quick snapshot of potential non-numeric issues
                    sample_cols = target_cols[:8]
                    print("[Diag] Sample values for first scoring cols:")
                    print(skater_df[sample_cols].head(3).to_string(index=False))
            except Exception as _vd:
                print(f"[Diag] Verbose diagnostics failed: {_vd}")

        # Save unified CSVs under data/ directory
        out_dir = os.path.join("data")
        os.makedirs(out_dir, exist_ok=True)
        # Restore persistence of raw skaters.csv for diagnostic visibility
        skater_pipeline.save(os.path.join(out_dir, "skaters.csv"))
        if not goalie_df.empty:
            goalie_df.to_csv(os.path.join(out_dir, "goalies.csv"), index=False)

        # Scoring logic - multi-league support
        session = get_session()
        try:
            leagues = session.query(League).all()
            if not leagues:
                print("[Warn] No leagues found in DB. Scoring with default T-score only.")
                leagues = [None] # Dummy league for default scoring
            
            for league in leagues:
                league_id_str = ""
                league_label = ""
                if league:
                    league_id_str = league.league_key.split(".l.")[-1]
                    league_label = f" for league {league_id_str}"
                
                print(f"Scoring skaters{league_label}...")
                indexer = PlayerIndexPipeline(skater_df, league=league)
                scored_skater_df = indexer.run()
                
                # If league has point values, apply FantasyPointsPipeline
                if league:
                    stat_info = {s.abbr: (float(s.value), s.group_code) for s in league.stats if s.value is not None}
                    if stat_info:
                        print(f"  Applying points scoring{league_label}")
                        points_pipeline = FantasyPointsPipeline(scored_skater_df, stat_info, league_id=league.id)
                        scored_skater_df = points_pipeline.run()

                # Persist scored skaters
                suffix = f"_{league_id_str}" if league_id_str else ""
                skater_scored_path = os.path.join(out_dir, f"skaters_scored{suffix}.csv")
                scored_skater_df.to_csv(skater_scored_path, index=False)
                print(f"  Saved scored skaters to {skater_scored_path}")

                # Scoring goalies
                print(f"Scoring goalies{league_label}...")
                # Use Yahoo-sourced goalie_df directly if available
                if not goalie_df.empty:
                    g_indexer = GoalieIndexPipeline(goalie_df, league=league)
                    scored_goalie_df = g_indexer.run()
                    
                    if league:
                        if stat_info:
                            # Re-use stat_info, assuming stat names match
                            points_pipeline = FantasyPointsPipeline(scored_goalie_df, stat_info, league_id=league.id)
                            scored_goalie_df = points_pipeline.run()
                    
                    goalie_scored_path = os.path.join(out_dir, f"goalies_scored{suffix}.csv")
                    scored_goalie_df.to_csv(goalie_scored_path, index=False)
                    print(f"  Saved scored goalies to {goalie_scored_path}")
                else:
                    print(f"  [Warn] goalie_df is empty; skipping scoring{league_label}")

                # For backward compatibility, also save to default path if it's the first league or only league
                if league == leagues[0]:
                    scored_skater_df.to_csv(os.path.join(out_dir, "skaters_scored.csv"), index=False)
                    if not goalie_df.empty:
                        scored_goalie_df.to_csv(os.path.join(out_dir, "goalies_scored.csv"), index=False)

        finally:
            session.close()

        # Optional prior-season pass (derive from env FROMSEASON/THRUSEASON)
        try:
            from_season = str(params.get("fromseason") or "").strip()
            thru_season = str(params.get("thruseason") or "").strip()
            if len(from_season) == 8 and from_season.isdigit() and from_season == thru_season:
                prior_label = previous_season_label(from_season)
                if prior_label and prior_label != from_season:
                    # Check for existing prior-season scored outputs
                    prior_sk_scored_path = os.path.join(out_dir, "skaters_scored_prior.csv")
                    prior_g_scored_path = os.path.join(out_dir, "goalies_scored_prior.csv")

                    if skip_prior:
                        print("[Info] Skipping prior-season fetch/scoring due to --skip-prior flag.")
                        return

                    if (not refresh_prior) and os.path.exists(prior_sk_scored_path) and os.path.exists(prior_g_scored_path):
                        print("[Info] Prior-season scored CSVs already exist; skipping re-fetch. Use --refresh-prior to force.")
                        return

                    prior_params = dict(params)
                    prior_params["fromseason"] = prior_label
                    prior_params["thruseason"] = prior_label
                    prior_base = {"tgps": [410, 7], "base_params": prior_params, "base_url": url}
                    prior_skater = StatPipelineFactory.create("skater", {**prior_base, **sk8r_config})
                    # Fetch prior season
                    prior_skater.run()
                    # Merge and save
                    prior_sk_df = prior_skater.merge()
                    prior_sk_path = os.path.join(out_dir, "skaters_prior.csv")
                    prior_skater.save(prior_sk_path)
                    # Score prior skaters
                    prior_indexer = PlayerIndexPipeline(prior_sk_df)
                    prior_scored = prior_indexer.run()
                    prior_scored_path = prior_sk_scored_path
                    prior_scored.to_csv(prior_scored_path, index=False)
                    print(f"Saved prior-season skaters to {prior_sk_path} and scored to {prior_scored_path}")
        except Exception as _pe:
            print(f"[Warn] Prior-season pass skipped due to error: {_pe}")

    except Exception as e:
        import traceback
        print(f"NST processing failed: {e}")
        traceback.print_exc()


def run_merge():
    if _run_merge_impl is None:
        print("Merge step unavailable: app.merger.run_merge could not be imported.")
        return
    try:
        _run_merge_impl()
    except Exception as e:
        print(f"Merge step failed: {e}")


def run_all():
    session = get_session()
    try:
        leagues = session.query(League).all()
        if not leagues:
            run_yahoo()
        else:
            for league in leagues:
                # Extract numeric ID from league_key (e.g. 12345 from nhl.l.12345)
                l_id = league.league_key.split(".l.")[-1]
                print(f"\n--- Running Yahoo for League {l_id} ---")
                run_yahoo(league_id=l_id)
    finally:
        session.close()
        
    run_nst()
    run_merge()


def run_roster_sync(league_key: str):
    """
    Refresh local roster state to reduce API overhead for all subsequent commands.
    """
    from infrastructure.persistence import (
        get_session, League, Team, Player, CurrentRoster, 
        upsert_team, upsert_player, upsert_league
    )
    from yahoo.yfs_client import YahooFantasyClient, parse_roster_xml, extract_num_teams, make_team_key
    from yahoo.yahoo_auth import get_oauth

    session = get_session()
    try:
        league = session.query(League).filter(League.league_key == league_key).one_or_none()
        if not league:
            print(f"League {league_key} not found in database. Registering...")
            game_key = league_key.split('.l.')[0]
            season = os.getenv("SEASON", "2025")
            league = upsert_league(session, game_key=game_key, league_key=league_key, season=season)
            session.commit()

        client_id = os.getenv("YAHOO_CLIENT_ID")
        client_secret = os.getenv("YAHOO_CLIENT_SECRET")
        client = YahooFantasyClient(get_oauth(client_id, client_secret))

        league_json = client.get_league(league_key)
        num_teams = extract_num_teams(league_json) or 12
        game_key = league_key.split('.l.')[0]

        print(f"Syncing rosters for {num_teams} teams...")
        session.query(CurrentRoster).filter(CurrentRoster.league_id == league.id).delete()

        for i in range(1, num_teams + 1):
            # team_id is the numeric part: e.g. 4
            tkey = make_team_key(game_key, league_key.split('.l.')[1], i)
            try:
                payload = client.get_team_roster(tkey)
                roster = parse_roster_xml(payload.get("_raw_xml", ""))
                team_name = roster.get("team_name")
                
                upsert_team(session, league_id=league.id, team_key=tkey, team_name=team_name)
                
                for p in roster.get("players", []):
                    pkey = f"{game_key}.p.{p['player_id']}"
                    upsert_player(session, pkey, p.get("name"), ";".join(p.get("positions", [])), status=p.get("status"))
                    session.add(CurrentRoster(league_id=league.id, player_key=pkey, team_key=tkey))
                
                print(f"  Synced {team_name}")
                session.commit()
            except Exception as e:
                print(f"  [Warn] Failed {tkey}: {e}")
                session.rollback()

        # Fetch Free Agents and Waiver players with potential IR/IR+ status
        # Using two-pass approach as requested:
        # Pass 1: sort=AR, sort_type=season (Top 25 skaters, 5 goalies)
        # Pass 2: sort=AR, sort_type=lastweek (Top 10 skaters, 2 goalies)
        print("Syncing Free Agents and Waiver players...")
        
        fa_configs = [
            {"sort_type": "season", "skater_count": 25, "goalie_count": 5},
            {"sort_type": "lastweek", "skater_count": 10, "goalie_count": 2},
        ]

        for cfg in fa_configs:
            sort_type = cfg["sort_type"]
            for pos_type, count in [("P", cfg["skater_count"]), ("G", cfg["goalie_count"])]:
                for status_filter in ["FA", "W"]:
                    try:
                        print(f"  Fetching {status_filter} {pos_type} (Top {count} by {sort_type})...")
                        payload = client.get_league_players(
                            league_key, 
                            status=status_filter, 
                            position=pos_type, 
                            sort="AR", 
                            sort_type=sort_type,
                            count=count
                        )
                        roster = parse_roster_xml(payload.get("_raw_xml", ""))
                        players_synced = 0
                        for p in roster.get("players", []):
                            pkey = f"{game_key}.p.{p['player_id']}"
                            # Only upsert if we have a name (avoid incomplete records)
                            if p.get("name"):
                                upsert_player(
                                    session, pkey, p.get("name"), 
                                    ";".join(p.get("positions", [])), 
                                    status=p.get("status")
                                )
                                players_synced += 1
                        session.commit()
                        if players_synced > 0:
                            print(f"    Synced {players_synced} players")
                    except Exception as e:
                        print(f"    [Warn] Failed to sync {status_filter} {pos_type} ({sort_type}): {e}")
                        session.rollback()
    finally:
        session.close()


def main():
    run_all()
