import os
from config import load_default_params
from pipelines import StatPipelineFactory, PlayerIndexPipeline

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


def run_yahoo():
    # If YahooFantasyApp failed to import, provide a clear message and return
    if YahooFantasyApp is None:
        print("Yahoo data source skipped: YahooFantasyApp could not be imported (check yahoo/app.py and dependencies).")
        return
    try:
        app = YahooFantasyApp()
        app.run()
    except Exception as err:
        print(f"Yahoo data source skipped due to error: {err}")


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
        goalie_pipeline = StatPipelineFactory.create("goalie", {**base_config, **goalie_config})

        # Warning: running these will fetch data from NST based on .env configuration
        skater_pipeline.run()
        goalie_pipeline.run()

        # Merge windows into one dataframe per role
        skater_df = skater_pipeline.merge()
        goalie_df = goalie_pipeline.merge()

        print(f"Merge complete for skaters: {skater_df.shape} ")
        print(f"Merge complete for goalies: {goalie_df.shape} ")

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
        goalie_pipeline.save(os.path.join(out_dir, "goalies.csv"))

        # compute player index on skaters
        if verbose:
            print("[Diag] Starting PlayerIndexPipeline scoring...")
        indexer = PlayerIndexPipeline(skater_df)
        try:
            scored_df = indexer.run()
        except Exception as se:
            if verbose:
                import traceback
                print("[Diag] Scoring failed with traceback:")
                print(traceback.format_exc())
            raise
        # persist scored skaters
        skater_scored_path = os.path.join(out_dir, "skaters_scored.csv")
        scored_df.to_csv(skater_scored_path, index=False)
        print(f"Saved scored skaters to {skater_scored_path}")

        # Score goalies (GA, SV%, GAA) similarly to skaters
        try:
            from pipelines import GoalieIndexPipeline
            g_indexer = GoalieIndexPipeline(goalie_df)
            g_scored = g_indexer.run()
            goalie_scored_path = os.path.join(out_dir, "goalies_scored.csv")
            g_scored.to_csv(goalie_scored_path, index=False)
            print(f"Saved scored goalies to {goalie_scored_path}")
        except Exception as ge:
            print(f"[Warn] Goalie scoring failed: {ge}")

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
                    prior_goalie = StatPipelineFactory.create("goalie", {**prior_base, **goalie_config})
                    # Fetch prior season
                    prior_skater.run()
                    prior_goalie.run()
                    # Merge and save
                    prior_sk_df = prior_skater.merge()
                    prior_g_df = prior_goalie.merge()
                    prior_sk_path = os.path.join(out_dir, "skaters_prior.csv")
                    prior_g_path = os.path.join(out_dir, "goalies_prior.csv")
                    prior_skater.save(prior_sk_path)
                    prior_goalie.save(prior_g_path)
                    # Score prior skaters
                    prior_indexer = PlayerIndexPipeline(prior_sk_df)
                    prior_scored = prior_indexer.run()
                    prior_scored_path = prior_sk_scored_path
                    prior_scored.to_csv(prior_scored_path, index=False)
                    print(f"Saved prior-season skaters to {prior_sk_path} and scored to {prior_scored_path}")
                    # Score prior goalies
                    try:
                        from pipelines import GoalieIndexPipeline as _GIP
                        prior_g_indexer = _GIP(prior_g_df)
                        prior_g_scored = prior_g_indexer.run()
                        # use previously defined scored path
                        prior_g_scored.to_csv(prior_g_scored_path, index=False)
                        print(f"Saved prior-season goalies to {prior_g_path} and scored to {prior_g_scored_path}")
                    except Exception as _gge:
                        print(f"[Warn] Prior-season goalie scoring failed: {_gge}")
        except Exception as _pe:
            print(f"[Warn] Prior-season pass skipped due to error: {_pe}")

    except Exception as e:
        print(f"NST processing failed: {e}")


def run_merge():
    if _run_merge_impl is None:
        print("Merge step unavailable: app.merger.run_merge could not be imported.")
        return
    try:
        _run_merge_impl()
    except Exception as e:
        print(f"Merge step failed: {e}")


def run_all():
    run_yahoo()
    run_nst()
    run_merge()


def run_roster_sync(league_key: str):
    """
    Refresh local roster state to reduce API overhead for all subsequent commands.
    """
    from infrastructure.persistence import (
        get_session, League, Team, Player, CurrentRoster, 
        upsert_team, upsert_player
    )
    from yahoo.yfs_client import YahooFantasyClient, parse_roster_xml, extract_num_teams, make_team_key
    from yahoo.yahoo_auth import get_oauth

    session = get_session()
    try:
        league = session.query(League).filter(League.league_key == league_key).one_or_none()
        if not league:
            print(f"League {league_key} not found. Run init-weeks first.")
            return

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
                    upsert_player(session, pkey, p.get("name"), ";".join(p.get("positions", [])))
                    session.add(CurrentRoster(league_id=league.id, player_key=pkey, team_key=tkey))
                
                print(f"  Synced {team_name}")
                session.commit()
            except Exception as e:
                print(f"  [Warn] Failed {tkey}: {e}")
                session.rollback()
    finally:
        session.close()


def main():
    run_all()
