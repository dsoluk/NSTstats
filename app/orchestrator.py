import os
from config import load_default_params
from pipelines import StatPipelineFactory, PlayerIndexPipeline

# Optional registry imports (added for Player/Team/Position registry)
try:
    from registries.factory import RegistryFactory
    from registries.reporting import SummaryReporter
    from registries.persistence import export_players_csv
    from adapters.nst.client import NSTPlayerAdapter
except Exception:
    # Defer import errors to runtime handling in main() for registry-only pieces
    RegistryFactory = None
    SummaryReporter = None
    export_players_csv = None
    NSTPlayerAdapter = None

# Import Yahoo app separately so a registry failure doesn't break Yahoo
try:
    from yahoo.app import YahooFantasyApp
except Exception:
    YahooFantasyApp = None


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


def run_registry_update():
    try:
        if RegistryFactory and SummaryReporter and NSTPlayerAdapter:
            factory = RegistryFactory()
            # Ensure team mappings are loaded from Team2TM.xlsx
            factory.teams.load()
            reporter = SummaryReporter()
            nst_adapter = NSTPlayerAdapter()
            factory.update_from_source(nst_adapter, reporter)
            # Export CSV snapshot of registry if helper is available
            if export_players_csv:
                export_players_csv(factory.names.all(), 'player_registry.csv')
            print("Registry update completed. See player_registry.json and summary.json.")
        else:
            print("Registry components not available (missing optional dependencies). Skipping registry update.")
    except Exception as re:
        print(f"Registry update skipped due to error: {re}")


def run_nst():
    try:
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

        # Save unified CSVs under data/ directory
        out_dir = os.path.join("data")
        os.makedirs(out_dir, exist_ok=True)
        skater_pipeline.save(os.path.join(out_dir, "skaters.csv"))
        goalie_pipeline.save(os.path.join(out_dir, "goalies.csv"))

        # Example: compute player index on skaters
        # indexer = PlayerIndexPipeline(skater_df)
        # scored_df = indexer.run()
        # TODO: persist scored_df if needed

    except Exception as e:
        print(f"NST processing failed: {e}")


def run_all():
    run_yahoo()
    run_registry_update()
    run_nst()


def main():
    run_all()
