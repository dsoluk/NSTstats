from numpy.ma.core import shape

from config import load_default_params
from pipelines import StatPipelineFactory, PlayerIndexPipeline

# Optional registry imports (added for Player/Team/Position registry)
try:
    from registries.factory import RegistryFactory
    from registries.reporting import SummaryReporter
    from registries.persistence import export_players_csv
    from adapters.nst import NSTPlayerAdapter
    from yahoo.app import YahooFantasyApp
except Exception:
    # Defer import errors to runtime handling in main()
    RegistryFactory = None
    SummaryReporter = None
    export_players_csv = None
    NSTPlayerAdapter = None
    YahooFantasyApp = None


def main():
    # Fetch and process Yahoo Fantasy data
    try:
        yah_app = YahooFantasyApp()
        yah_app.run()
    except Exception as err:
        print(f"Yahoo data source skipped due to error: {err}")

    # Update Player/Team/Position registry from NST player list (optional)
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

    # Fetch and process Natural Stat Trick player names
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

        # Save unified CSVs
        skater_pipeline.save("skaters.csv")
        goalie_pipeline.save("goalies.csv")

        # Example: compute player index on skaters
        # indexer = PlayerIndexPipeline(skater_df)
        # scored_df = indexer.run()
        # TODO: persist scored_df if needed

    except Exception as e:
        print(f"NST processing failed: {e}")

