from numpy.ma.core import shape

from config import load_default_params
from pipelines import StatPipelineFactory, PlayerIndexPipeline


def main():

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
        indexer = PlayerIndexPipeline(skater_df)
        scored_df = indexer.run()
        # TODO: persist scored_df if needed

    except Exception as e:
        print(f"NST processing failed: {e}")

