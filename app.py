from config import load_default_params
from pipelines import StatPipelineFactory, PlayerIndexPipeline


def main():

    # Fetch and process Natural Stat Trick player names
    try:
        params, url, weights, selected_columns, goalie_columns, column_renames = load_default_params()
        base_config = {
            "tgps": [410, 7],
            "base_params": params,
            "base_url": url,
            "weights": weights,
            "selected_columns": selected_columns,
            "goalie_columns": goalie_columns,
            "column_renames": column_renames
        }

        goalie_pipeline = StatPipelineFactory.create("goalie", base_config)
        skater_pipeline = StatPipelineFactory.create("skater", base_config)

        indexer = PlayerIndexPipeline(skater_df)
        scored_df = indexer.run()

        # TODO what to do with the pipeline results?  calcs, output, etc.

    except Exception as e:
        print(f"NST processing failed: {e}")

