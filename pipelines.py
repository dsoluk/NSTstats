import pandas as pd
import numpy as np

from helpers.extract import fetch_nst_html
from helpers.transform import basic_cleansing, basic_filtering



class SkaterStatsPipeline:
    def __init__(self, sits, tgps, base_params, base_url, selected_columns, rename_map=None):
        self.sits = sits
        self.tgps = tgps
        self.params = base_params.copy()
        self.url = base_url
        self.dataframes = {}
        self.selected_columns = selected_columns
        self.rename_map = rename_map or {}

    def fetch_html(self, sit, tgp):
        self.params["sit"] = sit
        self.params["tgp"] = tgp
        return fetch_nst_html(self.params)

    def process(self, html):
        clean = basic_cleansing(html)
        return basic_filtering(clean)

    def filter_columns(self, df):
        if self.selected_columns:
            df = df[self.selected_columns]
        # Apply rename AFTER selecting, so your mapping uses the original NST names
        if self.rename_map:
            df = df.rename(columns=self.rename_map)
        return df

    def run(self):
        for sit in self.sits:
            for tgp in self.tgps:
                html = self.fetch_html(sit, tgp)
                df = self.process(html)
                df["sit"] = sit
                df["tgp"] = tgp
                df = self.filter_columns(df)
                self.dataframes[f"{sit}_{tgp}"] = df

    def concat(self):
        return pd.concat(self.dataframes.values(), ignore_index=True)

    def save(self, path):
        self.concat().to_csv(path, index=False)


class GoalieStatsPipeline:
    def __init__(self, sits, tgps, base_params, base_url, goalie_columns):
        self.sits = sits
        self.tgps = tgps
        self.params = base_params.copy()
        self.url = base_url
        self.dataframes = {}
        self.goalie_columns = goalie_columns
        self.params["STDOI"] = "g"  # Goalie-specific flag

    def fetch_html(self, sit, tgp):
        self.params["sit"] = sit
        self.params["tgp"] = tgp
        return fetch_nst_html(self.params)

    def process(self, html):
        clean = basic_cleansing(html)
        return basic_filtering(clean)

    def filter_columns(self, df):
        # Select only desired columns
        if self.goalie_columns:
            df = df[self.goalie_columns]
        return df

    def run(self):
        for sit in self.sits:
            for tgp in self.tgps:
                try:
                    html = self.fetch_html(sit, tgp)
                    df = self.process(html)
                    df["sit"] = sit
                    df["tgp"] = tgp
                    df = self.filter_columns(df)
                    self.dataframes[f"{sit}_{tgp}"] = df
                except Exception as e:
                    print(f"Failed for sit={sit}, tgp={tgp}: {e}")

    def concat(self):
        return pd.concat(self.dataframes.values(), ignore_index=True)

    def save(self, path):
        self.concat().to_csv(path, index=False)


class StatPipelineFactory:
    @staticmethod
    def create(stat_type, config):
        if stat_type == "goalie":
            config["sits"] = ["all"]
            return GoalieStatsPipeline(**config)
        elif stat_type == "skater":
            config["sits"] = ["all", "pp"]
            return SkaterStatsPipeline(**config)
        # elif stat_type == "season_projection":
        #     return ExcelProjectionPipeline(**config)
        # elif stat_type == "faceoff_wins":
        #     return NHLApiFOWPipeline(**config)
        else:
            raise ValueError(f"Unknown stat_type: {stat_type}")


class PlayerIndexPipeline:
    def __init__(self, df):
        self.df = df.copy()
        self.offensive_stats = ["G", "A", "PPP", "SOG", "FOW"]
        self.banger_stats = ["HIT", "BLK", "PIM"]
        self.weights = {"offensive": 0.7, "banger": 0.3}

    def log_transform(self):
        for col in self.offensive_stats + self.banger_stats:
            self.df[f"log_{col}"] = np.log(self.df[col] + 1)

    def t_score(self, series):
        z = (series - series.mean()) / series.std()
        return 50 + 10 * z

    def calculate_t_scores(self):
        for col in self.offensive_stats + self.banger_stats:
            log_col = f"log_{col}"
            self.df[f"T_{col}"] = self.t_score(self.df[log_col])

    def segment_positions(self):
        self.df["pos_group"] = self.df["position"].apply(
            lambda x: "D" if x == "D" else "F"
        )

    def calculate_offensive_index(self):
        # TODO weighting?
        self.df["Offensive_Index"] = self.df.groupby("pos_group")[
            [f"T_{col}" for col in self.offensive_stats]
        ].transform("mean").mean(axis=1)

    def calculate_banger_index(self):
        self.df["Banger_Index"] = self.df[
            [f"T_{col}" for col in self.banger_stats]
        ].mean(axis=1)

    def calculate_composite_index(self):
        w_off = self.weights["offensive"]
        w_ban = self.weights["banger"]
        self.df["Composite_Index"] = (
            w_off * self.df["Offensive_Index"] + w_ban * self.df["Banger_Index"]
        )
        self.df["T_Composite_Index"] = self.t_score(self.df["Composite_Index"])

    def run(self):
        self.log_transform()
        self.calculate_t_scores()
        self.segment_positions()
        self.calculate_offensive_index()
        self.calculate_banger_index()
        self.calculate_composite_index()
        return self.df
