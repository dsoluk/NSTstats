import numpy as np
import pandas as pd


class PlayerIndexPipeline:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def log_transform(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_cols] = self.df[numeric_cols].apply(lambda x: np.log1p(x.clip(lower=0)))

    def t_score(self, series: pd.Series):
        return (series - series.mean()) / (series.std(ddof=0) + 1e-9)

    def calculate_t_scores(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            self.df[f"{col}_t"] = self.t_score(self.df[col])

    def segment_positions(self):
        self.df['is_d'] = self.df['position'].str.contains('D', na=False).astype(int)
        self.df['is_f'] = self.df['position'].str.contains('C|LW|RW', na=True, regex=True).astype(int)

    def calculate_offensive_index(self):
        cols = [c for c in self.df.columns if c.endswith('_t') and any(k in c for k in ['g', 'a', 'ppp', 'sog'])]
        if cols:
            self.df['off_idx'] = self.df[cols].mean(axis=1)

    def calculate_banger_index(self):
        cols = [c for c in self.df.columns if c.endswith('_t') and any(k in c for k in ['hit', 'blk', 'pim'])]
        if cols:
            self.df['bang_idx'] = self.df[cols].mean(axis=1)

    def calculate_composite_index(self):
        if 'off_idx' in self.df.columns and 'bang_idx' in self.df.columns:
            self.df['composite_idx'] = 0.7 * self.df['off_idx'] + 0.3 * self.df['bang_idx']
        elif 'off_idx' in self.df.columns:
            self.df['composite_idx'] = self.df['off_idx']
        elif 'bang_idx' in self.df.columns:
            self.df['composite_idx'] = self.df['bang_idx']

    def run(self):
        self.log_transform()
        self.calculate_t_scores()
        self.segment_positions()
        self.calculate_offensive_index()
        self.calculate_banger_index()
        self.calculate_composite_index()
        return self.df
