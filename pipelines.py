import os
import pandas as pd
import numpy as np
from pandas.api.types import is_string_dtype, is_object_dtype, is_categorical_dtype

from helpers.extract import fetch_nst_html
from helpers.transform import basic_cleansing, basic_filtering



class NSTPlayerPipeline:
    def __init__(self, sits, tgps, base_params, base_url, std_columns, ppp_columns=None, extra_params=None):
        # std_columns and ppp_columns are dicts: original NST header -> short name
        self.sits = sits
        self.tgps = tgps
        self.params = base_params.copy()
        # Allow caller (e.g., goalie mode) to inject extra NST params like STDOI/POS/rate
        if extra_params:
            try:
                self.params.update(extra_params)
            except Exception:
                pass
        self.url = base_url
        self.dataframes = {}
        self.std_columns = std_columns or {}
        self.ppp_columns = ppp_columns or {}

    def fetch_html(self, sit, tgp):
        self.params["sit"] = sit
        self.params["tgp"] = tgp
        # print(f"Fetching {sit} {tgp}")
        return fetch_nst_html(self.url, self.params)

    def process(self, html):
        clean = basic_cleansing(html)
        return basic_filtering(clean)

    def _col_map_for_sit(self, sit: str) -> dict:
        # Normalize and choose the correct mapping
        sit_value = (sit or "").strip().lower()
        return self.ppp_columns if sit_value == "pp" else self.std_columns

    def filter_columns(self, df: pd.DataFrame, sit: str) -> pd.DataFrame:
        col_map = self._col_map_for_sit(sit)
        if isinstance(col_map, dict):
            cols_to_keep = [c for c in col_map.keys() if c in df.columns]
            if cols_to_keep:
                return df[cols_to_keep]
        return df

    def rename_and_prefix(self, df: pd.DataFrame, sit: str, tgp):
        if not isinstance(df, pd.DataFrame) or df.empty:
            return df

        col_map = self._col_map_for_sit(sit)

        # Trim column names defensively (handles trailing/leading spaces from HTML)
        df = df.copy()
        df.columns = df.columns.str.strip()

        # Build the rename map only for columns present
        rename_map = {orig: short for orig, short in col_map.items() if orig in df.columns}
        # print(f"Rename map: {rename_map}")

        # Apply renaming
        renamed_df = df.rename(columns=rename_map) if rename_map else df
        # print(f"Renamed columns: {renamed_df.columns}")

        # Determine join keys using current and already processed frames
        # Use frames that are already renamed/prefixed in self.dataframes, plus the current renamed_df
        existing_frames = list(self.dataframes.values())
        frames_for_keys = [renamed_df] + existing_frames if existing_frames else [renamed_df]
        join_cols = self._detect_join_keys(frames_for_keys)
        # print(f"Join columns detected for prefixing: {join_cols}")

        # Prefix all columns that are NOT join columns
        prefix = 'l7_' if str(tgp) == '7' else 'szn_'
        def needs_prefix(col: str) -> bool:
            # Do not double-prefix if already has a known prefix
            if col.startswith('szn_') or col.startswith('l7_'):
                return False
            return col not in set(join_cols)

        new_columns = {col: (f"{prefix}{col}" if needs_prefix(col) else col) for col in renamed_df.columns}
        renamed_df = renamed_df.rename(columns=new_columns)

        return renamed_df

    def run(self):
        for sit in self.sits: # all,pp although pp=None for Goalie
            for tgp in self.tgps: # 410,7
                html = self.fetch_html(sit, tgp)
                df = self.process(html)
                # filter and rename/prefix using explicit sit/tgp parameters
                df = self.filter_columns(df, sit)
                df = self.rename_and_prefix(df, sit, tgp)
                self.dataframes[f"{sit}_{tgp}"] = df
                print(f"Run completed for: {sit} {tgp} {df.shape}")

    def _detect_join_keys(self, frames):
        """Infer join keys based solely on text-like dtype across frames.
        A column is a join key if it exists in all frames and its dtype is
        text-like (string/object/categorical) in every frame.
        """
        if not frames:
            return []

        # Build common column set across all frames
        common = set(frames[0].columns)
        for df in frames[1:]:
            common &= set(df.columns)

        candidate_cols = sorted(common)
        # print(f"Join keys (candidate columns): {candidate_cols}")

        def is_text_like(s: pd.Series) -> bool:
            # print(s.dtypes)

            # Treat pandas string dtype, generic object (often strings), and categoricals as text-like
            return is_string_dtype(s)

        keys = []
        for col in candidate_cols:
            try:
                if all(is_text_like(df[col]) for df in frames):
                    # print(f"Join key detected: {col}")
                    keys.append(col)
            except Exception:
                # If any issue arises accessing the column/dtype, skip it safely
                continue

        # print(f"Join keys (text-like heuristic): {keys}")
        return keys

    def merge(self):
        """Return a single wide table for skaters by left-joining windows:
        base = all_410, then left outer join all_7, pp_410, pp_7.
        """
        base_key = 'all_410'  # sit='all', tgp='410'
        if base_key not in self.dataframes:
            raise KeyError(
                "Missing base DataFrame 'all_410' . Ensure run() was called and tgps include 'all'.")

        def prep(df):
            # dropping helper columns remains safe; they simply won't exist now
            return df.drop(columns=['sit', 'tgp'], errors='ignore').copy()

        base = prep(self.dataframes[base_key])
        # Determine optional frames (they may be absent depending on config)
        to_join = []
        if 'all_7' in self.dataframes:
            to_join.append(('all_7', prep(self.dataframes['all_7'])))
        if 'pp_410' in self.dataframes:
            to_join.append(('pp_410', prep(self.dataframes['pp_410'])))
        if 'pp_7' in self.dataframes:
            to_join.append(('pp_7', prep(self.dataframes['pp_7'])))

        # Infer join keys from all participating frames
        frames_for_keys = [base] + [df for _, df in to_join]
        on_cols = self._detect_join_keys(frames_for_keys)
        if not on_cols:
            raise ValueError(
                "Could not infer join keys. Ensure identity columns (e.g., Player/Team/position) exist and are not prefixed.")

        merged = base
        for name, dfj in to_join:
            merged = merged.merge(dfj, how='left', on=on_cols)
        # Rename pandas default merge suffixes to requested labels:
        #   _x -> _all, _y -> _pp (only when they appear as terminal suffixes)
        def _rename_suffix(col: str) -> str:
            if isinstance(col, str):
                if col.endswith('_x'):
                    return col[:-2] + '_all'
                if col.endswith('_y'):
                    return col[:-2] + '_pp'
            return col
        merged.columns = [_rename_suffix(c) for c in merged.columns]
        return merged

    def save(self, path):
        # For skaters we use merge (joins multiple windows). For goalies (single-sit),
        # or if merge prerequisites aren't met, fall back to simple concat.
        try:
            df = self.merge()
        except Exception:
            # fallback for goalie/simple mode
            try:
                df = pd.concat(self.dataframes.values(), ignore_index=True)
            except Exception:
                # last resort: pick any existing frame
                df = next(iter(self.dataframes.values())) if self.dataframes else pd.DataFrame()
        # Format timedelta-like columns as HH:MM:SS strings for Excel compatibility
        try:
            formatted_cols = []

            def _fmt_seconds_from_seconds(sec_val):
                if pd.isna(sec_val):
                    return ""
                try:
                    import math
                    sec = float(sec_val)
                except Exception:
                    return ""
                neg = sec < 0
                # Use floor to handle negative fractional seconds correctly, then drop fraction
                sec = abs(int(math.floor(sec)))
                h = sec // 3600
                m = (sec % 3600) // 60
                s = sec % 60
                out = f"{h:02d}:{m:02d}:{s:02d}"
                return f"-{out}" if neg else out

            def _format_td_series(series: pd.Series) -> pd.Series:
                # Coerce any series (timedelta dtype or string/object) to timedeltas, then to seconds
                td = pd.to_timedelta(series, errors='coerce')
                secs = td.dt.total_seconds()
                return secs.apply(_fmt_seconds_from_seconds)

            # 1) Native timedelta columns
            td_native = list(df.select_dtypes(include=['timedelta64[ns]']).columns)
            for c in td_native:
                df[c] = _format_td_series(df[c])
                formatted_cols.append(c)

            # 2) Object columns that look like time deltas (e.g., '0 days 00:20:20.400000' or '01:23:45')
            obj_cols = [c for c in df.columns if df[c].dtype == 'object']
            for c in obj_cols:
                sample = df[c].dropna().astype(str).head(50)
                if sample.empty:
                    continue
                if not sample.str.contains(r"\bday\b|\d{1,2}:\d{2}:\d{2}", regex=True, case=False).any():
                    continue
                # Only convert if coercion yields at least one non-NaT value
                td_coerced = pd.to_timedelta(df[c], errors='coerce')
                if td_coerced.notna().any():
                    df[c] = _format_td_series(df[c])
                    formatted_cols.append(c)

            if formatted_cols:
                print(f"Formatted timedelta columns to HH:MM:SS: {sorted(set(formatted_cols))}")
        except Exception as e:
            # If any issue occurs, proceed without special formatting but log the problem
            print(f"[Warn] Timedelta formatting skipped due to error: {e}")
        # Ensure destination directory exists
        try:
            import os
            d = os.path.dirname(path)
            if d:
                os.makedirs(d, exist_ok=True)
        except Exception:
            pass
        df.to_csv(path, index=False)




class StatPipelineFactory:
    @staticmethod
    def create(stat_type, config):
        # Ensure we don't mutate caller's dict unexpectedly
        cfg = dict(config)
        if stat_type == "goalie":
            # Goalie uses only 'all' sit and a different NST param set
            cfg["sits"] = ["all"]
            # Columns for goalies should be supplied under 'goalie_columns'
            goalie_cols = cfg.pop("goalie_columns", {})
            # Use goalie mapping as standard columns; no PPP mapping for goalies
            cfg["std_columns"] = goalie_cols
            cfg["ppp_columns"] = None
            # Inject NST params specific to goalies
            extra = cfg.get("extra_params", {}) or {}
            extra.update({
                "stdoi": "g",  # goalie vs skater selector
                "pos": "g",
                "rate": "n",   # GA on counts at NST for goalies
            })
            cfg["extra_params"] = extra
            return NSTPlayerPipeline(**cfg)
        elif stat_type == "skater":
            cfg["sits"] = ["all", "pp"]
            return NSTPlayerPipeline(**cfg)
        # elif stat_type == "season_projection":
        #     return ExcelProjectionPipeline(**config)
        else:
            raise ValueError(f"Unknown stat_type: {stat_type}")


class PlayerIndexPipeline:
    def __init__(self, df):
        self.df = df.copy()
        # Base stat names (without prefixes)
        self.offensive_stats = ["G", "A", "PPP", "SOG", "FOW"]
        self.banger_stats = ["HIT", "BLK", "PIM"]
        self.weights = {"offensive": 0.7, "banger": 0.3}
        # Determine which prefixes/windows are available
        self.prefixes = self._detect_prefixes()
        # Determine the position column name
        self.pos_col = self._detect_position_col()

    def _detect_prefixes(self):
        has_szn = any(c.startswith("szn_") for c in self.df.columns)
        has_l7 = any(c.startswith("l7_") for c in self.df.columns)
        prefixes = []
        if has_szn:
            prefixes.append("szn_")
        if has_l7:
            prefixes.append("l7_")
        # Fallback to legacy (no-prefix) if neither present
        if not prefixes:
            prefixes = [""]
        return prefixes

    def _resolve_columns_for_prefix(self, base_names, prefix):
        cols = []
        for name in base_names:
            cand = f"{prefix}{name}" if prefix else name
            if cand in self.df.columns:
                cols.append(cand)
        return cols

    def _detect_position_col(self):
        for candidate in ["Position", "position", "POS", "pos"]:
            if candidate in self.df.columns:
                return candidate
        # If not found, create a default F group to avoid crashes
        self.df["Position"] = "F"
        return "Position"

    def _log_transform(self, cols):
        for col in cols:
            self.df[f"log_{col}"] = np.log(self.df[col].astype(float).clip(lower=0) + 1.0)

    def _t_score_grouped(self, series, groups):
        # Compute T-score within each group; if std=0, return 50 for that group
        grp = series.groupby(groups)
        mean = grp.transform("mean")
        std = grp.transform("std")
        z = (series - mean) / std.replace(0, np.nan)
        t = 50 + 10 * z
        return t.fillna(50)

    def calculate_t_scores(self):
        # Segment positions first
        self.df["pos_group"] = self.df[self.pos_col].apply(lambda x: "D" if str(x).strip().upper() == "D" else "F")
        # For each window/prefix, compute log and segmented T for the window's columns
        for prefix in self.prefixes:
            off_cols = self._resolve_columns_for_prefix(self.offensive_stats, prefix)
            ban_cols = self._resolve_columns_for_prefix(self.banger_stats, prefix)
            cols = off_cols + ban_cols
            if not cols:
                continue
            self._log_transform(cols)
            for col in cols:
                log_col = f"log_{col}"
                t_col = f"T_{col}"
                self.df[t_col] = self._t_score_grouped(self.df[log_col], self.df["pos_group"]) 

    def calculate_indexes_per_window(self):
        # Per window: mean of segmented T-scores for off/banger, and T of composite per window
        for prefix in self.prefixes:
            # Window label suffix: 'szn' or 'l7' or 'legacy'
            win = 'szn' if prefix == 'szn_' else ('l7' if prefix == 'l7_' else 'legacy')
            off_cols = self._resolve_columns_for_prefix(self.offensive_stats, prefix)
            ban_cols = self._resolve_columns_for_prefix(self.banger_stats, prefix)
            t_off = [f"T_{c}" for c in off_cols]
            t_ban = [f"T_{c}" for c in ban_cols]
            if t_off:
                self.df[f"Offensive_Index_{win}"] = self.df[t_off].mean(axis=1)
            else:
                self.df[f"Offensive_Index_{win}"] = np.nan
            if t_ban:
                self.df[f"Banger_Index_{win}"] = self.df[t_ban].mean(axis=1)
            else:
                self.df[f"Banger_Index_{win}"] = np.nan
            # Compute composite only to derive segmented T, then drop the raw composite
            composite = (
                self.weights["offensive"] * self.df[f"Offensive_Index_{win}"].astype(float) +
                self.weights["banger"] * self.df[f"Banger_Index_{win}"].astype(float)
            )
            self.df[f"T_Composite_Index_{win}"] = self._t_score_grouped(composite, self.df["pos_group"]) 
            # No persistent Composite_Index column per requirements

    def run(self):
        self.calculate_t_scores()
        self.calculate_indexes_per_window()
        # Remove any legacy/global index columns if they exist
        drop_candidates = [
            "Offensive_Index", "Banger_Index", "Composite_Index", "T_Composite_Index"
        ]
        existing = [c for c in drop_candidates if c in self.df.columns]
        if existing:
            self.df.drop(columns=existing, inplace=True)
        # Remove log-normal columns from final df
        log_cols = [c for c in self.df.columns if c.startswith("log_")]
        if log_cols:
            self.df.drop(columns=log_cols, inplace=True)
        return self.df
