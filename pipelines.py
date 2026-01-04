import os
import math
import pandas as pd
import numpy as np
from pandas.api.types import is_string_dtype, is_object_dtype, is_categorical_dtype

from helpers.extract import fetch_nst_html
from helpers.transform import basic_cleansing, basic_filtering
from config import load_default_params



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
        print(f"[Diag] Fetching NST {sit} (tgp={tgp})...")
        return fetch_nst_html(self.url, self.params)

    def process(self, html):
        if html is None:
            return pd.DataFrame()
        try:
            clean = basic_cleansing(html)
            return basic_filtering(clean)
        except Exception as e:
            print(f"[Warn] NST processing error: {e}")
            return pd.DataFrame()

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
            # Treat pandas string dtype, generic object (often strings), and categoricals as text-like
            return is_string_dtype(s) or is_object_dtype(s) or is_categorical_dtype(s)

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
            # Check if all frames are empty
            if all(df.empty for df in frames_for_keys):
                raise ValueError("Could not infer join keys because all NST dataframes are empty. Check your parameters and URL.")
            raise ValueError(
                f"Could not infer join keys. common columns: {list(set.intersection(*(set(df.columns) for df in frames_for_keys)))}. "
                "Ensure identity columns (e.g., Player/Team/position) exist and are not prefixed.")

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
    def __init__(self, df, league=None):
        self.df = df.copy()
        self.league = league
        # Load weights from config
        try:
            _, _, index_weights, _, _, _ = load_default_params()
        except Exception:
            index_weights = {
                "offensive": {"G": 1.0, "A": 1.0, "PPP": 1.0, "SOG": 1.0, "FOW": 1.0},
                "banger": {"HIT": 1.0, "BLK": 1.0, "PIM": 1.0},
                "composite": {"offensive": 0.7, "banger": 0.3},
            }
        self.index_weights = index_weights
        # Verbose scoring diagnostics via env flag
        self.verbose = str(os.getenv("VERBOSE_SCORING", "0")).lower() in {"1", "true", "yes", "y"}
        # Base stat names (without prefixes)
        self.offensive_stats = ["G", "A", "PPP", "SOG", "FOW"]
        self.banger_stats = ["HIT", "BLK", "PIM"]
        self.to_drop = []

        # Filter stats based on league if provided
        if self.league:
            league_abbrs = {s.abbr for s in self.league.stats}
            self.offensive_stats = [s for s in self.offensive_stats if s in league_abbrs]
            self.banger_stats = [s for s in self.banger_stats if s in league_abbrs]
            
            # Identify stats to drop (League 2 specifically wants to exclude these)
            if self.league.id == 2:
                self.to_drop = ["PIM", "FOW"]

        # Determine which prefixes/windows are available
        self.prefixes = self._detect_prefixes()
        # Determine the position column name
        self.pos_col = self._detect_position_col()
        # Prepare distribution mapping from best_fit_comparison
        self.best_fit = self._load_best_fit_mapping()

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

    def _load_best_fit_mapping(self):
        # Find latest dq directory
        dq_root = os.path.join("data", "dq")
        fit_path = None
        try:
            if os.path.isdir(dq_root):
                candidates = [d for d in os.listdir(dq_root) if os.path.isdir(os.path.join(dq_root, d))]
                candidates = sorted(candidates)
                for d in reversed(candidates):
                    p = os.path.join(dq_root, d, "best_fit_comparison.csv")
                    if os.path.isfile(p):
                        fit_path = p
                        break
        except Exception:
            fit_path = None
        mapping = {}
        if fit_path:
            try:
                fit_df = pd.read_csv(fit_path)
                # Expect columns: window, segment, metric, fit_current, ...
                for _, r in fit_df.iterrows():
                    w = str(r.get("window", "")).strip() or "szn"
                    s = str(r.get("segment", "")).strip().upper() or "F"
                    m = str(r.get("metric", "")).strip().upper()
                    f = str(r.get("fit_current", "")).strip()
                    if not m:
                        continue
                    mapping[(w, s, m)] = f
            except Exception:
                mapping = {}
        return mapping

    @staticmethod
    def _phi(z):
        # Vectorized standard normal CDF using error function
        # Prefer numpy's erfc (vectorized and widely available), fallback to vectorized math.erf
        try:
            import numpy as _np
            return 0.5 * _np.erfc(-_np.asarray(z, dtype=float) / _np.sqrt(2.0))
        except Exception:
            # Very unlikely path; ensure elementwise fallback using already-imported numpy as np
            arr = np.asarray(z, dtype=float)
            return 0.5 * (1.0 + np.vectorize(math.erf)(arr / math.sqrt(2.0)))

    def _percentile_from_family(self, x_series: pd.Series, family: str, group_mask: pd.Series):
        # Compute p = F(x) per group based on selected family by estimating parameters from group sample
        # Coerce to numeric first to avoid dtype issues (e.g., strings)
        x = pd.to_numeric(x_series, errors="coerce")
        # Extract group sample
        sample = x[group_mask]
        if sample.empty:
            return pd.Series(np.nan, index=x.index)
        fam = (family or "").lower()
        try:
            # Determine transformed variable y and its parameters
            if "normal(log1p)" in fam:
                y = np.log1p(sample.clip(lower=0))
                mu = y.mean()
                sd = y.std(ddof=0)
                if not np.isfinite(sd) or sd <= 0:
                    p = pd.Series(0.5, index=sample.index)
                else:
                    p = pd.Series(self._phi((np.log1p(sample.clip(lower=0)) - mu) / sd), index=sample.index)
            elif "lognormal" in fam:
                # use log1p for stability with zeros
                y = np.log1p(sample.clip(lower=0))
                mu = y.mean()
                sd = y.std(ddof=0)
                if not np.isfinite(sd) or sd <= 0:
                    p = pd.Series(0.5, index=sample.index)
                else:
                    p = pd.Series(self._phi((np.log1p(sample.clip(lower=0)) - mu) / sd), index=sample.index)
            elif "normal" in fam:
                mu = sample.mean()
                sd = sample.std(ddof=0)
                if not np.isfinite(sd) or sd <= 0:
                    p = pd.Series(0.5, index=sample.index)
                else:
                    p = pd.Series(self._phi((sample - mu) / sd), index=sample.index)
            else:
                # Fallback: empirical percentile within group
                ranks = sample.rank(method="average", pct=True)
                p = pd.Series(ranks, index=sample.index)
        except Exception as err:
            if self.verbose:
                nn = sample.notna().sum()
                print(f"[Scoring] Error computing CDF family='{family}' on {nn} non-null samples. Example values: {sample.head(5).tolist()} | err={err}")
            raise
        # Build full-length series with NaNs elsewhere
        out = pd.Series(np.nan, index=x.index)
        out.loc[sample.index] = p.values
        return out

    def _score_cols_for_window(self, prefix: str, window_key: str):
        # Compute percentile-based T_* scores per stat for a given window
        off_cols = self._resolve_columns_for_prefix(self.offensive_stats, prefix)
        ban_cols = self._resolve_columns_for_prefix(self.banger_stats, prefix)
        cols = off_cols + ban_cols
        if not cols:
            return
        # position segmentation: map to 'D' vs 'F' group
        self.df["pos_group"] = self.df[self.pos_col].apply(lambda x: "D" if str(x).strip().upper() == "D" else "F")
        for col in cols:
            base_metric = col.replace(prefix, "") if prefix else col
            metric_u = base_metric.upper()
            # Determine family: from mapping; if missing, fallback by simple heuristics
            fam = self.best_fit.get((window_key, "D", metric_u)) if False else None  # dummy to satisfy linter
            # We need per segment family; select later inside group loop
            # Initialize output series
            score_col = f"T_{col}"
            scores = pd.Series(np.nan, index=self.df.index)
            # Precompute numeric coercion for diagnostics
            if self.verbose:
                series_num = pd.to_numeric(self.df[col], errors="coerce")
                print(f"[Scoring] window={window_key} col={col} metric={metric_u} dtype={self.df[col].dtype} nonnull={series_num.notna().sum()} total={len(series_num)}")
            for seg in ["D", "F"]:
                seg_mask = self.df["pos_group"] == seg
                if not seg_mask.any():
                    continue
                # Choose family: if mapping contains (window, seg, metric) else fallback
                fam = self.best_fit.get((window_key, seg, metric_u))
                if not fam:
                    # Use szn mapping if l7 missing
                    fam = self.best_fit.get(("szn", seg, metric_u))
                if not fam:
                    # Heuristic: treat non-negative count-like stats as Normal(log1p)
                    fam = "Normal(log1p)"
                if self.verbose:
                    if 'series_num' in locals():
                        seg_nonnull = series_num[seg_mask].notna().sum()
                        print(f"[Scoring]  └─ seg={seg} family={fam} group_size={int(seg_mask.sum())} seg_nonnull={int(seg_nonnull)}")
                    else:
                        print(f"[Scoring]  └─ seg={seg} family={fam} group_size={int(seg_mask.sum())}")
                p = self._percentile_from_family(self.df[col], fam, seg_mask)
                # Scale to 0..99, round up to integer
                s = np.ceil(99.0 * p.clip(lower=0.0, upper=1.0)).astype("Int64")
                # Replace NaNs with 50
                s = s.fillna(50)
                scores.loc[seg_mask] = s.loc[seg_mask]
            self.df[score_col] = scores.astype("Int64")

    def calculate_t_scores(self):
        # For each window, compute percentile-based scores per requirement
        for prefix in self.prefixes:
            win = 'szn' if prefix == 'szn_' else ('l7' if prefix == 'l7_' else 'szn')
            self._score_cols_for_window(prefix, win)

    def calculate_indexes_per_window(self):
        # Per window: weighted averages of percentile T_* scores using configured weights
        off_w = self.index_weights.get("offensive", {})
        ban_w = self.index_weights.get("banger", {})
        comp_w = self.index_weights.get("composite", {"offensive": 0.7, "banger": 0.3})
        for prefix in self.prefixes:
            # Window label suffix: 'szn' or 'l7' or 'legacy'
            win = 'szn' if prefix == 'szn_' else ('l7' if prefix == 'l7_' else 'legacy')
            off_cols = self._resolve_columns_for_prefix(self.offensive_stats, prefix)
            ban_cols = self._resolve_columns_for_prefix(self.banger_stats, prefix)
            # Offensive index weighted by per-stat weights
            t_off_cols = [f"T_{c}" for c in off_cols]
            if t_off_cols:
                weights = np.array([off_w.get(c.replace(prefix, ""), 1.0) for c in off_cols], dtype=float)
                weights = np.where(np.isfinite(weights) & (weights >= 0), weights, 0.0)
                wsum = weights.sum()
                if wsum <= 0:
                    off_series = self.df[t_off_cols].mean(axis=1)
                else:
                    off_vals = self.df[t_off_cols].astype(float).values
                    off_series = pd.Series(np.dot(off_vals, weights) / wsum, index=self.df.index)
                self.df[f"Offensive_Index_{win}"] = off_series
            else:
                self.df[f"Offensive_Index_{win}"] = np.nan
            # Banger index weighted
            t_ban_cols = [f"T_{c}" for c in ban_cols]
            if t_ban_cols:
                weights = np.array([ban_w.get(c.replace(prefix, ""), 1.0) for c in ban_cols], dtype=float)
                weights = np.where(np.isfinite(weights) & (weights >= 0), weights, 0.0)
                wsum = weights.sum()
                if wsum <= 0:
                    ban_series = self.df[t_ban_cols].mean(axis=1)
                else:
                    ban_vals = self.df[t_ban_cols].astype(float).values
                    ban_series = pd.Series(np.dot(ban_vals, weights) / wsum, index=self.df.index)
                self.df[f"Banger_Index_{win}"] = ban_series
            else:
                self.df[f"Banger_Index_{win}"] = np.nan
            # Composite index as weighted combination of offensive and banger indices
            off_wt = float(comp_w.get("offensive", 0.7))
            ban_wt = float(comp_w.get("banger", 0.3))
            denom = off_wt + ban_wt if (off_wt + ban_wt) > 0 else 1.0
            comp_series = (off_wt * self.df[f"Offensive_Index_{win}"].astype(float) +
                           ban_wt * self.df[f"Banger_Index_{win}"].astype(float)) / denom
            self.df[f"Composite_Index_{win}"] = comp_series

    def run(self):
        self.calculate_t_scores()
        self.calculate_indexes_per_window()
        # Remove any legacy/global index columns if they exist
        drop_candidates = [
            "Offensive_Index", "Banger_Index", "Composite_Index", "T_Composite_Index"
        ]
        
        # Add league-specific exclusions
        if self.to_drop:
            for prefix in self.prefixes:
                for stat in self.to_drop:
                    drop_candidates.append(f"{prefix}{stat}")

        existing = [c for c in drop_candidates if c in self.df.columns]
        if existing:
            self.df.drop(columns=existing, inplace=True)
        # Remove log-normal columns from final df
        log_cols = [c for c in self.df.columns if c.startswith("log_")]
        if log_cols:
            self.df.drop(columns=log_cols, inplace=True)
        return self.df


class GoalieIndexPipeline:
    def __init__(self, df, league=None):
        self.df = df.copy()
        self.league = league
        # Verbose scoring diagnostics via env flag
        self.verbose = str(os.getenv("VERBOSE_SCORING", "0")).lower() in {"1", "true", "yes", "y"}
        # Goalie metrics to score (without prefixes)
        self.goalie_stats = ["GA", "SV%", "GAA", "W", "SV", "SHO"]
        self.to_drop = []
        if self.league:
            league_abbrs = {s.abbr for s in self.league.stats}
            self.goalie_stats = [s for s in self.goalie_stats if s in league_abbrs]
            
            # Identify stats to drop (League 2 specifically wants to exclude these)
            if self.league.id == 2:
                self.to_drop = ["SV%", "GAA"]

        # Which metrics are lower-better and should be inverted after percentile mapping
        self.lower_better = {"GA": True, "GAA": True, "SV%": False, "W": False, "SV": False, "SHO": False}
        # Determine which prefixes/windows are available
        self.prefixes = self._detect_prefixes()

    def _detect_prefixes(self):
        has_szn = any(c.startswith("szn_") for c in self.df.columns)
        has_l7 = any(c.startswith("l7_") for c in self.df.columns)
        prefixes = []
        if has_szn:
            prefixes.append("szn_")
        if has_l7:
            prefixes.append("l7_")
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

    @staticmethod
    def _phi(z):
        try:
            import numpy as _np
            return 0.5 * _np.erfc(-_np.asarray(z, dtype=float) / _np.sqrt(2.0))
        except Exception:
            arr = np.asarray(z, dtype=float)
            return 0.5 * (1.0 + np.vectorize(math.erf)(arr / math.sqrt(2.0)))

    def _percentile_from_family(self, x_series: pd.Series, family: str):
        x = pd.to_numeric(x_series, errors="coerce")
        sample = x.dropna()
        if sample.empty:
            return pd.Series(np.nan, index=x.index)
        fam = (family or "").lower()
        try:
            if "normal(log1p)" in fam:
                y = np.log1p(sample.clip(lower=0))
                mu = y.mean(); sd = y.std(ddof=0)
                p = pd.Series(0.5, index=sample.index) if (not np.isfinite(sd) or sd <= 0) else pd.Series(self._phi((np.log1p(sample.clip(lower=0)) - mu) / sd), index=sample.index)
            elif "lognormal" in fam:
                y = np.log1p(sample.clip(lower=0))
                mu = y.mean(); sd = y.std(ddof=0)
                p = pd.Series(0.5, index=sample.index) if (not np.isfinite(sd) or sd <= 0) else pd.Series(self._phi((np.log1p(sample.clip(lower=0)) - mu) / sd), index=sample.index)
            elif "normal" in fam:
                mu = sample.mean(); sd = sample.std(ddof=0)
                p = pd.Series(0.5, index=sample.index) if (not np.isfinite(sd) or sd <= 0) else pd.Series(self._phi((sample - mu) / sd), index=sample.index)
            else:
                p = sample.rank(method="average", pct=True)
        except Exception as err:
            if self.verbose:
                nn = sample.notna().sum()
                print(f"[GoalieScoring] Error computing CDF family='{family}' on {nn} samples. Example values: {sample.head(5).tolist()} | err={err}")
            raise
        out = pd.Series(np.nan, index=x.index)
        out.loc[sample.index] = p.values
        return out

    def _family_for_metric(self, metric_upper: str) -> str:
        # Heuristic families: counts/rates -> Normal(log1p); percentages -> Normal
        if metric_upper in {"GA", "GAA"}:
            return "Normal(log1p)"
        if metric_upper in {"SV%"}:
            return "Normal"
        return "Normal(log1p)"

    def calculate_t_scores(self):
        for prefix in self.prefixes:
            win = 'szn' if prefix == 'szn_' else ('l7' if prefix == 'l7_' else 'szn')
            cols = self._resolve_columns_for_prefix(self.goalie_stats, prefix)
            for col in cols:
                base_metric = col.replace(prefix, "") if prefix else col
                metric_u = base_metric.upper()
                fam = self._family_for_metric(metric_u)
                if self.verbose:
                    series_num = pd.to_numeric(self.df[col], errors="coerce")
                    print(f"[GoalieScoring] window={win} col={col} metric={metric_u} dtype={self.df[col].dtype} nonnull={series_num.notna().sum()} total={len(series_num)} family={fam}")
                p = self._percentile_from_family(self.df[col], fam)
                # Invert for lower-better metrics
                if self.lower_better.get(metric_u, False):
                    p = 1.0 - p
                s = np.ceil(99.0 * p.clip(lower=0.0, upper=1.0)).astype("Int64").fillna(50)
                self.df[f"T_{col}"] = s.astype("Int64")

    def calculate_indexes_per_window(self):
        # Build a single goalie index per window as average of T_ metrics present
        for prefix in self.prefixes:
            win = 'szn' if prefix == 'szn_' else ('l7' if prefix == 'l7_' else 'legacy')
            cols = self._resolve_columns_for_prefix(self.goalie_stats, prefix)
            t_cols = [f"T_{c}" for c in cols if f"T_{c}" in self.df.columns]
            if t_cols:
                self.df[f"Goalie_Index_{win}"] = self.df[t_cols].astype(float).mean(axis=1)
            else:
                self.df[f"Goalie_Index_{win}"] = np.nan

    def run(self):
        self.calculate_t_scores()
        self.calculate_indexes_per_window()
        
        # Add league-specific exclusions
        if self.to_drop:
            drop_candidates = []
            for prefix in self.prefixes:
                for stat in self.to_drop:
                    drop_candidates.append(f"{prefix}{stat}")
            
            existing = [c for c in drop_candidates if c in self.df.columns]
            if existing:
                self.df.drop(columns=existing, inplace=True)

        return self.df


class FantasyPointsPipeline:
    def __init__(self, df, stat_info: dict, league_id: int = None):
        self.df = df.copy()
        self.stat_info = stat_info  # abbr -> (value, group_code)
        self.league_id = league_id

    def _detect_prefixes(self):
        has_szn = any(c.startswith("szn_") for c in self.df.columns)
        has_l7 = any(c.startswith("l7_") for c in self.df.columns)
        prefixes = []
        if has_szn: prefixes.append("szn_")
        if has_l7: prefixes.append("l7_")
        if not prefixes: prefixes = [""]
        return prefixes

    @staticmethod
    def _phi(z):
        try:
            return 0.5 * np.erfc(-np.asarray(z, dtype=float) / np.sqrt(2.0))
        except Exception:
            arr = np.asarray(z, dtype=float)
            return 0.5 * (1.0 + np.vectorize(math.erf)(arr / math.sqrt(2.0)))

    def _percentile_score(self, x_series: pd.Series, group_mask: pd.Series):
        x = pd.to_numeric(x_series, errors="coerce")
        sample = x[group_mask].dropna()
        if sample.empty:
            return pd.Series(50, index=x.index, dtype="Int64")
        mu = sample.mean()
        sd = sample.std(ddof=0)
        if not np.isfinite(sd) or sd <= 0:
            p = pd.Series(0.5, index=sample.index)
        else:
            p = pd.Series(self._phi((sample - mu) / sd), index=sample.index)
        
        s = np.ceil(99.0 * p.clip(lower=0.0, upper=1.0)).astype("Int64").fillna(50)
        out = pd.Series(50, index=x.index, dtype="Int64")
        out.loc[sample.index] = s.values
        return out

    def run(self):
        prefixes = self._detect_prefixes()
        is_skater = "pos_group" in self.df.columns
        for prefix in prefixes:
            points_col = f"{prefix}FantasyPoints"
            self.df[points_col] = 0.0
            
            # Additional columns for league 2
            if self.league_id == 2:
                self.df[f"{prefix}Raw_Total_Points"] = 0.0
                self.df[f"{prefix}Offensive_Points"] = 0.0
                self.df[f"{prefix}Banger_Points"] = 0.0

            # Check for rates vs counts for skaters in League 2
            rate_param = os.getenv("RATE", "y")
            use_conversion = (self.league_id == 2 and is_skater and rate_param == 'y')
            
            stat_points_map = {}

            for stat, info in self.stat_info.items():
                if isinstance(info, (int, float)):
                    weight = float(info)
                    group = None
                else:
                    weight, group = info
                
                if weight is None or weight == 0:
                    continue
                
                col = f"{prefix}{stat}"
                if col in self.df.columns:
                    val = pd.to_numeric(self.df[col], errors="coerce").fillna(0)
                    
                    if use_conversion:
                        # Convert rate per 60 to total count
                        sit = 'pp' if stat in ['PPP', 'PPG', 'PPA'] else 'all'
                        gp_col = f"{prefix}GP_{sit}"
                        toi_col = f"{prefix}TOI/GP_{sit}"
                        
                        if gp_col in self.df.columns and toi_col in self.df.columns:
                            gp_val = pd.to_numeric(self.df[gp_col], errors="coerce").fillna(0)
                            toi_val = self.df[toi_col]
                            
                            # Ensure toi_val is Timedelta-like for total_seconds()
                            if not hasattr(toi_val, 'dt'):
                                toi_val = pd.to_timedelta(toi_val, errors='coerce')
                            
                            if toi_val.notna().any():
                                total_hours = (gp_val * toi_val.dt.total_seconds()) / 3600.0
                                val = (val * total_hours).fillna(0)
                    
                    contribution = val * float(weight)
                    self.df[points_col] += contribution
                    stat_points_map[stat] = contribution
                    
                    if self.league_id == 2:
                        self.df[f"{prefix}Raw_Total_Points"] += contribution
                        if group == "Offensive":
                            self.df[f"{prefix}Offensive_Points"] += contribution
                        elif group == "Banger":
                            self.df[f"{prefix}Banger_Points"] += contribution
            
            if self.league_id == 2:
                # Overall = Raw Total Points
                self.df[f"{prefix}Overall_Points"] = self.df[f"{prefix}Raw_Total_Points"]

                # FP/GP
                gp_col = f"{prefix}GP_all"
                if gp_col not in self.df.columns:
                    gp_col = f"{prefix}GP"
                if gp_col in self.df.columns:
                    gp_val = pd.to_numeric(self.df[gp_col], errors="coerce")
                    # Use Raw_Total_Points / GP. Replace 0 with NaN to avoid inf
                    self.df[f"{prefix}FP/GP"] = self.df[f"{prefix}Raw_Total_Points"] / gp_val.replace(0, np.nan)

                # Total Points T Score
                # Determine segmentation (D vs F vs Goalie)
                if is_skater:
                    # Skaters
                    scores = pd.Series(50, index=self.df.index, dtype="Int64")
                    for seg in ["D", "F"]:
                        mask = self.df["pos_group"] == seg
                        if mask.any():
                            scores.loc[mask] = self._percentile_score(self.df[f"{prefix}Raw_Total_Points"], mask).loc[mask]
                    self.df[f"{prefix}Total_Points_T_Score"] = scores
                else:
                    # Goalies
                    mask = pd.Series(True, index=self.df.index)
                    self.df[f"{prefix}Total_Points_T_Score"] = self._percentile_score(self.df[f"{prefix}Raw_Total_Points"], mask)
                    
                    # New requested columns for Goalies League 2
                    for s in ['W', 'GA', 'SV', 'SHO']:
                        if s in stat_points_map:
                            self.df[f"T_{prefix}{s}_FP"] = self._percentile_score(stat_points_map[s], mask)
                    
                    # /60 versions for Goalies
                    toi_col = f"{prefix}TOI"
                    if toi_col in self.df.columns:
                        toi_val = self.df[toi_col]
                        if not hasattr(toi_val, 'dt'):
                            toi_val = pd.to_timedelta(toi_val, errors='coerce')
                        
                        total_hours = toi_val.dt.total_seconds() / 3600.0
                        # Avoid div by zero
                        h_divisor = total_hours.replace(0, np.nan)
                        
                        for s in ['W', 'GA', 'SV', 'SHO']:
                            if s in stat_points_map:
                                rate_pts = stat_points_map[s] / h_divisor
                                self.df[f"T_{prefix}{s}_FP/60"] = self._percentile_score(rate_pts, mask)
                        
                        # Total FP/60
                        total_rate_pts = self.df[points_col] / h_divisor
                        self.df[f"T_{prefix}FP/60"] = self._percentile_score(total_rate_pts, mask)

                # Final renames for League 2
                self.df.rename(columns={
                    points_col: f"{prefix}FP",
                    f"{prefix}Total_Points_T_Score": f"T_{prefix}FP"
                }, inplace=True)
                
                # Column removals for Goalies only
                if not is_skater:
                    to_drop = [
                        f"{prefix}Raw_Total_Points",
                        f"{prefix}Offensive_Points",
                        f"{prefix}Banger_Points",
                        f"{prefix}Overall_Points"
                    ]
                    self.df.drop(columns=[c for c in to_drop if c in self.df.columns], inplace=True)

        return self.df
