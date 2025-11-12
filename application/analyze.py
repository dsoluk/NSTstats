import os
from typing import List, Dict

import numpy as np
import pandas as pd
import warnings

# Use non-interactive backend for headless PNG generation
try:
    import matplotlib
    matplotlib.use("Agg")  # safe for servers/CI
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:  # pragma: no cover - plotting is optional
    matplotlib = None
    plt = None
    sns = None

# Reuse stat groupings consistent with the project
POINTS_STATS = ["G", "A", "PPP", "SOG", "FOW"]
BANGERS_STATS = ["HIT", "BLK", "PIM"]
ALL_STATS = POINTS_STATS + BANGERS_STATS

# Overall score weights (user preference)
OVERALL_WEIGHTS = {"points": 0.7, "bangers": 0.3}


def _pos_group(val: str) -> str:
    s = str(val or "").strip().upper()
    return "D" if s == "D" else "F"


def _safe_sum(row: pd.Series, cols: List[str]) -> float:
    vals = [pd.to_numeric(row.get(c), errors="coerce") for c in cols]
    return float(np.nansum(vals))


def _season_total_cols(prefix: str, stat: str) -> List[str]:
    # Season total is ROW + NW + ROS for the given prefix ("", "Proj_", "LY_")
    return [f"{prefix}ROW_{stat}", f"{prefix}NW_{stat}", f"{prefix}ROS_{stat}"]


def build_season_totals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Compute season totals per stat and per source
    for stat in ALL_STATS:
        # Forecast
        out[f"SZN_{stat}"] = out.apply(lambda r: _safe_sum(r, _season_total_cols("", stat)), axis=1)
        # Projections
        out[f"SZN_Proj_{stat}"] = out.apply(lambda r: _safe_sum(r, _season_total_cols("Proj_", stat)), axis=1)
        # Last Year (optional columns may be missing)
        ly_cols = _season_total_cols("LY_", stat)
        if any(c in out.columns for c in ly_cols):
            out[f"SZN_LY_{stat}"] = out.apply(lambda r: _safe_sum(r, ly_cols), axis=1)
        else:
            out[f"SZN_LY_{stat}"] = np.nan

    # Per-GP normalized (82 GP baseline)
    for src in ["", "Proj_", "LY_"]:
        for stat in ALL_STATS:
            base = f"SZN_{src}{stat}" if src else f"SZN_{stat}"
            out[f"{base}_perGP"] = pd.to_numeric(out[base], errors="coerce") / 82.0

    # Position grouping F/D only
    pos_col = "Position" if "Position" in out.columns else ("position" if "position" in out.columns else None)
    if pos_col is None:
        out["Position"] = "F"
        pos_col = "Position"
    out["pos_group"] = out[pos_col].apply(_pos_group)

    # Scores (season totals)
    def _score(row: pd.Series, src_prefix: str, per_gp: bool = False) -> Dict[str, float]:
        suffix = "_perGP" if per_gp else ""
        pts = _safe_sum(row, [f"SZN_{src_prefix}{s}{suffix}" for s in POINTS_STATS])
        bgr = _safe_sum(row, [f"SZN_{src_prefix}{s}{suffix}" for s in BANGERS_STATS])
        overall = OVERALL_WEIGHTS["points"] * pts + OVERALL_WEIGHTS["bangers"] * bgr
        return {"points": pts, "bangers": bgr, "overall": overall}

    # Compute for forecast, projection, last year (where available)
    for per_gp in [False, True]:
        suf = "_perGP" if per_gp else ""
        # Forecast
        out[f"SZN_Points{suf}"] = out.apply(lambda r: _score(r, "", per_gp)["points"], axis=1)
        out[f"SZN_Bangers{suf}"] = out.apply(lambda r: _score(r, "", per_gp)["bangers"], axis=1)
        out[f"SZN_Overall{suf}"] = out.apply(lambda r: _score(r, "", per_gp)["overall"], axis=1)
        # Projection
        out[f"SZN_Points_Proj{suf}"] = out.apply(lambda r: _score(r, "Proj_", per_gp)["points"], axis=1)
        out[f"SZN_Bangers_Proj{suf}"] = out.apply(lambda r: _score(r, "Proj_", per_gp)["bangers"], axis=1)
        out[f"SZN_Overall_Proj{suf}"] = out.apply(lambda r: _score(r, "Proj_", per_gp)["overall"], axis=1)
        # Last Year
        out[f"SZN_Points_LY{suf}"] = out.apply(lambda r: _score(r, "LY_", per_gp)["points"], axis=1)
        out[f"SZN_Bangers_LY{suf}"] = out.apply(lambda r: _score(r, "LY_", per_gp)["bangers"], axis=1)
        out[f"SZN_Overall_LY{suf}"] = out.apply(lambda r: _score(r, "LY_", per_gp)["overall"], axis=1)

    return out


def metrics_table(pair_df: pd.DataFrame, ref_col: str, pred_col: str) -> Dict[str, float]:
    x = pd.to_numeric(pair_df[pred_col], errors="coerce")
    y = pd.to_numeric(pair_df[ref_col], errors="coerce")
    mask = ~(x.isna() | y.isna())
    x = x[mask]
    y = y[mask]
    n = int(len(x))
    if n == 0:
        return {"count": 0, "bias": np.nan, "mae": np.nan, "rmse": np.nan, "mape": np.nan, "pearson_r": np.nan, "cal_intercept": np.nan, "cal_slope": np.nan}
    err = x - y
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    # Safe MAPE: only on rows where reference != 0; if none, set NaN without warning
    y_nonzero = y.replace(0, np.nan)
    frac = np.abs(err) / y_nonzero
    if np.all(np.isnan(frac)):
        mape = np.nan
    else:
        mape = float(np.nanmean(frac))
    bias = float(np.mean(err))
    # Safe Pearson r: require at least 2 points and non-zero std on both vectors
    try:
        if n < 2 or float(np.nanstd(x)) == 0.0 or float(np.nanstd(y)) == 0.0:
            pearson_r = np.nan
        else:
            pearson_r = float(np.corrcoef(x, y)[0, 1])
    except Exception:
        pearson_r = np.nan
    # OLS calibration (y = a + b*x)
    X = np.column_stack([np.ones_like(x), x.values])
    coef, *_ = np.linalg.lstsq(X, y.values, rcond=None)
    return {
        "count": n,
        "bias": bias,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "pearson_r": pearson_r,
        "cal_intercept": float(coef[0]),
        "cal_slope": float(coef[1]),
    }


def evaluate(compare_csv: str = os.path.join("data", "compare.csv"), out_dir: str = os.path.join("data", "eval")) -> str:
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(compare_csv)

    # Build SZN totals and scores
    df2 = build_season_totals(df)

    # Load prior-season per-60 and TOI/GP to compute 82-game LY baseline (optional)
    def _norm_name(x: str) -> str:
        s = str(x or "").strip().lower()
        for ch in [".", ",", "'", '"', "`"]:
            s = s.replace(ch, "")
        return " ".join(s.split())

    def _parse_toi(v) -> float:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        s = str(v)
        if s == "" or s.lower() == "nan":
            return np.nan
        try:
            return float(s)
        except Exception:
            pass
        try:
            if "days" in s:
                parts = s.split()
                hms = parts[-1]
                hh, mm, ss = hms.split(":")
                return int(hh) * 60 + int(mm) + float(ss) / 60.0
            if s.count(":") == 2:
                hh, mm, ss = s.split(":")
                return int(hh) * 60 + int(mm) + float(ss) / 60.0
            if s.count(":") == 1:
                mm, ss = s.split(":")
                return int(mm) + float(ss) / 60.0
        except Exception:
            return np.nan
        return np.nan

    prior_path = os.path.join("data", "merged_skaters_prior.csv")
    ly82 = pd.DataFrame()
    if os.path.exists(prior_path):
        prior = pd.read_csv(prior_path)
        p = prior.copy()
        if "Player" in p.columns:
            p["_key_player"] = p["Player"].apply(_norm_name)
        else:
            p["_key_player"] = ""
        # Compute LY per-game from per-60 using TOI/GP
        # Parse TOI/GP strings directly; do not coerce to numeric before parsing, as formats like "0:20:57" would become NaN
        if "szn_TOI/GP_all" in p.columns:
            toi_all = p["szn_TOI/GP_all"].apply(_parse_toi)
        else:
            toi_all = pd.Series(np.nan, index=p.index)
        if "szn_TOI/GP_pp" in p.columns:
            toi_pp = p["szn_TOI/GP_pp"].apply(_parse_toi)
        else:
            toi_pp = pd.Series(np.nan, index=p.index)
        cols = {s: p.get(f"szn_{s}") for s in ALL_STATS}
        ly_pg: Dict[str, pd.Series] = {}
        for s, series in cols.items():
            rate = pd.to_numeric(series, errors="coerce")
            if s == "PPP":
                minutes = toi_pp.fillna(toi_all)
            else:
                minutes = toi_all
            ly_pg[s] = rate * (pd.to_numeric(minutes, errors="coerce") / 60.0)
        ly_df = pd.DataFrame({"_key_player": p["_key_player"]})
        for s in ALL_STATS:
            ly_df[f"LY82_{s}"] = (ly_pg[s] * 82.0).astype(float)
        # Group totals
        ly_df["LY82_Points"] = ly_df[[f"LY82_{s}" for s in POINTS_STATS]].sum(axis=1)
        ly_df["LY82_Bangers"] = ly_df[[f"LY82_{s}" for s in BANGERS_STATS]].sum(axis=1)
        ly_df["LY82_Overall"] = (
            OVERALL_WEIGHTS["points"] * ly_df["LY82_Points"] + OVERALL_WEIGHTS["bangers"] * ly_df["LY82_Bangers"]
        )
        ly82 = ly_df.groupby("_key_player", as_index=False).first()

        # Attach key to df2 for merge
        if "Player" in df2.columns:
            df2["_key_player"] = df2["Player"].apply(_norm_name)
            df2 = df2.merge(ly82, on="_key_player", how="left")
            df2.drop(columns=["_key_player"], inplace=True)

    # Focus on skaters, segment positions F/D
    id_cols = [c for c in ["Player", "Team", "Position"] if c in df2.columns]

    # Prepare pair tables for totals and scores (forecast vs projection)
    def _pair_cols(base: str, ly_col: str = None) -> pd.DataFrame:
        cols = id_cols + [f"SZN_{base}", f"SZN_{base}_Proj"]
        if ly_col and ly_col in df2.columns:
            cols.append(ly_col)
        sub = df2[[c for c in cols if c in df2.columns]].copy()
        rename_map = {f"SZN_{base}": "forecast", f"SZN_{base}_Proj": "projection"}
        if ly_col and ly_col in sub.columns:
            rename_map[ly_col] = "last_year"
        sub.rename(columns=rename_map, inplace=True)
        return sub

    pairs = {
        "Overall": _pair_cols("Overall", ly_col="LY82_Overall"),
        "Points": _pair_cols("Points", ly_col="LY82_Points"),
        "Bangers": _pair_cols("Bangers", ly_col="LY82_Bangers"),
    }
    # Also include stat-level totals
    for stat in ALL_STATS:
        base = stat
        cols = id_cols + [f"SZN_{base}", f"SZN_Proj_{base}"]
        if f"LY82_{base}" in df2.columns:
            cols.append(f"LY82_{base}")
        sub = df2[[c for c in cols if c in df2.columns]].copy()
        ren = {f"SZN_{base}": "forecast", f"SZN_Proj_{base}": "projection"}
        if f"LY82_{base}" in sub.columns:
            ren[f"LY82_{base}"] = "last_year"
        sub.rename(columns=ren, inplace=True)
        # Add explicit stat column for clarity in downstream evaluation
        sub.insert(len(id_cols), "stat", stat)
        pairs[f"STAT_{stat}"] = sub

    # Compute metrics overall, by pos_group, and by forecast deciles (using Overall)
    summaries = []
    by_pos_rows = []
    by_dec_rows = []

    overall_pair = pairs["Overall"].copy()
    if "Position" in overall_pair.columns:
        overall_pair["pos_group"] = overall_pair["Position"].apply(_pos_group)
    else:
        overall_pair["pos_group"] = "F"
    # Deciles by forecast overall
    tmp = overall_pair.copy()
    tmp["tier"] = pd.qcut(pd.to_numeric(tmp["forecast"], errors="coerce"), 10, labels=False, duplicates="drop")

    for name, sub in pairs.items():
        # Align pos_group and tier based on player
        sub = sub.merge(overall_pair[["Player", "pos_group"]], on="Player", how="left") if "Player" in sub.columns else sub
        # Overall
        m = metrics_table(sub, ref_col="projection", pred_col="forecast")
        summaries.append({"target": name, **m})
        # By position
        if "pos_group" in sub.columns:
            for pg, g in sub.groupby("pos_group"):
                m2 = metrics_table(g, ref_col="projection", pred_col="forecast")
                by_pos_rows.append({"target": name, "pos_group": pg, **m2})
        # By decile (use tiers defined from Overall forecast, join on Player)
        if "Player" in sub.columns and "Player" in tmp.columns:
            sub2 = sub.merge(tmp[["Player", "tier"]], on="Player", how="left")
            for t, g in sub2.groupby("tier"):
                m3 = metrics_table(g, ref_col="projection", pred_col="forecast")
                by_dec_rows.append({"target": name, "tier": int(t) if pd.notna(t) else None, **m3})

    summary_df = pd.DataFrame(summaries)
    by_pos_df = pd.DataFrame(by_pos_rows)
    by_dec_df = pd.DataFrame(by_dec_rows)

    # Save CSVs
    summary_csv = os.path.join(out_dir, "metrics_summary.csv")
    by_pos_csv = os.path.join(out_dir, "metrics_by_position.csv")
    by_dec_csv = os.path.join(out_dir, "metrics_by_decile.csv")
    summary_df.to_csv(summary_csv, index=False)
    by_pos_df.to_csv(by_pos_csv, index=False)
    by_dec_df.to_csv(by_dec_csv, index=False)

    # PNG plots
    plots_dir = os.path.join(out_dir, "_plots")
    os.makedirs(plots_dir, exist_ok=True)

    def _scatter_plot(dfp: pd.DataFrame, title: str, outfile: str):
        if plt is None or sns is None:
            return
        sub = dfp.copy()
        sub = sub.rename(columns={"forecast": "Forecast", "projection": "Projection"})
        sub = sub.dropna(subset=["Forecast", "Projection"])
        if sub.empty:
            return
        # Determine whether LOWESS is safe/meaningful (avoid statsmodels warnings on degenerate inputs)
        x = pd.to_numeric(sub["Forecast"], errors="coerce")
        y = pd.to_numeric(sub["Projection"], errors="coerce")
        n = len(sub)
        use_lowess = (
            n >= 10
            and np.isfinite(x).sum() >= 10
            and np.isfinite(y).sum() >= 10
            and (pd.Series(x).nunique(dropna=True) >= 5)
            and float(np.nanstd(x)) > 0.0
            and float(np.nanstd(y)) > 0.0
        )

        plt.figure(figsize=(6, 6))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            # If lowess isn't safe, fall back to a simple OLS fit (lowess=False)
            sns.regplot(
                data=sub,
                x="Forecast",
                y="Projection",
                lowess=use_lowess,
                scatter_kws={"alpha": 0.5, "s": 20},
                line_kws={"color": "tab:orange"},
            )
        mn = float(np.nanmin([sub["Forecast"].min(), sub["Projection"].min()]))
        mx = float(np.nanmax([sub["Forecast"].max(), sub["Projection"].max()]))
        plt.plot([mn, mx], [mn, mx], "k--", lw=1)
        plt.title(title)
        plt.xlabel("Forecast")
        plt.ylabel("Projection")
        plt.tight_layout()
        plt.savefig(outfile, dpi=150)
        plt.close()

    def _residuals_by_decile(dfp: pd.DataFrame, deciles_map: pd.DataFrame, title: str, outfile: str):
        if plt is None or sns is None:
            return
        sub = dfp.copy()
        if "Player" in sub.columns and "Player" in deciles_map.columns:
            sub = sub.merge(deciles_map[["Player", "tier"]], on="Player", how="left")
        sub["resid"] = pd.to_numeric(sub.get("forecast"), errors="coerce") - pd.to_numeric(sub.get("projection"), errors="coerce")
        sub = sub.dropna(subset=["resid", "tier"])
        if sub.empty:
            return
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=sub, x="tier", y="resid")
        plt.title(title)
        plt.xlabel("Forecast decile (Overall)")
        plt.ylabel("Residual (Forecast - Projection)")
        plt.tight_layout()
        plt.savefig(outfile, dpi=150)
        plt.close()

    # Build deciles map from Overall forecast once
    overall_deciles_map = tmp[["Player", "tier"]] if "Player" in tmp.columns else pd.DataFrame()

    for name, sub in pairs.items():
        base = name.replace("/", "-")
        _scatter_plot(sub, f"{name}: Forecast vs Projection (SZN)", os.path.join(plots_dir, f"{base}_scatter.png"))
        _residuals_by_decile(sub, overall_deciles_map, f"{name}: Residuals by Forecast Decile", os.path.join(plots_dir, f"{base}_residuals_by_decile.png"))

    # Build consolidated STAT_All sheet with explicit 'stat' column
    stat_frames = [df.assign(stat=stat) if 'stat' not in df.columns else df for key, df in pairs.items() if key.startswith('STAT_') for stat in [key.split('_', 1)[1]]]
    stat_all = pd.concat(stat_frames, ignore_index=True) if stat_frames else pd.DataFrame()

    # Excel workbook for pivots (rounded integers for forecast/projection/last_year)
    def _finalize_for_excel(name: str, sub: pd.DataFrame) -> pd.DataFrame:
        out = sub.copy()
        # Ensure last_year exists
        if "last_year" not in out.columns:
            # try to map appropriate LY column if present
            if name in ("Overall", "Points", "Bangers") and f"LY82_{name}" in df2.columns:
                out["last_year"] = df2.loc[out.index, f"LY82_{name}"]
        # Round up numeric columns
        for c in ["forecast", "projection", "last_year"]:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")
                out[c] = np.ceil(out[c]).astype("Int64")
        # Column order
        base_cols = [c for c in ["Player", "Team", "Position"] if c in out.columns]
        if name.startswith("STAT_"):
            order = base_cols + (["stat"] if "stat" in out.columns else []) + [c for c in ["forecast", "projection", "last_year"] if c in out.columns]
        else:
            order = base_cols + [c for c in ["forecast", "projection", "last_year"] if c in out.columns]
        return out[order]

    xlsx_path = os.path.join(out_dir, "season_total_eval.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as xw:
        # Pair sheets (e.g., Overall, Points, Bangers, STAT_G, ...)
        for name, sub in pairs.items():
            sub2 = _finalize_for_excel(name, sub)
            sub2.to_excel(xw, sheet_name=name[:31], index=False)
        # Consolidated stat sheet
        if not stat_all.empty:
            stat_all2 = _finalize_for_excel("STAT_All", stat_all)
            stat_all2.to_excel(xw, sheet_name="STAT_All", index=False)
        # Metrics sheets — use unique names to avoid case-insensitive clashes with pair sheets
        summary_df.to_excel(xw, sheet_name="metrics_overall", index=False)
        by_pos_df.to_excel(xw, sheet_name="metrics_by_position", index=False)
        by_dec_df.to_excel(xw, sheet_name="metrics_by_decile", index=False)

    return xlsx_path
