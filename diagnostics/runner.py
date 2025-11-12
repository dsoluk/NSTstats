import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .distributions import (
    FitResult,
    best_fit,
    fit_gamma,
    fit_lognormal,
    fit_normal,
    fit_normal_on_log1p,
)
from .plots import plot_hist_with_pdfs, qq_plot_normal

try:
    import scipy.stats as ss  # type: ignore
except Exception:  # pragma: no cover
    ss = None  # type: ignore


DEFAULT_METRICS = ["G", "A", "PPP", "SOG", "FOW", "HIT", "BLK", "PIM"]


def _detect_position_col(df: pd.DataFrame) -> str:
    for candidate in ["Position", "position", "POS", "pos"]:
        if candidate in df.columns:
            return candidate
    df["Position"] = "F"
    return "Position"


def _derive_pos_group(df: pd.DataFrame, pos_col: str) -> pd.Series:
    import re
    def f(x):
        s = str(x).strip().upper()
        # Common defense encodings seen across sources
        if s in {"D", "LD", "RD", "D/F", "F/D"}:
            return "D"
        # Split on common separators and whitespace, then look for a standalone 'D' token
        tokens = [t for t in re.split(r"[/,;|\s]+", s) if t]
        if "D" in tokens:
            return "D"
        # Default all other skater positions (C, LW, RW, F, etc.) to forwards
        return "F"
    return df[pos_col].apply(f)


def _pdf_builders(params: Dict[str, float]):
    # returns a dict of label -> callable(X)->pdf
    pdfs: Dict[str, callable] = {}
    if "mu" in params and "sigma" in params:
        mu = params["mu"]; sigma = max(params["sigma"], 1e-9)
        def n_pdf(X):
            return (1.0 / (np.sqrt(2*np.pi)*sigma)) * np.exp(-0.5*((X-mu)/sigma)**2)
        pdfs["Normal"] = n_pdf
    if "log_mu" in params and "log_sigma" in params:
        lm = params["log_mu"]; ls = max(params["log_sigma"], 1e-9)
        def ln_pdf(X):
            Xp = np.clip(X, 1e-12, None)
            return (1.0/(Xp*ls*np.sqrt(2*np.pi))) * np.exp(-0.5*((np.log(Xp)-lm)/ls)**2)
        pdfs["LogNormal"] = ln_pdf
    if "k" in params and "theta" in params:
        k = max(params["k"], 1e-9); th = max(params["theta"], 1e-12)
        # Gamma(k, theta) with support x>0
        def g_pdf(X):
            Xp = np.clip(X, 1e-12, None)
            # Use a numerically stable log-PDF and a robust gamma function implementation
            try:
                from scipy.special import gammaln  # type: ignore
                gln = gammaln
            except Exception:  # pragma: no cover
                import math
                gln = math.lgamma
            logY = (k - 1.0) * np.log(Xp) - (Xp / th) - k * np.log(th) - gln(k)
            return np.exp(logY)
        pdfs["Gamma"] = g_pdf
    return pdfs


def _fit_all(x: np.ndarray) -> Dict[str, FitResult]:
    results: Dict[str, FitResult] = {}
    if x.size == 0:
        return results
    # Raw fits
    results["Normal"] = fit_normal(x)
    # Only positive-support fits if data has positives
    if np.any(x > 0):
        results["LogNormal"] = fit_lognormal(x)
        results["Gamma"] = fit_gamma(x)
    # Normal on log1p
    results["Normal(log1p)"] = fit_normal_on_log1p(x)
    return results


essential_cols = ["Player", "Team"]


def run_dq(
    input_csv: str = os.path.join("data", "skaters_scored.csv"),
    out_dir: str = os.path.join("data", "dq"),
    metrics: List[str] = None,
    windows: List[str] = None,
    min_n: int = 25,
    prior_input_csv: str | None = None,
    # New options to support goalies and appending to an existing timestamp dir
    group_by_position: bool = True,
    segments: List[str] | None = None,
    existing_root: str | None = None,
) -> str:
    """
    Run distribution diagnostics for selected metrics and windows.
    - metrics: list of base metric names without prefix (defaults to DEFAULT_METRICS)
    - windows: e.g., ["szn"] (supported: "szn", "l7"). Defaults to ["szn"].
    - prior_input_csv: optional path to a prior-season CSV with the same schema to compare best-fit distributions.
    Returns the output directory path created.
    """
    if metrics is None:
        metrics = list(DEFAULT_METRICS)
    if windows is None:
        # Default to analyzing both season-to-date and last-7 windows
        windows = ["szn", "l7"]

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input not found: {input_csv}")
    df = pd.read_csv(input_csv)

    # Determine segmentation
    if group_by_position:
        # Prefer an existing pos_group column if already prepared by upstream pipelines
        if "pos_group" in df.columns:
            # Normalize formatting but do not re-derive
            df["pos_group"] = df["pos_group"].astype(str).str.strip().str.upper()
            if segments is not None:
                seg_values = segments
            else:
                # Derive segments from the data, keep only standard skater groups
                seg_values = [s for s in pd.unique(df["pos_group"]) if s in ("F", "D")]
                if not seg_values:
                    seg_values = ["F", "D"]
        else:
            pos_col = _detect_position_col(df)
            df["pos_group"] = _derive_pos_group(df, pos_col)
            seg_values = segments if segments is not None else ["F", "D"]
    else:
        # Single-segment dataset (e.g., goalies). Use provided segments or default to ["G"].
        seg_values = segments if segments is not None else ["G"]
        df["pos_group"] = seg_values[0] if len(seg_values) == 1 else "All"

    # Use existing root if provided (to append into same summary), else create a new timestamped folder
    if existing_root:
        root = existing_root
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = os.path.join(out_dir, ts)
    os.makedirs(root, exist_ok=True)

    # Prepare summary rows
    summary_rows: List[Dict[str, object]] = []

    for win in windows:
        prefix = f"{win}_"
        for metric in metrics:
            col = f"{prefix}{metric}"
            if col not in df.columns:
                continue
            for seg in seg_values:
                subset = df.loc[df["pos_group"] == seg, col]
                # Clean values: finite only
                x = subset.replace([np.inf, -np.inf], np.nan).dropna().astype(float).values
                if x.size < min_n:
                    continue
                # Optional outlier trimming could be added here; currently off

                fits = _fit_all(x)
                if not fits:
                    continue
                # Choose best by AIC
                choice = best_fit(fits, criterion="AIC")
                # Build PDFs for histogram overlay using chosen params and also others
                # Use individual PDFs
                pdfs = {}
                for name, res in fits.items():
                    pdfs[name] = _pdf_builders(res.params).get(name, None)
                # Remove None entries
                pdfs = {k: v for k, v in pdfs.items() if v is not None}

                # Save plots
                subdir = os.path.join(root, metric)
                os.makedirs(subdir, exist_ok=True)
                title = f"{metric} ({win}, {seg}) — best: {choice}"
                hist_path = os.path.join(subdir, f"{win}_{seg}_hist.png")
                plot_hist_with_pdfs(pd.Series(x), pdfs, bins=30, title=title, out_path=hist_path)

                # QQ plots for raw and log1p
                qq_raw_path = os.path.join(subdir, f"{win}_{seg}_qq_raw.png")
                qq_plot_normal(pd.Series(x), title=f"QQ Normal — {metric} ({win}, {seg}) raw", out_path=qq_raw_path)
                x_log1p = np.log1p(np.clip(x, 0, None))
                qq_log_path = os.path.join(subdir, f"{win}_{seg}_qq_log1p.png")
                qq_plot_normal(pd.Series(x_log1p), title=f"QQ Normal — {metric} ({win}, {seg}) log1p", out_path=qq_log_path)

                # Record summary per fit
                for name, res in fits.items():
                    row = {
                        "window": win,
                        "segment": seg,
                        "metric": metric,
                        "n": res.n,
                        "fit": name,
                        "loglik": res.loglik,
                        "aic": res.aic,
                        "bic": res.bic,
                        "ks_pvalue": res.ks_pvalue,
                        "chosen": 1 if name == choice else 0,
                        "season_label": "current",
                    }
                    for k, v in res.params.items():
                        row[f"param_{k}"] = v
                    summary_rows.append(row)

    # Optional prior-season comparison
    prior_summary_rows: List[Dict[str, object]] = []
    if prior_input_csv:
        if not os.path.exists(prior_input_csv):
            raise FileNotFoundError(f"Prior input not found: {prior_input_csv}")
        prior_df = pd.read_csv(prior_input_csv)
        if group_by_position:
            if "pos_group" in prior_df.columns:
                prior_df["pos_group"] = prior_df["pos_group"].astype(str).str.strip().str.upper()
                if segments is not None:
                    prior_seg_values = segments
                else:
                    prior_seg_values = [s for s in pd.unique(prior_df["pos_group"]) if s in ("F", "D")]
                    if not prior_seg_values:
                        prior_seg_values = ["F", "D"]
            else:
                pos_col_prior = _detect_position_col(prior_df)
                prior_df["pos_group"] = _derive_pos_group(prior_df, pos_col_prior)
                prior_seg_values = segments if segments is not None else ["F", "D"]
        else:
            prior_seg_values = segments if segments is not None else ["G"]
            prior_df["pos_group"] = prior_seg_values[0] if len(prior_seg_values) == 1 else "All"
        for win in windows:
            prefix = f"{win}_"
            for metric in metrics:
                col = f"{prefix}{metric}"
                if col not in prior_df.columns:
                    continue
                for seg in prior_seg_values:
                    subset = prior_df.loc[prior_df["pos_group"] == seg, col]
                    x = subset.replace([np.inf, -np.inf], np.nan).dropna().astype(float).values
                    if x.size < min_n:
                        continue
                    fits = _fit_all(x)
                    if not fits:
                        continue
                    choice = best_fit(fits, criterion="AIC")
                    for name, res in fits.items():
                        row = {
                            "window": win,
                            "segment": seg,
                            "metric": metric,
                            "n": res.n,
                            "fit": name,
                            "loglik": res.loglik,
                            "aic": res.aic,
                            "bic": res.bic,
                            "ks_pvalue": res.ks_pvalue,
                            "chosen": 1 if name == choice else 0,
                            "season_label": "prior",
                        }
                        for k, v in res.params.items():
                            row[f"param_{k}"] = v
                        prior_summary_rows.append(row)

    # Write summary files
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        csv_path = os.path.join(root, "summary.csv")
        if os.path.exists(csv_path):
            try:
                existing_df = pd.read_csv(csv_path)
                summary_df = pd.concat([existing_df, summary_df], ignore_index=True)
            except Exception:
                pass
        summary_df.to_csv(csv_path, index=False)
        # Also write JSON grouped compactly (append/merge if exists)
        out_json: Dict[Tuple[str, str, str], List[Dict[str, object]]] = {}
        for r in summary_rows:
            key = (r["window"], r["segment"], r["metric"])  # type: ignore
            out_json.setdefault(tuple(key), []).append(r)
        # Load existing json if present and merge
        json_path = os.path.join(root, "summary.json")
        merged_json: Dict[str, List[Dict[str, object]]] = {}
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    merged_json = json.load(f)
            except Exception:
                merged_json = {}
        # Convert tuple keys to strings and merge
        for (w, s, m), rows in out_json.items():
            k = f"{w}|{s}|{m}"
            merged_json.setdefault(k, [])
            merged_json[k].extend(rows)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(merged_json, f, indent=2)

    # Write prior summary if present
    if prior_summary_rows:
        prior_df_out = pd.DataFrame(prior_summary_rows)
        prior_csv_path = os.path.join(root, "summary_prior.csv")
        # Append if exists
        if os.path.exists(prior_csv_path):
            try:
                _existing = pd.read_csv(prior_csv_path)
                prior_df_out = pd.concat([_existing, prior_df_out], ignore_index=True)
            except Exception:
                pass
        prior_df_out.to_csv(prior_csv_path, index=False)
        # Comparison of best fits
        try:
            cur_best = (
                pd.DataFrame(summary_rows)
                .query("chosen == 1")
                .loc[:, ["window","segment","metric","fit"]]
                .rename(columns={"fit":"fit_current"})
            )
            prior_best = (
                prior_df_out
                .query("chosen == 1")
                .loc[:, ["window","segment","metric","fit"]]
                .rename(columns={"fit":"fit_prior"})
            )
            cmp_df = cur_best.merge(prior_best, on=["window","segment","metric"], how="outer")
            cmp_df["different"] = (cmp_df["fit_current"] != cmp_df["fit_prior"]).astype(bool)
            cmp_csv = os.path.join(root, "best_fit_comparison.csv")
            cmp_df.to_csv(cmp_csv, index=False)
        except Exception as _e:
            print(f"[Warn] Could not build comparison: {_e}")

    return root


def run_dq_prior(
    input_csv: str,
    out_dir: str = os.path.join("data", "dq", "prior"),
    metrics: List[str] | None = None,
    windows: List[str] | None = None,
    min_n: int = 25,
    group_by_position: bool = True,
    segments: List[str] | None = None,
    existing_ts: str | None = None,
) -> str:
    """
    Run distribution diagnostics for PRIOR-season inputs and write results under data/dq/prior.
    - Defaults to only the 'szn' window for prior (no 'l7').
    - Returns the output directory path created under out_dir/<ts>.
    """
    if metrics is None:
        metrics = list(DEFAULT_METRICS)
    # For prior, default to season-only unless explicitly specified
    if windows is None:
        windows = ["szn"]

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input not found: {input_csv}")
    df = pd.read_csv(input_csv)

    # Determine segmentation
    if group_by_position:
        # Prefer an existing pos_group if present in the prior CSV (skaters)
        if "pos_group" in df.columns:
            df["pos_group"] = df["pos_group"].astype(str).str.strip().str.upper()
            if segments is not None:
                seg_values = segments
            else:
                seg_values = [s for s in pd.unique(df["pos_group"]) if s in ("F", "D")]
                if not seg_values:
                    seg_values = ["F", "D"]
        else:
            pos_col = _detect_position_col(df)
            df["pos_group"] = _derive_pos_group(df, pos_col)
            seg_values = segments if segments is not None else ["F", "D"]
    else:
        seg_values = segments if segments is not None else ["G"]
        df["pos_group"] = seg_values[0] if len(seg_values) == 1 else "All"

    # Build root under dq/prior using provided ts for alignment if given
    ts = existing_ts or datetime.now().strftime("%Y%m%d_%H%M%S")
    root = os.path.join(out_dir, ts)
    os.makedirs(root, exist_ok=True)

    summary_rows: List[Dict[str, object]] = []

    for win in windows:
        prefix = f"{win}_"
        for metric in metrics:
            col = f"{prefix}{metric}"
            if col not in df.columns:
                continue
            for seg in seg_values:
                subset = df.loc[df["pos_group"] == seg, col]
                x = subset.replace([np.inf, -np.inf], np.nan).dropna().astype(float).values
                if x.size < min_n:
                    continue
                fits = _fit_all(x)
                if not fits:
                    continue
                choice = best_fit(fits, criterion="AIC")

                # Save plots (mirror current behavior) under prior root
                subdir = os.path.join(root, metric)
                os.makedirs(subdir, exist_ok=True)
                title = f"{metric} ({win}, {seg}) — best: {choice} [PRIOR]"
                hist_path = os.path.join(subdir, f"{win}_{seg}_hist.png")
                pdfs = {name: fn for name, res in fits.items() for n, fn in _pdf_builders(res.params).items() if name == n}
                plot_hist_with_pdfs(pd.Series(x), pdfs, bins=30, title=title, out_path=hist_path)
                qq_raw_path = os.path.join(subdir, f"{win}_{seg}_qq_raw.png")
                qq_plot_normal(pd.Series(x), title=f"QQ Normal — {metric} ({win}, {seg}) raw [PRIOR]", out_path=qq_raw_path)
                x_log1p = np.log1p(np.clip(x, 0, None))
                qq_log_path = os.path.join(subdir, f"{win}_{seg}_qq_log1p.png")
                qq_plot_normal(pd.Series(x_log1p), title=f"QQ Normal — {metric} ({win}, {seg}) log1p [PRIOR]", out_path=qq_log_path)

                # Record summary per fit
                for name, res in fits.items():
                    row = {
                        "window": win,
                        "segment": seg,
                        "metric": metric,
                        "n": res.n,
                        "fit": name,
                        "loglik": res.loglik,
                        "aic": res.aic,
                        "bic": res.bic,
                        "ks_pvalue": res.ks_pvalue,
                        "chosen": 1 if name == choice else 0,
                        "season_label": "prior",
                    }
                    for k, v in res.params.items():
                        row[f"param_{k}"] = v
                    summary_rows.append(row)

    # Write prior summary
    if summary_rows:
        prior_df_out = pd.DataFrame(summary_rows)
        prior_csv_path = os.path.join(root, "summary.csv")
        if os.path.exists(prior_csv_path):
            try:
                _existing = pd.read_csv(prior_csv_path)
                prior_df_out = pd.concat([_existing, prior_df_out], ignore_index=True)
            except Exception:
                pass
        prior_df_out.to_csv(prior_csv_path, index=False)

        # Also JSON grouped
        out_json: Dict[Tuple[str, str, str], List[Dict[str, object]]] = {}
        for r in summary_rows:
            key = (r["window"], r["segment"], r["metric"])  # type: ignore
            out_json.setdefault(tuple(key), []).append(r)
        json_path = os.path.join(root, "summary.json")
        merged_json: Dict[str, List[Dict[str, object]]] = {}
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    merged_json = json.load(f)
            except Exception:
                merged_json = {}
        for (w, s, m), rows in out_json.items():
            k = f"{w}|{s}|{m}"
            merged_json.setdefault(k, [])
            merged_json[k].extend(rows)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(merged_json, f, indent=2)

    return root
