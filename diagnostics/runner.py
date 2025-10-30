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
    def f(x):
        s = str(x).strip().upper()
        return "D" if s == "D" else "F"
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
) -> str:
    """
    Run distribution diagnostics for selected metrics and windows.
    - metrics: list of base metric names without prefix (defaults to DEFAULT_METRICS)
    - windows: e.g., ["szn"] (supported: "szn", "l7"). Defaults to ["szn"].
    Returns the output directory path created.
    """
    if metrics is None:
        metrics = list(DEFAULT_METRICS)
    if windows is None:
        windows = ["szn"]

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input not found: {input_csv}")
    df = pd.read_csv(input_csv)

    pos_col = _detect_position_col(df)
    df["pos_group"] = _derive_pos_group(df, pos_col)

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
            for seg in ["F", "D"]:
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
                    }
                    for k, v in res.params.items():
                        row[f"param_{k}"] = v
                    summary_rows.append(row)

    # Write summary files
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        csv_path = os.path.join(root, "summary.csv")
        summary_df.to_csv(csv_path, index=False)
        # Also write JSON grouped compactly
        out_json: Dict[Tuple[str, str, str], List[Dict[str, object]]] = {}
        for r in summary_rows:
            key = (r["window"], r["segment"], r["metric"])  # type: ignore
            out_json.setdefault(tuple(key), []).append(r)
        # Convert tuple keys to strings
        json_dict = {f"{w}|{s}|{m}": rows for (w, s, m), rows in out_json.items()}
        with open(os.path.join(root, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(json_dict, f, indent=2)
    return root
