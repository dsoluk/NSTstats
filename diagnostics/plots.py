import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

try:
    import seaborn as sns  # type: ignore
    _HAS_SNS = True
except Exception:  # pragma: no cover
    sns = None  # type: ignore
    _HAS_SNS = False


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)


def plot_hist_with_pdfs(series: pd.Series,
                        pdfs: Dict[str, np.ndarray],
                        bins: int,
                        title: str,
                        out_path: str,
                        xrange: Optional[tuple] = None) -> None:
    """
    series: raw numeric values used to estimate fits (after cleaning)
    pdfs: mapping from label -> pdf(x) evaluated on grid X (same for all)
    bins: number of histogram bins
    xrange: optional (xmin, xmax) to limit display
    """
    vals = series.dropna().astype(float).values
    if vals.size == 0:
        return
    X = None
    # Infer X from any provided pdf array length by reconstructing from range
    # We'll compute grid from data range to keep PDFs aligned
    xmin = np.min(vals) if xrange is None else xrange[0]
    xmax = np.max(vals) if xrange is None else xrange[1]
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin == xmax:
        # expand a bit to render a usable plot
        xmin = float(xmin) if np.isfinite(xmin) else 0.0
        xmax = float(xmax) if np.isfinite(xmax) else (xmin + 1.0)
        xmax = xmin + (abs(xmin) * 0.1 + 1.0)
    X = np.linspace(xmin, xmax, 400)

    _ensure_dir(out_path)
    plt.figure(figsize=(8, 5))
    if _HAS_SNS:
        sns.histplot(vals, bins=bins, stat='density', color='#c7d2fe', edgecolor='white')
    else:
        plt.hist(vals, bins=bins, density=True, color='#c7d2fe', edgecolor='white')

    for label, pdf in pdfs.items():
        # pdf may be a callable expecting X, or a precomputed array
        if callable(pdf):
            Y = pdf(X)
        else:
            # If array passed, assume it's aligned to X grid defined here is not possible, so recompute as callable required
            continue
        plt.plot(X, Y, label=label, linewidth=2)

    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def qq_plot_normal(series: pd.Series, title: str, out_path: str) -> None:
    vals = series.dropna().astype(float).values
    if vals.size < 3:
        return
    # Compute theoretical quantiles for N(0,1) and standardized sample quantiles
    mu = np.mean(vals)
    sd = np.std(vals, ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return
    z = (vals - mu) / sd
    z = np.sort(z)
    n = z.size
    probs = (np.arange(1, n + 1) - 0.5) / n
    try:
        from scipy.stats import norm  # type: ignore
        theo = norm.ppf(probs)
    except Exception:
        # Fallback using numpy inverse error function approximation
        from math import sqrt
        # Approximate inverse CDF via erfinv if available
        try:
            import numpy as _np
            theo = sqrt(2) * _np.erfinv(2 * probs - 1)
        except Exception:
            # Last resort: use a linear stretch to [-3,3]
            theo = 6 * probs - 3
    _ensure_dir(out_path)
    plt.figure(figsize=(6, 6))
    plt.scatter(theo, z, s=8, alpha=0.7)
    # 45-degree line
    lo = min(theo.min(), z.min())
    hi = max(theo.max(), z.max())
    plt.plot([lo, hi], [lo, hi], color='red', linestyle='--', linewidth=1)
    plt.title(title)
    plt.xlabel('Theoretical Quantiles (N(0,1))')
    plt.ylabel('Sample Quantiles (standardized)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
