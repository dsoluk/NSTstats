import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import scipy.stats as ss  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    ss = None  # type: ignore


@dataclass
class FitResult:
    name: str
    params: Dict[str, float]
    loglik: float
    aic: float
    bic: float
    n: int
    ks_pvalue: Optional[float] = None
    notes: Optional[str] = None


def _safe_log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log(np.clip(x, eps, None))


def _normal_fit(x: np.ndarray) -> Tuple[float, float]:
    mu = float(np.mean(x))
    # population std (MLE) uses ddof=0
    sigma = float(np.std(x, ddof=0))
    if sigma <= 0 or not np.isfinite(sigma):
        sigma = 1e-9
    return mu, sigma


def _normal_loglik(x: np.ndarray, mu: float, sigma: float) -> float:
    n = x.size
    var = sigma * sigma
    return float(-0.5 * n * (math.log(2 * math.pi * var) + np.mean((x - mu) ** 2) / var))


def fit_normal(x: np.ndarray) -> FitResult:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    mu, sigma = _normal_fit(x)
    ll = _normal_loglik(x, mu, sigma)
    k = 2  # mu, sigma
    aic = 2 * k - 2 * ll
    bic = k * math.log(n) - 2 * ll if n > 0 else float("inf")
    pval = None
    if ss is not None and n > 3:
        # KS test against normal with estimated parameters (Lilliefors-ish; approximate)
        try:
            z = (x - mu) / (sigma if sigma > 0 else 1e-9)
            d, p = ss.kstest(z, "norm")
            pval = float(p)
        except Exception:
            pval = None
    return FitResult("Normal", {"mu": mu, "sigma": sigma}, ll, aic, bic, n, pval)


def fit_lognormal(x: np.ndarray) -> FitResult:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    # Log-normal requires positive support; guard zeros with tiny epsilon
    eps = 1e-9
    xp = np.clip(x, eps, None)
    y = _safe_log(xp)
    mu, sigma = _normal_fit(y)
    # Log-likelihood under log-normal
    # ll = sum( -log(x) - 0.5*log(2*pi*sigma^2) - (log(x)-mu)^2/(2*sigma^2) )
    n = y.size
    if sigma <= 0:
        sigma = 1e-9
    ll = -np.sum(np.log(xp)) - 0.5 * n * (math.log(2 * math.pi * sigma * sigma) + np.mean((y - mu) ** 2) / (sigma * sigma))
    k = 2  # mu, sigma on log scale
    aic = 2 * k - 2 * float(ll)
    bic = k * math.log(n) - 2 * float(ll) if n > 0 else float("inf")
    pval = None
    if ss is not None and n > 3:
        try:
            # KS test on standardized log values
            z = (y - mu) / (sigma if sigma > 0 else 1e-9)
            d, p = ss.kstest(z, "norm")
            pval = float(p)
        except Exception:
            pval = None
    return FitResult("LogNormal", {"log_mu": mu, "log_sigma": sigma}, float(ll), aic, bic, n, pval)


def fit_gamma(x: np.ndarray) -> FitResult:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    eps = 1e-9
    xp = np.clip(x, eps, None)
    n = xp.size
    # Method of moments: k = (mean/var)^2, theta = var/mean
    mean = float(np.mean(xp))
    var = float(np.var(xp, ddof=0))
    if var <= 0 or not np.isfinite(var) or mean <= 0:
        # Degenerate case; approximate
        k = 1.0
        theta = max(mean, eps)
    else:
        k = (mean * mean) / var
        theta = var / mean
    # Log-likelihood for Gamma(k, theta) with x>0
    # ll = sum( (k-1)log(x) - x/theta - k log(theta) - log(Gamma(k)) )
    ll = float(np.sum((k - 1.0) * np.log(xp) - xp / theta) - n * (k * math.log(theta) + math.lgamma(k)))
    k_params = 2
    aic = 2 * k_params - 2 * ll
    bic = k_params * math.log(n) - 2 * ll if n > 0 else float("inf")
    pval = None
    if ss is not None and n > 3:
        try:
            # KS with estimated parameters via scipy when available for better CDF
            d, p = ss.kstest(xp, "gamma", args=(k, 0, theta))
            pval = float(p)
        except Exception:
            pval = None
    return FitResult("Gamma", {"k": float(k), "theta": float(theta)}, ll, aic, bic, n, pval)


def fit_normal_on_log1p(x: np.ndarray) -> FitResult:
    y = np.log1p(np.clip(np.asarray(x, dtype=float), 0, None))
    y = y[np.isfinite(y)]
    n = y.size
    mu, sigma = _normal_fit(y)
    ll = _normal_loglik(y, mu, sigma)
    k = 2
    aic = 2 * k - 2 * ll
    bic = k * math.log(n) - 2 * ll if n > 0 else float("inf")
    pval = None
    if ss is not None and n > 3:
        try:
            z = (y - mu) / (sigma if sigma > 0 else 1e-9)
            d, p = ss.kstest(z, "norm")
            pval = float(p)
        except Exception:
            pval = None
    return FitResult("Normal(log1p)", {"mu": mu, "sigma": sigma}, ll, aic, bic, n, pval)


def best_fit(results: Dict[str, FitResult], criterion: str = "AIC") -> str:
    crit = criterion.upper()
    if crit not in {"AIC", "BIC"}:
        crit = "AIC"
    key = lambda r: r.aic if crit == "AIC" else r.bic
    return min(results.values(), key=key).name
