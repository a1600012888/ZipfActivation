from __future__ import annotations

import numpy as np
import powerlaw
from dataclasses import dataclass, field

from .config import FittingConfig


@dataclass
class PowerLawFitResult:
    metric_name: str
    layer_name: str
    # Clauset method results
    alpha: float = 0.0
    xmin: float = 0.0
    sigma: float = 0.0
    ks_statistic: float = 0.0
    n_tail: int = 0
    n_total: int = 0
    # Comparisons vs alternative distributions
    comparisons: dict = field(default_factory=dict)
    # Log-log linear regression
    log_log_slope: float = 0.0
    log_log_intercept: float = 0.0
    log_log_r_squared: float = 0.0


def _subsample(values: np.ndarray, max_n: int, seed: int) -> np.ndarray:
    if len(values) <= max_n:
        return values
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(values), size=max_n, replace=False)
    return values[idx]


def fit_log_log_regression(values: np.ndarray) -> tuple[float, float, float]:
    """OLS on log(rank) vs log(value).  Returns (slope, intercept, R^2)."""
    sorted_vals = np.sort(np.abs(values))[::-1]
    sorted_vals = sorted_vals[sorted_vals > 0]
    if len(sorted_vals) < 10:
        return 0.0, 0.0, 0.0

    ranks = np.arange(1, len(sorted_vals) + 1, dtype=np.float64)
    log_ranks = np.log(ranks)
    log_vals = np.log(sorted_vals)

    slope, intercept = np.polyfit(log_ranks, log_vals, 1)
    fitted = slope * log_ranks + intercept
    ss_res = np.sum((log_vals - fitted) ** 2)
    ss_tot = np.sum((log_vals - np.mean(log_vals)) ** 2)
    r_squared = 1.0 - ss_res / (ss_tot + 1e-12)

    return float(slope), float(intercept), float(r_squared)


def fit_power_law(
    values: np.ndarray,
    metric_name: str,
    layer_name: str,
    config: FittingConfig,
) -> PowerLawFitResult:
    """Fit a power law using the Clauset et al. method (via powerlaw package)
    and also a simple log-log linear regression."""

    result = PowerLawFitResult(metric_name=metric_name, layer_name=layer_name)

    abs_values = np.abs(values.astype(np.float64))
    abs_values = abs_values[abs_values > 0]
    abs_values = abs_values[np.isfinite(abs_values)]

    if len(abs_values) < 50:
        return result

    result.n_total = len(abs_values)

    # Subsample for powerlaw fitting if too large
    sample = _subsample(abs_values, config.max_samples_for_fit, config.fit_seed)

    # Clauset method
    fit = powerlaw.Fit(sample, verbose=False)
    result.alpha = float(fit.power_law.alpha)
    result.xmin = float(fit.power_law.xmin)
    result.sigma = float(fit.power_law.sigma)
    try:
        result.ks_statistic = float(fit.power_law.KS())
    except (NameError, AttributeError):
        # powerlaw 2.0.0 has a bug in KS(); compute manually
        tail = sample[sample >= fit.power_law.xmin]
        if len(tail) > 0:
            from scipy.stats import kstest, pareto
            _, ks_p = kstest(tail / fit.power_law.xmin, pareto(b=fit.power_law.alpha - 1).cdf)
            result.ks_statistic = ks_p
    result.n_tail = int(np.sum(sample >= fit.power_law.xmin))

    # Comparisons
    for dist_name in config.compare_distributions:
        try:
            R, p = fit.distribution_compare("power_law", dist_name)
            result.comparisons[dist_name] = {"R": float(R), "p": float(p)}
        except Exception:
            result.comparisons[dist_name] = {"R": float("nan"), "p": float("nan")}

    # Log-log regression (on full data, not subsampled)
    slope, intercept, r_sq = fit_log_log_regression(abs_values)
    result.log_log_slope = slope
    result.log_log_intercept = intercept
    result.log_log_r_squared = r_sq

    return result
