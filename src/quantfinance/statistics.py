"""Descriptive statistics and likelihood tools for financial samples."""

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats


def descriptive_statistics(returns: ArrayLike) -> dict[str, float]:
    """Return mean, variance, skewness, excess kurtosis, and common quantiles."""
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("returns must be a finite one-dimensional sample")
    centered = values - values.mean()
    standard_deviation = values.std(ddof=1)
    if standard_deviation == 0:
        raise ValueError("returns must have non-zero variance")
    skewness = np.mean(centered**3) / standard_deviation**3
    kurtosis = np.mean(centered**4) / standard_deviation**4 - 3.0
    return {
        "mean": float(values.mean()),
        "variance": float(values.var(ddof=1)),
        "volatility": float(standard_deviation),
        "skewness": float(skewness),
        "excess_kurtosis": float(kurtosis),
        "quantile_01": float(np.quantile(values, 0.01)),
        "quantile_05": float(np.quantile(values, 0.05)),
        "quantile_50": float(np.quantile(values, 0.50)),
        "quantile_95": float(np.quantile(values, 0.95)),
        "quantile_99": float(np.quantile(values, 0.99)),
    }


def return_summary(returns: ArrayLike, confidence: float = 0.95) -> dict[str, float]:
    """Return descriptive statistics plus a lower-tail historical VaR quantile."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    descriptive = descriptive_statistics(returns)
    return {
        "mean": descriptive["mean"],
        "variance": descriptive["variance"],
        "volatility": descriptive["volatility"],
        "skewness": descriptive["skewness"],
        "excess_kurtosis": descriptive["excess_kurtosis"],
        "var": historical_quantile(returns, 1.0 - confidence),
    }


def historical_quantile(returns: ArrayLike, probability: float) -> float:
    """Estimate a return quantile empirically without a distributional assumption."""
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or values.size < 1 or not np.all(np.isfinite(values)):
        raise ValueError("returns must be a finite one-dimensional sample")
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between zero and one")
    return float(np.quantile(values, probability))


def normal_log_likelihood(returns: ArrayLike, mean: float, standard_deviation: float) -> float:
    """Return the Gaussian log-likelihood for a sample."""
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or values.size < 1 or not np.all(np.isfinite(values)):
        raise ValueError("returns must be a finite one-dimensional sample")
    if standard_deviation <= 0:
        raise ValueError("standard_deviation must be positive")
    return float(np.sum(stats.norm.logpdf(values, loc=mean, scale=standard_deviation)))


def fit_normal_mle(returns: ArrayLike) -> dict[str, float]:
    """Fit normal mean and scale by maximum likelihood."""
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("returns must be a finite one-dimensional sample")
    mean = float(values.mean())
    standard_deviation = float(values.std(ddof=0))
    if standard_deviation <= 0:
        raise ValueError("returns must have non-zero variance")
    return {
        "mean": mean,
        "standard_deviation": standard_deviation,
        "log_likelihood": normal_log_likelihood(values, mean, standard_deviation),
    }


def fit_student_t(returns: ArrayLike) -> dict[str, float]:
    """Fit Student-t degrees of freedom, location, and scale by likelihood."""
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or values.size < 3 or not np.all(np.isfinite(values)):
        raise ValueError("returns must be a finite one-dimensional sample")
    degrees_of_freedom, location, scale = stats.t.fit(values)
    log_likelihood = float(np.sum(stats.t.logpdf(
        values, degrees_of_freedom, loc=location, scale=scale
    )))
    return {
        "degrees_of_freedom": float(degrees_of_freedom),
        "location": float(location),
        "scale": float(scale),
        "log_likelihood": log_likelihood,
    }
