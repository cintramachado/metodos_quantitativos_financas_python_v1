"""Risk-adjusted performance measures."""

import numpy as np
from numpy.typing import ArrayLike


def sharpe_ratio(returns: ArrayLike, risk_free_rate: float = 0.0, periods: int = 252) -> float:
    """Return annualized Sharpe ratio for periodic returns."""
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or values.size < 2 or periods <= 0:
        raise ValueError("returns must contain at least two observations and periods must be positive")
    excess = values - risk_free_rate / periods
    deviation = excess.std(ddof=1)
    if deviation == 0:
        raise ValueError("Sharpe ratio is undefined for zero volatility")
    return float(excess.mean() / deviation * np.sqrt(periods))


def sortino_ratio(returns: ArrayLike, threshold: float = 0.0, periods: int = 252) -> float:
    """Return annualized Sortino ratio for periodic returns."""
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or values.size < 1 or periods <= 0:
        raise ValueError("returns must be one-dimensional and periods must be positive")
    periodic_threshold = threshold / periods
    downside = np.minimum(values - periodic_threshold, 0.0)
    downside_deviation = np.sqrt(np.mean(downside**2)) * np.sqrt(periods)
    if downside_deviation == 0:
        raise ValueError("Sortino ratio is undefined without downside deviation")
    return float((values.mean() - periodic_threshold) * periods / downside_deviation)


def omega_ratio(returns: ArrayLike, threshold: float = 0.0) -> float:
    """Return the Omega ratio relative to a periodic threshold."""
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or values.size < 1:
        raise ValueError("returns must be a non-empty one-dimensional array")
    gains = np.maximum(values - threshold, 0.0).mean()
    losses = np.maximum(threshold - values, 0.0).mean()
    if losses == 0:
        raise ValueError("Omega ratio is undefined without losses")
    return float(gains / losses)


def lower_partial_moment(
    returns: ArrayLike,
    threshold: float = 0.0,
    order: int = 2,
) -> float:
    """Return the lower partial moment E[max(threshold - R, 0)^order]."""
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or values.size < 1 or order < 1 or not np.all(np.isfinite(values)):
        raise ValueError("returns must be non-empty and order must be positive")
    downside = np.maximum(threshold - values, 0.0)
    return float(np.mean(downside**order))


def kappa_ratio(
    returns: ArrayLike,
    threshold: float = 0.0,
    order: int = 3,
    periods: int = 252,
) -> float:
    """Return the Kappa ratio using an order-n lower partial moment."""
    values = np.asarray(returns, dtype=float)
    if periods <= 0 or order < 1:
        raise ValueError("periods and order must be positive")
    moment = lower_partial_moment(values, threshold, order)
    if moment == 0:
        raise ValueError("Kappa ratio is undefined without downside observations")
    denominator = moment ** (1.0 / order) * periods ** (1.0 / order)
    numerator = (values.mean() - threshold) * periods
    return float(numerator / denominator)


def information_ratio(
    returns: ArrayLike,
    benchmark_returns: ArrayLike,
    periods: int = 252,
) -> float:
    """Return annualized active return divided by active risk."""
    values = np.asarray(returns, dtype=float)
    benchmark = np.asarray(benchmark_returns, dtype=float)
    if values.ndim != 1 or values.shape != benchmark.shape or values.size < 2:
        raise ValueError("returns and benchmark must be aligned one-dimensional samples")
    if periods <= 0:
        raise ValueError("periods must be positive")
    active = values - benchmark
    tracking_error = active.std(ddof=1)
    if tracking_error == 0:
        raise ValueError("information ratio is undefined without tracking error")
    return float(active.mean() / tracking_error * np.sqrt(periods))


def treynor_ratio(
    returns: ArrayLike,
    beta: float,
    risk_free_rate: float = 0.0,
    periods: int = 252,
) -> float:
    """Return excess return per unit of systematic beta risk."""
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or values.size < 1 or beta == 0 or periods <= 0:
        raise ValueError("returns, beta, and periods are invalid")
    return float((values.mean() - risk_free_rate / periods) * periods / beta)


def certainty_equivalent(returns: ArrayLike, risk_aversion: float = 3.0) -> float:
    """Return the mean-variance certainty equivalent."""
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or values.size < 2 or risk_aversion < 0:
        raise ValueError("returns and risk_aversion are invalid")
    return float(values.mean() - 0.5 * risk_aversion * values.var(ddof=1))
