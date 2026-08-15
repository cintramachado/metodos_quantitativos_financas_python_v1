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
