"""Return calculations for prices and portfolios."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def simple_return(prices: ArrayLike) -> NDArray[np.float64]:
    """Calculate simple period returns from strictly positive prices."""
    values = np.asarray(prices, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("prices must be a one-dimensional array with at least two values")
    if np.any(values <= 0) or not np.all(np.isfinite(values)):
        raise ValueError("prices must contain only finite positive values")
    return values[1:] / values[:-1] - 1.0


def log_return(prices: ArrayLike) -> NDArray[np.float64]:
    """Calculate continuously compounded returns from strictly positive prices."""
    values = np.asarray(prices, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("prices must be a one-dimensional array with at least two values")
    if np.any(values <= 0) or not np.all(np.isfinite(values)):
        raise ValueError("prices must contain only finite positive values")
    return np.diff(np.log(values))


def portfolio_return(returns: ArrayLike, weights: ArrayLike) -> NDArray[np.float64] | float:
    """Calculate a linear portfolio return from asset returns and weights."""
    observations = np.asarray(returns, dtype=float)
    vector = np.asarray(weights, dtype=float)
    if observations.ndim not in (1, 2) or vector.ndim != 1:
        raise ValueError("returns must be one- or two-dimensional and weights must be one-dimensional")
    if observations.shape[-1] != vector.size:
        raise ValueError("the number of assets must match the number of weights")
    if not np.all(np.isfinite(observations)) or not np.all(np.isfinite(vector)):
        raise ValueError("returns and weights must be finite")
    return observations @ vector


def pnl_long(quantity: float, entry_price: float, exit_price: float) -> float:
    """Calculate P&L for a long position."""
    if quantity < 0 or entry_price < 0 or exit_price < 0:
        raise ValueError("quantity and prices must be non-negative")
    return float(quantity * (exit_price - entry_price))


def pnl_short(quantity: float, entry_price: float, exit_price: float) -> float:
    """Calculate P&L for a short position."""
    return -pnl_long(quantity, entry_price, exit_price)
