"""Deterministic-seed Monte Carlo simulations."""

import numpy as np
from numpy.typing import NDArray


def gbm_terminal_prices(
    spot: float,
    drift: float,
    volatility: float,
    horizon: float,
    simulations: int = 100_000,
    seed: int = 123,
) -> NDArray[np.float64]:
    """Simulate terminal prices under geometric Brownian motion."""
    if spot <= 0 or volatility < 0 or horizon <= 0 or simulations < 1:
        raise ValueError("spot and horizon must be positive; other parameters are invalid")
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal(simulations)
    return spot * np.exp(
        (drift - 0.5 * volatility**2) * horizon
        + volatility * np.sqrt(horizon) * shocks
    )
