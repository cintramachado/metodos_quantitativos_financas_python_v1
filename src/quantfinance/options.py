"""European option pricing and implied volatility."""

import numpy as np
from scipy.stats import norm


def _validate_parameters(spot: float, strike: float, maturity: float, volatility: float) -> None:
    if spot <= 0 or strike <= 0 or maturity <= 0 or volatility <= 0:
        raise ValueError("spot, strike, maturity and volatility must be positive")


def black_scholes_call(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
) -> float:
    """Price a European call under the Black-Scholes model."""
    _validate_parameters(spot, strike, maturity, volatility)
    root_t = np.sqrt(maturity)
    d1 = (np.log(spot / strike) + (rate + 0.5 * volatility**2) * maturity) / (
        volatility * root_t
    )
    d2 = d1 - volatility * root_t
    return float(spot * norm.cdf(d1) - strike * np.exp(-rate * maturity) * norm.cdf(d2))


def implied_volatility(
    price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    initial_volatility: float = 0.2,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> float:
    """Solve a call's implied volatility with safeguarded Newton iterations."""
    if price < 0 or initial_volatility <= 0 or tolerance <= 0 or max_iterations < 1:
        raise ValueError("price, initial volatility, tolerance and iterations are invalid")
    _validate_parameters(spot, strike, maturity, initial_volatility)
    volatility = initial_volatility
    root_t = np.sqrt(maturity)
    for _ in range(max_iterations):
        value = black_scholes_call(spot, strike, maturity, rate, volatility)
        error = value - price
        if abs(error) < tolerance:
            return float(volatility)
        d1 = (np.log(spot / strike) + (rate + 0.5 * volatility**2) * maturity) / (
            volatility * root_t
        )
        vega = spot * norm.pdf(d1) * root_t
        if vega <= np.finfo(float).eps:
            break
        volatility -= error / vega
        if volatility <= 0 or not np.isfinite(volatility):
            break
    raise RuntimeError("implied volatility did not converge")
