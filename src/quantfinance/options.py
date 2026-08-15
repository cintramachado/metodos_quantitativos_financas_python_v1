"""Option pricing, numerical Greeks, trees, and Monte Carlo."""

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .numerical import bisection, newton_raphson


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


def black_scholes_put(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
) -> float:
    """Price a European put under Black-Scholes."""
    _validate_parameters(spot, strike, maturity, volatility)
    root_t = np.sqrt(maturity)
    d1 = (np.log(spot / strike) + (rate + 0.5 * volatility**2) * maturity) / (
        volatility * root_t
    )
    d2 = d1 - volatility * root_t
    return float(strike * np.exp(-rate * maturity) * norm.cdf(-d2) - spot * norm.cdf(-d1))


def _black_scholes_vega(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
) -> float:
    _validate_parameters(spot, strike, maturity, volatility)
    d1 = (np.log(spot / strike) + (rate + 0.5 * volatility**2) * maturity) / (
        volatility * np.sqrt(maturity)
    )
    return float(spot * norm.pdf(d1) * np.sqrt(maturity))


def implied_volatility_bisection(
    price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    lower: float = 1e-8,
    upper: float = 5.0,
    tolerance: float = 1e-10,
    max_iterations: int = 200,
) -> float:
    """Find call implied volatility with a bracketed bisection method."""
    if price < 0 or lower <= 0 or upper <= lower:
        raise ValueError("price and volatility bracket are invalid")
    return bisection(
        lambda volatility: black_scholes_call(spot, strike, maturity, rate, volatility) - price,
        lower,
        upper,
        tolerance,
        max_iterations,
    )


def implied_volatility_newton(
    price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    initial_volatility: float = 0.2,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> float:
    """Find call implied volatility with Newton-Raphson and Black-Scholes vega."""
    if price < 0:
        raise ValueError("price must be non-negative")
    return newton_raphson(
        lambda volatility: black_scholes_call(spot, strike, maturity, rate, volatility) - price,
        lambda volatility: _black_scholes_vega(spot, strike, maturity, rate, volatility),
        initial_volatility,
        tolerance,
        max_iterations,
    )


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
    return implied_volatility_newton(
        price,
        spot,
        strike,
        maturity,
        rate,
        initial_volatility,
        tolerance,
        max_iterations,
    )


def finite_difference_delta(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    step: float = 1e-3,
) -> float:
    """Estimate call delta with a central finite difference."""
    if step <= 0 or spot <= step:
        raise ValueError("step must be positive and smaller than spot")
    return float((
        black_scholes_call(spot + step, strike, maturity, rate, volatility)
        - black_scholes_call(spot - step, strike, maturity, rate, volatility)
    ) / (2 * step))


def finite_difference_gamma(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    step: float = 1e-2,
) -> float:
    """Estimate call gamma with a central second difference."""
    if step <= 0 or spot <= step:
        raise ValueError("step must be positive and smaller than spot")
    center = black_scholes_call(spot, strike, maturity, rate, volatility)
    return float((
        black_scholes_call(spot + step, strike, maturity, rate, volatility)
        - 2 * center
        + black_scholes_call(spot - step, strike, maturity, rate, volatility)
    ) / step**2)


def _binomial_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    steps: int,
    option: str,
    american: bool,
) -> float:
    _validate_parameters(spot, strike, maturity, volatility)
    if steps < 1 or option not in {"call", "put"}:
        raise ValueError("steps or option type is invalid")
    dt = maturity / steps
    up = np.exp(volatility * np.sqrt(dt))
    down = 1.0 / up
    probability = (np.exp(rate * dt) - down) / (up - down)
    if not 0 <= probability <= 1:
        raise ValueError("risk-neutral probability is outside [0, 1]")
    discount = np.exp(-rate * dt)
    indices = np.arange(steps + 1)
    terminal = spot * up**indices * down ** (steps - indices)
    if option == "call":
        values = np.maximum(terminal - strike, 0.0)
    else:
        values = np.maximum(strike - terminal, 0.0)
    for time in range(steps - 1, -1, -1):
        values = discount * (probability * values[1:time + 2] + (1 - probability) * values[:time + 1])
        if american:
            prices = spot * up**np.arange(time + 1) * down ** (time - np.arange(time + 1))
            intrinsic = np.maximum(prices - strike, 0.0) if option == "call" else np.maximum(strike - prices, 0.0)
            values = np.maximum(values, intrinsic)
    return float(values[0])


def binomial_european(
    spot: float, strike: float, maturity: float, rate: float, volatility: float,
    steps: int = 200, option: str = "call",
) -> float:
    """Price a European option with Cox-Ross-Rubinstein backward induction."""
    return _binomial_price(spot, strike, maturity, rate, volatility, steps, option, False)


def binomial_american(
    spot: float, strike: float, maturity: float, rate: float, volatility: float,
    steps: int = 200, option: str = "put",
) -> float:
    """Price an American option with intrinsic-value early exercise."""
    return _binomial_price(spot, strike, maturity, rate, volatility, steps, option, True)


@dataclass(frozen=True)
class MonteCarloResult:
    """Monte Carlo price estimate and sampling uncertainty."""

    price: float
    standard_error: float
    simulations: int


def monte_carlo_european_option(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    simulations: int = 100_000,
    seed: int = 123,
    option: str = "call",
) -> MonteCarloResult:
    """Price a European option under the risk-neutral GBM measure."""
    _validate_parameters(spot, strike, maturity, volatility)
    if simulations < 2 or option not in {"call", "put"}:
        raise ValueError("simulations or option type is invalid")
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal(simulations)
    terminal = spot * np.exp(
        (rate - 0.5 * volatility**2) * maturity
        + volatility * np.sqrt(maturity) * shocks
    )
    payoff = np.maximum(terminal - strike, 0.0) if option == "call" else np.maximum(strike - terminal, 0.0)
    discounted = np.exp(-rate * maturity) * payoff
    return MonteCarloResult(
        price=float(discounted.mean()),
        standard_error=float(discounted.std(ddof=1) / np.sqrt(simulations)),
        simulations=simulations,
    )
