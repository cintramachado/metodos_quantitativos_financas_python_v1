"""Deterministic-seed Monte Carlo simulations."""

import numpy as np
from numpy.typing import NDArray

from .linear_algebra import cholesky_factor


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


def simulate_random_walk(
    n_steps: int,
    drift: float = 0.0,
    volatility: float = 1.0,
    start: float = 0.0,
    seed: int = 123,
) -> NDArray[np.float64]:
    """Simulate an additive random walk with Gaussian innovations."""
    if n_steps < 1 or volatility < 0:
        raise ValueError("n_steps must be positive and volatility non-negative")
    rng = np.random.default_rng(seed)
    increments = drift + volatility * rng.standard_normal(n_steps)
    return np.concatenate(([start], start + np.cumsum(increments)))


def simulate_gbm(
    spot: float,
    drift: float,
    volatility: float,
    horizon: float,
    steps: int = 252,
    seed: int = 123,
) -> NDArray[np.float64]:
    """Simulate a geometric Brownian motion price path."""
    if spot <= 0 or volatility < 0 or horizon <= 0 or steps < 1:
        raise ValueError("spot and horizon must be positive; other parameters are invalid")
    rng = np.random.default_rng(seed)
    dt = horizon / steps
    shocks = rng.standard_normal(steps)
    log_increments = (
        (drift - 0.5 * volatility**2) * dt
        + volatility * np.sqrt(dt) * shocks
    )
    return spot * np.exp(np.concatenate(([0.0], np.cumsum(log_increments))))


def simulate_ou(
    mean_reversion: float,
    long_run_mean: float,
    volatility: float,
    horizon: float,
    steps: int = 252,
    start: float | None = None,
    seed: int = 123,
) -> NDArray[np.float64]:
    """Simulate an Ornstein-Uhlenbeck mean-reverting process by Euler steps."""
    if mean_reversion <= 0 or volatility < 0 or horizon <= 0 or steps < 1:
        raise ValueError("mean_reversion, horizon, and steps must be valid")
    rng = np.random.default_rng(seed)
    dt = horizon / steps
    path = np.empty(steps + 1)
    path[0] = long_run_mean if start is None else start
    for index in range(steps):
        path[index + 1] = (
            path[index]
            + mean_reversion * (long_run_mean - path[index]) * dt
            + volatility * np.sqrt(dt) * rng.normal()
        )
    return path


def simulate_jump_diffusion(
    spot: float,
    drift: float,
    volatility: float,
    jump_intensity: float,
    jump_mean: float,
    jump_volatility: float,
    horizon: float,
    steps: int = 252,
    seed: int = 123,
) -> NDArray[np.float64]:
    """Simulate a log-price diffusion with compound Poisson jumps."""
    if (
        spot <= 0 or volatility < 0 or jump_intensity < 0
        or jump_volatility < 0 or horizon <= 0 or steps < 1
    ):
        raise ValueError("jump-diffusion parameters are invalid")
    rng = np.random.default_rng(seed)
    dt = horizon / steps
    path = np.empty(steps + 1)
    path[0] = spot
    for index in range(steps):
        jump_count = rng.poisson(jump_intensity * dt)
        jumps = rng.normal(jump_mean, jump_volatility, jump_count).sum()
        log_increment = (
            (drift - 0.5 * volatility**2) * dt
            + volatility * np.sqrt(dt) * rng.normal()
            + jumps
        )
        path[index + 1] = path[index] * np.exp(log_increment)
    return path


def simulate_correlated_normal(
    correlation: np.ndarray,
    samples: int,
    seed: int = 123,
) -> NDArray[np.float64]:
    """Simulate multivariate standard Normal observations with Cholesky."""
    if samples < 1:
        raise ValueError("samples must be positive")
    factor = cholesky_factor(correlation)
    rng = np.random.default_rng(seed)
    independent = rng.standard_normal((factor.shape[0], samples))
    return (factor @ independent).T


def simulate_multivariate_student_t(
    correlation: np.ndarray,
    degrees_of_freedom: float,
    samples: int,
    seed: int = 123,
) -> NDArray[np.float64]:
    """Simulate a multivariate Student-t via a Normal/chi-square scale mixture."""
    if degrees_of_freedom <= 0 or samples < 1:
        raise ValueError("degrees_of_freedom and samples must be positive")
    normal = simulate_correlated_normal(correlation, samples, seed)
    rng = np.random.default_rng(seed + 1)
    chi_square = rng.chisquare(degrees_of_freedom, samples)
    return normal / np.sqrt(chi_square[:, None] / degrees_of_freedom)
