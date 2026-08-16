"""Portfolio construction and asset-pricing helpers."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

from .linear_algebra import covariance_quadratic_form


@dataclass(frozen=True)
class EfficientFrontierResult:
    """Weights and moments for a sequence of target returns."""

    target_returns: NDArray[np.float64]
    volatilities: NDArray[np.float64]
    weights: NDArray[np.float64]


def portfolio_variance(weights: ArrayLike, covariance: ArrayLike) -> float:
    """Return the variance of a portfolio."""
    return covariance_quadratic_form(weights, covariance)


def portfolio_volatility(weights: ArrayLike, covariance: ArrayLike) -> float:
    """Return portfolio volatility as the square root of variance."""
    variance = portfolio_variance(weights, covariance)
    if variance < 0:
        raise ValueError("portfolio variance cannot be negative")
    return float(np.sqrt(variance))


def gmv_weights(covariance: ArrayLike) -> NDArray[np.float64]:
    """Return unconstrained global minimum-variance portfolio weights."""
    matrix = np.asarray(covariance, dtype=float)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or not np.all(np.isfinite(matrix))
        or not np.allclose(matrix, matrix.T)
    ):
        raise ValueError("covariance must be finite, square, and symmetric")
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError("covariance must be positive definite") from error
    ones = np.ones(matrix.shape[0])
    solution = np.linalg.solve(matrix, ones)
    denominator = float(ones @ solution)
    if denominator <= 0:
        raise ValueError("covariance matrix does not produce valid GMV weights")
    return solution / denominator


def _validate_inputs(expected_returns: ArrayLike, covariance: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    returns = np.asarray(expected_returns, dtype=float)
    matrix = np.asarray(covariance, dtype=float)
    if returns.ndim != 1 or matrix.shape != (returns.size, returns.size):
        raise ValueError("expected returns and covariance dimensions are incompatible")
    if not np.all(np.isfinite(returns)) or not np.all(np.isfinite(matrix)):
        raise ValueError("expected returns and covariance must be finite")
    if not np.allclose(matrix, matrix.T):
        raise ValueError("covariance must be symmetric")
    return returns, matrix


def _bounds(n_assets: int, long_only: bool, bounds: tuple[float, float] | None) -> list[tuple[float, float]]:
    if bounds is not None and bounds[0] > bounds[1]:
        raise ValueError("bounds lower limit must not exceed upper limit")
    lower, upper = bounds if bounds is not None else ((0.0, 1.0) if long_only else (-np.inf, np.inf))
    if long_only and lower < 0:
        raise ValueError("long-only portfolios cannot have a negative lower bound")
    return [(lower, upper)] * n_assets


def global_minimum_variance(
    covariance: ArrayLike,
    long_only: bool = False,
    bounds: tuple[float, float] | None = None,
) -> NDArray[np.float64]:
    """Return GMV weights analytically or with constraints."""
    matrix = np.asarray(covariance, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.all(np.isfinite(matrix)) or not np.allclose(matrix, matrix.T):
        raise ValueError("covariance must be finite, square, and symmetric")
    if not long_only and bounds is None:
        return gmv_weights(matrix)
    n_assets = matrix.shape[0]
    initial = np.repeat(1.0 / n_assets, n_assets)
    result = minimize(
        lambda weights: portfolio_variance(weights, matrix),
        initial,
        method="SLSQP",
        bounds=_bounds(n_assets, long_only, bounds),
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
    )
    if not result.success:
        raise RuntimeError(f"GMV optimization failed: {result.message}")
    return result.x


def minimum_variance_target_return(
    expected_returns: ArrayLike,
    covariance: ArrayLike,
    target_return: float,
    long_only: bool = False,
    bounds: tuple[float, float] | None = None,
) -> NDArray[np.float64]:
    """Return minimum-variance weights subject to a target return."""
    returns, matrix = _validate_inputs(expected_returns, covariance)
    n_assets = returns.size
    initial = np.repeat(1.0 / n_assets, n_assets)
    result = minimize(
        lambda weights: portfolio_variance(weights, matrix),
        initial,
        method="SLSQP",
        bounds=_bounds(n_assets, long_only, bounds),
        constraints=[
            {"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
            {"type": "eq", "fun": lambda weights: weights @ returns - target_return},
        ],
        options={"ftol": 1e-12, "maxiter": 1_000},
    )
    if not result.success:
        raise RuntimeError(f"target-return optimization failed: {result.message}")
    return result.x


def efficient_frontier(
    expected_returns: ArrayLike,
    covariance: ArrayLike,
    target_returns: ArrayLike,
    long_only: bool = False,
    bounds: tuple[float, float] | None = None,
) -> EfficientFrontierResult:
    """Construct a minimum-variance frontier for target returns."""
    returns, matrix = _validate_inputs(expected_returns, covariance)
    targets = np.asarray(target_returns, dtype=float)
    if targets.ndim != 1 or targets.size < 2:
        raise ValueError("target_returns must be a one-dimensional grid")
    weights = np.array([
        minimum_variance_target_return(returns, matrix, target, long_only, bounds)
        for target in targets
    ])
    variances = np.einsum("ij,jk,ik->i", weights, matrix, weights)
    return EfficientFrontierResult(targets, np.sqrt(variances), weights)


def tangency_portfolio(
    expected_returns: ArrayLike,
    covariance: ArrayLike,
    risk_free_rate: float,
    long_only: bool = False,
    bounds: tuple[float, float] | None = None,
) -> NDArray[np.float64]:
    """Return the maximum-Sharpe risky portfolio under optional constraints."""
    returns, matrix = _validate_inputs(expected_returns, covariance)
    if not long_only and bounds is None:
        excess = returns - risk_free_rate
        raw = np.linalg.solve(matrix, excess)
        denominator = raw.sum()
        if abs(denominator) <= np.finfo(float).eps:
            raise ValueError("tangency portfolio cannot be normalized")
        return raw / denominator
    n_assets = returns.size
    initial = global_minimum_variance(matrix, long_only, bounds)
    result = minimize(
        lambda weights: -portfolio_sharpe(weights, returns, matrix, risk_free_rate),
        initial,
        method="SLSQP",
        bounds=_bounds(n_assets, long_only, bounds),
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"ftol": 1e-12, "maxiter": 1_000},
    )
    if not result.success:
        raise RuntimeError(f"tangency optimization failed: {result.message}")
    return result.x


def portfolio_sharpe(
    weights: ArrayLike,
    expected_returns: ArrayLike,
    covariance: ArrayLike,
    risk_free_rate: float = 0.0,
) -> float:
    """Return the portfolio Sharpe ratio from expected moments."""
    returns, matrix = _validate_inputs(expected_returns, covariance)
    vector = np.asarray(weights, dtype=float)
    volatility = portfolio_volatility(vector, matrix)
    if volatility <= 0:
        raise ValueError("portfolio volatility must be positive")
    return float((vector @ returns - risk_free_rate) / volatility)


def capm_expected_return(risk_free_rate: float, beta: float, market_return: float) -> float:
    """Return CAPM expected return."""
    return float(risk_free_rate + beta * (market_return - risk_free_rate))


def estimate_capm(
    asset_returns: ArrayLike,
    market_returns: ArrayLike,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """Estimate CAPM alpha, beta, R-squared, and residual risk."""
    asset = np.asarray(asset_returns, dtype=float)
    market = np.asarray(market_returns, dtype=float)
    if asset.ndim != 1 or asset.shape != market.shape:
        raise ValueError("asset and market returns must be aligned one-dimensional arrays")
    market_excess = market - risk_free_rate
    asset_excess = asset - risk_free_rate
    variance = np.var(market_excess, ddof=1)
    if variance <= 0:
        raise ValueError("market excess returns must have positive variance")
    beta = float(np.cov(asset_excess, market_excess, ddof=1)[0, 1] / variance)
    alpha = float(asset_excess.mean() - beta * market_excess.mean())
    fitted = alpha + beta * market_excess
    residuals = asset_excess - fitted
    total = np.sum((asset_excess - asset_excess.mean()) ** 2)
    r_squared = float(1.0 - np.sum(residuals**2) / total)
    return {
        "alpha": alpha,
        "beta": beta,
        "r_squared": r_squared,
        "residual_risk": float(residuals.std(ddof=1)),
    }


def jensen_alpha(
    asset_returns: ArrayLike,
    market_returns: ArrayLike,
    risk_free_rate: float = 0.0,
) -> float:
    """Return Jensen alpha estimated from CAPM moments."""
    return estimate_capm(asset_returns, market_returns, risk_free_rate)["alpha"]
