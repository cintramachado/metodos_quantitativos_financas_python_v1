"""Regression calculations used in finance examples."""

from dataclasses import dataclass

import numpy as np
import statsmodels.api as sm
from numpy.typing import ArrayLike, NDArray
from statsmodels.stats.diagnostic import (
    acorr_breusch_godfrey,
    het_breuschpagan,
    het_white,
)
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor


@dataclass(frozen=True)
class OLSResult:
    """Results from an ordinary least squares fit."""

    coefficients: NDArray[np.float64]
    fitted_values: NDArray[np.float64]
    residuals: NDArray[np.float64]
    standard_errors: NDArray[np.float64]
    t_statistics: NDArray[np.float64]
    rss: float
    tss: float
    r_squared: float
    adjusted_r_squared: float
    f_statistic: float
    degrees_of_freedom: int


def ols_coefficients(features: ArrayLike, target: ArrayLike) -> NDArray[np.float64]:
    """Estimate OLS coefficients with the linear system X'X beta = X'y."""
    design = np.asarray(features, dtype=float)
    response = np.asarray(target, dtype=float)
    if design.ndim != 2 or response.ndim != 1:
        raise ValueError("features must be 2-D and target must be 1-D")
    if design.shape[0] != response.size:
        raise ValueError("features and target must have the same number of rows")
    if not np.all(np.isfinite(design)) or not np.all(np.isfinite(response)):
        raise ValueError("features and target must be finite")
    gram = design.T @ design
    return np.linalg.solve(gram, design.T @ response)


def ols_numpy(
    features: ArrayLike,
    target: ArrayLike,
    add_intercept: bool = False,
) -> OLSResult:
    """Fit OLS using beta = (X'X)^(-1) X'y and return inference statistics."""
    design = np.asarray(features, dtype=float)
    response = np.asarray(target, dtype=float)
    if add_intercept:
        design = np.column_stack((np.ones(design.shape[0]), design))
    if design.ndim != 2 or response.ndim != 1:
        raise ValueError("features must be 2-D and target must be 1-D")
    if design.shape[0] != response.size:
        raise ValueError("features and target must have the same number of rows")
    if design.shape[0] <= design.shape[1] or not np.all(np.isfinite(design)):
        raise ValueError("design must have more observations than parameters")
    if not np.all(np.isfinite(response)):
        raise ValueError("target must be finite")
    gram = design.T @ design
    if np.linalg.matrix_rank(gram) < gram.shape[0]:
        raise ValueError("design matrix is singular or perfectly collinear")
    coefficients = np.linalg.inv(gram) @ design.T @ response
    fitted_values = design @ coefficients
    residuals = response - fitted_values
    observations, parameters = design.shape
    degrees_of_freedom = observations - parameters
    rss = float(residuals @ residuals)
    centered = response - response.mean()
    tss = float(centered @ centered)
    if tss <= 0:
        raise ValueError("target must have positive total variation")
    r_squared = 1.0 - rss / tss
    adjusted_r_squared = 1.0 - (1.0 - r_squared) * (observations - 1) / degrees_of_freedom
    residual_variance = rss / degrees_of_freedom
    covariance = residual_variance * np.linalg.inv(gram)
    standard_errors = np.sqrt(np.diag(covariance))
    t_statistics = coefficients / standard_errors
    explained = tss - rss
    f_statistic = float((explained / (parameters - 1)) / residual_variance) if parameters > 1 else np.nan
    return OLSResult(
        coefficients=coefficients,
        fitted_values=fitted_values,
        residuals=residuals,
        standard_errors=standard_errors,
        t_statistics=t_statistics,
        rss=rss,
        tss=tss,
        r_squared=float(r_squared),
        adjusted_r_squared=float(adjusted_r_squared),
        f_statistic=f_statistic,
        degrees_of_freedom=degrees_of_freedom,
    )


def regression_diagnostics(model: sm.regression.linear_model.RegressionResultsWrapper) -> dict[str, float]:
    """Return Durbin-Watson, Breusch-Godfrey, White, and Breusch-Pagan diagnostics."""
    white = het_white(model.resid, model.model.exog)
    breusch_pagan = het_breuschpagan(model.resid, model.model.exog)
    breusch_godfrey = acorr_breusch_godfrey(model, nlags=4)
    return {
        "durbin_watson": float(durbin_watson(model.resid)),
        "white_lm_pvalue": float(white[1]),
        "breusch_pagan_lm_pvalue": float(breusch_pagan[1]),
        "breusch_godfrey_lm_pvalue": float(breusch_godfrey[1]),
    }


def calculate_vif(features: ArrayLike) -> NDArray[np.float64]:
    """Calculate variance inflation factors for columns of a feature matrix."""
    values = np.asarray(features, dtype=float)
    if values.ndim != 2 or values.shape[0] <= values.shape[1]:
        raise ValueError("features must have more observations than columns")
    if not np.all(np.isfinite(values)):
        raise ValueError("features must be finite")
    return np.array([
        variance_inflation_factor(values, index)
        for index in range(values.shape[1])
    ])


def capm_regression(
    asset_returns: ArrayLike,
    market_returns: ArrayLike,
    risk_free_rate: float = 0.0,
) -> OLSResult:
    """Estimate Ri-Rf = alpha + beta (Rm-Rf) + epsilon."""
    asset = np.asarray(asset_returns, dtype=float)
    market = np.asarray(market_returns, dtype=float)
    if asset.shape != market.shape:
        raise ValueError("asset and market returns must have equal shapes")
    excess_asset = asset - risk_free_rate
    excess_market = market - risk_free_rate
    return ols_numpy(excess_market[:, None], excess_asset, add_intercept=True)


def minimum_variance_hedge_ratio(
    spot_changes: ArrayLike,
    futures_changes: ArrayLike,
) -> float:
    """Return h* = Cov(spot, futures) / Var(futures)."""
    spot = np.asarray(spot_changes, dtype=float)
    futures = np.asarray(futures_changes, dtype=float)
    if spot.ndim != 1 or futures.shape != spot.shape:
        raise ValueError("spot and futures changes must be one-dimensional and aligned")
    variance = np.var(futures, ddof=1)
    if variance <= 0:
        raise ValueError("futures changes must have positive variance")
    return float(np.cov(spot, futures, ddof=1)[0, 1] / variance)
