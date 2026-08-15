import numpy as np
import pytest

from quantfinance.linear_algebra import cholesky_factor
from quantfinance.options import black_scholes_call, implied_volatility
from quantfinance.performance import omega_ratio, sharpe_ratio, sortino_ratio
from quantfinance.portfolio import gmv_weights, portfolio_variance
from quantfinance.returns import log_return, simple_return
from quantfinance.simulation import gbm_terminal_prices


def test_simple_and_log_returns():
    prices = np.array([100.0, 110.0, 99.0])
    np.testing.assert_allclose(simple_return(prices), [0.10, -0.10])
    np.testing.assert_allclose(log_return(prices), np.log([1.10, 0.90]))


def test_cholesky_factor_reconstructs_matrix():
    covariance = np.array([[1.0, 0.6], [0.6, 1.0]])
    factor = cholesky_factor(covariance)
    np.testing.assert_allclose(factor @ factor.T, covariance)


def test_black_scholes_and_implied_volatility():
    price = black_scholes_call(100.0, 100.0, 1.0, 0.05, 0.30)
    recovered = implied_volatility(price, 100.0, 100.0, 1.0, 0.05)
    assert recovered == pytest.approx(0.30, abs=1e-8)


def test_monte_carlo_is_reproducible():
    first = gbm_terminal_prices(100.0, 0.08, 0.20, 1.0, simulations=1_000, seed=7)
    second = gbm_terminal_prices(100.0, 0.08, 0.20, 1.0, simulations=1_000, seed=7)
    np.testing.assert_array_equal(first, second)


def test_gmv_and_portfolio_variance():
    covariance = np.array([[0.04, 0.01], [0.01, 0.09]])
    weights = gmv_weights(covariance)
    assert weights.sum() == pytest.approx(1.0)
    assert portfolio_variance(weights, covariance) < portfolio_variance([1.0, 0.0], covariance)


def test_performance_ratios():
    returns = np.array([0.01, 0.02, -0.01, 0.015])
    assert sharpe_ratio(returns, periods=1) > 0
    assert sortino_ratio(returns, periods=1) > 0
    assert omega_ratio(returns) > 1


def test_invalid_prices_are_rejected():
    with pytest.raises(ValueError):
        simple_return([100.0, 0.0])
