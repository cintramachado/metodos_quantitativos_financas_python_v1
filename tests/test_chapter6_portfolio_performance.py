import numpy as np
import pytest

from quantfinance.performance import (
    certainty_equivalent,
    information_ratio,
    kappa_ratio,
    lower_partial_moment,
    omega_ratio,
    sharpe_ratio,
    sortino_ratio,
)
from quantfinance.portfolio import (
    capm_expected_return,
    efficient_frontier,
    estimate_capm,
    global_minimum_variance,
    jensen_alpha,
    minimum_variance_target_return,
    portfolio_sharpe,
    portfolio_variance,
    tangency_portfolio,
)


@pytest.fixture
def portfolio_inputs():
    expected_returns = np.array([0.06, 0.08, 0.10])
    covariance = np.array([
        [0.0400, 0.0060, 0.0040],
        [0.0060, 0.0625, 0.0080],
        [0.0040, 0.0080, 0.0900],
    ])
    return expected_returns, covariance


def test_gmv_formula_and_constrained_solution(portfolio_inputs):
    expected_returns, covariance = portfolio_inputs
    ones = np.ones(3)
    inverse_formula = np.linalg.inv(covariance) @ ones
    analytical = inverse_formula / (ones @ inverse_formula)
    result = global_minimum_variance(covariance)
    np.testing.assert_allclose(result, analytical)

    long_only = global_minimum_variance(covariance, long_only=True)
    assert long_only.sum() == pytest.approx(1.0)
    assert np.all(long_only >= -1e-8)
    assert portfolio_variance(long_only, covariance) >= portfolio_variance(result, covariance) - 1e-10


def test_frontier_target_and_tangency(portfolio_inputs):
    expected_returns, covariance = portfolio_inputs
    target = 0.08
    weights = minimum_variance_target_return(expected_returns, covariance, target)
    assert weights.sum() == pytest.approx(1.0)
    assert weights @ expected_returns == pytest.approx(target, abs=1e-8)

    frontier = efficient_frontier(expected_returns, covariance, [0.06, 0.08, 0.10])
    assert frontier.weights.shape == (3, 3)
    np.testing.assert_allclose(frontier.target_returns, [0.06, 0.08, 0.10])

    tangency = tangency_portfolio(expected_returns, covariance, 0.02)
    assert tangency.sum() == pytest.approx(1.0)
    assert portfolio_sharpe(tangency, expected_returns, covariance, 0.02) > 0


def test_capm_expected_return_and_jensen_alpha():
    rng = np.random.default_rng(61)
    market = rng.normal(0.001, 0.02, 1_000)
    asset = 0.0005 + 1.4 * market + rng.normal(0, 0.005, 1_000)
    estimate = estimate_capm(asset, market)
    assert estimate["beta"] == pytest.approx(np.cov(asset, market, ddof=1)[0, 1] / np.var(market, ddof=1))
    assert jensen_alpha(asset, market) == pytest.approx(estimate["alpha"])
    assert capm_expected_return(0.02, 1.4, 0.08) == pytest.approx(0.104)


def test_downside_and_active_performance_metrics():
    returns = np.array([0.01, 0.012, -0.004, 0.015, 0.008, -0.002])
    benchmark = np.full(returns.size, 0.005)
    assert lower_partial_moment(returns, order=2) > 0
    assert sharpe_ratio(returns, periods=1) > 0
    assert sortino_ratio(returns, periods=1) > 0
    assert omega_ratio(returns) > 1
    assert kappa_ratio(returns, order=3, periods=1) > 0
    assert information_ratio(returns, benchmark, periods=1) > 0
    assert certainty_equivalent(returns, risk_aversion=3.0) < returns.mean()


def test_similar_sharpe_can_have_different_downside():
    mean = 0.01
    standard_deviation = 0.02
    small_deviation = standard_deviation / np.sqrt(10)
    positively_skewed = np.array([mean - small_deviation] * 9 + [mean + 9 * small_deviation])
    negatively_skewed = np.array([mean - 9 * small_deviation] + [mean + small_deviation] * 9)
    assert sharpe_ratio(positively_skewed, periods=1) == pytest.approx(
        sharpe_ratio(negatively_skewed, periods=1), abs=1e-12
    )
    threshold = 0.005
    assert sortino_ratio(positively_skewed, threshold=threshold, periods=1) != pytest.approx(
        sortino_ratio(negatively_skewed, threshold=threshold, periods=1)
    )
    assert omega_ratio(positively_skewed, threshold=threshold) != pytest.approx(
        omega_ratio(negatively_skewed, threshold=threshold)
    )
