import numpy as np
import pytest
from scipy.stats import norm

from quantfinance.numerical import bisection, cubic_spline_curve, linear_interpolation, newton_raphson
from quantfinance.options import (
    black_scholes_call,
    black_scholes_put,
    binomial_american,
    binomial_european,
    finite_difference_delta,
    finite_difference_gamma,
    implied_volatility_bisection,
    implied_volatility_newton,
    monte_carlo_european_option,
)
from quantfinance.simulation import simulate_correlated_normal, simulate_multivariate_student_t


def test_root_finders_and_interpolation():
    function = lambda value: value**2 - 2
    derivative = lambda value: 2 * value
    assert bisection(function, 0.0, 2.0) == pytest.approx(np.sqrt(2), abs=1e-8)
    assert newton_raphson(function, derivative, 1.0) == pytest.approx(np.sqrt(2), abs=1e-8)
    assert linear_interpolation([0.0, 1.0], [0.0, 2.0], 0.25) == pytest.approx(0.5)
    assert cubic_spline_curve([0.0, 1.0, 2.0], [0.0, 1.0, 4.0], 1.0) == pytest.approx(1.0)


def test_black_scholes_parity_and_known_call():
    call = black_scholes_call(100.0, 100.0, 1.0, 0.05, 0.20)
    put = black_scholes_put(100.0, 100.0, 1.0, 0.05, 0.20)
    assert call == pytest.approx(10.45058357, abs=1e-6)
    assert call - put == pytest.approx(100.0 - 100.0 * np.exp(-0.05), abs=1e-10)


def test_implied_volatility_bisection_and_newton():
    target = black_scholes_call(100.0, 100.0, 1.0, 0.05, 0.30)
    assert implied_volatility_bisection(target, 100.0, 100.0, 1.0, 0.05) == pytest.approx(0.30, abs=1e-7)
    assert implied_volatility_newton(target, 100.0, 100.0, 1.0, 0.05) == pytest.approx(0.30, abs=1e-7)


def test_finite_difference_greeks_match_analytic_values():
    spot, strike, maturity, rate, volatility = 100.0, 100.0, 1.0, 0.05, 0.20
    d1 = (np.log(spot / strike) + (rate + volatility**2 / 2) * maturity) / (volatility * np.sqrt(maturity))
    analytic_delta = norm.cdf(d1)
    analytic_gamma = norm.pdf(d1) / (spot * volatility * np.sqrt(maturity))
    assert finite_difference_delta(spot, strike, maturity, rate, volatility) == pytest.approx(analytic_delta, abs=1e-5)
    assert finite_difference_gamma(spot, strike, maturity, rate, volatility) == pytest.approx(analytic_gamma, abs=1e-5)


def test_binomial_and_monte_carlo_prices():
    parameters = (100.0, 100.0, 1.0, 0.05, 0.20)
    european = binomial_european(*parameters, steps=500)
    analytic = black_scholes_call(*parameters)
    american_put = binomial_american(*parameters, steps=500, option="put")
    european_put = binomial_european(*parameters, steps=500, option="put")
    assert european == pytest.approx(analytic, abs=0.03)
    assert american_put >= european_put

    result = monte_carlo_european_option(*parameters, simulations=200_000, seed=9)
    assert abs(result.price - analytic) <= 4 * result.standard_error
    assert result.standard_error > 0


def test_correlated_normal_and_student_t_samples():
    correlation = np.array([[1.0, 0.6], [0.6, 1.0]])
    normal = simulate_correlated_normal(correlation, 100_000, seed=5)
    student_t = simulate_multivariate_student_t(correlation, 5.0, 100_000, seed=5)
    np.testing.assert_allclose(np.corrcoef(normal, rowvar=False), correlation, atol=0.01)
    np.testing.assert_allclose(np.corrcoef(student_t, rowvar=False), correlation, atol=0.02)
