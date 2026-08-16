import numpy as np
import pytest

from quantfinance.fixed_income import (
    continuous_compounding,
    convexity,
    future_value,
    macaulay_duration,
    modified_duration,
    present_value,
)
from quantfinance.returns import pnl_long, pnl_short, portfolio_return


def test_portfolio_return_and_pnl():
    returns = np.array([[0.10, 0.00], [0.00, 0.05]])
    np.testing.assert_allclose(portfolio_return(returns, [0.6, 0.4]), [0.06, 0.02])
    assert pnl_long(500, 30.0, 31.2) == pytest.approx(600.0)
    assert pnl_short(500, 30.0, 31.2) == pytest.approx(-600.0)


def test_discrete_and_continuous_compounding():
    assert future_value(100.0, 0.10, 2) == pytest.approx(121.0)
    assert present_value(121.0, 0.10, 2) == pytest.approx(100.0)
    assert continuous_compounding(100.0, 0.10, 2.0) == pytest.approx(100.0 * np.exp(0.2))
    assert future_value(100.0, -0.5, 2, compounds_per_year=2) == pytest.approx(100.0 * 0.75**4)
    with pytest.raises(ValueError):
        future_value(100.0, -2.0, 1, compounds_per_year=2)


def test_duration_and_convexity_are_positive():
    cash_flows = np.array([5.0, 5.0, 105.0])
    times = np.array([1.0, 2.0, 3.0])
    duration = macaulay_duration(cash_flows, times, 0.05)
    modified = modified_duration(cash_flows, times, 0.05)
    bond_convexity = convexity(cash_flows, times, 0.05)
    assert 2.8 < duration < 3.0
    assert modified < duration
    assert bond_convexity > 0
