import numpy as np
import pytest

from quantfinance.regression import ols_coefficients
from quantfinance.statistics import return_summary


def test_ols_coefficients_match_known_linear_model():
    features = np.column_stack([np.ones(4), [1.0, 2.0, 3.0, 4.0]])
    target = np.array([3.0, 5.0, 7.0, 9.0])
    np.testing.assert_allclose(ols_coefficients(features, target), [1.0, 2.0])


def test_return_summary_contains_moments_and_lower_var():
    summary = return_summary([0.01, -0.02, 0.03, 0.00], confidence=0.75)
    assert summary["mean"] == pytest.approx(0.005)
    assert summary["var"] == pytest.approx(-0.005)
    assert set(summary) == {
        "mean", "variance", "volatility", "skewness", "excess_kurtosis", "var"
    }
