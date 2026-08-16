import numpy as np
import pytest
import statsmodels.api as sm

from quantfinance.regression import (
    calculate_vif,
    capm_regression,
    minimum_variance_hedge_ratio,
    ols_numpy,
    regression_diagnostics,
)


def test_ols_numpy_matches_statsmodels_and_reports_inference():
    rng = np.random.default_rng(21)
    predictor = rng.normal(size=300)
    target = 0.4 + 1.7 * predictor + rng.normal(scale=0.5, size=300)
    design = sm.add_constant(predictor)
    manual = ols_numpy(design, target)
    reference = sm.OLS(target, design).fit()
    np.testing.assert_allclose(manual.coefficients, reference.params)
    np.testing.assert_allclose(manual.fitted_values, reference.fittedvalues)
    assert manual.r_squared == pytest.approx(reference.rsquared)
    assert manual.adjusted_r_squared == pytest.approx(reference.rsquared_adj)
    assert manual.f_statistic == pytest.approx(float(reference.fvalue))


def test_capm_beta_equals_covariance_formula():
    rng = np.random.default_rng(22)
    market = rng.normal(0.001, 0.02, 500)
    asset = 0.0002 + 1.35 * market + rng.normal(0.0, 0.01, 500)
    result = capm_regression(asset, market)
    beta_covariance = np.cov(asset, market, ddof=1)[0, 1] / np.var(market, ddof=1)
    assert result.coefficients[1] == pytest.approx(beta_covariance)


def test_vif_and_hedge_ratio():
    rng = np.random.default_rng(23)
    factor = rng.normal(size=500)
    features = np.column_stack([factor, 0.9 * factor + rng.normal(scale=0.2, size=500)])
    vif = calculate_vif(features)
    assert np.all(vif > 1)

    futures = rng.normal(size=500)
    spot = 0.8 * futures + rng.normal(scale=0.3, size=500)
    ratio = minimum_variance_hedge_ratio(spot, futures)
    assert ratio == pytest.approx(np.cov(spot, futures, ddof=1)[0, 1] / np.var(futures, ddof=1))


def test_regression_diagnostics_return_expected_metrics():
    rng = np.random.default_rng(24)
    predictor = rng.normal(size=200)
    target = 0.2 + 0.8 * predictor + rng.normal(size=200)
    model = sm.OLS(target, sm.add_constant(predictor)).fit()
    diagnostics = regression_diagnostics(model)
    assert set(diagnostics) == {
        "durbin_watson",
        "white_lm_pvalue",
        "breusch_pagan_lm_pvalue",
        "breusch_godfrey_lm_pvalue",
    }
    assert 0 <= diagnostics["durbin_watson"] <= 4


def test_ols_without_intercept_uses_uncentered_fit_statistics():
    predictor = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    target = 2.0 * predictor
    result = ols_numpy(predictor[:, None], target)
    assert result.coefficients[0] == pytest.approx(2.0)
    assert result.r_squared == pytest.approx(1.0)


def test_ols_rejects_singular_design():
    features = np.column_stack([np.ones(5), np.arange(5), np.arange(5)])
    with pytest.raises(ValueError):
        ols_numpy(features, np.arange(5, dtype=float))
