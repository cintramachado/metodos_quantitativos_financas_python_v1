import numpy as np
import pytest
from sklearn.decomposition import PCA

from quantfinance.linear_algebra import (
    cholesky_correlated_samples,
    cholesky_factor,
    correlation_to_covariance,
    covariance_to_correlation,
    is_positive_definite,
    nearest_valid_covariance,
    pca_from_correlation,
    pca_from_covariance,
)
from quantfinance.portfolio import portfolio_variance, portfolio_volatility


def test_covariance_correlation_identity():
    volatilities = np.array([0.10, 0.15, 0.20])
    correlation = np.array([
        [1.0, 0.2, -0.1],
        [0.2, 1.0, 0.4],
        [-0.1, 0.4, 1.0],
    ])
    covariance = correlation_to_covariance(correlation, volatilities)
    np.testing.assert_allclose(covariance_to_correlation(covariance), correlation)
    np.testing.assert_allclose(
        covariance,
        np.diag(volatilities) @ correlation @ np.diag(volatilities),
    )


def test_portfolio_variance_and_volatility_include_covariances():
    weights = np.array([0.5, 0.5])
    covariance = np.array([[0.04, 0.02], [0.02, 0.09]])
    expected = weights @ covariance @ weights
    assert portfolio_variance(weights, covariance) == pytest.approx(expected)
    assert portfolio_volatility(weights, covariance) == pytest.approx(np.sqrt(expected))
    naive_average = weights @ np.sqrt(np.diag(covariance))
    assert portfolio_volatility(weights, covariance) != pytest.approx(naive_average)


def test_positive_definite_and_nearest_covariance():
    positive_definite = np.array([[2.0, 0.5], [0.5, 1.0]])
    singular = np.array([[1.0, 1.0], [1.0, 1.0]])
    assert is_positive_definite(positive_definite)
    assert not is_positive_definite(singular)
    repaired = nearest_valid_covariance(singular)
    assert is_positive_definite(repaired)


def test_cholesky_and_correlated_samples():
    correlation = np.array([[1.0, 0.7], [0.7, 1.0]])
    factor = cholesky_factor(correlation)
    np.testing.assert_allclose(factor @ factor.T, correlation)
    samples = cholesky_correlated_samples(correlation, 100_000, seed=11)
    np.testing.assert_allclose(np.corrcoef(samples, rowvar=False), correlation, atol=0.01)


def test_manual_pca_matches_sklearn_variance():
    rng = np.random.default_rng(7)
    data = rng.normal(size=(500, 4)) @ np.array([
        [1.0, 0.4, 0.0, 0.0],
        [0.0, 0.9, 0.2, 0.0],
        [0.0, 0.0, 0.7, 0.3],
        [0.0, 0.0, 0.0, 0.5],
    ])
    covariance = np.cov(data, rowvar=False)
    manual = pca_from_covariance(covariance)
    sklearn_pca = PCA().fit(data)
    np.testing.assert_allclose(
        manual.explained_variance_ratio,
        sklearn_pca.explained_variance_ratio_,
    )
    correlation = covariance_to_correlation(covariance)
    correlation_pca = pca_from_correlation(correlation)
    assert correlation_pca.eigenvectors.shape == (4, 4)
