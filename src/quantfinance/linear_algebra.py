"""Linear algebra helpers used in quantitative finance."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class PCAResult:
    """Eigen-based principal component decomposition."""

    eigenvalues: NDArray[np.float64]
    eigenvectors: NDArray[np.float64]
    explained_variance_ratio: NDArray[np.float64]
    loadings: NDArray[np.float64]


def _square_symmetric_matrix(matrix: ArrayLike, name: str) -> NDArray[np.float64]:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"{name} must be square")
    if not np.all(np.isfinite(values)) or not np.allclose(values, values.T):
        raise ValueError(f"{name} must be finite and symmetric")
    return values


def covariance_quadratic_form(weights: ArrayLike, covariance: ArrayLike) -> float:
    """Return portfolio variance as $w' Sigma w$."""
    vector = np.asarray(weights, dtype=float)
    matrix = _square_symmetric_matrix(covariance, "covariance")
    if vector.ndim != 1 or matrix.shape != (vector.size, vector.size):
        raise ValueError("weights and covariance dimensions are incompatible")
    if not np.all(np.isfinite(vector)):
        raise ValueError("weights must be finite")
    return float(vector @ matrix @ vector)


def cholesky_factor(matrix: ArrayLike) -> NDArray[np.float64]:
    """Return the lower Cholesky factor of a positive-definite matrix."""
    values = _square_symmetric_matrix(matrix, "matrix")
    return np.linalg.cholesky(values)


def covariance_to_correlation(covariance: ArrayLike) -> NDArray[np.float64]:
    """Convert a covariance matrix Sigma into C = D^-1 Sigma D^-1."""
    matrix = _square_symmetric_matrix(covariance, "covariance")
    deviations = np.sqrt(np.diag(matrix))
    if np.any(deviations <= 0):
        raise ValueError("covariance diagonal must be strictly positive")
    return matrix / np.outer(deviations, deviations)


def correlation_to_covariance(
    correlation: ArrayLike,
    volatilities: ArrayLike,
) -> NDArray[np.float64]:
    """Convert correlation C into covariance Sigma = D C D."""
    matrix = _square_symmetric_matrix(correlation, "correlation")
    if not np.allclose(np.diag(matrix), 1.0) or np.any(np.abs(matrix) > 1.0):
        raise ValueError("correlation must have unit diagonal and entries in [-1, 1]")
    if np.min(np.linalg.eigvalsh(matrix)) < -1e-10:
        raise ValueError("correlation must be positive semidefinite")
    deviations = np.asarray(volatilities, dtype=float)
    if deviations.ndim != 1 or deviations.size != matrix.shape[0]:
        raise ValueError("volatilities must match the correlation dimensions")
    if np.any(deviations <= 0) or not np.all(np.isfinite(deviations)):
        raise ValueError("volatilities must be finite and strictly positive")
    return deviations[:, None] * matrix * deviations[None, :]


def is_positive_definite(matrix: ArrayLike, tolerance: float = 1e-12) -> bool:
    """Return whether a symmetric matrix is strictly positive definite."""
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    try:
        values = _square_symmetric_matrix(matrix, "matrix")
        return bool(np.all(np.linalg.eigvalsh(values) > tolerance))
    except ValueError:
        return False


def nearest_valid_covariance(
    covariance: ArrayLike,
    minimum_eigenvalue: float = 1e-10,
) -> NDArray[np.float64]:
    """Project a symmetric matrix to a positive-definite covariance matrix."""
    if minimum_eigenvalue <= 0:
        raise ValueError("minimum_eigenvalue must be positive")
    values = _square_symmetric_matrix(covariance, "covariance")
    eigenvalues, eigenvectors = np.linalg.eigh(values)
    clipped = np.maximum(eigenvalues, minimum_eigenvalue)
    return (eigenvectors * clipped) @ eigenvectors.T


def cholesky_correlated_samples(
    correlation: ArrayLike,
    n_samples: int,
    seed: int = 123,
) -> NDArray[np.float64]:
    """Generate standard normal samples with a target correlation matrix."""
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    factor = cholesky_factor(correlation)
    rng = np.random.default_rng(seed)
    independent = rng.standard_normal((factor.shape[0], n_samples))
    return (factor @ independent).T


def _pca_from_matrix(matrix: ArrayLike, n_components: int | None) -> PCAResult:
    values = _square_symmetric_matrix(matrix, "matrix")
    eigenvalues, eigenvectors = np.linalg.eigh(values)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    if np.any(eigenvalues < -1e-10):
        raise ValueError("PCA matrix must be positive semidefinite")
    if n_components is None:
        count = eigenvalues.size
    elif 1 <= n_components <= eigenvalues.size:
        count = n_components
    else:
        raise ValueError("n_components must be between one and the matrix size")
    eigenvalues = eigenvalues[:count]
    eigenvectors = eigenvectors[:, :count]
    total = float(np.sum(np.maximum(np.linalg.eigvalsh(values), 0.0)))
    ratio = eigenvalues / total if total > 0 else np.zeros_like(eigenvalues)
    loadings = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))
    return PCAResult(eigenvalues, eigenvectors, ratio, loadings)


def pca_from_covariance(
    covariance: ArrayLike,
    n_components: int | None = None,
) -> PCAResult:
    """Perform PCA from a covariance matrix using an eigendecomposition."""
    return _pca_from_matrix(covariance, n_components)


def pca_from_correlation(
    correlation: ArrayLike,
    n_components: int | None = None,
) -> PCAResult:
    """Perform PCA from a correlation matrix using an eigendecomposition."""
    return _pca_from_matrix(correlation, n_components)
