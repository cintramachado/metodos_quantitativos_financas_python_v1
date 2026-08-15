"""Linear algebra helpers used in quantitative finance."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def covariance_quadratic_form(weights: ArrayLike, covariance: ArrayLike) -> float:
    """Return portfolio variance as $w' Sigma w$."""
    vector = np.asarray(weights, dtype=float)
    matrix = np.asarray(covariance, dtype=float)
    if vector.ndim != 1 or matrix.shape != (vector.size, vector.size):
        raise ValueError("weights and covariance dimensions are incompatible")
    if not np.all(np.isfinite(vector)) or not np.all(np.isfinite(matrix)):
        raise ValueError("weights and covariance must be finite")
    return float(vector @ matrix @ vector)


def cholesky_factor(matrix: ArrayLike) -> NDArray[np.float64]:
    """Return the lower Cholesky factor of a positive-definite matrix."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("matrix must be square")
    if not np.allclose(values, values.T):
        raise ValueError("matrix must be symmetric")
    return np.linalg.cholesky(values)
