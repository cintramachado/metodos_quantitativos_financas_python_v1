"""Portfolio variance and minimum-variance weights."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .linear_algebra import covariance_quadratic_form


def portfolio_variance(weights: ArrayLike, covariance: ArrayLike) -> float:
    """Return the variance of a portfolio."""
    return covariance_quadratic_form(weights, covariance)


def gmv_weights(covariance: ArrayLike) -> NDArray[np.float64]:
    """Return unconstrained global minimum-variance portfolio weights."""
    matrix = np.asarray(covariance, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("covariance must be square")
    ones = np.ones(matrix.shape[0])
    solution = np.linalg.solve(matrix, ones)
    denominator = float(ones @ solution)
    if denominator <= 0:
        raise ValueError("covariance matrix does not produce valid GMV weights")
    return solution / denominator
