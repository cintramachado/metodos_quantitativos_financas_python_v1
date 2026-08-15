"""Regression calculations used in finance examples."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def ols_coefficients(features: ArrayLike, target: ArrayLike) -> NDArray[np.float64]:
    """Estimate OLS coefficients with the linear system X'X beta = X'y."""
    design = np.asarray(features, dtype=float)
    response = np.asarray(target, dtype=float)
    if design.ndim != 2 or response.ndim != 1:
        raise ValueError("features must be 2-D and target must be 1-D")
    if design.shape[0] != response.size:
        raise ValueError("features and target must have the same number of rows")
    if not np.all(np.isfinite(design)) or not np.all(np.isfinite(response)):
        raise ValueError("features and target must be finite")
    gram = design.T @ design
    return np.linalg.solve(gram, design.T @ response)
