"""Descriptive statistics for financial return samples."""

import numpy as np
from numpy.typing import ArrayLike


def return_summary(returns: ArrayLike, confidence: float = 0.95) -> dict[str, float]:
    """Return basic moments and lower-tail historical VaR."""
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("returns must be a finite one-dimensional sample")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    quantile = 1.0 - confidence
    centered = values - values.mean()
    standard_deviation = values.std(ddof=1)
    skewness = np.mean(centered**3) / standard_deviation**3
    kurtosis = np.mean(centered**4) / standard_deviation**4 - 3.0
    return {
        "mean": float(values.mean()),
        "variance": float(values.var(ddof=1)),
        "volatility": float(standard_deviation),
        "skewness": float(skewness),
        "excess_kurtosis": float(kurtosis),
        "var": float(np.quantile(values, quantile)),
    }
