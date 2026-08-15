"""General numerical methods used in quantitative finance."""

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import CubicSpline


def bisection(
    function: Callable[[float], float],
    lower: float,
    upper: float,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> float:
    """Find a root inside a bracket using bisection."""
    if lower >= upper or tolerance <= 0 or max_iterations < 1:
        raise ValueError("invalid bracket, tolerance, or max_iterations")
    left_value = function(lower)
    right_value = function(upper)
    if not np.isfinite(left_value) or not np.isfinite(right_value) or left_value * right_value > 0:
        raise ValueError("function values at the bracket endpoints must have opposite signs")
    for _ in range(max_iterations):
        midpoint = 0.5 * (lower + upper)
        midpoint_value = function(midpoint)
        if abs(midpoint_value) <= tolerance or 0.5 * (upper - lower) <= tolerance:
            return float(midpoint)
        if left_value * midpoint_value <= 0:
            upper = midpoint
            right_value = midpoint_value
        else:
            lower = midpoint
            left_value = midpoint_value
    raise RuntimeError("bisection did not converge")


def newton_raphson(
    function: Callable[[float], float],
    derivative: Callable[[float], float],
    initial: float,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> float:
    """Find a root using Newton-Raphson iterations."""
    if tolerance <= 0 or max_iterations < 1:
        raise ValueError("invalid tolerance or max_iterations")
    value = float(initial)
    for _ in range(max_iterations):
        function_value = function(value)
        slope = derivative(value)
        if abs(function_value) <= tolerance:
            return value
        if not np.isfinite(slope) or abs(slope) <= np.finfo(float).eps:
            raise RuntimeError("Newton-Raphson derivative is too small or invalid")
        value -= function_value / slope
        if not np.isfinite(value):
            raise RuntimeError("Newton-Raphson left the finite domain")
    raise RuntimeError("Newton-Raphson did not converge")


def linear_interpolation(
    x_values: ArrayLike,
    y_values: ArrayLike,
    points: ArrayLike | float,
) -> NDArray[np.float64] | float:
    """Interpolate linearly between ordered data points."""
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 2:
        raise ValueError("x and y must be one-dimensional arrays of equal length")
    if np.any(np.diff(x) <= 0) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("x must be finite and strictly increasing")
    return np.interp(points, x, y)


def cubic_spline_curve(
    x_values: ArrayLike,
    y_values: ArrayLike,
    points: ArrayLike | float,
) -> NDArray[np.float64] | float:
    """Evaluate a natural cubic spline through ordered data points."""
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 3:
        raise ValueError("x and y must be one-dimensional arrays with at least three values")
    if np.any(np.diff(x) <= 0) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("x must be finite and strictly increasing")
    return CubicSpline(x, y)(points)
