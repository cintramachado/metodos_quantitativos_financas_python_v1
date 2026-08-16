"""Fixed-income valuation, duration, and convexity calculations."""

import numpy as np
from numpy.typing import ArrayLike


def present_value(
    future_value: float,
    rate: float,
    periods: int,
    compounds_per_year: int = 1,
) -> float:
    """Discount a future value with discrete compounding."""
    if (
        future_value < 0
        or not np.isfinite(rate)
        or rate <= -compounds_per_year
        or periods < 0
        or compounds_per_year < 1
    ):
        raise ValueError("invalid future value, rate, periods, or compounding frequency")
    return float(future_value / (1.0 + rate / compounds_per_year) ** (periods * compounds_per_year))


def future_value(
    present_value_amount: float,
    rate: float,
    periods: int,
    compounds_per_year: int = 1,
) -> float:
    """Accumulate a present value with discrete compounding."""
    if (
        present_value_amount < 0
        or not np.isfinite(rate)
        or rate <= -compounds_per_year
        or periods < 0
        or compounds_per_year < 1
    ):
        raise ValueError("invalid present value, rate, periods, or compounding frequency")
    return float(present_value_amount * (1.0 + rate / compounds_per_year) ** (periods * compounds_per_year))


def continuous_compounding(amount: float, rate: float, time: float) -> float:
    """Accumulate an amount with continuous compounding, $FV=PV e^{rt}$."""
    if amount < 0 or time < 0 or not np.isfinite(rate):
        raise ValueError("amount and time must be non-negative and rate must be finite")
    return float(amount * np.exp(rate * time))


def macaulay_duration(
    cash_flows: ArrayLike,
    times: ArrayLike,
    yield_rate: float,
    compounds_per_year: int = 1,
) -> float:
    """Calculate Macaulay duration from promised cash flows."""
    flows = np.asarray(cash_flows, dtype=float)
    payment_times = np.asarray(times, dtype=float)
    if flows.ndim != 1 or payment_times.shape != flows.shape:
        raise ValueError("cash flows and times must be one-dimensional with equal lengths")
    if np.any(flows < 0) or np.any(payment_times < 0) or np.sum(flows) <= 0:
        raise ValueError("cash flows must be non-negative and not all zero")
    if yield_rate <= -compounds_per_year or compounds_per_year < 1:
        raise ValueError("yield rate or compounding frequency is invalid")
    discount = (1.0 + yield_rate / compounds_per_year) ** (compounds_per_year * payment_times)
    prices = flows / discount
    return float(np.sum(payment_times * prices) / np.sum(prices))


def modified_duration(
    cash_flows: ArrayLike,
    times: ArrayLike,
    yield_rate: float,
    compounds_per_year: int = 1,
) -> float:
    """Calculate modified duration from Macaulay duration."""
    duration = macaulay_duration(cash_flows, times, yield_rate, compounds_per_year)
    return float(duration / (1.0 + yield_rate / compounds_per_year))


def convexity(
    cash_flows: ArrayLike,
    times: ArrayLike,
    yield_rate: float,
    compounds_per_year: int = 1,
) -> float:
    """Calculate discrete-compounding price convexity."""
    flows = np.asarray(cash_flows, dtype=float)
    payment_times = np.asarray(times, dtype=float)
    if flows.ndim != 1 or payment_times.shape != flows.shape:
        raise ValueError("cash flows and times must be one-dimensional with equal lengths")
    if np.any(flows < 0) or np.any(payment_times < 0) or np.sum(flows) <= 0:
        raise ValueError("cash flows must be non-negative and not all zero")
    if yield_rate <= -compounds_per_year or compounds_per_year < 1:
        raise ValueError("yield rate or compounding frequency is invalid")
    discount = (1.0 + yield_rate / compounds_per_year) ** (compounds_per_year * payment_times)
    price = np.sum(flows / discount)
    numerator = payment_times * (payment_times + 1.0 / compounds_per_year) * flows
    return float(np.sum(numerator / discount) / ((1.0 + yield_rate / compounds_per_year) ** 2 * price))
