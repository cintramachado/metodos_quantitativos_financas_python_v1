import numpy as np
import pytest

from quantfinance.simulation import (
    simulate_gbm,
    simulate_jump_diffusion,
    simulate_ou,
    simulate_random_walk,
)
from quantfinance.statistics import (
    descriptive_statistics,
    fit_normal_mle,
    fit_student_t,
    historical_quantile,
    normal_log_likelihood,
)


def test_descriptive_statistics_and_historical_quantile():
    sample = np.array([-0.02, 0.00, 0.01, 0.03, 0.04])
    summary = descriptive_statistics(sample)
    assert summary["mean"] == pytest.approx(0.012)
    assert summary["variance"] == pytest.approx(np.var(sample, ddof=1))
    assert summary["quantile_50"] == pytest.approx(0.01)
    assert historical_quantile(sample, 0.0) == pytest.approx(-0.02)


def test_normal_mle_and_log_likelihood():
    rng = np.random.default_rng(10)
    sample = rng.normal(0.01, 0.02, 2_000)
    fit = fit_normal_mle(sample)
    assert fit["mean"] == pytest.approx(0.01, abs=0.002)
    assert fit["standard_deviation"] == pytest.approx(0.02, abs=0.002)
    assert fit["log_likelihood"] == pytest.approx(
        normal_log_likelihood(sample, fit["mean"], fit["standard_deviation"])
    )


def test_student_t_fit_recovers_heavy_tails():
    rng = np.random.default_rng(11)
    sample = 0.01 * rng.standard_t(df=5, size=2_000)
    fit = fit_student_t(sample)
    assert fit["degrees_of_freedom"] > 2
    assert fit["degrees_of_freedom"] < 10
    assert np.isfinite(fit["log_likelihood"])


def test_stochastic_processes_are_reproducible_and_valid():
    random_walk = simulate_random_walk(100, volatility=0.02, seed=3)
    assert random_walk.shape == (101,)
    np.testing.assert_array_equal(random_walk, simulate_random_walk(100, volatility=0.02, seed=3))

    gbm = simulate_gbm(100.0, 0.08, 0.20, 1.0, steps=100, seed=3)
    jump_diffusion = simulate_jump_diffusion(
        100.0, 0.08, 0.20, 2.0, -0.05, 0.10, 1.0, steps=100, seed=3
    )
    assert np.all(gbm > 0)
    assert np.all(jump_diffusion > 0)

    ou = simulate_ou(3.0, 0.0, 0.10, 1.0, steps=100, start=2.0, seed=3)
    assert ou.shape == (101,)
    assert abs(ou[-1]) < abs(ou[0])
