import numpy as np

from turbofan_rul.calibration import conformal_interval, conformal_quantile, nonconformity_scores


def test_nonconformity_scores_zero_for_perfect_mean():
    y = np.array([10.0, 20.0])
    mean = np.array([10.0, 20.0])
    scale = np.array([1.0, 2.0])
    assert np.allclose(nonconformity_scores(y, mean, scale), [0.0, 0.0])


def test_nonconformity_scores_normalizes_by_scale():
    y = np.array([12.0, 24.0])
    mean = np.array([10.0, 20.0])
    scale = np.array([2.0, 8.0])
    assert np.allclose(nonconformity_scores(y, mean, scale), [1.0, 0.5])


def test_conformal_quantile_increases_with_target_coverage():
    rng = np.random.default_rng(0)
    scores = rng.exponential(size=200)
    q50 = conformal_quantile(scores, 0.5)
    q90 = conformal_quantile(scores, 0.9)
    assert q90 > q50


def test_conformal_interval_achieves_target_coverage_on_holdout():
    # Calibrate on one i.i.d. sample, check empirical coverage on a fresh
    # holdout sample lands close to the 90% target (exchangeable data).
    rng = np.random.default_rng(42)
    n = 2000
    mean_cal = rng.normal(size=n)
    scale_cal = np.full(n, 2.0)
    y_cal = mean_cal + rng.normal(scale=2.0, size=n)

    scores = nonconformity_scores(y_cal, mean_cal, scale_cal)
    q_hat = conformal_quantile(scores, target_coverage=0.9)

    mean_test = rng.normal(size=n)
    scale_test = np.full(n, 2.0)
    y_test = mean_test + rng.normal(scale=2.0, size=n)

    lower, upper = conformal_interval(mean_test, scale_test, q_hat)
    empirical_coverage = np.mean((y_test >= lower) & (y_test <= upper))

    assert abs(empirical_coverage - 0.9) < 0.03
