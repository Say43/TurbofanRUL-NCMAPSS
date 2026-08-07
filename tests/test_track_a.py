import numpy as np
import pandas as pd

from turbofan_rul.track_a import (
    conformal_predict,
    cross_conformal_loo,
    fit_ngboost,
    predict_mean_scale,
    predict_with_interval,
)


def test_fit_and_predict_with_interval_shapes_and_ordering():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    y = X[:, 0] * 5 + rng.normal(scale=0.5, size=60)

    model = fit_ngboost(X, y, n_estimators=30)
    mean, lower, upper = predict_with_interval(model, X[:10], alpha=0.9)

    assert mean.shape == (10,)
    assert lower.shape == (10,)
    assert upper.shape == (10,)
    assert np.all(lower <= mean)
    assert np.all(mean <= upper)
    assert np.all(lower >= 0)


def test_predict_with_interval_clips_negative_lower_bound():
    # Regression test: near end-of-life, NGBoost's own scale routinely made
    # the lower bound go negative (e.g. -11.7 "cycles remaining"), which is
    # meaningless for RUL. Force that scenario with a target near zero and
    # confirm it's clipped away.
    rng = np.random.default_rng(2)
    X = rng.normal(size=(60, 3))
    y = np.abs(rng.normal(loc=1.0, scale=1.0, size=60))

    model = fit_ngboost(X, y, n_estimators=30)
    raw_mean, raw_scale = predict_mean_scale(model, X)
    from scipy.stats import norm

    raw_lower = raw_mean - norm.ppf(0.95) * raw_scale
    assert np.any(raw_lower < 0), "test setup should produce a negative raw lower bound"

    mean, lower, upper = predict_with_interval(model, X, alpha=0.9)
    assert np.all(mean >= 0)
    assert np.all(lower >= 0)
    assert np.all(upper >= 0)


def test_predict_with_interval_wider_at_higher_alpha():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 3))
    y = X[:, 0] * 5 + rng.normal(scale=0.5, size=60)

    model = fit_ngboost(X, y, n_estimators=30)
    _, lower_50, upper_50 = predict_with_interval(model, X[:10], alpha=0.5)
    _, lower_90, upper_90 = predict_with_interval(model, X[:10], alpha=0.9)

    assert np.all((upper_90 - lower_90) >= (upper_50 - lower_50))


def _make_synthetic_units_df(n_units=5, rows_per_unit=40, n_features=3, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(1, n_units + 1):
        X = rng.normal(size=(rows_per_unit, n_features))
        y = X[:, 0] * 5 + rng.normal(scale=1.0, size=rows_per_unit)
        for xi, yi in zip(X, y):
            row = {"a_unit": float(unit), "rul": yi}
            row.update({f"f{i}": v for i, v in enumerate(xi)})
            rows.append(row)
    return pd.DataFrame(rows)


def test_predict_mean_scale_shapes():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    y = X[:, 0] * 5 + rng.normal(scale=0.5, size=60)
    model = fit_ngboost(X, y, n_estimators=30)

    mean, scale = predict_mean_scale(model, X[:10])

    assert mean.shape == (10,)
    assert scale.shape == (10,)
    assert np.all(scale > 0)


def test_cross_conformal_loo_and_predict_end_to_end():
    df = _make_synthetic_units_df(n_units=5, rows_per_unit=40)
    feat_cols = ["f0", "f1", "f2"]

    model, q_hat = cross_conformal_loo(df, feat_cols, n_estimators=30, alpha=0.9)

    assert q_hat > 0

    mean, lower, upper = conformal_predict(model, q_hat, df[feat_cols].to_numpy())
    assert np.all(lower <= mean)
    assert np.all(mean <= upper)
    assert np.all(lower >= 0)


def test_conformal_predict_clips_negative_lower_bound():
    df = _make_synthetic_units_df(n_units=5, rows_per_unit=40)
    feat_cols = ["f0", "f1", "f2"]
    df["rul"] = np.abs(df["rul"]) * 0.1  # push targets near zero

    model, q_hat = cross_conformal_loo(df, feat_cols, n_estimators=30, alpha=0.9)
    mean, lower, upper = conformal_predict(model, q_hat, df[feat_cols].to_numpy())

    assert np.all(mean >= 0)
    assert np.all(lower >= 0)
    assert np.all(upper >= 0)


def test_cross_conformal_loo_higher_alpha_gives_wider_interval():
    df = _make_synthetic_units_df(n_units=5, rows_per_unit=40)
    feat_cols = ["f0", "f1", "f2"]

    _, q_hat_50 = cross_conformal_loo(df, feat_cols, n_estimators=30, alpha=0.5)
    _, q_hat_90 = cross_conformal_loo(df, feat_cols, n_estimators=30, alpha=0.9)

    assert q_hat_90 >= q_hat_50
