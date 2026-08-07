import numpy as np
import pandas as pd

from turbofan_rul.sequences import make_windows, standardize_features


def _make_raw_unit_df(unit, n_rows, start_rul, cycle=1.0):
    return pd.DataFrame(
        {
            "a_unit": float(unit),
            "a_cycle": cycle,
            "rul": np.arange(start_rul, start_rul - n_rows, -1, dtype=float),
            "f0": np.arange(n_rows, dtype=float),
            "f1": np.arange(n_rows, dtype=float) * 2,
        }
    )


def test_make_windows_shape_and_no_subsampling():
    df = _make_raw_unit_df(unit=1, n_rows=20, start_rul=19)
    X, y, units, cycles = make_windows(df, ["f0", "f1"], window=5, stride=1, subsample=1)

    n_expected = 20 - 5 + 1
    assert X.shape == (n_expected, 5, 2)
    assert y.shape == (n_expected,)
    assert units.shape == (n_expected,)
    assert cycles.shape == (n_expected,)
    assert np.all(units == 1.0)
    assert np.all(cycles == 1.0)


def test_make_windows_label_is_rul_at_last_timestep():
    df = _make_raw_unit_df(unit=1, n_rows=10, start_rul=9)
    X, y, _, _ = make_windows(df, ["f0", "f1"], window=3, stride=1, subsample=1)

    # first window covers rows 0..2, label should be rul at row 2
    assert y[0] == df["rul"].iloc[2]
    np.testing.assert_array_equal(X[0], df[["f0", "f1"]].iloc[0:3].to_numpy())


def test_make_windows_does_not_cross_unit_boundary():
    df = pd.concat(
        [_make_raw_unit_df(unit=1, n_rows=10, start_rul=9), _make_raw_unit_df(unit=2, n_rows=10, start_rul=9)],
        ignore_index=True,
    )
    X, y, units, _ = make_windows(df, ["f0", "f1"], window=4, stride=1, subsample=1)

    assert set(np.unique(units)) == {1.0, 2.0}
    n_per_unit = 10 - 4 + 1
    assert (units == 1.0).sum() == n_per_unit
    assert (units == 2.0).sum() == n_per_unit


def test_make_windows_subsample_reduces_row_count():
    df = _make_raw_unit_df(unit=1, n_rows=100, start_rul=99)
    X_full, _, _, _ = make_windows(df, ["f0", "f1"], window=5, stride=1, subsample=1)
    X_sub, _, _, _ = make_windows(df, ["f0", "f1"], window=5, stride=1, subsample=10)

    assert X_sub.shape[0] < X_full.shape[0]
    assert X_sub.shape[0] == (100 // 10) - 5 + 1


def test_make_windows_skips_unit_shorter_than_window():
    df = _make_raw_unit_df(unit=1, n_rows=3, start_rul=2)
    X, y, units, cycles = make_windows(df, ["f0", "f1"], window=5, stride=1, subsample=1)

    assert X.shape == (0, 5, 2)
    assert len(y) == 0
    assert len(cycles) == 0


def test_make_windows_cycle_tracks_last_timestep_cycle():
    df = pd.concat(
        [
            _make_raw_unit_df(unit=1, n_rows=5, start_rul=9, cycle=1.0),
            _make_raw_unit_df(unit=1, n_rows=5, start_rul=4, cycle=2.0),
        ],
        ignore_index=True,
    )
    X, y, units, cycles = make_windows(df, ["f0", "f1"], window=3, stride=1, subsample=1)

    # window ending at row index 4 (0-based) is the last row of cycle 1
    assert cycles[2] == 1.0
    # window ending at row index 5 straddles into cycle 2
    assert cycles[3] == 2.0


def test_standardize_features_fit_gives_zero_mean_unit_std():
    rng = np.random.default_rng(0)
    X = rng.normal(loc=500.0, scale=50.0, size=(200, 10, 3)).astype(np.float32)

    X_scaled, mean, std = standardize_features(X)

    assert np.allclose(X_scaled.mean(axis=(0, 1)), 0.0, atol=1e-3)
    assert np.allclose(X_scaled.std(axis=(0, 1)), 1.0, atol=1e-2)
    assert mean.shape == (3,)
    assert std.shape == (3,)


def test_standardize_features_reuses_given_stats_without_refitting():
    rng = np.random.default_rng(0)
    X_train = rng.normal(loc=500.0, scale=50.0, size=(200, 10, 3)).astype(np.float32)
    _, mean, std = standardize_features(X_train)

    X_other = rng.normal(loc=800.0, scale=50.0, size=(20, 10, 3)).astype(np.float32)
    X_other_scaled, mean_out, std_out = standardize_features(X_other, mean=mean, std=std)

    # scaled with train stats, so it should NOT be zero-mean itself
    assert not np.allclose(X_other_scaled.mean(axis=(0, 1)), 0.0, atol=1e-1)
    np.testing.assert_array_equal(mean_out, mean)
    np.testing.assert_array_equal(std_out, std)


def test_standardize_features_handles_zero_variance_column():
    X = np.zeros((50, 5, 2), dtype=np.float32)
    X[:, :, 1] = 3.0  # constant column, std = 0

    X_scaled, mean, std = standardize_features(X)

    assert np.all(np.isfinite(X_scaled))
    assert std[1] == 0.0
