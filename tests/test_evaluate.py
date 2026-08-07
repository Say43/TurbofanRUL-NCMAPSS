import numpy as np

from turbofan_rul.evaluate import clip_rul, coverage, nasa_score, rmse


def test_rmse_zero_for_perfect_predictions():
    y = np.array([10.0, 20.0, 30.0])
    assert rmse(y, y) == 0.0


def test_rmse_known_value():
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([3.0, 4.0])
    assert rmse(y_true, y_pred) == 3.5355339059327378


def test_nasa_score_zero_for_perfect_predictions():
    y = np.array([10.0, 20.0, 30.0])
    assert nasa_score(y, y) == 0.0


def test_nasa_score_penalizes_late_predictions_more_than_early():
    y_true = np.array([50.0])
    early_pred = np.array([40.0])  # d = -10
    late_pred = np.array([60.0])  # d = +10
    assert nasa_score(y_true, late_pred) > nasa_score(y_true, early_pred)


def test_coverage_all_within_interval():
    y_true = np.array([1.0, 2.0, 3.0])
    lower = np.array([0.0, 0.0, 0.0])
    upper = np.array([5.0, 5.0, 5.0])
    assert coverage(y_true, lower, upper) == 1.0


def test_coverage_partial():
    y_true = np.array([1.0, 10.0])
    lower = np.array([0.0, 0.0])
    upper = np.array([5.0, 5.0])
    assert coverage(y_true, lower, upper) == 0.5


def test_clip_rul_clips_negative_values_to_zero():
    mean = np.array([-2.0, 5.0])
    lower = np.array([-11.7, 3.0])
    upper = np.array([20.9, -1.0])

    mean_c, lower_c, upper_c = clip_rul(mean, lower, upper)

    np.testing.assert_array_equal(mean_c, [0.0, 5.0])
    np.testing.assert_array_equal(lower_c, [0.0, 3.0])
    np.testing.assert_array_equal(upper_c, [20.9, 0.0])


def test_clip_rul_leaves_nonnegative_values_unchanged():
    mean = np.array([4.6, 15.8])
    lower = np.array([0.1, 2.0])
    upper = np.array([20.9, 30.0])

    mean_c, lower_c, upper_c = clip_rul(mean, lower, upper)

    np.testing.assert_array_equal(mean_c, mean)
    np.testing.assert_array_equal(lower_c, lower)
    np.testing.assert_array_equal(upper_c, upper)


def test_clip_rul_does_not_change_coverage_result():
    # Clipping a negative lower bound to 0 must not change whether a
    # non-negative y_true counts as covered -- y_true >= lower already held
    # for any negative lower, so coverage is invariant to this clip.
    y_true = np.array([16.0])
    mean = np.array([15.6])
    lower = np.array([-5.0])
    upper = np.array([31.1])

    before = coverage(y_true, lower, upper)
    _, lower_c, upper_c = clip_rul(mean, lower, upper)
    after = coverage(y_true, lower_c, upper_c)

    assert before == after == 1.0
