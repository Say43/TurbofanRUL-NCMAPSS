import numpy as np
import pytest

from turbofan_rul.track_b import pick_device, predict_ensemble, predict_member, train_ensemble, train_member


def test_pick_device_returns_cpu_when_no_cuda():
    # This test machine has no GPU, so pick_device must fall back to "cpu"
    # via the `not torch.cuda.is_available()` branch.
    assert pick_device() == "cpu"


def _make_synthetic_sequences(n=40, window=6, n_features=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, window, n_features)).astype(np.float32)
    y = (X[:, -1, 0] * 5 + rng.normal(scale=0.5, size=n)).astype(np.float32)
    return X, y


def test_train_member_and_predict_shapes():
    X, y = _make_synthetic_sequences()
    model = train_member(X, y, n_epochs=2, batch_size=8, hidden=4, seed=0)

    mu, var = predict_member(model, X)
    assert mu.shape == (40,)
    assert var.shape == (40,)
    assert np.all(var > 0)


def test_train_ensemble_produces_n_distinct_members():
    X, y = _make_synthetic_sequences()
    models = train_ensemble(X, y, n_members=3, n_epochs=2, batch_size=8, hidden=4)

    assert len(models) == 3
    # different seeds/init -> members should not all produce identical predictions
    preds = [predict_member(m, X[:5])[0] for m in models]
    assert not np.allclose(preds[0], preds[1])


def test_predict_ensemble_returns_mean_and_positive_scale():
    X, y = _make_synthetic_sequences()
    models = train_ensemble(X, y, n_members=3, n_epochs=2, batch_size=8, hidden=4)

    mean, scale = predict_ensemble(models, X)
    assert mean.shape == (40,)
    assert scale.shape == (40,)
    assert np.all(scale > 0)


def test_predict_ensemble_mean_matches_average_of_member_means():
    X, y = _make_synthetic_sequences()
    models = train_ensemble(X, y, n_members=4, n_epochs=2, batch_size=8, hidden=4)

    mean, _ = predict_ensemble(models, X)
    member_means = np.stack([predict_member(m, X)[0] for m in models])
    np.testing.assert_allclose(mean, member_means.mean(axis=0), rtol=1e-5)


def test_train_member_fails_fast_on_unnormalized_inputs():
    # Regression test for a real incident: raw (unstandardized) sensor-scale
    # inputs made the loss go to NaN from epoch 1 on a real Kaggle run,
    # silently "succeeding" after burning 30 minutes of compute. Large,
    # unnormalized inputs should now raise immediately instead.
    rng = np.random.default_rng(0)
    X = rng.normal(loc=1e6, scale=5e5, size=(64, 50, 32)).astype(np.float32)
    y = rng.normal(loc=50.0, scale=10.0, size=64).astype(np.float32)

    with pytest.raises(RuntimeError, match="Non-finite loss"):
        train_member(X, y, n_epochs=5, batch_size=8, hidden=32, lr=1e-2, seed=0)
