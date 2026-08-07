"""Track A: classical ML + uncertainty quantification via NGBoost.

NGBoost fits a full predictive distribution (Normal by default) rather than
a point estimate, which gives calibrated-in-principle prediction intervals
"for free" instead of requiring separate quantile models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from ngboost import NGBRegressor

from turbofan_rul.calibration import conformal_interval, conformal_quantile, nonconformity_scores
from turbofan_rul.evaluate import clip_rul
from turbofan_rul.features import unit_group_kfold


def fit_ngboost(X: np.ndarray, y: np.ndarray, **kwargs) -> NGBRegressor:
    kwargs.setdefault("verbose", False)
    model = NGBRegressor(**kwargs)
    model.fit(X, y)
    return model


def predict_with_interval(
    model: NGBRegressor, X: np.ndarray, alpha: float = 0.9
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean, lower, upper) with `alpha` central prediction interval,
    taken at face value from NGBoost's own fitted Normal distribution.

    This is the "naive" interval — see `cross_conformal_loo` for a
    calibrated alternative with an actual coverage guarantee. RUL cannot be
    negative, so the result is clipped at 0 (see `evaluate.clip_rul`).
    """
    mean, scale = predict_mean_scale(model, X)
    dist = model.pred_dist(X)
    lower, upper = dist.dist.interval(alpha)
    return clip_rul(mean, lower, upper)


def predict_mean_scale(model: NGBRegressor, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return NGBoost's (mean, scale) of its fitted Normal, without collapsing
    it to a fixed-alpha interval — needed as the raw ingredient for conformal
    calibration."""
    dist = model.pred_dist(X)
    return dist.params["loc"], dist.params["scale"]


def cross_conformal_loo(
    df: pd.DataFrame,
    feat_cols: list[str],
    target_col: str = "rul",
    alpha: float = 0.9,
    **ngb_kwargs,
) -> tuple[NGBRegressor, float]:
    """Fit a conformally-calibrated NGBoost model via nested leave-one-unit-out.

    Inner leave-one-unit-out CV over `df`'s units produces out-of-fold
    (mean, scale) predictions, from which pooled nonconformity scores give a
    single calibration quantile `q_hat`. A final model is then fit on all of
    `df`. Combine the two with `conformal_predict` on new data: the interval
    `mean +/- q_hat * scale` has a distribution-free coverage guarantee at
    `alpha`, *provided the new data is exchangeable with `df`* (see module
    docstring in `calibration.py` for why that assumption can fail).
    """
    ngb_kwargs.setdefault("verbose", False)

    pooled_scores = []
    for train_idx, val_idx in unit_group_kfold(df):
        X_train = df.iloc[train_idx][feat_cols].to_numpy()
        y_train = df.iloc[train_idx][target_col].to_numpy(dtype=float)
        X_val = df.iloc[val_idx][feat_cols].to_numpy()
        y_val = df.iloc[val_idx][target_col].to_numpy(dtype=float)

        model = fit_ngboost(X_train, y_train, **ngb_kwargs)
        mean, scale = predict_mean_scale(model, X_val)
        pooled_scores.append(nonconformity_scores(y_val, mean, scale))

    q_hat = conformal_quantile(np.concatenate(pooled_scores), target_coverage=alpha)

    final_model = fit_ngboost(
        df[feat_cols].to_numpy(), df[target_col].to_numpy(dtype=float), **ngb_kwargs
    )
    return final_model, q_hat


def conformal_predict(
    model: NGBRegressor, q_hat: float, X: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean, lower, upper) using a conformal `q_hat` from
    `cross_conformal_loo`. RUL cannot be negative, so the result is clipped
    at 0 (see `evaluate.clip_rul`)."""
    mean, scale = predict_mean_scale(model, X)
    lower, upper = conformal_interval(mean, scale, q_hat)
    return clip_rul(mean, lower, upper)
