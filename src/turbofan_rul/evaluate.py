"""Evaluation metrics for RUL prognostics.

`nasa_score` follows Saxena et al. (2008), the standard asymmetric scoring
function used across the C-MAPSS / N-CMAPSS literature and PHM Data
Challenges: late predictions (predicted RUL too high, i.e. maintenance
would happen after actual failure) are penalized more heavily than early
ones, since they represent the operationally dangerous case.
"""

from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def nasa_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha_early: float = 13.0,
    alpha_late: float = 10.0,
) -> float:
    """Sum of the asymmetric PHM scoring function over all predictions.

    d = y_pred - y_true. d < 0 (early) uses the gentler `alpha_early` decay,
    d >= 0 (late) uses the steeper `alpha_late` decay. Lower is better;
    unlike RMSE this is not symmetric and not on an interpretable physical
    scale, so report it alongside RMSE rather than instead of it.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    d = y_pred - y_true
    early = np.exp(-d[d < 0] / alpha_early) - 1
    late = np.exp(d[d >= 0] / alpha_late) - 1
    return float(np.sum(early) + np.sum(late))


def coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of true values falling within [lower, upper] (prediction interval)."""
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def clip_rul(
    mean: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clip a point prediction and its interval bounds to >= 0.

    RUL cannot be negative, but neither NGBoost's Normal-distribution
    intervals nor a Gaussian/conformal interval built from an unconstrained
    (mean, scale) know that — near end-of-life, where scale is often still
    sizeable relative to the true RUL, the lower bound routinely dips below
    zero. Apply this right before reporting/plotting predictions, not before
    computing metrics: `coverage` already treats a negative `lower` exactly
    like a clipped one (since `y_true >= lower` for any non-negative
    `y_true` and any `lower <= 0`), so clipping earlier would just be
    cosmetic there — but an unclipped "-11.7 cycles remaining" is
    meaningless to a maintenance planner.
    """
    return np.maximum(mean, 0), np.maximum(lower, 0), np.maximum(upper, 0)
