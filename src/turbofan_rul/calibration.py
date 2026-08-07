"""Split-conformal calibration for NGBoost's (mean, scale) predictions.

NGBoost's own prediction intervals assume its fitted Normal distribution is
correct, which is optimistic in practice (see docs/results.md — observed
coverage was 0.46-0.89 against a 0.90 target). Conformal calibration fixes
this by picking the interval half-width from held-out residuals instead of
trusting the model's own scale estimate, which gives a distribution-free
finite-sample coverage guarantee *as long as calibration and test data are
exchangeable*. That assumption is worth stating explicitly: it does not
hold across a distribution shift (e.g. an unseen flight class), so
calibrating on in-distribution data does not guarantee nominal coverage
under such a shift.
"""

from __future__ import annotations

import numpy as np


def nonconformity_scores(y_true: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Scale-normalized absolute residual: how many `scale` units off was the mean?"""
    y_true = np.asarray(y_true, dtype=float)
    mean = np.asarray(mean, dtype=float)
    scale = np.asarray(scale, dtype=float)
    return np.abs(y_true - mean) / scale


def conformal_quantile(scores: np.ndarray, target_coverage: float = 0.9) -> float:
    """Split-conformal quantile with the standard finite-sample correction.

    Uses the ceil((n+1) * target_coverage) / n empirical quantile (Vovk et
    al.), not the plain sample quantile, so the guarantee holds for finite
    calibration sets rather than only asymptotically.
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    level = min(np.ceil((n + 1) * target_coverage) / n, 1.0)
    return float(np.quantile(scores, level, method="higher"))


def conformal_interval(
    mean: np.ndarray, scale: np.ndarray, q_hat: float
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(mean, dtype=float)
    scale = np.asarray(scale, dtype=float)
    return mean - q_hat * scale, mean + q_hat * scale
