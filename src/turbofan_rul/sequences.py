"""Sliding-window sequence construction for Track B (deep learning).

Unlike Track A's per-cycle aggregation, Track B trains on the raw 1 Hz
signal directly. Each unit is one continuous run-to-failure trajectory
(its cycles back to back), so windows are built per unit across its full
subsampled history and never cross a unit boundary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    window: int = 50,
    stride: int = 1,
    subsample: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Slide a window over each unit's subsampled 1 Hz trajectory.

    The label for a window is the RUL at its last timestep. Units with
    fewer than `window` (subsampled) rows are skipped.

    Returns (X, y, units, cycles): X has shape (n_windows, window,
    n_features); `cycles` is the `a_cycle` value at each window's last
    timestep, useful for plotting predictions against flight progression.
    """
    X_parts: list[np.ndarray] = []
    y_parts: list[float] = []
    unit_parts: list[float] = []
    cycle_parts: list[float] = []

    for unit, g in df.groupby("a_unit", sort=True):
        g = g.iloc[::subsample]
        feats = g[feature_cols].to_numpy(dtype=np.float32)
        rul = g["rul"].to_numpy(dtype=np.float32)
        cycle = g["a_cycle"].to_numpy(dtype=np.float32)
        n = len(g)
        if n < window:
            continue
        for start in range(0, n - window + 1, stride):
            end = start + window
            X_parts.append(feats[start:end])
            y_parts.append(rul[end - 1])
            unit_parts.append(unit)
            cycle_parts.append(cycle[end - 1])

    if not X_parts:
        n_features = len(feature_cols)
        return (
            np.empty((0, window, n_features), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.float32),
        )

    X = np.stack(X_parts).astype(np.float32)
    y = np.array(y_parts, dtype=np.float32)
    units = np.array(unit_parts, dtype=np.float64)
    cycles = np.array(cycle_parts, dtype=np.float32)
    return X, y, units, cycles


def standardize_features(
    X: np.ndarray, mean: np.ndarray | None = None, std: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-feature zero-mean/unit-variance scaling, fit on `X` if `mean`/`std`
    are not given.

    Raw N-CMAPSS sensor channels span wildly different physical scales
    (temperatures in the hundreds, pressures, speeds in the thousands RPM).
    Feeding that directly into a gradient-trained CNN causes activations to
    blow up and the loss to go to NaN within the first step — always fit
    `mean`/`std` on the training split only, then reuse them (don't refit)
    for calibration/test data to avoid leakage.
    """
    if mean is None or std is None:
        mean = X.mean(axis=(0, 1))
        std = X.std(axis=(0, 1))
    std_safe = np.where(std < 1e-8, 1.0, std)
    return (X - mean) / std_safe, mean, std
