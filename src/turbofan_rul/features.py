"""Cycle-level feature engineering for Track A (classical ML)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

_ID_COLS = ["a_unit", "a_cycle"]
_CONST_COLS = ["a_Fc", "a_hs", "rul"]
_AGG_FUNCS = ("mean", "std", "min", "max")

# T-block columns encode the (otherwise unobservable) ground-truth degradation
# state used by the simulator. Real deployments would not have this at
# inference time, so Track A must not train on it as a feature.
_EXCLUDE_PREFIXES = ("t_",)


def aggregate_cycles(df: pd.DataFrame, agg_funcs: tuple[str, ...] = _AGG_FUNCS) -> pd.DataFrame:
    """Collapse 1 Hz rows to one row per (unit, cycle).

    `a_Fc`, `a_hs`, and `rul` are constant within a cycle (verified against
    real DS02 data) and are carried through unchanged. All other numeric
    columns (W, X_s, X_v, T) are aggregated with `agg_funcs`. `cycle_len`
    records the number of 1 Hz samples in the cycle (flight duration proxy).
    """
    value_cols = [
        c for c in df.columns if c not in _ID_COLS + _CONST_COLS
    ]

    grouped = df.groupby(_ID_COLS, sort=True)
    aggregated = grouped[value_cols].agg(list(agg_funcs))
    aggregated.columns = [f"{col}_{func}" for col, func in aggregated.columns]

    const = grouped[_CONST_COLS].first()
    cycle_len = grouped.size().rename("cycle_len")

    result = pd.concat([const, cycle_len, aggregated], axis=1).reset_index()
    return result


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Feature columns usable for modeling: everything except ids, target, T-block,
    and `a_hs`/`a_Fc`.

    `a_hs` (health-state flag) is derived from the same unobservable
    degradation state that produces RUL and flips deterministically near
    end-of-life — using it as an input is leakage, the same reasoning that
    excludes the T-block. `a_Fc` (flight class) is constant within DS02's
    training units, so keeping it in only invites spurious extrapolation
    behavior on the test units, which span other flight classes.
    """
    exclude = set(_ID_COLS) | set(_CONST_COLS)
    return [
        c
        for c in df.columns
        if c not in exclude and not c.startswith(_EXCLUDE_PREFIXES)
    ]


def unit_group_kfold(
    df: pd.DataFrame, n_splits: int | None = None
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Group K-Fold over `a_unit` so no unit appears in both train and val.

    Defaults to leave-one-unit-out (n_splits = number of distinct units),
    the most data-efficient choice given DS02's small unit count (6 dev
    units across 2 fault modes).
    """
    units = df["a_unit"].to_numpy()
    n_units = len(np.unique(units))
    if n_splits is None:
        n_splits = n_units
    if n_splits > n_units:
        raise ValueError(f"n_splits ({n_splits}) cannot exceed unit count ({n_units})")

    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(df, groups=units))
