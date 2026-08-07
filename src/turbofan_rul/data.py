"""Loader for NASA N-CMAPSS HDF5 files.

Schema per Arias Chao et al. (2021), "Aircraft Engine Run-to-Failure Dataset
under Real Flight Conditions for Prognostics and Diagnostics": each file has
row-aligned arrays W, X_s, X_v, T, A, Y (1 Hz samples) plus <name>_var arrays
holding the column names for W, X_s, X_v, T, A as fixed-width byte strings.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

_BLOCKS = ("W", "X_s", "X_v", "T", "A")


def _decode_var_names(raw: np.ndarray) -> list[str]:
    flat = raw.reshape(-1)
    return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in flat]


def load_ncmapss_h5(path: str | Path, split: str = "dev") -> pd.DataFrame:
    """Load one split ("dev" or "test") of an N-CMAPSS HDF5 file into a DataFrame.

    Columns are prefixed by block (w_, xs_, xv_, t_, a_) to keep origin
    traceable; the RUL label is returned as column "rul".
    """
    if split not in ("dev", "test"):
        raise ValueError(f"split must be 'dev' or 'test', got {split!r}")

    prefix = {"W": "w_", "X_s": "xs_", "X_v": "xv_", "T": "t_", "A": "a_"}
    frames: dict[str, np.ndarray] = {}

    with h5py.File(path, "r") as f:
        for block in _BLOCKS:
            data = f[f"{block}_{split}"][:]
            names = _decode_var_names(f[f"{block}_var"][:])
            for col_idx, name in enumerate(names):
                frames[f"{prefix[block]}{name}"] = data[:, col_idx]
        frames["rul"] = f[f"Y_{split}"][:, 0]

    return pd.DataFrame(frames)


def list_units(df: pd.DataFrame) -> list[int]:
    return sorted(df["a_unit"].unique().tolist())
