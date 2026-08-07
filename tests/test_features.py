import numpy as np
import pandas as pd

from turbofan_rul.features import aggregate_cycles, feature_columns, unit_group_kfold


def _make_synthetic_raw(n_units=3, cycles_per_unit=4, samples_per_cycle=5):
    rng = np.random.default_rng(0)
    rows = []
    for unit in range(1, n_units + 1):
        for cycle in range(1, cycles_per_unit + 1):
            rul = (cycles_per_unit - cycle) * 10
            for _ in range(samples_per_cycle):
                rows.append(
                    {
                        "a_unit": float(unit),
                        "a_cycle": float(cycle),
                        "a_Fc": 3.0,
                        "a_hs": 1.0 if rul > 0 else 0.0,
                        "rul": rul,
                        "w_alt": rng.normal(),
                        "xs_T24": rng.normal(),
                        "t_fan_eff_mod": 0.0,
                    }
                )
    return pd.DataFrame(rows)


def test_aggregate_cycles_shape_and_constants():
    df = _make_synthetic_raw(n_units=3, cycles_per_unit=4, samples_per_cycle=5)
    agg = aggregate_cycles(df)

    assert len(agg) == 3 * 4
    assert (agg["cycle_len"] == 5).all()
    assert {"w_alt_mean", "w_alt_std", "xs_T24_min", "xs_T24_max"} <= set(agg.columns)
    # rul/Fc/hs carried through unchanged, not aggregated
    assert "rul_mean" not in agg.columns


def test_feature_columns_excludes_ids_target_and_t_block():
    df = _make_synthetic_raw()
    agg = aggregate_cycles(df)
    cols = feature_columns(agg)

    assert "a_unit" not in cols
    assert "a_cycle" not in cols
    assert "rul" not in cols
    assert "a_Fc" not in cols
    assert "a_hs" not in cols
    assert not any(c.startswith("t_") for c in cols)
    assert any(c.startswith("w_alt") for c in cols)


def test_unit_group_kfold_leave_one_unit_out_by_default():
    df = _make_synthetic_raw(n_units=4, cycles_per_unit=3, samples_per_cycle=2)
    agg = aggregate_cycles(df)

    splits = unit_group_kfold(agg)

    assert len(splits) == 4
    for train_idx, val_idx in splits:
        val_units = agg.iloc[val_idx]["a_unit"].unique()
        train_units = agg.iloc[train_idx]["a_unit"].unique()
        assert len(val_units) == 1
        assert set(val_units).isdisjoint(set(train_units))


def test_unit_group_kfold_rejects_too_many_splits():
    df = _make_synthetic_raw(n_units=2, cycles_per_unit=3, samples_per_cycle=2)
    agg = aggregate_cycles(df)

    try:
        unit_group_kfold(agg, n_splits=3)
        assert False, "expected ValueError"
    except ValueError:
        pass
