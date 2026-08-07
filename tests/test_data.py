import h5py
import numpy as np

from turbofan_rul.data import list_units, load_ncmapss_h5

_BLOCK_VARS = {
    "W": ["alt", "Mach", "TRA", "T2"],
    "X_s": ["T24", "T30"],
    "X_v": ["T40", "P30"],
    "T": ["fan_eff_mod"],
    "A": ["unit", "cycle", "Fc", "hs"],
}


def _make_synthetic_ncmapss(path, n_rows=20):
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        for split, n in (("dev", n_rows), ("test", n_rows // 2)):
            for block, names in _BLOCK_VARS.items():
                if block == "A":
                    unit = np.repeat([1, 2], n // 2 + 1)[:n]
                    cycle = np.tile(np.arange(1, n // 2 + 2), 2)[:n]
                    fc = np.ones(n)
                    hs = np.ones(n)
                    data = np.stack([unit, cycle, fc, hs], axis=1).astype(float)
                else:
                    data = rng.normal(size=(n, len(names)))
                f.create_dataset(f"{block}_{split}", data=data)
            f.create_dataset(f"Y_{split}", data=rng.uniform(0, 100, size=(n, 1)))
        for block, names in _BLOCK_VARS.items():
            f.create_dataset(
                f"{block}_var",
                data=np.array([n.encode("utf-8") for n in names]),
            )


def test_load_ncmapss_h5_dev(tmp_path):
    h5_path = tmp_path / "synthetic.h5"
    _make_synthetic_ncmapss(h5_path, n_rows=20)

    df = load_ncmapss_h5(h5_path, split="dev")

    assert len(df) == 20
    assert {"w_alt", "xs_T24", "xv_T40", "t_fan_eff_mod", "a_unit", "rul"} <= set(df.columns)
    assert list_units(df) == [1, 2]


def test_load_ncmapss_h5_test_split(tmp_path):
    h5_path = tmp_path / "synthetic.h5"
    _make_synthetic_ncmapss(h5_path, n_rows=20)

    df = load_ncmapss_h5(h5_path, split="test")

    assert len(df) == 10


def test_load_ncmapss_h5_invalid_split(tmp_path):
    h5_path = tmp_path / "synthetic.h5"
    _make_synthetic_ncmapss(h5_path, n_rows=4)

    try:
        load_ncmapss_h5(h5_path, split="bogus")
        assert False, "expected ValueError"
    except ValueError:
        pass
