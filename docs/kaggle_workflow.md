# Kaggle workflow for Track B (deep learning)

The Kaggle CLI is configured locally (`~/.kaggle/kaggle.json`) — everything
below can be driven from the command line with the `kaggle` command, no
manual clicking in the Kaggle UI needed.

## 1. Upload datasets (one-time, then only on changes)

**DS02 raw data** (2.45 GB, see [data_download.md](data_download.md) for
provenance):

```bash
kaggle datasets init -p data/       # creates data/dataset-metadata.json once
# adjust id/title in it, then:
kaggle datasets create -p "<absolute Windows path>\data" -t
```

**Important (Windows):** pass `-p` an *absolute* Windows path
(`C:\...\data`), not a relative path with a forward slash (`data/`) — the
latter triggers a path-mixing bug in the Kaggle CLI
(`No such file or directory: '...\\.kaggle/uploads\\data/...json'`).

**Code utility dataset** (`src/turbofan_rul/*.py`, small, seconds to upload):

```bash
kaggle datasets init -p src/
kaggle datasets create -p "<absolute Windows path>\src" -r zip
```

`-r zip` is needed because `src/` only contains a subdirectory
(`turbofan_rul/`) — in the default mode (`skip`) the Kaggle CLI would ignore
directories and upload nothing. **Kaggle flattens the zip structure on
extraction:** the `.py` files end up directly under
`/kaggle/input/turbofan-rul-src/`, not under `.../turbofan_rul/`. Since our
code uses `from turbofan_rul.xxx import yyy` (absolute package imports), the
kernel notebook has to fix that up itself in the first setup cell (copy the
files to `/kaggle/working/pkg/turbofan_rul/`, then
`sys.path.insert(0, "/kaggle/working/pkg")` — see
`kaggle_kernel/track_b_training.ipynb`).

## 2. Create and push the kernel (notebook)

```bash
kaggle kernels init -p kaggle_kernel/
```

`kaggle_kernel/kernel-metadata.json` references both datasets via
`dataset_sources`, sets `enable_gpu: true`, `enable_internet: false` (all
required packages — numpy/pandas/torch/scipy/h5py — are pre-installed in
Kaggle's standard image).

**Gotcha:** a notebook built programmatically via `nbformat` has **no
`kernelspec` metadata** by default. Locally, `jupyter nbconvert --execute`
runs it anyway, but Kaggle's papermill-based executor aborts with
`ValueError: No kernel name found in notebook`. Fix: make sure
`nb.metadata.kernelspec` is set (`python3`, `display_name: Python 3`)
before pushing.

```bash
kaggle kernels push -p kaggle_kernel/
```

The title and the `id` slug in `kernel-metadata.json` should match each
other (otherwise Kaggle creates a different slug and you have to guess it
when checking status).

## 3. Monitor the run

```bash
kaggle kernels status says43/turbofan-rul-track-b
```

On failure (`KernelWorkerStatus.ERROR`), pull the log:

```bash
kaggle kernels output says43/turbofan-rul-track-b -p <target-folder>
```

The log is a JSON-Lines file (`stream_name`/`data` pairs) — search it with
`grep -i "error\|traceback"`.

## 4. Retrieve results

After a successful run, `track_b_summary.csv`, `track_b_test_results.csv`
and a PNG with the degradation curves are under `/kaggle/working/` (written
by the notebook itself). Download with the same `kaggle kernels output`
command as above, then place the CSVs locally under `outputs/` and compare
against Track A in `docs/results.md`.
