# TurbofanRUL-NCMAPSS

Predictive maintenance: Remaining Useful Life (RUL) prediction for turbofan
engines based on NASA's N-CMAPSS dataset (DS02, realistic flight profiles
instead of the synthetic cycles used in classic C-MAPSS).

## Goal

RUL prediction on N-CMAPSS DS02 with two modeling approaches developed and
evaluated in parallel, including calibrated uncertainty quantification (UQ)
instead of plain point estimates.

## Methodology

1. **Data:** N-CMAPSS DS02 (HDF5, 1 Hz sensor data over full flights,
   run-to-failure, multiple units/failure modes, RUL label already included).
2. **Preprocessing:** aggregation of the 1 Hz raw data into cycle-level
   features (flight-phase statistics) plus optional sliding-window sequences
   for the deep-learning track. Split **by unit**, not by time (no leakage).
3. **Track A — classical ML + UQ:** gradient boosting with quantile loss /
   NGBoost for prediction intervals.
4. **Track B — deep learning + UQ:** CNN/LSTM baseline with deep ensembles
   or MC dropout for uncertainty. Training runs on Kaggle (T4 GPU), not
   locally — see [docs/kaggle_workflow.md](docs/kaggle_workflow.md).
5. **Evaluation:** RMSE, NASA PHM score (asymmetric penalty function,
   punishing late predictions harder), coverage/calibration of the
   prediction intervals, cross-validation across flight classes/operating
   conditions.
6. **Comparison & report:** side-by-side comparison of both tracks under
   [docs/results.md](docs/results.md) — Track A and Track B are both done.

## Setup

```powershell
uv sync
```

For Track B additionally (only needed locally if not training on Kaggle):

```powershell
uv sync --extra deep
```

## Data

DS02 is not version-controlled (`data/` is gitignored). Download
instructions: [docs/data_download.md](docs/data_download.md).

## Project structure

```
src/turbofan_rul/   Python package (preprocessing, models, evaluation)
notebooks/          EDA and results notebooks
docs/                Download guide, Kaggle workflow, results report
data/                Raw data (gitignored)
outputs/             Trained models, metrics (gitignored)
tests/               pytest tests
```

## License & data source

Code and documentation in this repo are licensed under the [MIT
License](LICENSE).

The dataset used (NASA N-CMAPSS DS02) is not part of this repo (see
[docs/data_download.md](docs/data_download.md) for download and citation)
and comes from the NASA Prognostics Center of Excellence Data Set
Repository:

> Arias Chao, M.; Kulkarni, C.; Goebel, K.; Fink, O. (2021). *Aircraft
> Engine Run-to-Failure Dataset under Real Flight Conditions for
> Prognostics and Diagnostics.* Data, 6(1), 5.
> DOI: [10.3390/data6010005](https://doi.org/10.3390/data6010005)
