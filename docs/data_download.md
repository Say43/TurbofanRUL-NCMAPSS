# Data acquisition: N-CMAPSS DS02

The N-CMAPSS dataset is distributed as a ZIP via the NASA PCoE S3 bucket and
contains all subsets (DS01–DS08) as individual HDF5 files. **There is no
official standalone download for DS02 only** — confirmed via the
[mohyunho/N-CMAPSS_DL](https://github.com/mohyunho/N-CMAPSS_DL) reference
implementation, which points to the same NASA source. The full package is
**15.76 GB** (checked via HTTP HEAD, as of 2026-08-05).

Unofficial fallback (only if the NASA link is ever unreachable, provenance/
freshness not guaranteed): Google Drive mirror linked in the README of the
GitHub repo mentioned above.

## Download

```powershell
Invoke-WebRequest `
  -Uri "https://phm-datasets.s3.amazonaws.com/NASA/17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2.zip" `
  -OutFile "data/ncmapss_full.zip"
```

Afterwards, do **not** extract the whole archive (the other 7 subsets waste
disk space) — instead pull out only the DS02 file. From the repo root, using
`unzip` (available in Git Bash):

```bash
unzip -l data/ncmapss_full.zip | grep -i DS02   # check the path inside the archive
unzip -j data/ncmapss_full.zip "*N-CMAPSS_DS02*.h5" -d data/
mv data/N-CMAPSS_DS02*.h5 data/N-CMAPSS_DS02.h5
rm data/ncmapss_full.zip
```

`-j` discards the folder path from the archive (flat extraction), so the
file ends up directly under `data/`.

## Alternative sources

- Official PCoE repository (overview/metadata):
  https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
- Mirror: https://data.phmsociety.org/nasa/

## Source & citation

The dataset comes from the NASA Prognostics Center of Excellence (PCoE)
Data Set Repository. Usage is unrestricted (NASA data as a U.S. Government
Work, de facto public domain); the repository's only request is that the
dataset and its authors be cited in any publication that uses it:

> Arias Chao, M.; Kulkarni, C.; Goebel, K.; Fink, O. (2021). *Aircraft
> Engine Run-to-Failure Dataset under Real Flight Conditions for
> Prognostics and Diagnostics.* Data, 6(1), 5.
> DOI: [10.3390/data6010005](https://doi.org/10.3390/data6010005)
> (open access, CC BY 4.0). Also mirrored as a NASA Technical Report:
> [ntrs.nasa.gov](https://ntrs.nasa.gov/api/citations/20205001125/downloads/Run_to_Failure_Simulation_Under_Real_Flight_Conditions_Dataset.pdf).

A local copy of the paper and NASA's own example notebook live (if
downloaded) under `docs/reference/` — this folder is deliberately
**gitignored**: these are third-party works kept here only for personal
reference, not part of this repo. Anyone who needs them can download them
from the links above.

## HDF5 file structure

Each N-CMAPSS file contains (per the paper cited above), among others, the
following groups:

- `W` — operating conditions (scenario descriptors: altitude, Mach, TRA, T2)
- `X_s` — measurable sensors (physical channels)
- `X_v` — virtual/model-internal sensors
- `T` — unobservable health parameters (for analysis/debugging only, do not use as a feature)
- `A` — auxiliary: `unit`, `cycle`, `Fc` (flight class), `hs` (health state)
- `Y` — RUL label per row

All arrays are row-wise aligned across 1 Hz measurement points (same number
of rows in `W`, `X_s`, `X_v`, `A`, `Y`).

## Kaggle upload (for Track B)

After the local download, upload `N-CMAPSS_DS02.h5` as a private Kaggle
dataset (Kaggle → Datasets → New Dataset → Upload), so the deep-learning
notebook (see [kaggle_workflow.md](kaggle_workflow.md)) can pull it in
directly without re-downloading it every session.
