# TurbofanRUL-NCMAPSS

Predictive-Maintenance-Portfolioprojekt: Restlebensdauer-Prognose (RUL) für
Turbofan-Triebwerke auf Basis des NASA N-CMAPSS-Datensatzes (DS02,
realistische Flugprofile statt synthetischer Zyklen wie im klassischen
C-MAPSS).

## Ziel

RUL-Prognose auf N-CMAPSS DS02 mit zwei parallel entwickelten und
gegeneinander evaluierten Modellierungsansätzen, inkl. kalibrierter
Unsicherheitsquantifizierung (UQ) statt reiner Punktschätzung.

## Methodik

1. **Daten:** N-CMAPSS DS02 (HDF5, 1 Hz-Sensordaten über volle Flüge,
   run-to-failure, mehrere Einheiten/Fehlermodi, RUL-Label bereits enthalten).
2. **Preprocessing:** Aggregation der 1 Hz-Rohdaten auf Zyklus-Level-Features
   (Flugphasen-Statistiken) + optional Sliding-Window-Sequenzen für die
   Deep-Learning-Spur. Split **nach Unit**, nicht nach Zeit (kein Leakage).
3. **Track A — klassisches ML + UQ:** Gradient Boosting mit Quantilverlust
   bzw. NGBoost für Prediction Intervals.
4. **Track B — Deep Learning + UQ:** CNN/LSTM-Baseline mit Deep Ensembles
   oder MC-Dropout für Unsicherheit. Training läuft auf Kaggle (T4-GPU),
   nicht lokal — siehe [docs/kaggle_workflow.md](docs/kaggle_workflow.md).
5. **Evaluation:** RMSE, NASA-PHM-Score (asymmetrische Straffunktion,
   verspätete Vorhersagen härter bestraft), Coverage/Kalibrierung der
   Prediction Intervals, Cross-Validation über Flight Classes/Betriebs-
   bedingungen.
6. **Vergleich & Report:** Gegenüberstellung beider Tracks unter
   [docs/results.md](docs/results.md) — Track A steht, Track B folgt.

## Setup

```powershell
uv sync
```

Für Track B zusätzlich (nur lokal nötig, falls nicht auf Kaggle trainiert wird):

```powershell
uv sync --extra deep
```

## Daten

DS02 wird nicht mitversioniert (`data/` ist gitignored). Download-Anleitung:
[docs/data_download.md](docs/data_download.md).

## Projektstruktur

```
src/turbofan_rul/   Python-Package (Preprocessing, Modelle, Evaluation)
notebooks/          EDA und Ergebnis-Notebooks
docs/                Download-Anleitung, Kaggle-Workflow, Ergebnisreport
data/                Rohdaten (gitignored)
outputs/             Trainierte Modelle, Metriken (gitignored)
tests/               pytest-Tests
```

## Lizenz & Datenquelle

Code und Dokumentation in diesem Repo stehen unter der [MIT-Lizenz](LICENSE).

Der verwendete Datensatz (NASA N-CMAPSS DS02) ist nicht Teil dieses Repos
(siehe [docs/data_download.md](docs/data_download.md) für Download und
Zitation) und stammt aus dem NASA Prognostics Center of Excellence Data Set
Repository:

> Arias Chao, M.; Kulkarni, C.; Goebel, K.; Fink, O. (2021). *Aircraft
> Engine Run-to-Failure Dataset under Real Flight Conditions for
> Prognostics and Diagnostics.* Data, 6(1), 5.
> DOI: [10.3390/data6010005](https://doi.org/10.3390/data6010005)
