# Kaggle-Workflow für Track B (Deep Learning)

Kaggle-CLI ist lokal konfiguriert (`~/.kaggle/kaggle.json`) — alles unten
lässt sich per `kaggle`-Befehl von der Kommandozeile treiben, kein manuelles
Klicken im Kaggle-UI nötig.

## 1. Datasets hochladen (einmalig, danach nur bei Änderungen)

**DS02-Rohdaten** (2,45 GB, siehe [data_download.md](data_download.md) für
die Herkunft):

```bash
kaggle datasets init -p data/       # erzeugt data/dataset-metadata.json einmalig
# id/title darin anpassen, dann:
kaggle datasets create -p "<absoluter Windows-Pfad>\data" -t
```

**Wichtig (Windows):** `-p` mit einem *absoluten* Windows-Pfad
(`C:\...\data`) übergeben, nicht mit einem relativen Pfad mit
Forward-Slash (`data/`) — Letzteres löst einen Pfad-Mixing-Bug in der
Kaggle-CLI aus (`No such file or directory: '...\\.kaggle/uploads\\data/...json'`).

**Code-Utility-Dataset** (`src/turbofan_rul/*.py`, klein, sekundenschnell):

```bash
kaggle datasets init -p src/
kaggle datasets create -p "<absoluter Windows-Pfad>\src" -r zip
```

`-r zip` ist nötig, da `src/` nur ein Unterverzeichnis (`turbofan_rul/`)
enthält — im Default-Modus (`skip`) würde die Kaggle-CLI Verzeichnisse
ignorieren und nichts hochladen. **Kaggle flacht die Zip-Struktur beim
Extrahieren ab:** die `.py`-Dateien landen direkt unter
`/kaggle/input/turbofan-rul-src/`, nicht unter `.../turbofan_rul/`. Da
unser Code `from turbofan_rul.xxx import yyy` (absolute Package-Imports)
verwendet, muss das Kernel-Notebook das im ersten Setup-Cell selbst
geradebiegen (Dateien nach `/kaggle/working/pkg/turbofan_rul/` kopieren,
dann `sys.path.insert(0, "/kaggle/working/pkg")` — siehe
`kaggle_kernel/track_b_training.ipynb`).

## 2. Kernel (Notebook) erstellen und pushen

```bash
kaggle kernels init -p kaggle_kernel/
```

`kaggle_kernel/kernel-metadata.json` referenziert beide Datasets über
`dataset_sources`, setzt `enable_gpu: true`, `enable_internet: false`
(alle gebrauchten Pakete — numpy/pandas/torch/scipy/h5py — sind im
Kaggle-Standardimage vorinstalliert).

**Stolperstein:** ein per `nbformat` programmatisch gebautes Notebook hat
standardmäßig **keine `kernelspec`-Metadata**. Lokal führt `jupyter
nbconvert --execute` das trotzdem aus, Kaggles Papermill-basierter Executor
bricht aber mit `ValueError: No kernel name found in notebook` ab. Fix: vor
dem Push sicherstellen, dass `nb.metadata.kernelspec` gesetzt ist (`python3`,
`display_name: Python 3`).

```bash
kaggle kernels push -p kaggle_kernel/
```

Titel und `id`-Slug im `kernel-metadata.json` sollten zueinander passen
(sonst legt Kaggle einen abweichenden Slug an und man muss den beim Status-
Check erraten).

## 3. Lauf überwachen

```bash
kaggle kernels status says43/turbofan-rul-track-b
```

Bei Fehlern (`KernelWorkerStatus.ERROR`) Log ziehen:

```bash
kaggle kernels output says43/turbofan-rul-track-b -p <ziel-ordner>
```

Das Log liegt als JSON-Lines-Datei vor (`stream_name`/`data`-Paare) — mit
`grep -i "error\|traceback"` durchsuchen.

## 4. Ergebnisse zurückholen

Nach erfolgreichem Lauf liegen `track_b_summary.csv`,
`track_b_test_results.csv` und ein PNG mit den Degradationskurven unter
`/kaggle/working/` (im Notebook selbst geschrieben). Herunterladen mit
demselben `kaggle kernels output`-Befehl wie oben, dann die CSVs lokal
unter `outputs/` ablegen und in `docs/results.md` mit Track A vergleichen.
