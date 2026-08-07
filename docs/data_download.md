# Datenbeschaffung: N-CMAPSS DS02

Der N-CMAPSS-Datensatz wird als ZIP über den NASA-PCoE-S3-Bucket vertrieben
und enthält alle Subsets (DS01–DS08) als einzelne HDF5-Dateien. **Es gibt
keinen offiziellen Einzeldownload nur für DS02** — bestätigt über die
[mohyunho/N-CMAPSS_DL](https://github.com/mohyunho/N-CMAPSS_DL)
Referenzimplementierung, die auf dieselbe NASA-Quelle verweist. Das
Gesamtpaket ist **15,76 GB** groß (per HTTP HEAD geprüft, Stand 2026-08-05).

Inoffizieller Fallback (nur falls der NASA-Link mal nicht erreichbar ist,
Herkunft/Aktualität nicht garantiert): Google-Drive-Mirror, verlinkt im
README des genannten GitHub-Repos.

## Download

```powershell
Invoke-WebRequest `
  -Uri "https://phm-datasets.s3.amazonaws.com/NASA/17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2.zip" `
  -OutFile "data/ncmapss_full.zip"
```

Danach **nicht** das komplette Archiv entpacken (die anderen 7 Subsets
kosten unnötig Speicherplatz) — stattdessen nur die DS02-Datei gezielt
herausziehen. Im Repo-Root mit `unzip` (in Git Bash verfügbar):

```bash
unzip -l data/ncmapss_full.zip | grep -i DS02   # Pfad im Archiv prüfen
unzip -j data/ncmapss_full.zip "*N-CMAPSS_DS02*.h5" -d data/
mv data/N-CMAPSS_DS02*.h5 data/N-CMAPSS_DS02.h5
rm data/ncmapss_full.zip
```

`-j` verwirft den Ordnerpfad aus dem Archiv (flaches Extrahieren), sodass
die Datei direkt unter `data/` landet.

## Alternative Quellen

- Offizielles PCoE-Repository (Übersicht/Metadaten):
  https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
- Mirror: https://data.phmsociety.org/nasa/

## Quelle & Zitation

Der Datensatz stammt aus dem NASA Prognostics Center of Excellence (PCoE)
Data Set Repository. Nutzung ist frei (NASA-Daten als U.S. Government Work,
faktisch public domain); die einzige Bitte des Repositories ist eine
Zitierung des Datensatzes bzw. der Autoren in Publikationen, die damit
arbeiten:

> Arias Chao, M.; Kulkarni, C.; Goebel, K.; Fink, O. (2021). *Aircraft
> Engine Run-to-Failure Dataset under Real Flight Conditions for
> Prognostics and Diagnostics.* Data, 6(1), 5.
> DOI: [10.3390/data6010005](https://doi.org/10.3390/data6010005)
> (Open Access, CC BY 4.0). Auch als NASA Technical Report gespiegelt:
> [ntrs.nasa.gov](https://ntrs.nasa.gov/api/citations/20205001125/downloads/Run_to_Failure_Simulation_Under_Real_Flight_Conditions_Dataset.pdf).

Eine lokale Kopie des Papers sowie NASAs eigenes Beispiel-Notebook liegen
(falls heruntergeladen) unter `docs/reference/` — dieser Ordner ist bewusst
**gitignored**: es sind Fremdwerke, die hier nur zum eigenen Nachschlagen
liegen, nicht Teil dieses Repos. Wer sie braucht, lädt sie über die Links
oben selbst herunter.

## Struktur der HDF5-Datei

Jede N-CMAPSS-Datei enthält (laut oben zitiertem Paper) u.a. folgende
Gruppen:

- `W` — Betriebsbedingungen (Scenario-Descriptors: Altitude, Mach, TRA, T2)
- `X_s` — messbare Sensoren (physikalische Kanäle)
- `X_v` — virtuelle/modellinterne Sensoren
- `T` — unbeobachtbare Gesundheitsparameter (nur für Analyse/Debugging, nicht als Feature verwenden)
- `A` — Auxiliary: `unit`, `cycle`, `Fc` (Flight Class), `hs` (Health State)
- `Y` — RUL-Label pro Zeile

Alle Arrays sind zeilenweise über 1 Hz-Messpunkte hinweg parallel indiziert
(gleiche Zeilenzahl in `W`, `X_s`, `X_v`, `A`, `Y`).

## Kaggle-Upload (für Track B)

Nach dem lokalen Download `N-CMAPSS_DS02.h5` als privates Kaggle-Dataset
hochladen (Kaggle → Datasets → New Dataset → Upload), damit das
Deep-Learning-Notebook (siehe [kaggle_workflow.md](kaggle_workflow.md)) es
direkt einbinden kann, ohne bei jeder Session neu herunterzuladen.
