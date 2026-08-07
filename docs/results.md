# Ergebnisse: Track A (NGBoost)

Siehe `notebooks/02_track_a_baseline.ipynb` für den vollständigen Lauf.

## Setup

- Features: 129 zyklus-aggregierte Statistiken (mean/std/min/max je Sensor-
  und Betriebsbedingungskanal + Zykluslänge) aus `aggregate_cycles`.
  **Hinweis:** ein früherer Lauf enthielt versehentlich `a_hs`
  (Health-State-Flag) als Feature — das ist Data Leakage, da `a_hs` aus dem
  unbeobachtbaren Degradationszustand abgeleitet ist und kurz vor Ausfall
  deterministisch kippt. Behoben in `features.feature_columns`; alle Zahlen
  unten sind der bereinigte, leakage-freie Lauf.
- Modell: NGBoost (Normal-Verteilung, 300 Estimatoren).
- CV: Leave-one-unit-out über die 6 Dev-Einheiten (alle Flight Class 3).
- Test: offizielles DS02-Testset (Units 11/14/15) — spannt alle drei Flight
  Classes auf (Fc3/Fc1/Fc2), obwohl das Training nur Fc3 gesehen hat.

## CV-Ergebnisse (Dev, Leave-one-unit-out)

| Metrik | Mittelwert über Folds |
|---|---|
| RMSE (Zyklen) | 12.60 |
| NASA-Score (Summe pro Fold) | 173.76 |
| Coverage @ 90 %-Intervall | 0.74 |

Starke Streuung zwischen Folds (RMSE 9.8–16.6, Coverage 0.45–0.88) — bei nur
6 Einheiten ist jede Leave-one-out-Schätzung stark vom Fehlermodus/Alter der
jeweils ausgelassenen Einheit geprägt.

## Test-Ergebnisse (offizielles Testset, über Flight Classes hinweg)

| Unit | Flight Class | RMSE | NASA-Score | Coverage @ 90 % |
|---|---|---|---|---|
| 11 | 3 (gesehen) | 8.72 | 75.37 | 0.97 |
| 14 | 1 (ungesehen) | 12.31 | 175.40 | 0.83 |
| 15 | 2 (ungesehen) | 10.21 | 112.36 | 0.91 |
| **Gesamt** | | **10.67** | **363.12** | **0.90** |

## Beobachtungen (naive NGBoost-Intervalle)

- RMSE auf dem Testset (10.67) liegt in der gleichen Größenordnung wie der
  CV-Mittelwert (12.60) — die Generalisierung auf ungesehene Flight Classes
  ist für den Punktschätzer plausibel, wenn auch nicht überragend.
- **Coverage liegt in der CV klar unter dem nominellen 90 %-Ziel** (0.45–0.88,
  Ø 0.74). Auf dem Testset liegt die naive Coverage bereits nahe am Ziel
  (0.90) — ohne die geleakte Feature war die alte Testset-Zahl (0.76)
  irreführend optimistisch in die falsche Richtung.

## Nachbesserung: nested cross-conformal Kalibrierung

`turbofan_rul.calibration` + `track_a.cross_conformal_loo` ersetzen NGBoosts
eigene Skalenschätzung durch einen Faktor `q_hat`, der aus out-of-fold-
Residuen bestimmt wird (nested Leave-one-unit-out, da nur 6 Einheiten
verfügbar sind). Ergebnis:

| | Coverage @ 90%-Ziel, naiv | Coverage @ 90%-Ziel, conformal |
|---|---|---|
| CV (Dev, Ø über Folds) | 0.74 | **0.95** |
| Offizielles Testset (gesamt) | 0.90 | **0.99** |

RMSE/NASA-Score ändern sich nicht (Kalibrierung betrifft nur die
Intervallbreite, nicht den Punktschätzer).

**Einordnung:** Coverage von 0.95 (CV) und 0.99 (Test) statt exakt 0.90 kommt
aus der finite-sample-Korrektur der Conformal-Quantile (rundet bei kleinen
Kalibrierungsmengen konservativ auf) plus dem Distribution Shift zwischen
Kalibrierung (nur Flight Class 3) und Testset (Fc1/Fc2/Fc3) — die
Conformal-Garantie setzt Austauschbarkeit voraus, die hier verletzt ist.
Dass die Intervalle trotzdem eher zu breit als zu eng sind, ist das sicherere
Versagen, aber **keine bewiesene 90%-Garantie unter Shift**.

**Hinweis zur Reproduzierbarkeit:** `fit_ngboost`/`cross_conformal_loo` fixen
aktuell keinen `random_state`. Wiederholte Läufe schwanken deshalb leicht
(gesehen: RMSE 10.67–10.71, NASA-Score 363–365 auf demselben Testset) — die
Größenordnung und alle qualitativen Aussagen oben sind über mehrere Läufe
stabil, einzelne Nachkommastellen nicht.

## Produktions-Fix: negative RUL-Werte geclippt

Reale Kalibrierungsintervalle können nahe dem Lebensende unterhalb von 0
rutschen — beobachtet z.B. bei Testset-Unit 15, Zyklus 67 (wahre RUL 0):
das Rohmodell lieferte eine untere Intervallgrenze von **−11,7 Zyklen**,
physikalisch unmöglich und für eine Instandhaltungsanzeige nicht
präsentierbar. `evaluate.clip_rul` klemmt Punktvorhersage und beide
Intervallgrenzen jetzt bei 0 fest; angewendet in `track_a.predict_with_interval`
und `track_a.conformal_predict` (Track B analog im Kaggle-Notebook). Ändert
RMSE/NASA-Score nicht (der Punktschätzer war nie negativ) und auch nicht
die Coverage (ein y≥0 lag ohnehin schon oberhalb jeder negativen Grenze) —
reiner Darstellungs-/Produktions-Fix, keine Ergebniskorrektur.

## Track B (Deep Ensemble, Kaggle T4)

Siehe `kaggle_kernel/track_b_training.ipynb` für den vollständigen Lauf,
`docs/kaggle_workflow.md` für den Infra-Ablauf (inkl. der einigen echten
Stolpersteine unterwegs — fehlende Kernelspec-Metadata, verschachtelter
`/kaggle/input`-Pfad, veraltete Dataset-Version, fehlende Feature-
Standardisierung → NaN-Loss, P100/T4-Kompatibilität).

### Setup

- Modell: 1D-CNN (2 Conv-Layer, hidden 32/64) + Gaussian-NLL-Kopf, 5er
  Deep Ensemble (Kombination via Mixture-of-Gaussians Moment-Matching,
  Lakshminarayanan et al. 2017).
- Daten: rohe 1Hz-Sensorik, subsampled auf 0,1 Hz, Sliding-Window (Länge 50,
  Stride 1) über die volle Unit-Historie hinweg (Zyklen aneinandergereiht,
  siehe `sequences.make_windows`). Features **standardisiert** (Mittelwert/
  Std aus dem Trainings-Split, siehe `sequences.standardize_features`).
- Split: Unit 2 komplett als Kalibrierungs-Einheit zurückgehalten (Split-
  Conformal, kein Nested-CV nötig bei dieser Datenmenge), Units 5/10/16/18/20
  fürs Training. Test: dasselbe offizielle DS02-Testset wie Track A
  (Units 11/14/15, Fc3/Fc1/Fc2).
- Training: 15 Epochen/Mitglied, Batch 256, Adam (lr 1e-3), Gradient-Clipping
  (max\_norm 5) — **~16 Minuten** für alle 5 Mitglieder auf einer Tesla T4
  (Kaggle `machine_shape: NvidiaTeslaT4` explizit erzwungen, siehe
  `docs/kaggle_workflow.md`). Loss fiel sauber von ~8–9 auf ~1.4.
- Punktvorhersage und Intervallgrenzen sind mit `evaluate.clip_rul` bei 0
  gekappt (siehe Abschnitt "Produktions-Fix" oben — dieselbe Begründung
  gilt hier: negative "verbleibende Zyklen" sind unmöglich).

### Ergebnisse (offizielles Testset)

| Unit | Flight Class | n Fenster | RMSE | NASA-Score (Summe) | Coverage naiv | Coverage conformal |
|---|---|---|---|---|---|---|
| 11 | 3 (gesehen) | 66.301 | 8.12 | 77.913 | 0.46 | 0.90 |
| 14 | 1 (ungesehen) | 15.629 | 5.48 | 6.975 | 0.88 | 1.00 |
| 15 | 2 (ungesehen) | 43.298 | 3.25 | 11.143 | 0.95 | 0.99 |
| **Gesamt** | | **125.228** | **6.50** | **96.031** | **0.68** | **0.94** |

`q_hat` (Split-Conformal, aus Kalibrierungs-Unit 2): 3.25.

**Achtung Skaleneffekt beim NASA-Score:** die Summe ist über 125.228 Fenster
gebildet, nicht über 202 Zyklen wie bei Track A — die rohe Summe ist daher
*nicht* direkt mit Track As 363 vergleichbar. Fair vergleichbar ist der
Mittelwert pro Vorhersage: 96.031 / 125.228 ≈ **0,77** (Track B) vs.
363 / 202 ≈ **1,80** (Track A).

**Hinweis zur Reproduzierbarkeit:** wie bei Track A ist auch das
CNN-Training nicht geseedet — ein vorheriger Lauf (vor dem Clipping-Fix,
sonst identischer Code) ergab RMSE 7.21 statt 6.50 auf demselben Testset.
Größenordnung und qualitative Aussagen sind stabil, einzelne Zahlen
schwanken von Lauf zu Lauf um grob ±10-15%.

### Vergleich Track A vs. Track B

| Metrik | Track A (NGBoost, Zyklus-Level) | Track B (Deep Ensemble, Fenster-Level) |
|---|---|---|
| RMSE (Test, gesamt) | 10.67 | 6.50 |
| NASA-Score Ø pro Vorhersage | 1.80 | 0.77 |
| Coverage @ 90 %, naiv | 0.90 | 0.68 |
| Coverage @ 90 %, conformal | 0.99 | 0.94 |
| Trainingszeit | Sekunden (CPU) | ~16 Min (T4-GPU) |

**Wichtiger Vergleichbarkeits-Vorbehalt:** Track A sagt einmal pro Zyklus
vorher, Track B einmal pro (überlappendem) Zeitfenster — bei Stride 1
überlappen benachbarte Fenster in 49 von 50 Zeitschritten, sind also stark
autokorreliert. Die 125.228 "Fenster" sind damit statistisch **keine**
125.228 unabhängigen Beobachtungen; RMSE/Coverage bleiben aussagekräftige
deskriptive Kennzahlen für das Modell auf diesen Daten, aber ein direkter
Signifikanztest zwischen den Tracks wäre auf dieser Basis nicht valide.

**Beobachtungen:**
- Track B erreicht einen niedrigeren RMSE und einen besseren
  Ø-NASA-Score als Track A — plausibel, da es auf viel mehr, feiner
  aufgelösten Trainingsdaten (Rohsignal statt Zyklus-Aggregate) lernt.
- Umgekehrtes Coverage-Bild: Track As naive Intervalle sind in der CV klar
  zu eng, im Testset zufällig nah am Ziel (0.90); Track Bs naive Intervalle
  sind durchgängig zu eng (0.68) — nach Konformal-Kalibrierung liegen beide
  nah am oder leicht über dem 90 %-Ziel.
- Innerhalb Track B ist die Coverage für Unit 11 (gesehene Flight Class 3)
  naiv am schlechtesten (0.46), während die ungesehenen Flight Classes
  1/2 naiv besser abschneiden (0.88/0.95) — kontraintuitiv auf den ersten
  Blick, aber die Kalibrierungs-Einheit (Unit 2) ist ebenfalls Fc3, sodass
  Stichprobenvarianz zwischen einzelnen Einheiten hier mehr Einfluss haben
  könnte als der Flight-Class-Unterschied selbst. Kein Widerspruch zur
  Kernaussage, aber ein Hinweis, dass "gesehene vs. ungesehene Flight
  Class" allein nicht die ganze Geschichte ist.

## Offen

- 129 Features bei ~446 Trainings-Zyklen bleibt ein ungünstiges Verhältnis
  für Bäume — Feature-Selektion oder eine einfachere lineare Baseline zum
  Vergleich sind naheliegende nächste Schritte für Track A.
- Track B nutzt bislang nur eine einzelne Kalibrierungs-Einheit für
  Split-Conformal; ein Vergleich mit einer Nested-Variante (wie bei Track A)
  wäre ein sauberer nächster Schritt, ist aber wegen der Trainingskosten
  pro Fold deutlich teurer.
- Kein direkter statistischer Signifikanztest zwischen den Tracks (siehe
  Autokorrelations-Vorbehalt oben) — für eine Bewerbungsunterlage reicht die
  deskriptive Gegenüberstellung, für eine wissenschaftliche Arbeit wäre das
  nachzuholen.
