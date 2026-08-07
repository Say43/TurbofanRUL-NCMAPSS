# Results: Track A (NGBoost)

See `notebooks/02_track_a_baseline.ipynb` for the full run.

## Setup

- Features: 129 cycle-aggregated statistics (mean/std/min/max per sensor
  and operating-condition channel + cycle length) from `aggregate_cycles`.
  **Note:** an earlier run accidentally included `a_hs` (health-state flag)
  as a feature — that is data leakage, since `a_hs` is derived from the
  unobservable degradation state and flips deterministically shortly before
  failure. Fixed in `features.feature_columns`; all numbers below are from
  the corrected, leakage-free run.
- Model: NGBoost (Normal distribution, 300 estimators).
- CV: leave-one-unit-out over the 6 dev units (all flight class 3).
- Test: official DS02 test set (units 11/14/15) — spans all three flight
  classes (Fc3/Fc1/Fc2), even though training only saw Fc3.

## CV results (dev, leave-one-unit-out)

| Metric | Mean across folds |
|---|---|
| RMSE (cycles) | 12.60 |
| NASA score (sum per fold) | 173.76 |
| Coverage @ 90% interval | 0.74 |

Strong spread across folds (RMSE 9.8–16.6, coverage 0.45–0.88) — with only
6 units, each leave-one-out estimate is heavily shaped by the failure
mode/age of the specific held-out unit.

## Test results (official test set, across flight classes)

| Unit | Flight class | RMSE | NASA score | Coverage @ 90% |
|---|---|---|---|---|
| 11 | 3 (seen) | 8.72 | 75.37 | 0.97 |
| 14 | 1 (unseen) | 12.31 | 175.40 | 0.83 |
| 15 | 2 (unseen) | 10.21 | 112.36 | 0.91 |
| **Overall** | | **10.67** | **363.12** | **0.90** |

## Observations (naive NGBoost intervals)

- RMSE on the test set (10.67) is in the same range as the CV mean
  (12.60) — generalization to unseen flight classes is plausible for the
  point estimator, if not outstanding.
- **Coverage in CV is clearly below the nominal 90% target** (0.45–0.88,
  avg 0.74). On the test set, naive coverage is already close to the target
  (0.90) — without the leaked feature, the old test-set number (0.76) was
  misleadingly optimistic in the wrong direction.

## Follow-up fix: nested cross-conformal calibration

`turbofan_rul.calibration` + `track_a.cross_conformal_loo` replace
NGBoost's own scale estimate with a factor `q_hat` derived from out-of-fold
residuals (nested leave-one-unit-out, since only 6 units are available).
Result:

| | Coverage @ 90% target, naive | Coverage @ 90% target, conformal |
|---|---|---|
| CV (dev, avg across folds) | 0.74 | **0.95** |
| Official test set (overall) | 0.90 | **0.99** |

RMSE/NASA score don't change (calibration only affects interval width, not
the point estimate).

**Interpretation:** coverage of 0.95 (CV) and 0.99 (test) instead of
exactly 0.90 comes from the finite-sample correction of the conformal
quantile (rounds up conservatively for small calibration sets) plus the
distribution shift between calibration (flight class 3 only) and the test
set (Fc1/Fc2/Fc3) — the conformal guarantee assumes exchangeability, which
is violated here. The fact that the intervals end up too wide rather than
too narrow is the safer failure mode, but it is **not a proven 90%
guarantee under shift**.

**Reproducibility note:** `fit_ngboost`/`cross_conformal_loo` currently
don't fix a `random_state`. Repeated runs therefore fluctuate slightly
(observed: RMSE 10.67–10.71, NASA score 363–365 on the same test set) — the
order of magnitude and all qualitative statements above are stable across
runs, individual decimal places are not.

## Production fix: negative RUL values clipped

Real calibration intervals can dip below 0 near end-of-life — observed
e.g. for test-set unit 15, cycle 67 (true RUL 0): the raw model produced a
lower interval bound of **−11.7 cycles**, physically impossible and not
presentable for a maintenance display. `evaluate.clip_rul` now clamps the
point prediction and both interval bounds at 0; applied in
`track_a.predict_with_interval` and `track_a.conformal_predict` (Track B
analogously in the Kaggle notebook). Does not change RMSE/NASA score (the
point estimate was never negative) nor coverage (a y≥0 was already above
any negative bound) — a pure presentation/production fix, not a result
correction.

## Track B (deep ensemble, Kaggle T4)

See `kaggle_kernel/track_b_training.ipynb` for the full run,
`docs/kaggle_workflow.md` for the infrastructure workflow (including the
various real gotchas along the way — missing kernelspec metadata, nested
`/kaggle/input` path, stale dataset version, missing feature
standardization → NaN loss, P100/T4 compatibility).

### Setup

- Model: 1D CNN (2 conv layers, hidden 32/64) + Gaussian NLL head, 5-member
  deep ensemble (combined via mixture-of-Gaussians moment matching,
  Lakshminarayanan et al. 2017).
- Data: raw 1 Hz sensor readings, subsampled to 0.1 Hz, sliding window
  (length 50, stride 1) over the full unit history (cycles concatenated,
  see `sequences.make_windows`). Features **standardized** (mean/std from
  the training split, see `sequences.standardize_features`).
- Split: unit 2 held out entirely as the calibration unit (split conformal,
  no nested CV needed at this data volume), units 5/10/16/18/20 for
  training. Test: the same official DS02 test set as Track A (units
  11/14/15, Fc3/Fc1/Fc2).
- Training: 15 epochs/member, batch 256, Adam (lr 1e-3), gradient clipping
  (max_norm 5) — **~16 minutes** for all 5 members on a Tesla T4 (Kaggle
  `machine_shape: NvidiaTeslaT4` explicitly forced, see
  `docs/kaggle_workflow.md`). Loss dropped cleanly from ~8–9 to ~1.4.
- Point prediction and interval bounds are clamped at 0 via
  `evaluate.clip_rul` (see "Production fix" section above — the same
  reasoning applies here: negative "cycles remaining" is impossible).

### Results (official test set)

| Unit | Flight class | n windows | RMSE | NASA score (sum) | Coverage naive | Coverage conformal |
|---|---|---|---|---|---|---|
| 11 | 3 (seen) | 66,301 | 8.12 | 77,913 | 0.46 | 0.90 |
| 14 | 1 (unseen) | 15,629 | 5.48 | 6,975 | 0.88 | 1.00 |
| 15 | 2 (unseen) | 43,298 | 3.25 | 11,143 | 0.95 | 0.99 |
| **Overall** | | **125,228** | **6.50** | **96,031** | **0.68** | **0.94** |

`q_hat` (split conformal, from calibration unit 2): 3.25.

**Note on the scale effect in the NASA score:** the sum is computed over
125,228 windows, not over 202 cycles as in Track A — the raw sum is
therefore *not* directly comparable to Track A's 363. The fair comparison
is the mean per prediction: 96,031 / 125,228 ≈ **0.77** (Track B) vs.
363 / 202 ≈ **1.80** (Track A).

**Reproducibility note:** as with Track A, the CNN training is not seeded
either — an earlier run (before the clipping fix, otherwise identical code)
produced RMSE 7.21 instead of 6.50 on the same test set. The order of
magnitude and qualitative statements are stable, individual numbers
fluctuate run to run by roughly ±10–15%.

### Comparison: Track A vs. Track B

| Metric | Track A (NGBoost, cycle level) | Track B (deep ensemble, window level) |
|---|---|---|
| RMSE (test, overall) | 10.67 | 6.50 |
| NASA score, avg per prediction | 1.80 | 0.77 |
| Coverage @ 90%, naive | 0.90 | 0.68 |
| Coverage @ 90%, conformal | 0.99 | 0.94 |
| Training time | seconds (CPU) | ~16 min (T4 GPU) |

**Important comparability caveat:** Track A predicts once per cycle, Track
B once per (overlapping) time window — at stride 1, neighboring windows
overlap in 49 of 50 time steps, so they are strongly autocorrelated. The
125,228 "windows" are therefore statistically **not** 125,228 independent
observations; RMSE/coverage remain meaningful descriptive metrics for the
model on this data, but a direct significance test between the tracks
would not be valid on this basis.

**Observations:**
- Track B achieves a lower RMSE and a better average NASA score than
  Track A — plausible, since it learns from much more, finer-grained
  training data (raw signal instead of cycle aggregates).
- Reversed coverage picture: Track A's naive intervals are clearly too
  narrow in CV, coincidentally close to the target (0.90) on the test set;
  Track B's naive intervals are consistently too narrow (0.68) — after
  conformal calibration, both end up near or slightly above the 90% target.
- Within Track B, coverage for unit 11 (seen flight class 3) is naively
  the worst (0.46), while the unseen flight classes 1/2 perform naively
  better (0.88/0.95) — counterintuitive at first glance, but the
  calibration unit (unit 2) is also Fc3, so sample variance between
  individual units may matter more here than the flight-class difference
  itself. Not a contradiction of the core finding, but a hint that "seen
  vs. unseen flight class" alone isn't the whole story.

## Open items

- 129 features with ~446 training cycles remains an unfavorable ratio for
  trees — feature selection or a simpler linear baseline for comparison are
  obvious next steps for Track A.
- Track B currently uses only a single calibration unit for split
  conformal; comparing against a nested variant (as in Track A) would be a
  clean next step, but is considerably more expensive due to per-fold
  training cost.
- No direct statistical significance test between the tracks (see the
  autocorrelation caveat above) — for a job-application portfolio, the
  descriptive comparison is sufficient; for an academic paper, this would
  need to be addressed.
