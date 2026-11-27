# Step 10 - follow-up analyses of the NormBlend benchmark (descriptive)

Written on 26 November 2026, after Step 9 and before any Step 10 analysis or job was run. Script step10_followup.py, SHA-256
055ec3e9ea7c3841c1301b653d4871c8a6fd0eac517fe805acc4a759d51d4853; it imports the unchanged step9_extension.py (309e5009bdc91cf13fa392821fc0240f47649d4469be6ae4dbf2c4ff516376be), step8_benchmark.py (d47cb43793c3df61732bfe7241cac172c6f8f545deeff7f72963c2cdb81d0696), step5_confirm.py (f3492a1c5edd50c845b4509ed4108883f64e8390ac4748c4164f32d72a08658d)
and step7_robustness.py (ea8893efbbf37f8cade41a0afdcf929de3dd1295d044d84d6adc9f8fa42e8979). Step 10 has no confirmatory decision rule; every result is reported.

## Why
A reading of the manuscript (26 November 2026) found that (1) the paper describes errors on the scale of Equation 2,
whereas Steps 7-9 report them on a reporting scale; (2) the comparison with validation selection does not separate
per-variable choice from blending; (3) under the C-MAPSS shift the guard flags every window, so the shift test shows
the fallback and says nothing about moderate shifts; (4) the Wilcoxon tests over 96 settings treat settings that
share data as independent; and (5) the paper has no engineering case.

## Part A (stored predictions, no training)
* A1: every error of Steps 8 and 9 recomputed on the Equation 2 scale of its setting (target-domain scales for the
  shift test), next to the reporting scale; the recomputed reporting-scale RMSE must equal the stored tables.
* A2: per-variable selection (weight 0 or 1 from the calibration MSE of each variable; pooled choice for variables
  without calibration positions) with and without the guard; global standardization with the guard as fallback.
* A3: sums of squared and absolute errors per test unit for every method, for a cluster bootstrap over test units
  within each seed (the same draw for all settings of a dataset).
* A4: MetroPT-3, four main conditions: errors in physical units (signed, squared, absolute) per test day, failure
  flag (window overlaps one of the four documented air-leak failures) and variable for global, seq, normblend, nb_noguard, valsel, sel_var, global_guard; full windows (truth,
  mask, imputations) for the failure windows and up to two windows on either side for SAITS, Transformer, ImputeFormer.

## Part B (new runs)
* KNN, SAITS, Transformer with G and S on CMAPSS, MetroPT, the four main conditions, seeds [0, 1, 2, 3, 4]; Step 8 protocol, seeds and settings;
  both views trained in one job. The test imputations without shift must equal the stored Step 7/8 imputations.
* Shifted test windows: offset of k standard deviations (Equation 2 scale) in one variable per window (variable
  (window index + seed) mod D, as in Step 7) or in every variable; k in [0.5, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50, 100]; truth shifted alike.
* Variants: G, S, NormBlend without guard, NormBlend with the window guard at kappa in [0.1, 0.25, 0.5, 1.0, 2.0, 4.0] (kappa = 1 is the
  method), per-variable guard (weight 0 only for the flagged variables) at the same margins, per-variable
  selection with the guard, global standardization with the guard as fallback. RMSE over all removed positions, over
  those of the shifted variables and over the others; guard shares.
