# Step 9 - seed-ensemble control and a new industrial dataset (descriptive)

Written on 2026-11-22, after Step 8 and before any Step 9 job was run. Script step9_extension.py, SHA-256 309e5009bdc91cf13fa392821fc0240f47649d4469be6ae4dbf2c4ff516376be; it imports
the unchanged step8_benchmark.py (d47cb43793c3df61732bfe7241cac172c6f8f545deeff7f72963c2cdb81d0696), step5_confirm.py (f3492a1c5edd50c845b4509ed4108883f64e8390ac4748c4164f32d72a08658d) and step7_robustness.py (ea8893efbbf37f8cade41a0afdcf929de3dd1295d044d84d6adc9f8fa42e8979).
Step 9 has no confirmatory decision rule; every result is reported.

## Why
A reading of the NormBlend manuscript asked (1) how much of the gain a two-seed ensemble of one normalization would
give at the same cost, and (2) for a second real industrial dataset that played no part in designing the method.

## Design
* E6: SAITS, TimesNet, ImputeFormer, Transformer trained a second time with G and S on CMAPSS, CMAPSS_M, MetroPT, four conditions, seeds [0, 1, 2, 3, 4]; Step 8 splits,
  masks, early-stopping and calibration halves; only the model initialization differs (seed key "step9-init2").
  Compared: NormBlend (2 trainings), ens_global and ens_seq (2), ens_valsel (3), nb_2seed (4); calibrated
  per-variable weights (NormBlend formula) and equal weights; oracle choices marked as such. C-MAPSS-SC models also
  impute the C-MAPSS-MC windows. KNN (deterministic) and BRITS (training time) are not part of E6.
* E7: SKAB, GitHub waico/SKAB at commit b2c0d46c2971dcbfe71e26087b6d231998bb91c2; signals Accelerometer1RMS, Accelerometer2RMS, Current, Pressure, Temperature, Thermocouple, Voltage, Volume Flow RateRMS; units = 34 experiment runs plus the normal-mode
  record cut into blocks of 1200 records; windows of 60 consecutive records within a unit, dropped if a time step
  exceeds 5.0 s; gaps of nominal length 12. Six imputers with G and S, the five single-model alternatives,
  four conditions, seeds [0, 1, 2, 3, 4], Step 8 protocol, hyperparameters and kappa = 1. Nothing is tuned on SKAB.

Jobs: 820
