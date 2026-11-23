# NormBlend

Code and results for

> Song, S., & Jung, H. NormBlend: error-optimal blending of global and per-sequence normalization for deep
> imputation of industrial time series. Manuscript submitted to the *Journal of Computational Design and Engineering*.

NormBlend trains a deep imputer twice, once on globally standardized windows and once on windows normalized by the
statistics of their visible values, and blends the two imputations per variable with weights that minimize the
squared error on held-out calibration units (closed form). A range guard sends windows outside the training range
to the per-sequence imputation.

## Reproducing the numbers, tables and figures from the stored results

The result tables of the benchmark are in `step8_outputs/` and `step9_outputs/` (one row per run). On a CPU, in
about a minute:

```
python paper_tables.py          # paper_outputs/summary.json (numbers in the text), paper_outputs/tables.md (Tables 2-5)
python make_figdata.py          # figures/src/figdata2.json
cd figures/src
python figT.py; python figM.py; python figP.py; python figW.py; python figS.py; python ga2.py
```

| Paper | Source |
| --- | --- |
| Tables 2, 3 and 5, Figures 1 and 5, Supplementary Tables S1-S4 | `step8_outputs/step8_runs.csv`, `step8_cross.csv`; SKAB: `step9_outputs/step9_runs.csv` (`exp` = E7) |
| Table 4, Supplementary Table S6 (seed-ensemble control) | `step9_outputs/step9_runs.csv` (`exp` = E6), `step9_cross.csv` |
| Figure 3 (Proposition 1) | `step8_outputs/step8_variables.csv`, `step9_outputs/step9_variables_skab.csv` |
| Figure 4 (weights) | `step8_outputs/step8_weights.csv`, `step9_outputs/step9_weights.csv` |
| Range-guard activation | `step8_outputs/step8_guard.csv`, `step9_outputs/step9_guard.csv` |
| Run times and model sizes (Supplementary Table S5) | `step8_outputs/step8_jobs.csv`, `step9_outputs/step9_jobs.csv` |

## Rerunning the benchmark

Requirements: Python 3.12, PyTorch 2.11 with CUDA 12.8, PyPOTS 1.5, NumPy, pandas, SciPy, scikit-learn
(`requirements.txt`). The runs in the paper used one NVIDIA GeForce RTX 5090 shared by six worker processes and
deterministic algorithms; the full benchmark takes several days on such a machine.

```
python step5_confirm.py fetch       # downloads C-MAPSS (NASA) and MetroPT-3 (UCI) into data/
python step5_confirm.py prepare     # C-MAPSS-SC and MetroPT-3 windows
python step7_robustness.py plan     # Step 8 reuses the Step 7 runs of KNN, SAITS and the transformer on MetroPT-3
python step7_robustness.py run
python step8_benchmark.py prepare   # C-MAPSS-MC windows
python step8_benchmark.py check     # KNN on C-MAPSS must reproduce Step 7 exactly
python step8_benchmark.py run       # parallel: --worker k --n-workers K, one process per worker
python step8_benchmark.py report
python step9_extension.py fetch     # SKAB at a pinned commit
python step9_extension.py prepare
python step9_extension.py run
python step9_extension.py report
python step9_variables.py           # variable-level errors of the SKAB runs (step9_outputs/step9_variables_skab.csv)
```

`python step8_benchmark.py status` (likewise for Steps 7 and 9) shows the progress. The per-run prediction files (`results/`) and the window caches are
not stored in this repository; the scripts recreate them.

## Files

| File | Content |
| --- | --- |
| `step8_benchmark.py` | Benchmark: six imputers, C-MAPSS-SC, C-MAPSS-MC, MetroPT-3, four missingness conditions, shift from C-MAPSS-SC to C-MAPSS-MC, single-model alternatives, missing-rate sweep |
| `step9_extension.py` | SKAB benchmark (E7) and seed-ensemble control (E6) |
| `step9_variables.py` | Variable-level calibration and test errors of the SKAB runs |
| `step5_confirm.py`, `step7_robustness.py` | Data download and windowing, splits, masks, NormBlend weights and the clean protocol, imported by Steps 8 and 9 |
| `dtin_core.py`, `dtin_v2.py`, `dtin_v2_eval.py`, `dtin_improve.py` | Shared utilities and model code; the mask-aware transformer of the paper is `DTINv` in `dtin_v2.py` |
| `dtin_outputs/config.json`, `dtin_outputs/final_v2/selected_saits.json` | Configuration read by the scripts; the SAITS configuration (two layers, width 256) was fixed before the benchmark |
| `stepN_outputs/PLAN_stepN.md`, `REPORT_stepN.md` | Plan written before the first run of the step (with the SHA-256 of every script involved) and the generated report |
| `paper_tables.py`, `make_figdata.py`, `figures/src/` | Numbers, tables and figures of the paper from the result tables |

The scripts keep the names under which they were run; their step numbers come from the project in which they were
developed. Steps 5 and 7 are included because Steps 8 and 9 import their code; their own analyses are not part of
the paper. The job tables (`step8_jobs.csv`, `step9_jobs.csv`) record the SHA-256 of the script that ran each job,
and the plans record the hashes of the imported scripts, so the scripts must not be edited.

## Data

The datasets are not redistributed; the scripts download them.

- C-MAPSS: NASA Prognostics Center of Excellence data set repository (Saxena & Goebel, 2008).
- MetroPT-3: UCI Machine Learning Repository, https://doi.org/10.24432/C5VW3R (Davari et al., 2021; CC BY 4.0).
- SKAB: https://github.com/waico/SKAB, commit b2c0d46c2971dcbfe71e26087b6d231998bb91c2 (Katser & Kozitsin, 2020; GPL-3.0).

## License

Code: MIT (see `LICENSE`). The datasets are under their own licenses.
