Figures for the JCDE manuscript (NormBlend).

Data
  figdata2.json   per-setting results of the benchmark, written by make_figdata.py (repository root) from
                  step8_runs.csv, step8_cross.csv, step8_weights.csv, step8_variables.csv (C-MAPSS-SC, C-MAPSS-MC,
                  MetroPT-3) and step9_runs.csv, step9_weights.csv, step9_variables_skab.csv (SKAB)

Scripts (run from this folder: python <script>.py; output goes to figures/ unless FIG_OUT is set)
  figT.py  -> fig_1_A, fig_1_B   Figure 1: no normalization dominates; NormBlend vs the better normalization
  figM.py  -> fig_2              Figure 2: the method (dual-view training, blending, range guard)
  figP.py  -> fig_3_A, fig_3_B   Figure 3: Proposition 1 on the benchmark data
  figW.py  -> fig_4              Figure 4: learned weights a_d per imputer and variable (four datasets)
  figS.py  -> fig_5              Figure 5: RMSE against the missing rate (MCAR 10-90%)
  ga2.py   -> Graphical_abstract
  jstyle.py  shared style (fonts, colours, save helpers)

Needs matplotlib, numpy, pandas and the Liberation Sans font (metric-compatible with Arial).
Submission files: fig_N.pdf/.png/.tif (600 dpi); multi-panel figures as fig_N_A, fig_N_B with no panel
labels inside (JCDE checklist 6.7). preview/fig_N.png combines the panels with (A)/(B) below each, for the
draft only.
