# Step 8 - benchmark extension of NormBlend (descriptive)

Written on 19 November 2026, after Steps 1-7 and the first full draft of the paper, before any Step 8 job was run.
Script step8_benchmark.py, SHA-256 d47cb43793c3df61732bfe7241cac172c6f8f545deeff7f72963c2cdb81d0696; it imports the unchanged step5_confirm.py (f3492a1c5edd50c845b4509ed4108883f64e8390ac4748c4164f32d72a08658d) and
step7_robustness.py (ea8893efbbf37f8cade41a0afdcf929de3dd1295d044d84d6adc9f8fa42e8979). Step 8 has no confirmatory decision rule; every result is reported.

## Why
A pre-submission reading of the draft asked for (1) more imputers, (2) data with changing operating conditions,
(3) a test under a real distribution shift, (4) comparisons with methods that handle normalization inside a single
model, and (5) results over the missing rate.

## Design
* Protocol: Step 7 clean protocol (observed-only training statistics; calibration half of the validation units for
  the NormBlend weights; early-stopping half for checkpoints). NormBlend as frozen in Step 4 (per-variable weights,
  window guard with kappa = 1). Seeds [0, 1, 2, 3, 4]; seeds [0, 1, 2] are run first.
* Datasets: C-MAPSS FD001 + FD003 and MetroPT-3 (Steps 5 and 7) and C-MAPSS FD002 + FD004 (new; same 14 sensors,
  30-cycle windows, units = engines, 30% test, 10% validation).
* Imputers with global (G) and per-sequence (S) normalization: KNN, SAITS, the transformer, BRITS, TimesNet,
  ImputeFormer. Hyperparameters of the added models: {"BRITS": {"rnn_hidden_size": 64}, "TimesNet": {"n_layers": 2, "top_k": 3, "d_model": 64, "d_ffn": 64, "n_kernels": 6, "dropout": 0.1}, "ImputeFormer": {"n_layers": 3, "d_input_embed": 32, "d_learnable_embed": 96, "d_proj": 8, "d_ffn": 256, "n_temporal_heads": 4, "dropout": 0.1}, "NSTransformer": {"n_layers": 2, "d_model": 128, "n_heads": 4, "d_ffn": 256, "d_projector_hidden": [128, 128], "n_projector_hidden_layers": 2, "dropout": 0.1}, "RevIN-SCINet": {"n_stacks": 1, "n_levels": 1, "n_groups": 1, "n_decoder_layers": 1, "d_hidden": 1, "dropout": 0.1}, "TimesNet-NS": {"n_layers": 2, "top_k": 3, "d_model": 64, "d_ffn": 64, "n_kernels": 6, "dropout": 0.1, "apply_nonstationary_norm": true}}. Training budget of the PyPOTS models: {"epochs": 300, "patience": 20, "batch_size": 32, "lr": 0.001, "weight_decay": 0.0}.
* Single-model alternatives (one training): SAITS and ImputeFormer with the visible mean and log standard deviation
  of each variable as extra input channels (G+stats); Non-stationary Transformer; RevIN-SCINet; TimesNet with its
  non-stationary normalization (TimesNet-NS).
* Cross-domain: the C-MAPSS FD001 + FD003 models impute the FD002 + FD004 test windows of the same seed with the
  masks of the FD002 + FD004 jobs; errors on the FD002 + FD004 reporting scale.
* Missing-rate sweep: MCAR 10%, 50%, 90% for KNN, SAITS and ImputeFormer on C-MAPSS and MetroPT.
* Reused: the MetroPT results of linear regression, KNN, SAITS and the transformer under the Step 5 conditions are
  read from step7_outputs. C-MAPSS is rerun (needed for the cross-domain test) and must reproduce Step 7.
* Reported: RMSE and MAE at the removed positions; G, S, NormBlend, NormBlend without guard, pooled and equal
  weights, validation-selected normalization, oracle; weights; guard activation; run time and parameter counts.

Jobs: 1150

Written: 2026-11-19 18:01:13
