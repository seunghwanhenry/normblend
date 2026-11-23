# Step 9 report (descriptive)

Script versions in the results: ['309e5009bdc91cf13fa392821fc0240f47649d4469be6ae4dbf2c4ff516376be']

## E6 seed-ensemble control: geometric-mean ratio to the oracle, settings below the better ensemble

```
normblend      ratio to oracle 0.968  NB below it in 0 of 48
ens_global     ratio to oracle 0.979  NB below it in 20 of 48
ens_seq        ratio to oracle 1.567  NB below it in 46 of 48
ens_better     ratio to oracle 0.953  NB below it in 18 of 48
ens_valsel     ratio to oracle 0.959  NB below it in 19 of 48
ens_global_eq  ratio to oracle 1.002  NB below it in 25 of 48
ens_seq_eq     ratio to oracle 1.592  NB below it in 46 of 48
nb_2seed       ratio to oracle 0.951  NB below it in 8 of 48
```

```
variant                normblend  ens_better  ens_valsel  nb_2seed
dataset  model                                                    
CMAPSS   ImputeFormer      0.988       0.988       0.990     0.981
         SAITS             0.980       0.978       0.979     0.971
         TimesNet          0.994       0.998       1.003     0.995
         Transformer       0.995       0.993       0.993     0.993
CMAPSS_M ImputeFormer      0.996       0.965       0.965     0.968
         SAITS             0.883       0.918       0.918     0.830
         TimesNet          1.000       0.939       0.939     1.072
         Transformer       0.984       0.973       0.973     0.965
MetroPT  ImputeFormer      0.973       0.809       0.809     0.897
         SAITS             0.928       0.951       0.968     0.881
         TimesNet          0.925       0.974       1.015     0.934
         Transformer       0.972       0.968       0.968     0.945
```

## E6 cross-domain (C-MAPSS-SC -> C-MAPSS-MC), mean RMSE

```
variant       ens_global  ens_global_eq  ens_seq  ens_seq_eq  ens_valsel  global  nb_2seed  normblend    seq
model                                                                                                       
ImputeFormer       1.617          1.616    0.896       0.893       1.467   1.613     0.893      0.895  0.895
SAITS              1.740          1.738    1.012       1.010       1.675   1.737     1.010      1.019  1.019
TimesNet           1.716          1.716    1.084       1.084       1.122   1.716     1.084      1.086  1.086
Transformer        1.724          1.723    1.010       1.012       1.724   1.723     1.012      1.025  1.025
```

## E7 SKAB

Settings: 24; NormBlend below the better normalization in 24; geometric mean 0.980; worst 0.999; global better in 0

```
variant             global   gstat  nb_equal  nb_noguard  nb_pooled  normblend     own     seq  valsel
model                                                                                                 
BRITS               0.5990     NaN    0.5598      0.5476     0.5484     0.5457     NaN  0.5517  0.5619
ImputeFormer        0.5829     NaN    0.5624      0.5556     0.5576     0.5556     NaN  0.5611  0.5630
ImputeFormer+stats     NaN  0.5896       NaN         NaN        NaN        NaN     NaN     NaN     NaN
KNN                 0.6472     NaN    0.5991      0.5894     0.5948     0.5898     NaN  0.6158  0.6347
NSTransformer          NaN     NaN       NaN         NaN        NaN        NaN  0.5457     NaN     NaN
RevIN-SCINet           NaN     NaN       NaN         NaN        NaN        NaN  0.5657     NaN     NaN
SAITS               0.5890     NaN    0.5532      0.5399     0.5428     0.5399     NaN  0.5471  0.5464
SAITS+stats            NaN  0.6190       NaN         NaN        NaN        NaN     NaN     NaN     NaN
TimesNet            0.7005     NaN    0.5999      0.5590     0.5848     0.5591     NaN  0.5726  0.5913
TimesNet-NS            NaN     NaN       NaN         NaN        NaN        NaN  0.6009     NaN     NaN
Transformer         0.5937     NaN    0.5566      0.5428     0.5454     0.5428     NaN  0.5524  0.5683
```

Guard activation on SKAB: cond; GAP 0.4     0.0169; MCAR 0.3    0.0169; MCAR 0.7    0.0169; MNAR 0.3    0.0169

Runs without strict determinism: 0 of 820
