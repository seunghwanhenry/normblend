# Step 8 report (benchmark extension, descriptive)

Script versions in the results: ['d47cb43793c3df61732bfe7241cac172c6f8f545deeff7f72963c2cdb81d0696']

## Mean RMSE over conditions and seeds

```
variant                global  normblend     seq  valsel  oracle  NB/oracle
dataset  model                                                             
CMAPSS   BRITS         0.4858     0.4742  0.4870  0.4821  0.4858     0.9761
         ImputeFormer  0.4818     0.4750  0.4857  0.4795  0.4818     0.9860
         KNN           0.5400     0.5008  0.5255  0.5271  0.5255     0.9529
         SAITS         0.4703     0.4608  0.4822  0.4702  0.4703     0.9798
         TimesNet      0.6287     0.5012  0.5042  0.5068  0.5042     0.9941
         Transformer   0.4546     0.4523  0.4791  0.4546  0.4546     0.9951
CMAPSS_M BRITS         0.4334     0.4316  0.5169  0.4334  0.4334     0.9957
         ImputeFormer  0.0566     0.0563  0.1953  0.0566  0.0566     0.9960
         KNN           0.8317     0.8115  0.8642  0.8325  0.8317     0.9757
         SAITS         0.0989     0.0905  0.1825  0.0989  0.0989     0.9147
         TimesNet      0.2563     0.2563  0.6072  0.2563  0.2563     1.0000
         Transformer   0.0522     0.0516  0.1930  0.0522  0.0522     0.9879
MetroPT  BRITS         0.2205     0.2087  0.2866  0.2205  0.2205     0.9466
         ImputeFormer  0.2719     0.2632  0.4243  0.2719  0.2719     0.9679
         KNN           0.5007     0.4648  0.5151  0.4984  0.5007     0.9284
         SAITS         0.1744     0.1618  0.2255  0.1760  0.1744     0.9275
         TimesNet      0.6399     0.5202  0.5720  0.5724  0.5720     0.9094
         Transformer   0.1593     0.1545  0.2866  0.1593  0.1593     0.9699
```

Cells: 72; NormBlend <= 1.02 x oracle in 72; below the oracle in 67; geometric mean ratio 0.966; worst 1.013

## Single-model alternatives (mean RMSE)

```
dataset             CMAPSS  CMAPSS_M  MetroPT
model                                        
ImputeFormer+stats  0.4964    0.0654   0.3017
Linear regression   0.5034    1.1384   0.8639
NSTransformer       0.4808    0.1906   0.2972
RevIN-SCINet        0.4884    0.5482   0.3899
SAITS+stats         0.4863    0.0855   0.2342
TimesNet-NS         0.6513    0.6794   0.6739
```

## Cross-domain C-MAPSS FD001+FD003 -> FD002+FD004 (mean RMSE, FD002+FD004 scale)

```
variant             global   gstat  nb_equal  nb_noguard  nb_pooled  normblend     own     seq  valsel
model                                                                                                 
BRITS               3.9626     NaN    2.0967      2.4119     1.0472     1.0472     NaN  1.0472  2.4733
ImputeFormer        1.6130     NaN    1.0960      1.2611     0.8952     0.8952     NaN  0.8952  1.4615
ImputeFormer+stats     NaN  1.6467       NaN         NaN        NaN        NaN     NaN     NaN     NaN
KNN                 1.7443     NaN    1.2315      1.2313     1.1697     1.1697     NaN  1.1697  1.3810
NSTransformer          NaN     NaN       NaN         NaN        NaN        NaN  1.1490     NaN     NaN
RevIN-SCINet           NaN     NaN       NaN         NaN        NaN        NaN  1.0859     NaN     NaN
SAITS               1.7371     NaN    1.1808      1.3138     1.0194     1.0194     NaN  1.0194  1.6713
SAITS+stats            NaN  1.7218       NaN         NaN        NaN        NaN     NaN     NaN     NaN
TimesNet            1.7163     NaN    1.2073      1.1480     1.0862     1.0862     NaN  1.0862  1.1242
TimesNet-NS            NaN     NaN       NaN         NaN        NaN        NaN  1.3577     NaN     NaN
Transformer         1.7232     NaN    1.1774      1.4825     1.0249     1.0249     NaN  1.0249  1.7232
```

## Missing-rate sweep (mean RMSE)

```
cond                            MCAR 0.1  MCAR 0.3  MCAR 0.5  MCAR 0.7  MCAR 0.9
dataset model        variant                                                    
CMAPSS  ImputeFormer global       0.4530    0.4695    0.4769    0.4911    0.5276
                     normblend    0.4495    0.4649    0.4712    0.4830    0.5216
                     seq          0.4583    0.4717    0.4820    0.5006    0.6131
        KNN          global       0.5028    0.5185    0.5274    0.5592    0.8217
                     normblend    0.4796    0.4896    0.4915    0.5028    0.6050
                     seq          0.5080    0.5158    0.5185    0.5319    0.6503
        SAITS        global       0.4429    0.4497    0.4549    0.4743    0.5214
                     normblend    0.4376    0.4454    0.4507    0.4665    0.5114
                     seq          0.4501    0.4669    0.4769    0.4978    0.6273
MetroPT ImputeFormer global       0.1640    0.1568    0.1913    0.4047    0.7151
                     normblend    0.1621    0.1543    0.1870    0.3811    0.7081
                     seq          0.2073    0.3021    0.4209    0.5532    0.8604
        KNN          global       0.4023    0.4201    0.4567    0.5394    0.9491
                     normblend    0.3619    0.3807    0.4207    0.5045    0.8241
                     seq          0.3807    0.4114    0.4642    0.5611    0.8695
        SAITS        global       0.1172    0.1223    0.1529    0.2117    0.5349
                     normblend    0.1054    0.1150    0.1388    0.1954    0.5058
                     seq          0.1157    0.1389    0.1787    0.2882    0.6986
```

## Guard activation (share of test windows)

```
cond              GAP 0.4  MCAR 0.1  MCAR 0.3  MCAR 0.5  MCAR 0.7  MCAR 0.9  MNAR 0.3
dataset                                                                              
CMAPSS                0.0       0.0       0.0       0.0    0.0009       0.0       0.0
CMAPSS->CMAPSS_M      1.0       NaN       1.0       NaN    1.0000       NaN       1.0
CMAPSS_M              0.0       NaN       0.0       NaN    0.0000       NaN       0.0
MetroPT               0.0       0.0       0.0       0.0    0.0000       0.0       0.0
```

Reproduction of Step 7 on C-MAPSS: 120 comparisons, largest |difference| 0.00e+00

Runs without strict determinism: 0 of 1150
