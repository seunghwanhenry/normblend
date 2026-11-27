"""
step8_analysis.py - per-variable quantities behind Proposition 1 of the NormBlend paper (CPU, reads step8_outputs).

For every setting, imputer and variable: the calibration mean squares E_G, E_S and cross-product C of the two
normalizations (the inputs of the NormBlend weight), the weight, and the test mean squared errors of global,
per-sequence and NormBlend (without guard) imputation. Output: step8_outputs/step8_variables.csv

  python step8_analysis.py
"""
import importlib.util, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("step8_benchmark", os.path.join(HERE, "step8_benchmark.py"))
s8 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s8)
s5 = s8.s5


def main():
    dc, _, _ = s5.modules(); rows = []
    for seed in s8.SEEDS:
        for ds in s8.DATASETS:
            for mech, lv in s8.CONDS:
                S, Mtr, Mte, es, cal, mu, sd = s8.parts(dc, ds, mech, lv, seed)
                y = S["Xte"][Mte]; _, _, d = np.nonzero(Mte); D = S["Xa"].shape[2]
                y_c = S["Xa"][cal][Mtr[cal]]; _, _, d_c = np.nonzero(Mtr[cal])
                for mdl in s8.BACKBONES:
                    (zg, _), (zs, _) = (s8._load_pred(s8._job(ds, mech, lv, seed, mdl, n, 0)) for n in s8.NORMS)
                    if zg is None or zs is None: continue
                    cg, cs = zg["cal"].astype(float) - y_c, zs["cal"].astype(float) - y_c
                    g, s_ = zg["te"].astype(float) - y, zs["te"].astype(float) - y
                    a = s5.dual_weights(y_c, zg["cal"].astype(float), zs["cal"].astype(float), d_c, D)
                    for k in range(D):
                        c, t = d_c == k, d == k
                        if c.sum() == 0 or t.sum() == 0: continue
                        nb = a[k] * g[t] + (1 - a[k]) * s_[t]
                        rows.append(dict(dataset=ds, cond=f"{mech} {lv:g}", seed=seed, model=mdl, var=k, n_cal=int(c.sum()),
                                         Eg=float((cg[c] ** 2).mean()), Es=float((cs[c] ** 2).mean()), C=float((cg[c] * cs[c]).mean()),
                                         weight=float(a[k]), n_test=int(t.sum()), mse_g=float((g[t] ** 2).mean()),
                                         mse_s=float((s_[t] ** 2).mean()), mse_nb=float((nb ** 2).mean()),
                                         cross_test=float((g[t] * s_[t]).mean())))
    out = pd.DataFrame(rows); p = os.path.join(s8.OUT, "step8_variables.csv"); out.to_csv(p, index=False)
    print(f"wrote {p}: {len(out)} rows")


if __name__ == "__main__":
    main()
