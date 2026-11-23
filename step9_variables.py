"""Variable-level calibration and test errors of the SKAB runs (Proposition 1 check, Figures 3 and 4).
Run from the folder that holds step9_extension.py after `python step9_extension.py run`; writes step9_outputs/step9_variables_skab.csv
and checks the global-view RMSE against step9_runs.csv."""
import importlib.util, numpy as np, pandas as pd
spec = importlib.util.spec_from_file_location("s9", "step9_extension.py"); s9 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s9)
s8, s5 = s9.s8, s9.s5
dc, _, _ = s5.modules(); rows = []; chk = []
for seed in s9.SEEDS:
    for mech, lv in s9.CONDS:
        S, Mtr, Mte, es, cal, mu, sd = s8.parts(dc, "SKAB", mech, lv, seed)
        y = S["Xte"][Mte]; _, _, d = np.nonzero(Mte); D = S["Xa"].shape[2]
        y_c = S["Xa"][cal][Mtr[cal]]; _, _, d_c = np.nonzero(Mtr[cal])
        for mdl in s8.BACKBONES:
            zg, _ = s9._load9(s9._job("SKAB", mech, lv, seed, mdl, "global", 0)); zs, _ = s9._load9(s9._job("SKAB", mech, lv, seed, mdl, "seq", 0))
            if zg is None or zs is None: print("missing", seed, mech, lv, mdl); continue
            g, s_ = zg["te"].astype(float) - y, zs["te"].astype(float) - y
            cg, cs = zg["cal"].astype(float) - y_c, zs["cal"].astype(float) - y_c
            chk.append(dict(cond=f"{mech} {lv:g}", seed=seed, model=mdl, rmse_g=float(np.sqrt((g**2).mean()))))
            a = s5.dual_weights(y_c, zg["cal"].astype(float), zs["cal"].astype(float), d_c, D)
            for k in range(D):
                c, t = d_c == k, d == k
                if c.sum() == 0 or t.sum() == 0: continue
                nb = a[k] * g[t] + (1 - a[k]) * s_[t]
                rows.append(dict(dataset="SKAB", cond=f"{mech} {lv:g}", seed=seed, model=mdl, var=k, n_cal=int(c.sum()),
                                 Eg=float((cg[c]**2).mean()), Es=float((cs[c]**2).mean()), C=float((cg[c]*cs[c]).mean()),
                                 weight=float(a[k]), n_test=int(t.sum()), mse_g=float((g[t]**2).mean()), mse_s=float((s_[t]**2).mean()),
                                 mse_nb=float((nb**2).mean()), cross_test=float((g[t]*s_[t]).mean())))
pd.DataFrame(rows).to_csv("step9_outputs/step9_variables_skab.csv", index=False)
C = pd.DataFrame(chk); R = pd.read_csv("step9_outputs/step9_runs.csv")
R = R[(R.dataset == "SKAB") & (R.variant == "global")][["cond", "seed", "model", "rmse"]]
m = C.merge(R, on=["cond", "seed", "model"]); print(len(rows), len(m), "max |rmse diff|", float((m.rmse_g - m.rmse).abs().max()))
