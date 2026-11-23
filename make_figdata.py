"""Build figures/src/figdata2.json, the input of the figure scripts, from the result tables of Steps 8 and 9.
Run from the repository root:  python make_figdata.py"""
import json, os
import numpy as np
import pandas as pd

S8, S9 = "step8_outputs", "step9_outputs"
MAIN = ["MCAR 0.3", "MCAR 0.7", "MNAR 0.3", "GAP 0.4"]
SWEEP = ["MCAR 0.1", "MCAR 0.3", "MCAR 0.5", "MCAR 0.7", "MCAR 0.9"]
MODELS = ["KNN", "BRITS", "SAITS", "TimesNet", "ImputeFormer", "Transformer"]

r8 = pd.read_csv(os.path.join(S8, "step8_runs.csv")); r9 = pd.read_csv(os.path.join(S9, "step9_runs.csv"))
x8 = pd.read_csv(os.path.join(S8, "step8_cross.csv"))
w8 = pd.read_csv(os.path.join(S8, "step8_weights.csv")); w9 = pd.read_csv(os.path.join(S9, "step9_weights.csv"))
v8 = pd.read_csv(os.path.join(S8, "step8_variables.csv")); v9 = pd.read_csv(os.path.join(S9, "step9_variables_skab.csv"))
runs = pd.concat([r8, r9[r9.dataset == "SKAB"]], ignore_index=True)
main = runs[runs.cond.isin(MAIN) & runs.model.isin(MODELS)]
m = main.groupby(["dataset", "cond", "model", "variant"]).rmse.mean().unstack()
D = {}
D["teaserA2"] = [dict(ds=d, cond=c, model=k, r=float(r.seq / r["global"])) for (d, c, k), r in m.iterrows()]
xm = x8[x8.model.isin(MODELS)].groupby(["cond", "model", "variant"]).rmse.mean().unstack()
D["teaserA2"] += [dict(ds="SC2MC", cond=c, model=k, r=float(r.seq / r["global"])) for (c, k), r in xm.iterrows()]
orc = m[["global", "seq"]].min(axis=1)
D["teaserB"] = [dict(ds=d, cond=c, model=k, r=float(r.normblend / o), g=float(r["global"] / o), s=float(r.seq / o))
                for ((d, c, k), r), o in zip(m.iterrows(), orc.values)]
V = pd.concat([v8[v8.cond.isin(MAIN)], v9], ignore_index=True)
Emin, Emax = V[["Eg", "Es"]].min(axis=1), V[["Eg", "Es"]].max(axis=1); den = V.Eg + V.Es - 2 * V.C
inter = (V.weight > 0) & (V.weight < 1)
fav = np.where(V.Eg <= V.Es, V.mse_g, V.mse_s)
V["test"] = 1 - V.mse_nb / fav
V["pred"] = np.where(inter, (Emin - V.C) ** 2 / den / Emin, 0.0)
V["q"] = (V.C / np.sqrt(V.Eg * V.Es)) / np.sqrt(Emin / Emax)
D["prop"] = [dict(dataset=r.dataset, model=r.model, q=float(r.q), pred=float(r.pred), test=float(r.test)) for r in V.itertuples()]
sw = r8[r8.cond.isin(SWEEP) & r8.dataset.isin(["CMAPSS", "MetroPT"]) & r8.model.isin(["KNN", "SAITS", "ImputeFormer"])
        & r8.variant.isin(["global", "seq", "normblend"])]
sw = sw.groupby(["dataset", "model", "variant", "cond", "seed"]).rmse.mean().groupby(["dataset", "model", "variant", "cond"]).agg(["mean", "std"])
D["sweep"] = [dict(dataset=d, model=k, variant=v, cond=c, mean=float(r["mean"]), std=float(r["std"])) for (d, k, v, c), r in sw.iterrows()]
W = pd.concat([w8[w8.cond.isin(MAIN)], w9[w9.dataset == "SKAB"]], ignore_index=True)
W = W[W.model.isin(MODELS)].groupby(["dataset", "model", "var"]).weight.mean()
D["weights"] = [dict(dataset=d, model=k, var=int(v), weight=float(w)) for (d, k, v), w in W.items()]
D["_sources"] = {"teaserA2, teaserB": "step8_runs.csv, step8_cross.csv, step9_runs.csv (SKAB)",
                 "prop": "step8_variables.csv, step9_variables_skab.csv", "sweep": "step8_runs.csv",
                 "weights": "step8_weights.csv, step9_weights.csv (SKAB)"}
json.dump(D, open(os.path.join("figures", "src", "figdata2.json"), "w"))
print({k: len(v) for k, v in D.items()})
