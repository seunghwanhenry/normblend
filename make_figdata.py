"""Build figures/src/figdata2.json, the input of the figure scripts, from the result tables of Steps 8-10.
Run from the repository root:  python make_figdata.py"""
import json, os
import numpy as np
import pandas as pd

S8, S9, S10 = "step8_outputs", "step9_outputs", "step10_outputs"
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
# case study: MetroPT-3 air leaks (Step 10, part A)
SIG = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current"]
F = pd.read_csv(os.path.join(S10, "step10_metropt_failure.csv.gz"))
F = F[F.variant.isin(["global", "seq", "normblend"])].assign(sig=lambda q: q["var"].map(dict(enumerate(SIG))))
g = F.groupby(["sig", "model", "failure", "variant"])[["n", "sum_e2"]].sum()
g = np.sqrt(g.sum_e2 / g.n).unstack("variant")
D["case"] = {"rmse": [dict(sig=s, model=k, failure=int(f), **{v: float(r[v]) for v in g.columns}) for (s, k, f), r in g.iterrows()]}
E = np.load(os.path.join(S10, "step10_metropt_windows.npz"))
cond, seed, model, day, h0, h1, leak = "GAP 0.4", 0, "SAITS", "2020-06-05", 8, 15, 10
st = pd.to_datetime(E[f"{cond}|{seed}|start"]); pick = [i for i, s in enumerate(st) if str(s)[:10] == day and h0 <= s.hour < h1]
minute = np.concatenate([(st[i].hour - h0) * 60 + np.arange(60) for i in pick])
tr = {"minute": minute.tolist(), "hour0": h0, "leak_hour": leak, "day": "5 June 2020", "seed": seed, "model": model,
      "truth": {}, "mask": {}, "gaps": {}, "G": {}, "S": {}, "NB": {}}
for v in ("Reservoirs", "Oil_temperature"):
    j = SIG.index(v); mk = np.concatenate([E[f"{cond}|{seed}|mask"][i, :, j] for i in pick])
    tr["truth"][v] = np.concatenate([E[f"{cond}|{seed}|truth"][i, :, j] for i in pick]).round(4).tolist(); tr["mask"][v] = mk.astype(int).tolist()
    e = np.flatnonzero(np.diff(np.r_[0, mk.astype(int), 0])); tr["gaps"][v] = [[int(a), int(b) - 1] for a, b in zip(e[::2], e[1::2])]
    for key, var in (("G", "global"), ("S", "seq"), ("NB", "normblend")):
        x = np.concatenate([E[f"{cond}|{seed}|{model}|{var}"][i, :, j] for i in pick])
        tr[key][v] = [None if not np.isfinite(q) else round(float(q), 4) for q in x]
D["case"]["trace"] = tr
# shift magnitude (Step 10, part B)
K = pd.read_csv(os.path.join(S10, "step10_shift_runs.csv"))
K0 = K[K.scenario == "none"]; K = pd.concat([K[K.scenario != "none"]] + [K0.assign(scenario=sc) for sc in ("one", "all")])
kv = ["global", "seq", "nb_noguard", "normblend", "nb_k0.25", "nbvar_k1", "global_guard"]
cell = K[K.variant.isin(kv)].groupby(["dataset", "scenario", "k", "cond", "model", "variant"]).rmse.mean().unstack()
rat = cell.div(cell.seq, axis=0)
gmr = np.exp(np.log(rat).groupby(level=["dataset", "scenario", "k"]).mean())
gs = K.groupby(["dataset", "scenario", "k"])[["guard_k1", "guard_k0.25"]].mean()
D["shiftk"] = [dict(dataset=d, scenario=sc, k=float(k), **{v: float(r[v]) for v in kv}, guard1=float(gs.loc[(d, sc, k), "guard_k1"]),
                    guard025=float(gs.loc[(d, sc, k), "guard_k0.25"])) for (d, sc, k), r in gmr.iterrows()]
# graphical abstract: geometric-mean ratio to the better normalization (Step 10 unit errors, evaluation scale)
U = pd.read_csv(os.path.join(S10, "step10_unit_errors.csv.gz"))
U = U[(U.part == "main") & U.model.isin(["KNN", "BRITS", "SAITS", "TimesNet", "ImputeFormer", "Transformer"])
      & U.variant.isin(["global", "seq", "normblend", "valsel", "sel_var"])]
g = U.groupby(["dataset", "cond", "seed", "model", "variant"])[["n", "sse_rep"]].sum()
Pga = np.sqrt(g.sse_rep / g.n).groupby(["dataset", "cond", "model", "variant"]).mean().unstack()
orc = Pga[["global", "seq"]].min(axis=1)
D["ga"] = {v: float(np.exp(np.log(Pga[v] / orc).mean())) for v in ["normblend", "sel_var", "valsel", "global", "seq"]}
D["_sources"] = {"teaserA2, teaserB": "step8_runs.csv, step8_cross.csv, step9_runs.csv (SKAB)",
                 "prop": "step8_variables.csv, step9_variables_skab.csv", "sweep": "step8_runs.csv",
                 "weights": "step8_weights.csv, step9_weights.csv (SKAB)",
                 "case": "step10_metropt_failure.csv.gz, step10_metropt_windows.npz",
                 "shiftk": "step10_shift_runs.csv", "ga": "step10_unit_errors.csv.gz"}
json.dump(D, open(os.path.join("figures", "src", "figdata2.json"), "w"))
print({k: len(v) for k, v in D.items()})
