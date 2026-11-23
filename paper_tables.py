"""Numbers and Tables 2-5 of the paper from the result tables of Steps 8 and 9.
Run from the repository root:  python paper_tables.py
Writes paper_outputs/summary.json (statistics quoted in the text) and paper_outputs/tables.md (Tables 2-5)."""
import json, os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

S8, S9, OUT = "step8_outputs", "step9_outputs", "paper_outputs"
os.makedirs(OUT, exist_ok=True)
gm = lambda x: float(np.exp(np.log(np.asarray(x, float)).mean()))
MAIN = ["MCAR 0.3", "MCAR 0.7", "MNAR 0.3", "GAP 0.4"]
BB = ["KNN", "BRITS", "SAITS", "TimesNet", "ImputeFormer", "Transformer"]
DS = [("CMAPSS", "C-MAPSS-SC"), ("CMAPSS_M", "C-MAPSS-MC"), ("MetroPT", "MetroPT-3"), ("SKAB", "SKAB")]
rd = lambda f: pd.read_csv(f)
R8, R9 = rd(os.path.join(S8, "step8_runs.csv")), rd(os.path.join(S9, "step9_runs.csv"))
X8, X9 = rd(os.path.join(S8, "step8_cross.csv")), rd(os.path.join(S9, "step9_cross.csv"))
R = pd.concat([R8[R8.cond.isin(MAIN)].drop(columns=["source"], errors="ignore"),
               R9[R9.exp == "E7"].drop(columns=["exp"])], ignore_index=True)       # benchmark: 4 datasets, 4 conditions

# ---------------------------------------------------------------- benchmark (96 settings)
out = {}
c = R[R.model.isin(BB)].groupby(["dataset", "cond", "model", "variant"])[["rmse", "mae"]].mean()
P, Q = c.rmse.unstack(), c.mae.unstack()
P["orc"] = P[["global", "seq"]].min(axis=1); Q["orc"] = Q[["global", "seq"]].min(axis=1)
r = P.normblend / P.orc
out.update(n=len(P), below=int((r < 1).sum()), worst=float(r.max()), gm_orc=gm(r), gm_glob=gm(P.normblend / P["global"]),
           gm_seq=gm(P.normblend / P.seq), glob_better=int((P["global"] < P.seq).sum()), seq_better=int((P.seq < P["global"]).sum()),
           seq_over_glob_max=float((P.seq / P["global"]).max()))
out["glob_better_by_ds"] = (P["global"] < P.seq).groupby(level="dataset").sum().astype(int).to_dict()
out["gm_by_ds"] = r.groupby(level="dataset").apply(gm).round(3).to_dict()
out["gm_by_model"] = r.groupby(level="model").apply(gm).round(3).to_dict()
out.update(valsel_gm=gm(P.normblend / P.valsel), below_valsel=int((P.normblend < P.valsel).sum()), valsel_worst=float((P.valsel / P.orc).max()))
out["wilcoxon_p"] = {v: float(wilcoxon(P.normblend, P[v]).pvalue) for v in ["global", "seq", "valsel"]}
rq = Q.normblend / Q.orc
out.update(mae_below=int((rq < 1).sum()), mae_gm=gm(rq), mae_worst=float(rq.max()))
s = R[R.model.isin(BB)].pivot_table(index=["dataset", "cond", "model", "seed"], columns="variant", values="rmse").dropna(subset=["global", "seq", "normblend"])
out.update(seed_le=float((s.normblend <= s[["global", "seq"]].min(axis=1)).mean()), n_seed=len(s))
strat = ["normblend", "nb_noguard", "nb_pooled", "nb_equal", "valsel", "global", "seq"]
rk = P[strat].rank(axis=1).mean()
out["ablation"] = {v: dict(gm=gm(P[v] / P.orc), worst=float((P[v] / P.orc).max()), below=int((P[v] < P.orc).sum()), rank=float(rk[v])) for v in strat}
d = P.normblend / P.nb_noguard - 1
out.update(guard_changed=int((d.abs() > 1e-9).sum()), guard_change_range=[float(d.min()), float(d.max())])
G = pd.concat([rd(os.path.join(S8, "step8_guard.csv")), rd(os.path.join(S9, "step9_guard.csv"))])
out["guard_share"] = G.groupby("dataset").guard_share.agg(["mean", "max"]).to_dict()
# cross-domain (C-MAPSS-SC -> C-MAPSS-MC), mean over conditions and seeds
xc = X8.groupby(["model", "variant"]).rmse.mean()
out["cross_ablation"] = {v: float(xc.xs(v, level="variant").loc[BB].mean()) for v in strat}
# Proposition 1 (variable-level blends)
V = pd.concat([rd(os.path.join(S8, "step8_variables.csv")).query("cond in @MAIN"), rd(os.path.join(S9, "step9_variables_skab.csv"))], ignore_index=True)
Emin, Emax = V[["Eg", "Es"]].min(axis=1), V[["Eg", "Es"]].max(axis=1); den = V.Eg + V.Es - 2 * V.C
inter = (V.weight > 0) & (V.weight < 1)
gain = 1 - V.mse_nb / np.where(V.Eg <= V.Es, V.mse_g, V.mse_s)
pred = np.where(inter, (Emin - V.C) ** 2 / den / Emin, 0.0)
ratio = (V.C / np.sqrt(V.Eg * V.Es)) / np.sqrt(Emin / Emax)
out["prop"] = dict(n=len(V), interior=float(inter.mean()), beat=float((gain[inter] > 0).mean()), med_gain=float(np.median(gain[inter])),
                   spearman=float(spearmanr(pred, gain).correlation), low=float(np.median(gain[ratio <= 0.25])),
                   high=float(np.median(gain[(ratio > 0.9) & (ratio <= 1)])))
out["prop_by_ds"] = {ds: dict(beat=float((gain[inter & (V.dataset == ds)] > 0).mean()),
                              spearman=float(spearmanr(pred[V.dataset == ds], gain[V.dataset == ds]).correlation)) for ds, _ in DS}
# weights
W = pd.concat([rd(os.path.join(S8, "step8_weights.csv")), rd(os.path.join(S9, "step9_weights.csv"))], ignore_index=True)
W = W[W.cond.isin(MAIN)]
out["w_means"] = W.groupby(["dataset", "model"]).weight.mean().unstack(0).round(3).to_dict()
out["w_strict"] = float(((W.weight > 0.02) & (W.weight < 0.98)).mean())

# ---------------------------------------------------------------- seed-ensemble control (E6, 48 settings)
ce = R9[R9.exp == "E6"].groupby(["dataset", "cond", "model", "variant"]).rmse.mean().unstack()
ce["orc"] = ce[["global", "seq"]].min(axis=1); ce["ens_better"] = ce[["ens_global", "ens_seq"]].min(axis=1)
E6V = ["normblend", "ens_global", "ens_seq", "ens_better", "ens_valsel", "ens_global_eq", "ens_seq_eq", "nb_2seed"]
e6 = {v: dict(gm=gm(ce[v] / ce.orc), worst=float((ce[v] / ce.orc).max()), nb_below=int((ce.normblend < ce[v]).sum())) for v in E6V}
x9 = X9.groupby(["model", "variant"]).rmse.mean().unstack()
for v in E6V:
    if v in x9: e6[v]["cross"] = float(x9[v].mean())
# oracle ensemble under the shift: in each C-MAPSS-SC setting, the ensemble that was better in distribution
pick = ce.loc["CMAPSS"].apply(lambda q: "ens_global" if q.ens_global <= q.ens_seq else "ens_seq", axis=1)
xs = X9.groupby(["cond", "model", "variant"]).rmse.mean()
e6["ens_better"]["cross"] = float(np.mean([xs.loc[(cnd, mdl, v)] for (cnd, mdl), v in pick.items()]))
e6["wilcoxon_nb_vs_ens_global"] = float(wilcoxon(ce.normblend, ce.ens_global).pvalue)
e6["nb_2seed_below_ens_better"] = int((ce.nb_2seed < ce.ens_better).sum())
e6["global_better"] = int((ce["global"] < ce.seq).sum())
out["e6"] = e6
ce.to_csv(os.path.join(OUT, "e6_settings.csv"))
json.dump(out, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=float)

# ---------------------------------------------------------------- tables
B = lambda s, on: f"**{s}**" if on else s
def table2():
    sm = R[R.model.isin(BB) & R.variant.isin(["global", "seq", "normblend"])].groupby(["dataset", "model", "variant", "seed"]).rmse.mean()
    T = sm.groupby(["dataset", "model", "variant"]).agg(["mean", "std"])
    rows = ["| Dataset | Imputer | Global | Per-sequence | NormBlend | NormBlend vs better |", "|---|---|---|---|---|---|"]
    for ds, lab in DS:
        for k, m in enumerate(BB):
            v = {x: T.loc[(ds, m, x)] for x in ["global", "seq", "normblend"]}
            lo = min(round(v[x]["mean"], 4) for x in v)
            cells = [B(f"{v[x]['mean']:.4f} ± {v[x]['std']:.4f}", round(v[x]["mean"], 4) == lo) for x in v]
            dd = 100 * (v["normblend"]["mean"] / min(v["global"]["mean"], v["seq"]["mean"]) - 1)
            t = "0.0%" if abs(dd) < 0.05 else ("−" if dd < 0 else "+") + f"{abs(dd):.1f}%"
            rows.append(f"| {lab if k == 0 else ''} | {m} | " + " | ".join(cells) + f" | {t} |")
    return "\n".join(rows)
def table3():
    m = R.groupby(["model", "variant", "dataset"]).rmse.mean()
    val = lambda mdl, v, ds: m.loc[(mdl, v, ds)]
    blocks = [("SAITS", [("Global standardization", 1, "SAITS", "global"), ("Per-sequence normalization", 1, "SAITS", "seq"), ("Validation selection", 2, "SAITS", "valsel"),
                         ("Window statistics as inputs", 1, "SAITS+stats", "gstat"), ("NormBlend", 2, "SAITS", "normblend")]),
              ("ImputeFormer", [("Global standardization", 1, "ImputeFormer", "global"), ("Per-sequence normalization", 1, "ImputeFormer", "seq"),
                                ("Validation selection", 2, "ImputeFormer", "valsel"), ("Window statistics as inputs", 1, "ImputeFormer+stats", "gstat"),
                                ("NormBlend", 2, "ImputeFormer", "normblend")]),
              ("TimesNet", [("Global standardization", 1, "TimesNet", "global"), ("Per-sequence normalization", 1, "TimesNet", "seq"), ("Validation selection", 2, "TimesNet", "valsel"),
                            ("Built-in non-stationary normalization", 1, "TimesNet-NS", "own"), ("NormBlend", 2, "TimesNet", "normblend")]),
              ("Normalization-aware models", [("Non-stationary Transformer", 1, "NSTransformer", "own"), ("RevIN-SCINet", 1, "RevIN-SCINet", "own")])]
    rows = ["| Method | Trainings | C-MAPSS-SC | C-MAPSS-MC | MetroPT-3 | SKAB | SC → MC |", "|---|---|---|---|---|---|---|"]
    for name, items in blocks:
        rows.append(f"| **{name}** | | | | | | |")
        vals = {lab: [val(mdl, v, ds) for ds, _ in DS] + [xc.loc[(mdl, v)]] for lab, _, mdl, v in items}
        lo = [min(round(vals[lab][i], 4 if i < 4 else 3) for lab in vals) for i in range(5)]
        for lab, tr, mdl, v in items:
            x = vals[lab]
            cells = [B(f"{x[i]:.4f}", round(x[i], 4) == lo[i]) for i in range(4)] + [B(f"{x[4]:.3f}", round(x[4], 3) == lo[4])]
            rows.append(f"| {lab} | {tr} | " + " | ".join(cells) + " |")
    return "\n".join(rows)
def table4():
    names = [("normblend", "NormBlend", "2", "no"), ("ens_global", "Two-seed ensemble, global standardization", "2", "no"),
             ("ens_seq", "Two-seed ensemble, per-sequence normalization", "2", "no"), ("ens_better", "Two-seed ensemble, better normalization (oracle)", "2", "yes"),
             ("ens_valsel", "Two-seed ensemble, validation-selected normalization", "3", "no"), ("nb_2seed", "NormBlend over two-seed members", "4", "no")]
    vals = {k: [round(e6[k]["gm"], 3), round(e6[k]["worst"], 3), round(e6[k]["cross"], 3)] for k, *_ in names}
    best = [min(v[i] for v in vals.values()) for i in range(3)]
    rows = ["| Method | Trainings | Needs the better view | Ratio, geometric mean | Ratio, worst | NormBlend lower | SC → MC |", "|---|---|---|---|---|---|---|"]
    for k, lab, tr, nd in names:
        v = vals[k]; nbl = "–" if k == "normblend" else f"{e6[k]['nb_below']} of {len(ce)}"
        rows.append(f"| {lab} | {tr} | {nd} | {B(f'{v[0]:.3f}', v[0] == best[0])} | {B(f'{v[1]:.3f}', v[1] == best[1])} | {nbl} | {B(f'{v[2]:.3f}', v[2] == best[2])} |")
    return "\n".join(rows)
def table5():
    A, X = out["ablation"], out["cross_ablation"]
    names = [("normblend", "NormBlend (per-variable weights and guard)"), ("nb_noguard", "without the guard"), ("nb_pooled", "one pooled weight"),
             ("nb_equal", "equal weights (0.5)"), ("valsel", "validation selection"), ("global", "global standardization"), ("seq", "per-sequence normalization")]
    vals = {k: [round(A[k]["gm"], 3), round(A[k]["worst"], 3), A[k]["below"], round(A[k]["rank"], 2), round(X[k], 3)] for k, _ in names}
    best = [min(v[0] for v in vals.values()), min(v[1] for v in vals.values()), max(v[2] for v in vals.values()),
            min(v[3] for v in vals.values()), min(v[4] for v in vals.values())]
    rows = ["| Variant | Ratio, geometric mean | Ratio, worst | Below the better normalization | Mean rank | SC → MC |", "|---|---|---|---|---|---|"]
    for k, lab in names:
        v = vals[k]; f = [f"{v[0]:.3f}", f"{v[1]:.3f}", str(v[2]), f"{v[3]:.2f}", f"{v[4]:.3f}"]
        rows.append(f"| {lab} | " + " | ".join(B(f[i], v[i] == best[i]) for i in range(5)) + " |")
    return "\n".join(rows)
md = "\n\n".join(f"Table {n}\n\n{f()}" for n, f in [(2, table2), (3, table3), (4, table4), (5, table5)])
open(os.path.join(OUT, "tables.md"), "w", encoding="utf-8").write(md + "\n")
print(md)
print(f"\n{out['below']} of {out['n']} settings below the better normalization; geometric mean {out['gm_orc']:.3f}")
