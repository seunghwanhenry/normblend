"""Supplementary Tables S1-S9 of the paper from the result tables of Steps 8, 9 and 10.
Run from the repository root after paper_tables.py (Table S7 reads its bootstrap intervals from paper_outputs/summary.json):
    python paper_tables.py
    python supp_tables.py
Writes paper_outputs/supplementary_tables.md."""
import json, os
import numpy as np
import pandas as pd

S8, S9, S10, OUT = "step8_outputs", "step9_outputs", "step10_outputs", "paper_outputs"
rd = lambda *p: pd.read_csv(os.path.join(*p))
gm = lambda x: float(np.exp(np.log(np.asarray(x, float)).mean()))
DSN = {"CMAPSS": "C-MAPSS-SC", "CMAPSS_M": "C-MAPSS-MC", "MetroPT": "MetroPT-3", "SKAB": "SKAB"}
CON = {"MCAR 0.3": "MCAR 30%", "MCAR 0.7": "MCAR 70%", "MNAR 0.3": "MNAR 30%", "GAP 0.4": "Gaps 40%",
       "MCAR 0.1": "MCAR 10%", "MCAR 0.5": "MCAR 50%", "MCAR 0.9": "MCAR 90%"}
MAIN = ["MCAR 0.3", "MCAR 0.7", "MNAR 0.3", "GAP 0.4"]
BB = ["KNN", "BRITS", "SAITS", "TimesNet", "ImputeFormer", "Transformer"]
E6M = ["SAITS", "TimesNet", "ImputeFormer", "Transformer"]
ALT = ["SAITS+stats", "ImputeFormer+stats", "TimesNet-NS", "NSTransformer", "RevIN-SCINet"]
ALTN = {"SAITS+stats": "SAITS + stats", "ImputeFormer+stats": "ImputeFormer + stats", "TimesNet-NS": "TimesNet-NS",
        "NSTransformer": "NS Transformer", "RevIN-SCINet": "RevIN-SCINet"}
f = lambda x, d=4: "" if pd.isna(x) else f"{x:.{d}f}"

R8, R9 = rd(S8, "step8_runs.csv"), rd(S9, "step9_runs.csv")
X8, X9 = rd(S8, "step8_cross.csv"), rd(S9, "step9_cross.csv")
J8, J9 = rd(S8, "step8_jobs.csv"), rd(S9, "step9_jobs.csv")
R = pd.concat([R8[R8.cond.isin(MAIN)].drop(columns=["source"], errors="ignore"),
               R9[R9.exp == "E7"].drop(columns=["exp"])], ignore_index=True)        # benchmark: 4 datasets x 4 conditions
md = []


def table(title, header, rows):
    md.extend([f"## {title}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)])
    md.extend("| " + " | ".join(str(x) for x in r) + " |" for r in rows)
    md.append("")


def lead(keys, prev):
    return [k if keys[:i + 1] != prev[:i + 1] else "" for i, k in enumerate(keys)]


def ordered(df, conds, cols=("dataset", "cond", "model")):
    df = df.reset_index()
    sk = {"dataset": list(DSN), "cond": conds, "model": BB + ALT}
    for c in cols: df[c + "_o"] = df[c].map({v: i for i, v in enumerate(sk[c])})
    return df.sort_values([c + "_o" for c in cols])


# S1: per setting, RMSE and MAE
m = R[R.model.isin(BB)].groupby(["dataset", "cond", "model", "variant"])[["rmse", "mae"]].mean()
P, Q = m.rmse.unstack(), m.mae.unstack()
t = pd.DataFrame({"g": P["global"], "s": P.seq, "v": P.valsel, "nb": P.normblend, "ga": Q["global"], "sa": Q.seq, "nba": Q.normblend})
rows, prev = [], []
for _, r in ordered(t, MAIN).iterrows():
    key = [DSN[r.dataset], CON[r.cond], r.model]
    rows.append(lead(key, prev) + [f(r.g), f(r.s), f(r.v), f(r.nb), f(r.nb / min(r.g, r.s), 3),
                                   f(r.ga), f(r.sa), f(r.nba), f(r.nba / min(r.ga, r.sa), 3)]); prev = key
table("Table S1: Results per setting, mean over seeds 0-4",
      ["Dataset", "Condition", "Imputer", "RMSE, global", "RMSE, per-seq.", "RMSE, val. sel.", "RMSE, NB", "RMSE ratio",
       "MAE, global", "MAE, per-seq.", "MAE, NB", "MAE ratio"], rows)

# S2: single-model alternatives per setting
a = R.copy(); a["key"] = np.where(a.variant.isin(["gstat", "own"]), a.model, a.model + "|" + a.variant)
k = a.groupby(["dataset", "cond", "key"]).rmse.mean().unstack()
cols = ["SAITS|normblend", "SAITS+stats", "ImputeFormer|normblend", "ImputeFormer+stats", "TimesNet|normblend", "TimesNet-NS", "NSTransformer", "RevIN-SCINet"]
rows, prev = [], []
for _, r in ordered(k[cols], MAIN, ("dataset", "cond")).iterrows():
    key = [DSN[r["dataset"]], CON[r["cond"]]]
    rows.append(lead(key, prev) + [f(r[c]) for c in cols]); prev = key
table("Table S2: RMSE of single-model alternatives per setting", ["Dataset", "Condition"] + cols, rows)

# S3: shift of operating conditions (C-MAPSS-SC -> C-MAPSS-MC)
xc = X8.groupby(["model", "variant"]).rmse.mean().unstack()
ind = R8[(R8.dataset == "CMAPSS_M") & R8.cond.isin(MAIN)].groupby(["model", "variant"]).rmse.mean().unstack()
rows = [[mdl] + [f(xc.loc[mdl, v], 3) for v in ("global", "seq", "valsel", "nb_equal", "nb_pooled", "nb_noguard", "normblend")]
        + [f"{100 * (1 - xc.loc[mdl, 'normblend'] / xc.loc[mdl, 'global']):.0f}%", f(ind.loc[mdl, "normblend"])] for mdl in BB]
table("Table S3 (A): Shift of operating conditions, wrapped imputers",
      ["Imputer", "Global", "Per-seq.", "Val. selection", "Equal weights", "Pooled weight", "NormBlend, no guard", "NormBlend", "Reduction vs global", "NormBlend, in-domain"], rows)
xa = X8[X8.variant.isin(["gstat", "own"])].groupby("model").rmse.mean()
ia = R8[(R8.dataset == "CMAPSS_M") & R8.cond.isin(MAIN) & R8.variant.isin(["gstat", "own"])].groupby("model").rmse.mean()
table("Table S3 (B): Shift of operating conditions, single models", ["Single model", "Cross-domain", "In-domain"],
      [[ALTN[mdl], f(xa[mdl], 3), f(ia[mdl])] for mdl in ALT])

# S4: missing-rate sweep
SW = ["MCAR 0.1", "MCAR 0.3", "MCAR 0.5", "MCAR 0.7", "MCAR 0.9"]
s = R8[R8.cond.isin(SW) & R8.dataset.isin(["CMAPSS", "MetroPT"]) & R8.model.isin(["KNN", "SAITS", "ImputeFormer"]) & R8.variant.isin(["global", "seq", "normblend"])]
s = s.groupby(["dataset", "model", "cond", "variant"]).rmse.mean().unstack()
rows, prev = [], []
for _, r in ordered(s, SW, ("dataset", "model", "cond")).iterrows():
    key = [DSN[r["dataset"]], r["model"], CON[r["cond"]]]
    rows.append(lead(key, prev) + [f(r["global"]), f(r.seq), f(r.normblend), f(r.normblend / min(r["global"], r.seq), 3)]); prev = key
table("Table S4: RMSE against the MCAR rate, mean over seeds 0-4", ["Dataset", "Imputer", "Missing rate", "Global", "Per-sequence", "NormBlend", "NormBlend / better"], rows)

# S5: training time
JJ = pd.concat([J8, J9[J9.exp == "E7"]]); JJ = JJ[JJ.method != "Linear regression"].copy()
JJ["t"] = JJ.train_seconds.fillna(JJ.seconds_fit_impute).fillna(JJ.seconds)
JJ["m"] = np.where(JJ.norm == "gstat", JJ.method + "+stats", JJ.method)
JJ["nk"] = JJ.norm.map({"global": "g", "seq": "s", "gstat": "own", "own": "own"})
q = JJ.groupby(["m", "dataset", "nk"]).t.median().unstack().join(JJ.groupby(["m", "dataset"]).size().rename("jobs"))
t1 = lambda x: "" if pd.isna(x) else (f"{x:,.0f}" if x >= 100 else f"{x:.1f}")
rows, prev = [], None
for mdl in BB + ALT:
    for ds in DSN:
        if (mdl, ds) not in q.index: continue
        r = q.loc[(mdl, ds)]
        rows.append([ALTN.get(mdl, mdl) if mdl != prev else "", DSN[ds], int(r.jobs), t1(r.get("g")), t1(r.get("s")), t1(r.get("own"))]); prev = mdl
table("Table S5: Median wall-clock time of one training in seconds", ["Method", "Dataset", "Jobs", "Global", "Per-sequence", "Single model"], rows)

# S6: seed-ensemble control (E6)
E = R9[R9.exp == "E6"]; ce = E.groupby(["dataset", "cond", "model", "variant"]).rmse.mean().unstack()
ce["orc"] = ce[["global", "seq"]].min(axis=1)
V6 = ["orc", "normblend", "ens_global", "ens_global_eq", "ens_seq", "ens_seq_eq", "ens_valsel", "nb_2seed"]
rows, prev = [], []
for _, r in ordered(ce[V6], MAIN).iterrows():
    key = [DSN[r["dataset"]], CON[r["cond"]], r["model"]]
    rows.append(lead(key, prev) + [f(r[v]) for v in V6]); prev = key
H6 = ["Better norm.", "NB", "Ens. G, cal.", "Ens. G, eq.", "Ens. S, cal.", "Ens. S, eq.", "Ens., val. sel.", "NB, 2 seeds"]
table("Table S6 (A): Seed-ensemble control, RMSE per setting", ["Dataset", "Condition", "Imputer"] + H6, rows)
x6 = X9.groupby(["model", "variant"]).rmse.mean().unstack()
V6x = ["global", "seq", "normblend", "ens_global", "ens_global_eq", "ens_seq", "ens_seq_eq", "ens_valsel", "nb_2seed"]
table("Table S6 (B): Seed-ensemble control, RMSE under the shift", ["Imputer", "Global", "Per-seq."] + H6[1:],
      [[mdl] + [f(x6.loc[mdl, v], 3) for v in V6x] for mdl in E6M])
md.append(f"Calibrated against equal weights, geometric mean over the {len(ce)} settings: "
          f"global {gm(ce.ens_global / ce.ens_global_eq):.4f}, per-sequence {gm(ce.ens_seq / ce.ens_seq_eq):.4f}.\n")

# S7: both scales, bootstrap intervals (paper_outputs/summary.json, written by paper_tables.py)
S = json.load(open(os.path.join(OUT, "summary.json")))["step10"]
rows = []
for sc, lab in (("rep", "z° (evaluation)"), ("eq2", "z (Equation 2)")):
    B, o = S[sc]["boot"], S[sc]
    for ds in [None] + list(DSN):
        suf = "" if ds is None else f"|{ds}"
        ci = lambda kk: f"{B[kk + suf]['est']:.3f} ({B[kk + suf]['lo']:.3f}-{B[kk + suf]['hi']:.3f})"
        below = o["below"] if ds is None else o["below_by_ds"][ds]
        rows.append([lab if ds is None else "", "All" if ds is None else DSN[ds], f"{below} of {96 if ds is None else 24}"]
                    + [ci(kk) for kk in ("nb/orc", "nb/global", "nb/seq", "nb/valsel", "nb/sel_var")])
table("Table S7: Geometric-mean ratios of NormBlend (NB) on both scales, 95% cluster-bootstrap intervals",
      ["Scale", "Dataset", "Below", "NB / better", "NB / global", "NB / per-sequence", "NB / val. selection", "NB / per-variable selection"], rows)

# S8: MetroPT-3 air leaks
SIG = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current"]
SIGN = {"TP2": "TP2 (compressor)", "TP3": "TP3 (pneumatic panel)", "H1": "H1 (cyclonic separator)", "DV_pressure": "DV pressure (air dryer)",
        "Reservoirs": "Reservoirs", "Oil_temperature": "Oil temperature", "Motor_current": "Motor current"}
UNIT = {"Oil_temperature": "°C", "Motor_current": "A"}
F = rd(S10, "step10_metropt_failure.csv.gz"); F = F[F.variant.isin(["global", "seq", "normblend"])]
g = F.groupby(["var", "failure", "variant"])[["n", "sum_e", "sum_e2"]].sum()
g["rmse"] = np.sqrt(g.sum_e2 / g.n); g["bias"] = g.sum_e / g.n
VV = ("global", "seq", "normblend")
rows = [[SIGN[sg], UNIT.get(sg, "bar")] + [f(g.loc[(j, 0, v), "rmse"], 3) for v in VV] + [f(g.loc[(j, 1, v), "rmse"], 3) for v in VV]
        + [f"{g.loc[(j, 1, v), 'bias']:+.3f}" for v in VV] for j, sg in enumerate(SIG)]
table("Table S8: MetroPT-3 air leaks, RMSE in physical units and mean signed error in leak windows",
      ["Signal", "Unit", "Normal, G", "Normal, S", "Normal, NB", "Leak, G", "Leak, S", "Leak, NB", "Bias, G", "Bias, S", "Bias, NB"], rows)
nw = F[(F.variant == "global") & (F.model == "KNN")].groupby("failure").n.sum()
md.append(f"Removed values per imputer and method: {int(nw[0]):,} in normal windows (all seeds), {int(nw[1]):,} in leak windows (seeds 0, 1, 2 and 4).\n")

# S9: graded shift
K = rd(S10, "step10_shift_runs.csv")
K0 = K[K.scenario == "none"]; K = pd.concat([K[K.scenario != "none"]] + [K0.assign(scenario=sc) for sc in ("one", "all")])
cell = K.groupby(["dataset", "scenario", "k", "cond", "model", "variant"]).rmse.mean().unstack()
rat = np.exp(np.log(cell.div(cell.seq, axis=0)).groupby(level=["dataset", "scenario", "k"]).mean())
gsh = K.groupby(["dataset", "scenario", "k"])[["guard_k1", "guard_k0.25"]].mean()
V9 = ["global", "nb_noguard", "nb_k2", "normblend", "nb_k0.5", "nb_k0.25", "nb_k0.1", "nbvar_k1"]
rows = []
for ds in ["CMAPSS", "MetroPT"]:
    for sc, scn in (("one", "one variable"), ("all", "every variable")):
        for i, kk in enumerate(sorted(rat.loc[(ds, sc)].index)):
            r, gg = rat.loc[(ds, sc, kk)], gsh.loc[(ds, sc, kk)]
            rows.append([DSN[ds] if (sc == "one" and i == 0) else "", scn if i == 0 else "", f"{kk:g}"] + [f(r[v], 2) for v in V9]
                        + [f(gg["guard_k1"], 3), f(gg["guard_k0.25"], 3)])
table("Table S9: Graded shift, RMSE relative to per-sequence normalization (geometric mean over 12 settings) and flagged share",
      ["Dataset", "Offset in", "k", "Global", "NB, no guard", "NB, κ = 2", "NB, κ = 1", "NB, κ = 0.5", "NB, κ = 0.25", "NB, κ = 0.1",
       "Per-variable fallback", "Flagged, κ = 1", "Flagged, κ = 0.25"], rows)

os.makedirs(OUT, exist_ok=True)
open(os.path.join(OUT, "supplementary_tables.md"), "w", encoding="utf-8").write("\n".join(md))
print("wrote", os.path.join(OUT, "supplementary_tables.md"))
