"""
step10_followup.py - Step 10: follow-up analyses of the NormBlend benchmark (descriptive; run after Step 9).

A  Re-analysis of the stored predictions of Steps 7-9 (CPU, no training):
   A1 Scale. Steps 7-9 report errors on the reporting scale of each seed (mean and sd of all values of the training
      and validation units before removal; used by no model). Every error is also computed on the scale of
      Equation 2 of the paper (mean and sd of the observed values of the masked training windows), the scale the
      models work on. The shift test keeps the target-domain scales.
   A2 Per-variable selection: in each variable, the view with the lower calibration MSE (weight 0 or 1), with and
      without the range guard; and global standardization with the guard as the only fallback.
   A3 Squared and absolute errors summed per test unit (engine, day, run, block) for every method of Steps 8 and 9,
      for a cluster bootstrap over test units.
   A4 MetroPT-3 air-leak failures: errors in physical units for the test windows that overlap one of the four
      documented failures and for the other windows, and full windows around the failures.
B  Shift magnitude (GPU): KNN, SAITS and the transformer on C-MAPSS-SC and MetroPT-3, four conditions, seeds 0-4,
   trained exactly as in Step 8 (the test imputations must equal the stored ones). The test windows are offset by
   k standard deviations of the observed training values (Equation 2), either in one variable per window (as in
   Step 7) or in every variable, for k in K_GRID; the ground truth moves with them. In each job both views are trained
   and evaluated together: G, S, NormBlend without guard, NormBlend with the window guard at several margins kappa,
   a per-variable guard, per-variable selection and global standardization with the guard as fallback.

Commands (from the DTIN folder, next to step9_extension.py):
  python step10_followup.py plan        # writes step10_outputs/PLAN_step10.md (once, before anything else)
  python step10_followup.py analyze     # part A (CPU, about 20-40 min)
  python step10_followup.py check       # part B reproduction on CPU: KNN in two settings must equal Step 8
  python step10_followup.py smoke       # part B on a tiny subset with two epochs (a few minutes)
  python step10_followup.py run [--worker k --n-workers K]
  python step10_followup.py status
  python step10_followup.py report      # part B tables and REPORT_step10.md
"""
import argparse, glob, hashlib, importlib.util, json, os, sys, time, traceback
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("step9_extension", os.path.join(HERE, "step9_extension.py"))
s9 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s9)        # also lets s8.parts read SKAB
s8, s5, s7 = s9.s8, s9.s5, s9.s7

OUT = "step10_outputs"
SEEDS = [0, 1, 2, 3, 4]
SHIFT_DATASETS = ["CMAPSS", "MetroPT"]
SHIFT_MODELS = ["KNN", "SAITS", "Transformer"]
K_GRID = [0.5, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50, 100]
SCENARIOS = ["one", "all"]
KAPPAS = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
EX_VARIANTS = ["global", "seq", "normblend", "nb_noguard", "valsel", "sel_var", "global_guard"]   # A4 variants
EX_MODELS = ["SAITS", "Transformer", "ImputeFormer"]                                             # A4 full windows


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def script_sha():
    return sha_file(os.path.abspath(__file__))


PLAN = """# Step 10 - follow-up analyses of the NormBlend benchmark (descriptive)

Written on {date}, after Step 9 and before any Step 10 analysis or job was run. Script step10_followup.py, SHA-256
{sha}; it imports the unchanged step9_extension.py ({sha9}), step8_benchmark.py ({sha8}), step5_confirm.py ({sha5})
and step7_robustness.py ({sha7}). Step 10 has no confirmatory decision rule; every result is reported.

## Why
A reading of the manuscript (26 November 2026) found that (1) the paper describes errors on the scale of Equation 2,
whereas Steps 7-9 report them on a reporting scale; (2) the comparison with validation selection does not separate
per-variable choice from blending; (3) under the C-MAPSS shift the guard flags every window, so the shift test shows
the fallback and says nothing about moderate shifts; (4) the Wilcoxon tests over 96 settings treat settings that
share data as independent; and (5) the paper has no engineering case.

## Part A (stored predictions, no training)
* A1: every error of Steps 8 and 9 recomputed on the Equation 2 scale of its setting (target-domain scales for the
  shift test), next to the reporting scale; the recomputed reporting-scale RMSE must equal the stored tables.
* A2: per-variable selection (weight 0 or 1 from the calibration MSE of each variable; pooled choice for variables
  without calibration positions) with and without the guard; global standardization with the guard as fallback.
* A3: sums of squared and absolute errors per test unit for every method, for a cluster bootstrap over test units
  within each seed (the same draw for all settings of a dataset).
* A4: MetroPT-3, four main conditions: errors in physical units (signed, squared, absolute) per test day, failure
  flag (window overlaps one of the four documented air-leak failures) and variable for {exv}; full windows (truth,
  mask, imputations) for the failure windows and up to two windows on either side for {exm}.

## Part B (new runs)
* {models} with G and S on {sds}, the four main conditions, seeds {seeds}; Step 8 protocol, seeds and settings;
  both views trained in one job. The test imputations without shift must equal the stored Step 7/8 imputations.
* Shifted test windows: offset of k standard deviations (Equation 2 scale) in one variable per window (variable
  (window index + seed) mod D, as in Step 7) or in every variable; k in {ks}; truth shifted alike.
* Variants: G, S, NormBlend without guard, NormBlend with the window guard at kappa in {kappas} (kappa = 1 is the
  method), per-variable guard (weight 0 only for the flagged variables) at the same margins, per-variable
  selection with the guard, global standardization with the guard as fallback. RMSE over all removed positions, over
  those of the shifted variables and over the others; guard shares.
"""


def plan():
    os.makedirs(OUT, exist_ok=True); p = os.path.join(OUT, "PLAN_step10.md")
    if os.path.exists(p): sys.exit(f"{p} exists; the plan is written once")
    txt = PLAN.format(date=time.strftime("%d %B %Y"), sha=script_sha(), sha9=s9.script_sha(), sha8=s8.script_sha(),
                      sha5=sha_file(os.path.join(HERE, "step5_confirm.py")), sha7=sha_file(os.path.join(HERE, "step7_robustness.py")),
                      exv=", ".join(EX_VARIANTS), exm=", ".join(EX_MODELS), models=", ".join(SHIFT_MODELS),
                      sds=", ".join(SHIFT_DATASETS), seeds=SEEDS, ks=K_GRID, kappas=KAPPAS)
    open(p, "w", encoding="utf-8").write(txt); print(txt)


# ------------------------------------------------------------------------------------------- shared pieces
def sel_weights(ys, gs, ss, d_st, D):
    """per-variable selection: 1 where the global view has the lower calibration MSE (pooled choice as fallback)"""
    eg, es = gs - ys, ss - ys
    a = np.full(D, 1.0 if (eg ** 2).mean() <= (es ** 2).mean() else 0.0)
    for d in range(D):
        k = d_st == d
        if k.sum() > 0: a[d] = 1.0 if (eg[k] ** 2).mean() <= (es[k] ** 2).mean() else 0.0
    return a


def extra_variants(g, s_, a_sel, d, gn):
    an = np.where(gn, 0.0, a_sel[d])
    return {"sel_var": an * g + (1 - an) * s_, "sel_var_noguard": a_sel[d] * g + (1 - a_sel[d]) * s_,
            "global_guard": np.where(gn, s_, g)}


def exceedance(Xtr_m_train, Xte_m):
    """(N, D) exceedance of the guard statistics in range widths (Equation 14); -inf where fewer than two values"""
    m_tr, l_tr, c_tr = s5.vis_stats(Xtr_m_train); ok = c_tr >= 2
    mlo, mhi = np.where(ok, m_tr, np.inf).min(0), np.where(ok, m_tr, -np.inf).max(0)
    llo, lhi = np.where(ok, l_tr, np.inf).min(0), np.where(ok, l_tr, -np.inf).max(0)
    m, l, c = s5.vis_stats(Xte_m)
    exm = np.maximum(mlo - m, m - mhi) / np.maximum(mhi - mlo, s5.RANGE_FLOOR)
    exl = np.maximum(llo - l, l - lhi) / np.maximum(lhi - llo, s5.RANGE_FLOOR)
    return np.where(c >= 2, np.maximum(exm, exl), -np.inf)


# ------------------------------------------------------------------------------------------- part A
class Sink:
    """per-unit sums of the errors of one evaluation context"""
    def __init__(self):
        self.rows, self.keys = [], []

    def add(self, key, e, ui, nu, sd_eq2_d):
        q = e / sd_eq2_d
        s = np.stack([np.bincount(ui, minlength=nu).astype(float), np.bincount(ui, e * e, nu), np.bincount(ui, np.abs(e), nu),
                      np.bincount(ui, q * q, nu), np.bincount(ui, np.abs(q), nu)], 1)
        self.rows.append(s); self.keys.append(key)

    def frame(self, cols):
        out = []
        for key, s in zip(self.keys, self.rows):
            k, units = key[:-1], key[-1]; keep = s[:, 0] > 0
            f = pd.DataFrame(s[keep], columns=["n", "sse_rep", "sae_rep", "sse_eq2", "sae_eq2"]); f.insert(0, "unit", units[keep])
            for c, v in zip(reversed(cols), reversed(k)): f.insert(0, c, v)
            out.append(f)
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _units(S, n):
    u = S["meta"].unit.to_numpy()[S["ite"]]; uu, inv = np.unique(u, return_inverse=True)
    return inv[n], len(uu), uu


def _load(ds, mech, lv, seed, method, variant, init2=False):
    if ds == s9.SKAB or init2: return s9._load9(s9._job(ds, mech, lv, seed, method, variant, 0, init2=init2))[0]
    return s8._load_pred(s8._job(ds, mech, lv, seed, method, variant, 0))[0]


def analyze():
    os.makedirs(OUT, exist_ok=True)
    if not os.path.exists(os.path.join(OUT, "PLAN_step10.md")): sys.exit("The plan is missing: python step10_followup.py plan")
    dc, _, _ = s5.modules(); t0 = time.time()
    sink = Sink(); cols = ["part", "dataset", "cond", "seed", "model", "variant"]
    frows, ex, scales = [], {}, []
    settings = [(ds, mech, lv, seed, "main") for seed in SEEDS for ds in s8.DATASETS + [s9.SKAB] for (mech, lv) in s8.CONDS]
    settings += [(ds, mech, lv, seed, "sweep") for seed in SEEDS for ds in s8.SWEEP_DS for (mech, lv) in s8.SWEEP]
    for i, (ds, mech, lv, seed, part) in enumerate(settings, 1):
        cond = f"{mech} {lv:g}"
        S, Mtr, Mte, es, cal, mu, sd = s8.parts(dc, ds, mech, lv, seed)
        mur, sdr = s8.scale_of(ds, seed, S)
        scales += [dict(dataset=ds, cond=cond, seed=seed, var=k, mu_eq2=mu[k], sd_eq2=sd[k], mu_rep=mur[k], sd_rep=sdr[k]) for k in range(len(sd))]
        Za_m = dc.apply_mask((S["Xa"] - mu) / sd, Mtr); Zte_m = dc.apply_mask((S["Xte"] - mu) / sd, Mte)
        y = S["Xte"][Mte]; n, _, d = np.nonzero(Mte); D = S["Xa"].shape[2]
        y_c = S["Xa"][cal][Mtr[cal]]; _, _, d_c = np.nonzero(Mtr[cal])
        guard = s5.window_guard(Za_m[S["tr"]], Zte_m); ui, nu, uu = _units(S, n); sdd = sd[d]
        xd = None
        if ds == "CMAPSS" and part == "main":
            Xx, Mx, Sm, (mu_c, sd_c, mu_m, sd_m) = s8.cross_domain(dc, mech, lv, seed)
            gx = s5.window_guard(Za_m[S["tr"]], dc.apply_mask((Xx - mu) / sd, Mx)); nx, _, dxv = np.nonzero(Mx)
            to_m = lambda Q, dxv=dxv: ((Q * sd_c[dxv] + mu_c[dxv]) - mu_m[dxv]) / sd_m[dxv]
            sd_t = s8.parts(dc, "CMAPSS_M", mech, lv, seed)[6]; uix, nux, uux = _units(Sm, nx)
            xd = (gx, nx, dxv, Sm["Xte"][Mx], to_m, uix, nux, uux, sd_t[dxv])
        fail = None
        if ds == "MetroPT" and part == "main":
            fw = S["meta"].failure.to_numpy()[S["ite"]]; fail = fw[n]
            near = np.flatnonzero(fw == 1); idx = S["ite"]
            pick = sorted({w for f in near for w in range(len(idx)) if abs(int(idx[w]) - int(idx[f])) <= 2})
            if pick:
                ex[f"{cond}|{seed}|windows"] = np.array(pick); ex[f"{cond}|{seed}|start"] = S["meta"].start.to_numpy()[idx[pick]].astype(str)
                ex[f"{cond}|{seed}|truth"] = (S["Xte"][pick] * sdr + mur).astype(np.float32); ex[f"{cond}|{seed}|mask"] = Mte[pick]
                ex[f"{cond}|{seed}|failure"] = fw[pick]
        base = (part, ds, cond, seed)
        def put(mdl, v, p, keyset=None):
            e = p - y; sink.add(base + (mdl, v, uu), e, ui, nu, sdd)
            if fail is not None and v in EX_VARIANTS:
                er = e * sdr[d]; key = (ui * 2 + fail) * D + d; L = nu * 2 * D
                agg = np.stack([np.bincount(key, minlength=L).astype(float), np.bincount(key, er, L), np.bincount(key, er * er, L),
                                np.bincount(key, np.abs(er), L)], 1)
                for kk in np.flatnonzero(agg[:, 0] > 0):
                    u_, r_ = divmod(kk, 2 * D); f_, v_ = divmod(r_, D)
                    frows.append((cond, seed, mdl, v, uu[u_], f_, v_, *agg[kk]))
                if mdl in EX_MODELS and f"{cond}|{seed}|windows" in ex and v in ("global", "seq", "normblend"):
                    pk = ex[f"{cond}|{seed}|windows"]; F = np.full(S["Xte"].shape, np.nan); F[Mte] = p
                    ex[f"{cond}|{seed}|{mdl}|{v}"] = (F[pk] * sdr + mur).astype(np.float32)
        z = _load(ds, mech, lv, seed, "Linear regression", "global") if ds != s9.SKAB else None
        if z is not None: put("Linear regression", "-", z["te"].astype(float))
        for mdl in (s8.SWEEP_MODELS if part == "sweep" else s8.BACKBONES):
            zg, zs = _load(ds, mech, lv, seed, mdl, "global"), _load(ds, mech, lv, seed, mdl, "seq")
            if zg is None or zs is None: continue
            g, s_, cg, cs = zg["te"].astype(float), zs["te"].astype(float), zg["cal"].astype(float), zs["cal"].astype(float)
            a = s5.dual_weights(y_c, cg, cs, d_c, D); asel = sel_weights(y_c, cg, cs, d_c, D)
            V = s8._variants(g, s_, a, d, guard[n], cg, cs, y_c, d_c); V.update(extra_variants(g, s_, asel, d, guard[n]))
            for v, p in V.items(): put(mdl, v, p)
            if xd is not None and "xd" in zg.files and "xd" in zs.files:
                gx, nx, dxv, y_x, to_m, uix, nux, uux, sdx = xd
                gg, ss = to_m(zg["xd"].astype(float)), to_m(zs["xd"].astype(float))
                Vx = s8._variants(gg, ss, a, dxv, gx[nx], cg, cs, y_c, d_c); Vx.update(extra_variants(gg, ss, asel, dxv, gx[nx]))
                for v, p in Vx.items(): sink.add(("shift", "CMAPSS->CMAPSS_M", cond, seed, mdl, v, uux), p - y_x, uix, nux, sdx)
        if part == "main":
            for mdl, var in [(m, "gstat") for m in s8.GSTAT] + [(m, "own") for m in s8.OWN]:
                z = _load(ds, mech, lv, seed, mdl, var)
                if z is None: continue
                name = mdl if var == "own" else f"{mdl}+stats"; put(name, var, z["te"].astype(float))
                if xd is not None and "xd" in z.files:
                    gx, nx, dxv, y_x, to_m, uix, nux, uux, sdx = xd
                    sink.add(("shift", "CMAPSS->CMAPSS_M", cond, seed, name, var, uux), to_m(z["xd"].astype(float)) - y_x, uix, nux, sdx)
        if part == "main" and ds in s9.ENS_DATASETS:                                      # E6
            for mdl in s9.ENS_MODELS:
                zz = {"g1": _load(ds, mech, lv, seed, mdl, "global"), "s1": _load(ds, mech, lv, seed, mdl, "seq"),
                      "g2": _load(ds, mech, lv, seed, mdl, "global", True), "s2": _load(ds, mech, lv, seed, mdl, "seq", True)}
                if any(v is None for v in zz.values()): continue
                P = {"te": {k: v["te"].astype(float) for k, v in zz.items()}, "cal": {k: v["cal"].astype(float) for k, v in zz.items()}}
                for v, p in s9._ens_variants(P, y_c, d_c, D, d, guard[n]).items():
                    sink.add(("E6", ds, cond, seed, mdl, v, uu), p - y, ui, nu, sdd)
                if xd is not None and all("xd" in v.files for v in zz.values()):
                    gx, nx, dxv, y_x, to_m, uix, nux, uux, sdx = xd
                    Px = {"te": {k: to_m(v["xd"].astype(float)) for k, v in zz.items()}, "cal": P["cal"]}
                    for v, p in s9._ens_variants(Px, y_c, d_c, D, dxv, gx[nx]).items():
                        sink.add(("E6shift", "CMAPSS->CMAPSS_M", cond, seed, mdl, v, uux), p - y_x, uix, nux, sdx)
        if i % 10 == 0 or i == len(settings): print(f"[{i}/{len(settings)}] {ds} {cond} seed {seed}: {time.time() - t0:.0f} s", flush=True)
    U = sink.frame(cols); U.to_csv(os.path.join(OUT, "step10_unit_errors.csv.gz"), index=False)
    pd.DataFrame(scales).to_csv(os.path.join(OUT, "step10_scales.csv"), index=False)
    pd.DataFrame(frows, columns=["cond", "seed", "model", "variant", "unit", "failure", "var", "n", "sum_e", "sum_e2", "sum_abs"]).to_csv(
        os.path.join(OUT, "step10_metropt_failure.csv.gz"), index=False)
    np.savez_compressed(os.path.join(OUT, "step10_metropt_windows.npz"), **ex)
    check_against_stored(U)
    print(f"part A done in {time.time() - t0:.0f} s; rows {len(U)}")


def check_against_stored(U):
    """the RMSE and MAE recomputed from the unit sums must equal the stored Step 8 and 9 tables"""
    g = U.groupby(["part", "dataset", "cond", "seed", "model", "variant"])[["n", "sse_rep", "sae_rep"]].sum()
    g["rmse"] = np.sqrt(g.sse_rep / g.n); g["mae"] = g.sae_rep / g.n; g = g.reset_index()
    R8, X8 = pd.read_csv(os.path.join(s8.OUT, "step8_runs.csv")), pd.read_csv(os.path.join(s8.OUT, "step8_cross.csv"))
    R9, X9 = pd.read_csv(os.path.join(s9.OUT, "step9_runs.csv")), pd.read_csv(os.path.join(s9.OUT, "step9_cross.csv"))
    sweep = {f"{m} {l:g}" for m, l in s8.SWEEP}
    ref = {"main": pd.concat([R8[~R8.cond.isin(sweep)], R9[R9.exp == "E7"]]), "sweep": R8[R8.cond.isin(sweep)],
           "shift": X8, "E6": R9[R9.exp == "E6"], "E6shift": X9}
    out = []
    for part, r in ref.items():
        m = g[g.part == part].merge(r, on=["dataset", "cond", "seed", "model", "variant"], suffixes=("", "_ref"))
        out.append(dict(part=part, matched=len(m), stored=len(r), max_rmse_diff=float((m.rmse - m.rmse_ref).abs().max()) if len(m) else np.nan,
                        max_mae_diff=float((m.mae - m.mae_ref).abs().max()) if len(m) else np.nan))
    C = pd.DataFrame(out); C.to_csv(os.path.join(OUT, "step10_check.csv"), index=False); print(C.to_string(index=False))
    ok = bool((C.matched == C.stored).all() and (C.max_rmse_diff < 1e-9).all() and (C.max_mae_diff < 1e-9).all())
    print("part A check against the stored tables:", "passed" if ok else "FAILED")


# ------------------------------------------------------------------------------------------- part B
def _job(ds, mech, lv, seed, method):
    return dict(dataset=ds, mech=mech, level=lv, seed=seed, method=method, id=f"{ds}|{mech}|{lv}|{seed}|{method}|shift")


def job_list():
    jobs = []
    for seed in SEEDS:
        for (mech, lv) in s8.CONDS:
            for ds in SHIFT_DATASETS:
                jobs += [_job(ds, mech, lv, seed, m) for m in SHIFT_MODELS]
    return jobs


def res_path(j, ext, out=None):
    return os.path.join(out or OUT, "results", hashlib.md5(j["id"].encode()).hexdigest() + ext)


def shifted(Z, dw, k, scen):
    return s7.stressed(Z, dw, k) if scen == "one" else Z + k


def impute_view(dc, v2, ev, method, norm, Za_m, Za, Mtr, tr, es, inputs, s):
    """the Step 8 training and imputation of one view (s8.run_job without saving); returns imputations on the z scale"""
    if norm == "global":
        Ztr, Zes_true, Mes = Za_m, Za[es], Mtr[es]
        prep = lambda Z: (Z, (lambda W: W))
    else:
        m_tr, s_tr = s5.seq_stats(Za_m); Ztr = (Za_m - m_tr) / s_tr
        m_es, s_es = s5.seq_stats(Za_m[es]); Zes_true = (Za[es] - m_es) / s_es; Mes = Mtr[es]
        def prep(Z):
            m, sc = s5.seq_stats(Z)
            return (Z - m) / sc, (lambda W, m=m, sc=sc: W * sc + m)
    P = {n: prep(Z) for n, Z in inputs.items()}
    if method == "KNN":
        ref = Ztr[np.r_[tr, es]]; Zi = {n: dc.impute_knn(ref, z, dc.CFG["knn_k"]) for n, (z, _) in P.items()}
    else:
        Zi, _ = s8.fit_impute(dc, v2, ev, method, Ztr[tr], Zes_true, Mes, {n: z for n, (z, _) in P.items()}, s)
    return {n: P[n][1](Zi[n]) for n in P}


def run_job(j, dc, v2, ev, out=None):
    ds, mech, lv, seed, method = j["dataset"], j["mech"], j["level"], j["seed"], j["method"]
    S, Mtr, Mte, es, cal, mu, sd = s8.parts(dc, ds, mech, lv, seed)
    Xa, Xte, tr = S["Xa"], S["Xte"], S["tr"]
    if s8.FAST["on"]: tr, es, cal, Xte, Mte = tr[:64], es[:16], cal[:16], Xte[:24], Mte[:24]
    fw = lambda Z: (Z - mu) / sd
    bw = lambda Z: Z * sd + mu
    Za, Zte = fw(Xa), fw(Xte); D = Xa.shape[2]
    Za_m, Zte_m = dc.apply_mask(Za, Mtr), dc.apply_mask(Zte, Mte)
    dw = (np.arange(len(Xte)) + seed) % D                                                       # as in Step 7
    inputs = {"te": Zte_m, "cal": Za_m[cal]}
    for sc in SCENARIOS:
        for k in K_GRID: inputs[f"{sc}{k:g}"] = dc.apply_mask(shifted(Zte, dw, k, sc), Mte)
    s = s5.sseed("step5-init", ds, mech, lv, seed, method); t0 = time.time()
    imp = {}
    for norm in s8.NORMS:
        Z = impute_view(dc, v2, ev, method, norm, Za_m, Za, Mtr, tr, es, inputs, s)
        imp[norm] = {n: np.where(np.isnan(inputs[n]), bw(Z[n]), bw(inputs[n])) for n in Z}      # reporting scale
    secs = time.time() - t0
    masks = {n: (Mtr[cal] if n == "cal" else Mte) for n in inputs}
    G = {n: imp["global"][n][masks[n]] for n in inputs}; Sv = {n: imp["seq"][n][masks[n]] for n in inputs}
    res = dict(j, seconds_train_impute=secs, n_test_windows=int(len(Xte)))
    if not s8.FAST["on"]:                                                                      # reproduction of Step 7/8
        f32 = lambda x: x.astype(np.float32).astype(float)                                     # stored as float32
        for norm, A in (("global", G), ("seq", Sv)):
            z, src = s8._load_pred(s8._job(ds, mech, lv, seed, method, norm, 0))
            res[f"repro_{norm}"] = float(np.abs(z["te"].astype(float) - f32(A["te"])).max()) if z is not None else None
            res[f"repro_{norm}_source"] = src
            q = s7.res_path(dict(id=f"{ds}|{mech}|{lv}|{seed}|{method}|{norm}"), ".npz")
            if os.path.exists(q):
                z7 = np.load(q)
                for k in s7.K_STRESS:
                    if f"k{k}" in z7.files and f"one{k:g}" in A:
                        res[f"repro_step7_k{k}_{norm}"] = float(np.abs(z7[f"k{k}"].astype(float) - f32(A[f"one{k:g}"])).max())
    y = Xte[Mte]; n, _, d = np.nonzero(Mte)
    y_c = Xa[cal][Mtr[cal]]; _, _, d_c = np.nonzero(Mtr[cal])
    a = s5.dual_weights(y_c, G["cal"], Sv["cal"], d_c, D); asel = sel_weights(y_c, G["cal"], Sv["cal"], d_c, D)
    Eg, Es = ((G["cal"] - y_c) ** 2).mean(), ((Sv["cal"] - y_c) ** 2).mean()
    rows = []
    for name in ["te"] + [f"{sc}{k:g}" for sc in SCENARIOS for k in K_GRID]:
        sc, k = ("none", 0.0) if name == "te" else (name[:3], float(name[3:]))
        hit = np.ones(len(y), bool) if sc == "all" else (d == dw[n]) if sc == "one" else np.zeros(len(y), bool)
        yk = y + np.where(hit, k * sd[d], 0.0)
        ex = exceedance(Za_m[tr], inputs[name]); wmax = ex.max(1)
        assert np.array_equal(wmax > 1.0, s5.window_guard(Za_m[tr], inputs[name]))
        g, s_ = G[name], Sv[name]
        V = {"global": g, "seq": s_, "nb_noguard": a[d] * g + (1 - a[d]) * s_, "valsel": g if Eg <= Es else s_}
        info = {}
        for kap in KAPPAS:
            gw = (wmax > kap)[n]; gv = (ex > kap)[n, d]
            aw, av = np.where(gw, 0.0, a[d]), np.where(gv, 0.0, a[d])
            V[f"nb_k{kap:g}"] = aw * g + (1 - aw) * s_; V[f"nbvar_k{kap:g}"] = av * g + (1 - av) * s_
            info[f"guard_k{kap:g}"] = float((wmax > kap).mean()); info[f"guardvar_k{kap:g}"] = float((ex > kap).mean())
        gw1 = (wmax > 1.0)[n]
        V["normblend"] = V["nb_k1"]; V.update(extra_variants(g, s_, asel, d, gw1))
        for v, p in V.items():
            e2 = (p - yk) ** 2
            rows.append(dict(scenario=sc, k=k, variant=v, rmse=float(np.sqrt(e2.mean())), mae=float(np.abs(p - yk).mean()),
                             rmse_shifted=float(np.sqrt(e2[hit].mean())) if hit.any() else np.nan,
                             rmse_other=float(np.sqrt(e2[~hit].mean())) if (~hit).any() else np.nan, **info))
    res["rows"] = rows
    return res


def _attempt(j, dc, v2, ev, out, errdir):
    res = None
    for strict in (True, False):
        s5.strict_mode(strict)
        try:
            res = run_job(j, dc, v2, ev, out); res["deterministic"] = strict; break
        except RuntimeError as e:
            if strict and "deterministic" in str(e).lower(): continue
            open(os.path.join(errdir, hashlib.md5(j["id"].encode()).hexdigest() + ".txt"), "w").write(j["id"] + "\n" + traceback.format_exc()); break
        except Exception:
            open(os.path.join(errdir, hashlib.md5(j["id"].encode()).hexdigest() + ".txt"), "w").write(j["id"] + "\n" + traceback.format_exc()); break
    return res


def run(worker=0, n_workers=1):
    plan_p = os.path.join(OUT, "PLAN_step10.md")
    if not os.path.exists(plan_p): sys.exit("The plan is missing: python step10_followup.py plan")
    sha = script_sha()
    if sha not in open(plan_p, encoding="utf-8").read():
        print("WARNING: this script differs from the version recorded in PLAN_step10.md; document the change.")
    os.makedirs(os.path.join(OUT, "results"), exist_ok=True); os.makedirs(os.path.join(OUT, "errors"), exist_ok=True)
    dc, v2, ev = s5.modules(); import torch
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // max(1, n_workers)))
    jobs = [j for i, j in enumerate(job_list()) if i % n_workers == worker]
    for i, j in enumerate(jobs, 1):
        p = res_path(j, ".json")
        if os.path.exists(p): continue
        print(f"[{i}/{len(jobs)}] {j['id']}", flush=True); t0 = time.time()
        res = _attempt(j, dc, v2, ev, None, os.path.join(OUT, "errors"))
        if res is not None:
            res.update(seconds=round(time.time() - t0, 1), script_sha256=sha); tmp = p + ".tmp"
            json.dump(res, open(tmp, "w"), indent=1, default=float); os.replace(tmp, p)
    json.dump(s8.environment(), open(os.path.join(OUT, "environment.json"), "w"), indent=1)


def status():
    jobs = job_list(); done = [os.path.exists(res_path(j, ".json")) for j in jobs]
    errs = glob.glob(os.path.join(OUT, "errors", "*.txt"))
    print(f"finished {sum(done)} of {len(jobs)} jobs; errors: {len(errs)}")
    print(pd.DataFrame(jobs).assign(done=done).groupby(["dataset", "method"]).done.agg(["sum", "count"]).to_string())
    for e in errs[:5]: print("  ", open(e).read().splitlines()[0])


def check():
    """CPU: KNN in two settings; the unshifted test imputations must equal the stored Step 8 imputations"""
    dc, _, _ = s5.modules(); bad = 0
    for ds, mech, lv, seed in (("CMAPSS", "MNAR", 0.3, 0), ("MetroPT", "MCAR", 0.7, 3)):
        r = run_job(_job(ds, mech, lv, seed, "KNN"), dc, None, None)
        for norm in s8.NORMS:
            dlt = r.get(f"repro_{norm}"); bad += (dlt is None) or (dlt > 1e-6)
            print(f"{ds} {mech} {lv} seed {seed} KNN {norm}: max |difference| to the stored imputation = {dlt} ({r.get(f'repro_{norm}_source')})")
        st7 = {k: v for k, v in r.items() if k.startswith("repro_step7")}
        if st7: print("   Step 7 stress files:", st7); bad += any(v > 1e-6 for v in st7.values())
        te = [x for x in r["rows"] if x["scenario"] == "none" and x["variant"] in ("global", "seq", "normblend")]
        print("   unshifted RMSE:", {x["variant"]: round(x["rmse"], 6) for x in te})
    print("reproduction check:", "passed" if bad == 0 else "FAILED")


def smoke():
    s8.FAST["on"] = True
    dc, v2, ev = s5.modules(); out = os.path.join(OUT, "smoke"); err = os.path.join(out, "errors")
    os.makedirs(err, exist_ok=True); ok = 0
    todo = [_job("CMAPSS", "MCAR", 0.3, 0, m) for m in SHIFT_MODELS] + [_job("MetroPT", "GAP", 0.4, 1, "SAITS")]
    for j in todo:
        t0 = time.time(); r = _attempt(j, dc, v2, ev, out, err)
        if r is None: print(f"FAILED  {j['id']} (see {err})"); continue
        ok += 1; q = pd.DataFrame(r["rows"]); q = q[(q.scenario == "one") & (q.k == 10)].set_index("variant").rmse
        print(f"ok      {j['id']}: {time.time() - t0:5.1f} s, one-variable shift k=10: G {q['global']:.3f} S {q['seq']:.3f} NB {q['normblend']:.3f}")
    print(f"smoke test: {ok} of {len(todo)} passed")


def report():
    jr = [json.load(open(p)) for p in glob.glob(os.path.join(OUT, "results", "*.json"))]
    if not jr: sys.exit("no results yet")
    rows = [dict(dataset=r["dataset"], cond=f"{r['mech']} {r['level']:g}", seed=r["seed"], model=r["method"], **x) for r in jr for x in r["rows"]]
    R = pd.DataFrame(rows); R.to_csv(os.path.join(OUT, "step10_shift_runs.csv"), index=False)
    J = pd.DataFrame([{k: v for k, v in r.items() if k != "rows"} for r in jr]); J.to_csv(os.path.join(OUT, "step10_jobs.csv"), index=False)
    rep = [c for c in J.columns if c.startswith("repro_") and not c.endswith("_source")]
    L = ["# Step 10 report, part B (shift magnitude; descriptive)", "", f"Script versions in the results: {sorted(J.script_sha256.unique())}",
         f"Jobs: {len(J)} of {len(job_list())}; runs without strict determinism: {int((~J.deterministic.astype(bool)).sum())}", "",
         "Reproduction (max |difference| to the stored imputations): " + "; ".join(f"{c} {J[c].max():.2e}" for c in rep if J[c].notna().any()), ""]
    R0 = R[R.scenario == "none"]
    Rk = pd.concat([R[R.scenario != "none"]] + [R0.assign(scenario=sc) for sc in SCENARIOS])     # k = 0 in both
    m = Rk.groupby(["dataset", "scenario", "k", "model", "variant"]).rmse.mean().unstack()
    for sc in ["one", "all"]:
        L += [f"## Mean RMSE, offset in {'one variable per window' if sc == 'one' else 'every variable'}", "", "```",
              m.xs(sc, level="scenario")[["global", "seq", "nb_noguard", "normblend", "nbvar_k1", "nb_k0.25", "global_guard", "sel_var"]].round(3).to_string(),
              "```", ""]
    gcols = [c for c in R.columns if c.startswith("guard")]
    L += ["## Guard shares (mean over conditions and seeds)", "", "```", R.groupby(["dataset", "scenario", "k"])[gcols].mean().round(3).to_string(), "```", ""]
    txt = "\n".join(L); open(os.path.join(OUT, "REPORT_step10.md"), "w", encoding="utf-8").write(txt); print(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "analyze", "check", "smoke", "run", "status", "report"])
    ap.add_argument("--worker", type=int, default=0); ap.add_argument("--n-workers", type=int, default=1)
    a = ap.parse_args()
    if a.cmd == "plan": plan()
    elif a.cmd == "analyze": analyze()
    elif a.cmd == "check": check()
    elif a.cmd == "smoke": smoke()
    elif a.cmd == "run": run(a.worker, a.n_workers)
    elif a.cmd == "status": status()
    else: report()
