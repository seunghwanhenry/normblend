"""
step7_robustness.py - Step 7: post hoc robustness re-analysis of Step 5 (written after a pre-submission review, before
any Step 7 run). Uses the Step 5 data, splits, masks, seeds, models and model seeds (functions imported unchanged from
step5_confirm.py, SHA-256 f3492a1c5edd50c845b4509ed4108883f64e8390ac4748c4164f32d72a08658d).

What changes against Step 5 ("clean" protocol)
  1. Global standardization uses the mean and standard deviation of the OBSERVED values of the TRAINING units only
     (Step 5: all values of the training and early-stopping windows, including the values removed later).
  2. The early-stopping units are split into two halves by unit: one half selects the checkpoint of SAITS and the
     transformer (and is the KNN reference set together with the training windows), the other half ("calibration")
     only estimates the dual weights (Step 5: the same early-stopping windows did both).
  Everything is reported on the Step 5 reporting scale (Step 5 standardization), which no model uses, so that the
  numbers are comparable with Step 5.
Stress test (post hoc, descriptive): every model also imputes copies of the test windows in which one variable per
window (variable (window index + seed) mod D) is shifted by k standard deviations of the observed training values,
k in K_STRESS. The ground truth is shifted in the same way.

Commands (from the DTIN folder):
  python step7_robustness.py plan       # writes step7_outputs/PLAN_step7.md (done once, before any run)
  python step7_robustness.py check      # reproduction check: Step 5 protocol, KNN, two settings; must equal Step 5
  python step7_robustness.py units      # CPU, a few minutes: unit-level analysis of the existing Step 5 predictions
  python step7_robustness.py run [--worker k --n-workers K]
  python step7_robustness.py status
  python step7_robustness.py report
"""
import argparse, glob, hashlib, importlib.util, json, os, platform, sys, time, traceback
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("step5_confirm", os.path.join(HERE, "step5_confirm.py"))
s5 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(s5)

OUT = "step7_outputs"
STEP5_SHA = "f3492a1c5edd50c845b4509ed4108883f64e8390ac4748c4164f32d72a08658d"
K_STRESS = [3, 10, 100]
N_BOOT = 2000
CHECK_KEYS = ["CMAPSS|MNAR|0.3|0", "MetroPT|MCAR|0.3|3"]          # reproduction check settings
WITHIN, WITHIN_SHARE, GAIN, WORST = s5.WITHIN, s5.WITHIN_SHARE, s5.GAIN, s5.WORST     # 1.02, 0.90, 0.98, 1.10

PLAN = """# Step 7 - robustness re-analysis of Step 5 (post hoc; decision rules fixed before any Step 7 run)

Written after a pre-submission review of the manuscript and before any Step 7 job was run. Script step7_robustness.py,
SHA-256 {sha}; it imports the unchanged step5_confirm.py (SHA-256 {sha5}).

## Status of the data
The Step 5 data (C-MAPSS FD001 + FD003, MetroPT-3) and the Step 5 and Step 6 results are known. Step 7 is therefore
not an independent confirmation; it asks whether the Step 5 conclusions depend on features of the Step 5 protocol.
Done before this plan, from existing files only: the size of the statistics difference (below), validation-selected
baselines from the Step 5 and Step 6 predictions (reported separately as post hoc), and the reproduction check of this
script (`check`: Step 5 protocol, KNN, four jobs; the test imputations equal the Step 5 files exactly). No job of the
clean protocol had been run on real data.

## Why
1. In Step 5 the global mean and standard deviation of each variable were computed from all values of the training
   and early-stopping windows before any value was removed. A deployed system only has the observed values, and the
   early-stopping windows also selected the checkpoints and estimated the dual weights. Under MNAR, the observed-only
   statistics of the training units differ by a median of about -0.2 standard deviations in the mean and a factor of
   about 0.9 in the standard deviation; under MCAR and gaps by at most 0.07 standard deviations and a factor of 0.89-1.08
   (computed from the data and masks only; no model was run).
2. The same early-stopping windows chose the checkpoints and estimated the dual weights.
3. The window guard was almost never active in Step 5, so its effect on new data was not tested.

## Protocol ("clean"; everything not listed is identical to Step 5)
* Global standardization: mean and standard deviation of the observed (unmasked) values of the training units' windows
  (standard deviation 1 if it is 0 or undefined, mean 0 if no value is observed).
* The early-stopping units of each seed are split by unit into two halves (seeded permutation; the calibration half has
  floor(n/2) units): ES windows select the checkpoint of SAITS and the transformer and, with the training windows, form
  the KNN reference set; CAL windows are imputed by every model and only estimate the dual weights.
* Dual weights, window guard (kappa = 1, ranges from the masked training windows) and combination as in Step 5.
* Same model seeds as Step 5 (common random numbers). Metric: RMSE at the removed test positions on the Step 5 reporting
  scale (Step 5 standardization, used by no model).
* Stress test (descriptive): copies of the test windows with one variable per window shifted by k standard deviations of
  the observed training values, k in {ks}; every model imputes them without retraining.

## Decision rules (24 cells = dataset x condition x model, mean RMSE over seeds 0-9; oracle = better of global and seq)
* R1: dual <= {w} x oracle in at least {ws:.0%} of the cells.
* R2: geometric mean over the cells of dual / seq <= {g} and of dual / global <= {g}.
* R3: dual / oracle <= {wo} in every cell.
If R1-R3 hold, the manuscript states that the Step 5 conclusions do not depend on the Step 5 statistics or on the shared
early-stopping windows. If any fails, the manuscript reports the Step 7 result next to Step 5 and qualifies the claims.

## Reported descriptively
Ratio of every variant's cell RMSE to Step 5; dual weights; guard activation on the normal and the stressed test windows;
stress-test RMSE of global, seq, dual without guard and dual (all removed positions, and the shifted variable only).
Unit level, for Step 5 and Step 7: per-unit RMSE; bootstrap over test units (resampled within each seed, the same draw
for all cells of a dataset; {nb} replicates; percentile 95% intervals) of the geometric-mean ratios per dataset and of
the number of cells within {w} x oracle; median, 95th percentile and maximum over units of dual / better normalization.
"""


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def script_sha():
    return sha_file(os.path.abspath(__file__))


# ------------------------------------------------------------------------------------------- protocol pieces
def es_cal(ds, S, seed):
    """split the early-stopping windows by unit: (ES windows, CAL windows), indices into S['Xa']"""
    st = S["st"]; units = S["meta"].unit.to_numpy()[S["itrain"]][st]
    u = np.random.default_rng(s5.sseed("step7-calib", ds, seed)).permutation(np.sort(np.unique(units)))
    cal_u = set(u[: len(u) // 2].tolist())
    is_cal = np.array([x in cal_u for x in units])
    return st[~is_cal], st[is_cal]


def clean_stats(Xa, Mtr, tr):
    """mean and sd of the observed values of the training windows (on the Step 5 reporting scale)"""
    Z = Xa[tr].astype(float).copy(); Z[Mtr[tr]] = np.nan
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mu = np.nanmean(Z, axis=(0, 1)); sd = np.nanstd(Z, axis=(0, 1))
    mu = np.where(np.isfinite(mu), mu, 0.0); sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
    return mu, sd


def protocol_parts(dc, ds, mech, lv, seed, protocol):
    S = s5.setting(dc, ds, seed); Mtr, Mte = s5.masks(dc, ds, S, mech, lv, seed); D = S["Xa"].shape[2]
    if protocol == "clean":
        es, cal = es_cal(ds, S, seed); mu, sd = clean_stats(S["Xa"], Mtr, S["tr"])
    else:                                                         # Step 5 protocol (reproduction)
        es = cal = S["st"]; mu, sd = np.zeros(D), np.ones(D)
    dw = (np.arange(len(S["Xte"])) + seed) % D                    # stressed variable of each test window
    return S, Mtr, Mte, es, cal, mu, sd, dw


def stressed(Z, dw, k):
    Zk = Z.copy(); Zk[np.arange(len(Z)), :, dw] += k
    return Zk


def job_list():
    return s5.job_list()                                          # same 560 jobs and ids as Step 5


def res_path(j, ext, out=None):
    return os.path.join(out or OUT, "results", hashlib.md5(j["id"].encode()).hexdigest() + ext)


# ------------------------------------------------------------------------------------------- one job
def run_job(j, dc, v2, ev, protocol="clean", out=None):
    ds, mech, lv, seed, method, norm = j["dataset"], j["mech"], j["level"], j["seed"], j["method"], j["norm"]
    S, Mtr, Mte, es, cal, mu, sd, dw = protocol_parts(dc, ds, mech, lv, seed, protocol)
    Xa, Xte, tr, st = S["Xa"], S["Xte"], S["tr"], S["st"]
    fw = lambda Z: (Z - mu) / sd
    bw = lambda Z: Z * sd + mu
    Za, Zte = fw(Xa), fw(Xte)
    Za_m, Zte_m = dc.apply_mask(Za, Mtr), dc.apply_mask(Zte, Mte)
    inputs = {"te": Zte_m, "cal": Za_m[cal]}
    if protocol == "clean":
        for k in K_STRESS: inputs[f"k{k}"] = dc.apply_mask(stressed(Zte, dw, k), Mte)
    s = s5.sseed("step5-init", ds, mech, lv, seed, method); info = {}
    if method == "Linear regression":
        imp = {n: dc.impute_lr_time(Z) for n, Z in inputs.items()}
    else:
        if norm == "global":
            Ztr, Zes_true = Za_m, Za[es]
            prep = lambda Z: (Z, (lambda W: W))
        else:
            m_tr, s_tr = s5.seq_stats(Za_m); Ztr = (Za_m - m_tr) / s_tr
            m_es, s_es = s5.seq_stats(Za_m[es]); Zes_true = (Za[es] - m_es) / s_es
            def prep(Z):
                m, sc = s5.seq_stats(Z)
                return (Z - m) / sc, (lambda W, m=m, sc=sc: W * sc + m)
        P = {n: prep(Z) for n, Z in inputs.items()}
        if method == "KNN":
            kk = dc.CFG["knn_k"]
            if protocol == "clean":
                ref = Ztr[np.r_[tr, es]]; Zi = {n: dc.impute_knn(ref, z, kk) for n, (z, _) in P.items()}
            else:
                nst = np.setdiff1d(np.arange(len(Xa)), st)
                Zi = {n: dc.impute_knn(Ztr[nst] if n == "cal" else Ztr, z, kk) for n, (z, _) in P.items()}
        elif method == "SAITS":
            names = list(P); sizes = [len(P[n][0]) for n in names]
            Zall, val = ev.pypots_fit_impute("SAITS", ev.select_saits()["cfg"], Ztr[tr], Zes_true, Mtr[es],
                                             np.concatenate([P[n][0] for n in names]), s)
            Zi = dict(zip(names, np.split(Zall, np.cumsum(sizes)[:-1]))); info["val_es_rmse_model_space"] = val
        elif method == "Transformer":
            model = v2.DTINv2Imputer(v2.imputation_config(**s5.TRANSFORMER), seed=s).fit(Ztr[tr], Zes_true, Mtr[es])
            Zi = {n: model.impute(z) for n, (z, _) in P.items()}; del model
        else:
            raise ValueError(method)
        imp = {n: P[n][1](Zi[n]) for n in P}
    imp = {n: np.where(np.isnan(inputs[n]), bw(imp[n]), bw(inputs[n])) for n in imp}   # back to the reporting scale
    masks = {"te": Mte, "cal": Mtr[cal], **{f"k{k}": Mte for k in K_STRESS}}
    arrs = {n: imp[n][masks[n]].astype(np.float32) for n in imp}
    if method != "Linear regression":
        try:
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except ImportError:
            pass
    os.makedirs(os.path.join(out or OUT, "results"), exist_ok=True)
    np.savez_compressed(res_path(j, ".npz", out), **arrs)
    res = dict(j, protocol=protocol, rmse=dc.rmse_at(Xte, imp["te"], Mte), rmse_cal=dc.rmse_at(Xa[cal], imp["cal"], Mtr[cal]),
               n_es_windows=int(len(es)), n_cal_windows=int(len(cal)), mu_shift_max=float(np.abs(mu).max()),
               sd_ratio_min=float(sd.min()), sd_ratio_max=float(sd.max()), **info)
    for k in (K_STRESS if protocol == "clean" else []):
        shift = np.zeros_like(Xte); shift[np.arange(len(Xte)), :, dw] = k * sd[dw][:, None]
        res[f"rmse_k{k}"] = dc.rmse_at(Xte + shift, imp[f"k{k}"], Mte)
    return res


# ------------------------------------------------------------------------------------------- running
def run(worker=0, n_workers=1):
    plan = os.path.join(OUT, "PLAN_step7.md")
    if not os.path.exists(plan): sys.exit("The plan is missing: python step7_robustness.py plan")
    sha = script_sha()
    if sha not in open(plan, encoding="utf-8").read():
        print("WARNING: this script differs from the version recorded in PLAN_step7.md; document the change as an amendment.")
    if sha_file(os.path.join(HERE, "step5_confirm.py")) != STEP5_SHA:
        sys.exit("step5_confirm.py differs from the Step 5 version; restore it before running Step 7.")
    os.makedirs(os.path.join(OUT, "results"), exist_ok=True); os.makedirs(os.path.join(OUT, "errors"), exist_ok=True)
    dc, v2, ev = s5.modules(); import torch
    jobs = [j for i, j in enumerate(job_list()) if i % n_workers == worker]
    for i, j in enumerate(jobs, 1):
        p = res_path(j, ".json")
        if os.path.exists(p): continue
        print(f"[{i}/{len(jobs)}] {j['id']}", flush=True); t0 = time.time(); res = None
        for strict in (True, False):
            s5.strict_mode(strict)
            try:
                res = run_job(j, dc, v2, ev); res["deterministic"] = strict; break
            except RuntimeError as e:
                if strict and "deterministic" in str(e).lower(): continue
                open(os.path.join(OUT, "errors", os.path.basename(p) + ".txt"), "w").write(j["id"] + "\n" + traceback.format_exc()); break
            except Exception:
                open(os.path.join(OUT, "errors", os.path.basename(p) + ".txt"), "w").write(j["id"] + "\n" + traceback.format_exc()); break
        if res is not None:
            res.update(seconds=round(time.time() - t0, 1), script_sha256=sha); tmp = p + ".tmp"
            json.dump(res, open(tmp, "w"), indent=1, default=float); os.replace(tmp, p)
    json.dump(dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                   gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
              open(os.path.join(OUT, "environment.json"), "w"), indent=1)


def status():
    jobs = job_list(); done = sum(os.path.exists(res_path(j, ".json")) for j in jobs)
    errs = glob.glob(os.path.join(OUT, "errors", "*.txt"))
    print(f"finished {done} of {len(jobs)} jobs; errors: {len(errs)}")
    for e in errs[:5]: print("  ", open(e).read().splitlines()[0])


def check(dc=None):
    """Step 5 protocol, KNN global and seq, two settings: the saved test imputations must equal Step 5's"""
    if dc is None: dc, _, _ = s5.modules()
    out = os.path.join(OUT, "check"); bad = 0
    jobs = {j["id"]: j for j in job_list()}
    for key in CHECK_KEYS:
        for norm in ("global", "seq"):
            j = jobs[f"{key}|KNN|{norm}"]; r = run_job(j, dc, None, None, protocol="step5", out=out)
            a = np.load(res_path(j, ".npz", out))["te"]; b = np.load(s5.res_path(j, ".npz"))["te"]
            d = float(np.abs(a.astype(float) - b.astype(float)).max()); bad += d > 1e-6
            print(f"{j['id']}: max |difference| to Step 5 = {d:.2e}; RMSE {r['rmse']:.6f}")
    print("reproduction check:", "passed" if bad == 0 else "FAILED")
    return bad == 0


# ------------------------------------------------------------------------------------------- evaluation
def gmean(x):
    x = np.asarray(x, float); return float(np.exp(np.log(x).mean()))


def evaluate(dc, protocol):
    """per-seed RMSE rows, per-unit SSE rows, stress rows and guard rows for 'step5' (Step 5 files) or 'clean'"""
    jobs = {j["id"]: j for j in job_list()}
    rows, urows, srows, grows = [], [], [], []
    for ds in s5.DATASETS:
        for mech, lv in s5.CONDITIONS:
            cond = f"{mech} {lv:g}"
            for seed in s5.SEEDS:
                S, Mtr, Mte, es, cal, mu, sd, dw = protocol_parts(dc, ds, mech, lv, seed, protocol)
                Za_m = dc.apply_mask((S["Xa"] - mu) / sd, Mtr); Zte = (S["Xte"] - mu) / sd
                y = S["Xte"][Mte]; n, _, d = np.nonzero(Mte); D = S["Xa"].shape[2]
                y_c = S["Xa"][cal][Mtr[cal]]; _, _, d_c = np.nonzero(Mtr[cal])
                guard = s5.window_guard(Za_m[S["tr"]], dc.apply_mask(Zte, Mte))
                grows.append(dict(dataset=ds, cond=cond, seed=seed, k=0, guard_share=float(guard.mean())))
                gk = {}
                if protocol == "clean":
                    for k in K_STRESS:
                        gk[k] = s5.window_guard(Za_m[S["tr"]], dc.apply_mask(stressed(Zte, dw, k), Mte))
                        grows.append(dict(dataset=ds, cond=cond, seed=seed, k=k, guard_share=float(gk[k].mean())))
                unit = S["meta"].unit.to_numpy()[S["ite"]][n]
                def load(method, norm):
                    j = jobs[f"{ds}|{mech}|{lv}|{seed}|{method}|{norm}"]
                    p = s5.res_path(j, ".npz") if protocol == "step5" else res_path(j, ".npz")
                    return np.load(p) if os.path.exists(p) else None
                ckey = "st" if protocol == "step5" else "cal"
                for mdl in s5.MODELS:
                    zg, zs = load(mdl, "global"), load(mdl, "seq")
                    if zg is None or zs is None: continue
                    g, s_ = zg["te"].astype(float), zs["te"].astype(float)
                    a = s5.dual_weights(y_c, zg[ckey].astype(float), zs[ckey].astype(float), d_c, D)
                    an = np.where(guard[n], 0.0, a[d])
                    preds = {"global": g, "seq": s_, "dual_noguard": a[d] * g + (1 - a[d]) * s_, "dual": an * g + (1 - an) * s_}
                    for v, p in preds.items():
                        e2 = (p - y) ** 2
                        rows.append(dict(dataset=ds, cond=cond, seed=seed, model=mdl, variant=v, rmse=float(np.sqrt(e2.mean())),
                                         weight_mean=float(a.mean()) if v == "dual" else np.nan))
                        u = pd.DataFrame(dict(unit=unit, e2=e2)).groupby("unit").e2.agg(["sum", "count"]).reset_index()
                        for r in u.itertuples(index=False):
                            urows.append((ds, cond, seed, mdl, v, r.unit, r.sum, r.count))
                    if protocol == "clean":
                        hit = d == dw[n]
                        for k in K_STRESS:
                            yk = y + np.where(hit, k * sd[d], 0.0)
                            gks, sks = zg[f"k{k}"].astype(float), zs[f"k{k}"].astype(float)
                            ak = np.where(gk[k][n], 0.0, a[d])
                            for v, p in {"global": gks, "seq": sks, "dual_noguard": a[d] * gks + (1 - a[d]) * sks, "dual": ak * gks + (1 - ak) * sks}.items():
                                e2 = (p - yk) ** 2
                                srows.append(dict(dataset=ds, cond=cond, seed=seed, model=mdl, k=k, variant=v,
                                                  rmse=float(np.sqrt(e2.mean())), rmse_shifted=float(np.sqrt(e2[hit].mean())),
                                                  rmse_other=float(np.sqrt(e2[~hit].mean()))))
    U = pd.DataFrame(urows, columns=["dataset", "cond", "seed", "model", "variant", "unit", "sse", "n"])
    return pd.DataFrame(rows), U, pd.DataFrame(srows), pd.DataFrame(grows)


def cells(R):
    m = R.groupby(["dataset", "cond", "model", "variant"]).rmse.mean().unstack()
    m["oracle"] = m[["global", "seq"]].min(axis=1); m["ratio"] = m.dual / m.oracle
    return m


def rules(m, need=24):
    complete = len(m.dropna(subset=["global", "seq", "dual"])) == need
    n1 = int((m.ratio <= WITHIN).sum()); g_s, g_g = gmean(m.dual / m.seq), gmean(m.dual / m["global"])
    v = lambda ok: ("supported" if ok else "not supported") if complete else f"no decision yet ({len(m)} of {need} cells)"
    return [f"* R1: cells with dual <= {WITHIN} x oracle: {n1} of {len(m)} (rule: at least {WITHIN_SHARE:.0%} of {need}) -> **{v(n1 >= WITHIN_SHARE * need)}**",
            f"* R2: geometric mean dual / seq = {g_s:.3f}, dual / global = {g_g:.3f} (rule: both <= {GAIN}) -> **{v(g_s <= GAIN and g_g <= GAIN)}**",
            f"* R3: worst dual / oracle = {m.ratio.max():.3f} (rule: every cell <= {WORST}) -> **{v(bool((m.ratio <= WORST).all()))}**"]


def bootstrap(U, n_boot=N_BOOT, rng_seed=20261117):
    """cluster bootstrap over test units within each seed (same draw for all cells of a dataset)"""
    out = []
    rng = np.random.default_rng(rng_seed)
    for ds, Ud in U.groupby("dataset"):
        cl = sorted(Ud[["cond", "model"]].drop_duplicates().itertuples(index=False, name=None))
        V = ["global", "seq", "dual"]; seeds = sorted(Ud.seed.unique())
        reps = {c: [] for c in ("dual/oracle", "dual/global", "dual/seq", "cells_within")}
        draws = {s: None for s in seeds}; units = {}
        A = {}
        for s in seeds:
            us = np.sort(Ud[Ud.seed == s].unit.unique()); units[s] = us
            for c in cl:
                for v in V:
                    q = Ud[(Ud.seed == s) & (Ud.cond == c[0]) & (Ud.model == c[1]) & (Ud.variant == v)].set_index("unit").reindex(us)
                    A[(s, c, v)] = (q.sse.to_numpy(float), q.n.to_numpy(float))
        for b in range(n_boot + 1):
            w = {s: (np.ones(len(units[s])) if b == 0 else rng.multinomial(len(units[s]), np.full(len(units[s]), 1 / len(units[s]))).astype(float)) for s in seeds}
            cm = {}
            for c in cl:
                for v in V:
                    vals = []
                    for s in seeds:
                        sse, nn = A[(s, c, v)]; ok = ~np.isnan(sse)
                        vals.append(np.sqrt((w[s][ok] * sse[ok]).sum() / (w[s][ok] * nn[ok]).sum()))
                    cm[(c, v)] = np.mean(vals)
            r_or = [cm[(c, "dual")] / min(cm[(c, "global")], cm[(c, "seq")]) for c in cl]
            res = {"dual/oracle": gmean(r_or), "dual/global": gmean([cm[(c, "dual")] / cm[(c, "global")] for c in cl]),
                   "dual/seq": gmean([cm[(c, "dual")] / cm[(c, "seq")] for c in cl]), "cells_within": int(np.sum(np.array(r_or) <= WITHIN))}
            for k, x in res.items(): reps[k].append(x)
        for k, x in reps.items():
            x = np.asarray(x, float)
            out.append(dict(dataset=ds, stat=k, n_cells=len(cl), estimate=x[0], lo=np.percentile(x[1:], 2.5), hi=np.percentile(x[1:], 97.5)))
    return pd.DataFrame(out)


def unit_ratios(U):
    """per cell: dual / better normalization (chosen at cell level) for every test unit and seed"""
    U = U.assign(rmse=np.sqrt(U.sse / U.n))
    P = U.pivot_table(index=["dataset", "cond", "model", "seed", "unit"], columns="variant", values="rmse")
    better = P.groupby(["dataset", "cond", "model"])[["global", "seq"]].mean().idxmin(axis=1)
    P = P.join(better.rename("better"), on=["dataset", "cond", "model"])
    P["ratio"] = P.dual / np.where(P.better == "global", P["global"], P.seq)
    g = P.groupby(["dataset", "cond", "model"]).ratio
    return pd.DataFrame(dict(median=g.median(), p95=g.quantile(0.95), max=g.max(), share_above_102=g.apply(lambda r: float((r > WITHIN).mean())),
                             n_units=g.size()))


def units(dc=None):
    """unit-level analysis of the Step 5 predictions (CPU)"""
    if dc is None: dc, _, _ = s5.modules()
    os.makedirs(OUT, exist_ok=True)
    R, U, _, G = evaluate(dc, "step5")
    U.to_csv(os.path.join(OUT, "step5_unit_errors.csv"), index=False)
    m = cells(R); B = bootstrap(U); Q = unit_ratios(U)
    B.to_csv(os.path.join(OUT, "step5_bootstrap.csv"), index=False); Q.to_csv(os.path.join(OUT, "step5_unit_ratios.csv"))
    L = ["# Step 5 - unit-level analysis (descriptive; written by step7_robustness.py units)", "",
         "Recomputed cell results (must equal REPORT_step5.md):", ""] + rules(m) + ["", "## Bootstrap over test units", "", "```",
         B.round(4).to_string(index=False), "```", "", "## Unit level: dual / better normalization of the cell", "", "```",
         Q.round(3).to_string(), "```", ""]
    txt = "\n".join(L); open(os.path.join(OUT, "REPORT_units_step5.md"), "w", encoding="utf-8").write(txt); print(txt)


def report(dc=None):
    if dc is None: dc, _, _ = s5.modules()
    R, U, SR, G = evaluate(dc, "clean")
    if R.empty: sys.exit("no Step 7 results yet")
    m = cells(R)
    R.to_csv(os.path.join(OUT, "step7_runs.csv"), index=False); U.to_csv(os.path.join(OUT, "step7_unit_errors.csv"), index=False)
    SR.to_csv(os.path.join(OUT, "step7_stress.csv"), index=False); G.to_csv(os.path.join(OUT, "step7_guard.csv"), index=False)
    sha = sorted({json.load(open(p)).get("script_sha256") for p in glob.glob(os.path.join(OUT, "results", "*.json"))})
    L = ["# Step 7 report (robustness re-analysis of Step 5; post hoc)", "", f"Script versions in the results: {sha}", "",
         "## Mean RMSE per cell (clean protocol; Step 5 reporting scale)", "", "```",
         m[["global", "seq", "dual_noguard", "dual", "oracle", "ratio"]].round(4).to_string(), "```", ""] + rules(m) + [""]
    p5 = os.path.join(OUT, "step5_unit_errors.csv")
    if os.path.exists(p5):
        U5 = pd.read_csv(p5); R5 = (U5.groupby(["dataset", "cond", "model", "variant", "seed"]).apply(lambda g: np.sqrt(g.sse.sum() / g.n.sum()), include_groups=False)
                                    .rename("rmse").reset_index())
        m5 = cells(R5); cmp_ = (m[["global", "seq", "dual"]] / m5[["global", "seq", "dual"]]).add_suffix(" 7/5")
        L += ["## Step 7 / Step 5 cell RMSE", "", "```", cmp_.round(3).to_string(), "```", "",
              f"Geometric mean Step 7 / Step 5: global {gmean(cmp_['global 7/5'].dropna()):.3f}, seq {gmean(cmp_['seq 7/5'].dropna()):.3f}, dual {gmean(cmp_['dual 7/5'].dropna()):.3f}", ""]
    else:
        L += ["(run `python step7_robustness.py units` first for the comparison with Step 5)", ""]
    B = bootstrap(U); Q = unit_ratios(U); B.to_csv(os.path.join(OUT, "step7_bootstrap.csv"), index=False)
    L += ["## Bootstrap over test units (clean protocol)", "", "```", B.round(4).to_string(index=False), "```", "",
          "## Unit level: dual / better normalization of the cell (clean protocol)", "", "```", Q.round(3).to_string(), "```", "",
          "## Window guard: share of test windows using seq only", "", "```",
          G.groupby(["dataset", "cond", "k"]).guard_share.mean().unstack("k").round(4).to_string(), "```", "",
          "## Mean dual weight on global (calibration half)", "", "```",
          R.dropna(subset=["weight_mean"]).groupby(["dataset", "cond", "model"]).weight_mean.mean().unstack().round(2).to_string(), "```", ""]
    if not SR.empty:
        st_ = SR.groupby(["dataset", "k", "model", "variant"])[["rmse", "rmse_shifted", "rmse_other"]].mean().unstack("variant")
        L += ["## Stress test (mean over conditions and seeds)", "", "```", st_.round(3).to_string(), "```", ""]
    jr = [json.load(open(p)) for p in glob.glob(os.path.join(OUT, "results", "*.json"))]
    if jr:
        J = pd.DataFrame(jr); J.to_csv(os.path.join(OUT, "step7_jobs.csv"), index=False)
        L += [f"Runs without strict determinism: {int((~J.deterministic.astype(bool)).sum())}"]
    txt = "\n".join(L); open(os.path.join(OUT, "REPORT_step7.md"), "w", encoding="utf-8").write(txt); print(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["plan", "check", "units", "run", "status", "report"])
    ap.add_argument("--worker", type=int, default=0); ap.add_argument("--n-workers", type=int, default=1)
    a = ap.parse_args()
    if a.cmd == "plan":
        os.makedirs(OUT, exist_ok=True); p = os.path.join(OUT, "PLAN_step7.md")
        if os.path.exists(p): sys.exit(f"{p} already exists; it must not be changed after results exist")
        txt = PLAN.format(sha=script_sha(), sha5=STEP5_SHA, ks=K_STRESS, w=WITHIN, ws=WITHIN_SHARE, g=GAIN, wo=WORST, nb=N_BOOT)
        open(p, "w", encoding="utf-8").write(txt + f"\nWritten: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        print(open(p, encoding="utf-8").read()); print("SHA-256 of the plan:", sha_file(p))
    elif a.cmd == "check": check()
    elif a.cmd == "units": units()
    elif a.cmd == "run": run(a.worker, a.n_workers)
    elif a.cmd == "status": status()
    else: report()
