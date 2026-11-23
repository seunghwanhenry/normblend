"""
step9_extension.py - Step 9: two additions to the NormBlend benchmark (descriptive; run after Step 8).

E6  Seed-ensemble control. SAITS, TimesNet, ImputeFormer and the mask-aware transformer are trained a second time with
    each normalization on C-MAPSS-SC, C-MAPSS-MC and MetroPT-3 (four conditions, seeds 0-4). Splits, masks,
    early-stopping and calibration halves are those of Step 8; only the model initialization differs. The second
    models are combined with the Step 8 models, without further training, into
      ens_global / ens_seq   two-seed ensembles of one normalization (2 trainings), calibrated per-variable weights
                             (the NormBlend formula) and, as a check, equal weights
      ens_valsel             two-seed ensemble of the normalization with the lower calibration RMSE (3 trainings)
      nb_2seed               NormBlend on the seed-averaged members of each normalization (4 trainings)
    and compared with NormBlend (2 trainings). The C-MAPSS-SC models also impute the C-MAPSS-MC windows (shift).
    KNN is deterministic and BRITS is left out because of its training time.
E7  A real industrial dataset not used before: SKAB (Skoltech Anomaly Benchmark; water-circulation testbed, 8 sensors
    at 1 Hz). Units are the 34 experiment runs and the normal-mode record cut into 20-min blocks; windows of 60
    consecutive records within a unit, without a time step longer than 5 s. Full Step 8 benchmark on it: six imputers
    with G and S, the five single-model alternatives, four conditions, seeds 0-4, the Step 8 protocol and settings.
    Nothing is tuned on SKAB; kappa = 1 as before.

Commands (from the DTIN folder, next to step8_benchmark.py):
  python step9_extension.py plan        # writes step9_outputs/PLAN_step9.md (once, before any job)
  python step9_extension.py fetch       # downloads SKAB (GitHub, pinned commit) to data/skab
  python step9_extension.py prepare     # builds the SKAB windows
  python step9_extension.py check       # CPU, about 1 min: Step 8 reproduction (KNN) and the seed/mask separation
  python step9_extension.py smoke       # every new job type once on a tiny subset (a few minutes)
  python step9_extension.py run [--worker k --n-workers K]
  python step9_extension.py status
  python step9_extension.py report
"""
import argparse, glob, hashlib, importlib.util, io, json, os, sys, time, urllib.request, zipfile
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("step8_benchmark", os.path.join(HERE, "step8_benchmark.py"))
s8 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s8)
s5, s7 = s8.s5, s8.s7

OUT = "step9_outputs"
SEEDS = [0, 1, 2, 3, 4]
FIRST_SEEDS = [0, 1, 2]
CONDS = s8.CONDS
ENS_MODELS = ["SAITS", "TimesNet", "ImputeFormer", "Transformer"]
ENS_DATASETS = ["CMAPSS", "CMAPSS_M", "MetroPT"]
SKAB = "SKAB"
SKAB_DIR = os.path.join("data", "skab")
SKAB_COMMIT = "b2c0d46c2971dcbfe71e26087b6d231998bb91c2"
SKAB_URL = f"https://github.com/waico/SKAB/archive/{SKAB_COMMIT}.zip"
SKAB_SIGNALS = ["Accelerometer1RMS", "Accelerometer2RMS", "Current", "Pressure", "Temperature", "Thermocouple",
                "Voltage", "Volume Flow RateRMS"]
SKAB_T, SKAB_GAP, SKAB_MAXSTEP, SKAB_BLOCK = 60, 12, 5.0, 1200
s5.T_LEN[SKAB] = SKAB_T; s5.GAP_L[SKAB] = SKAB_GAP
INIT2 = "step9-init2"


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def script_sha():
    return sha_file(os.path.abspath(__file__))


PLAN = """# Step 9 - seed-ensemble control and a new industrial dataset (descriptive)

Written on {date}, after Step 8 and before any Step 9 job was run. Script step9_extension.py, SHA-256 {sha}; it imports
the unchanged step8_benchmark.py ({sha8}), step5_confirm.py ({sha5}) and step7_robustness.py ({sha7}).
Step 9 has no confirmatory decision rule; every result is reported.

## Why
A reading of the NormBlend manuscript asked (1) how much of the gain a two-seed ensemble of one normalization would
give at the same cost, and (2) for a second real industrial dataset that played no part in designing the method.

## Design
* E6: {ens_models} trained a second time with G and S on {ens_ds}, four conditions, seeds {seeds}; Step 8 splits,
  masks, early-stopping and calibration halves; only the model initialization differs (seed key "{init2}").
  Compared: NormBlend (2 trainings), ens_global and ens_seq (2), ens_valsel (3), nb_2seed (4); calibrated
  per-variable weights (NormBlend formula) and equal weights; oracle choices marked as such. C-MAPSS-SC models also
  impute the C-MAPSS-MC windows. KNN (deterministic) and BRITS (training time) are not part of E6.
* E7: SKAB, GitHub waico/SKAB at commit {commit}; signals {signals}; units = 34 experiment runs plus the normal-mode
  record cut into blocks of {block} records; windows of {T} consecutive records within a unit, dropped if a time step
  exceeds {maxstep} s; gaps of nominal length {gap}. Six imputers with G and S, the five single-model alternatives,
  four conditions, seeds {seeds}, Step 8 protocol, hyperparameters and kappa = 1. Nothing is tuned on SKAB.
"""


# ------------------------------------------------------------------------------------------- SKAB data
def fetch():
    os.makedirs(SKAB_DIR, exist_ok=True)
    if glob.glob(os.path.join(SKAB_DIR, "**", "valve1", "*.csv"), recursive=True):
        print("SKAB already present in", SKAB_DIR); return
    print("downloading", SKAB_URL)
    raw = urllib.request.urlopen(SKAB_URL, timeout=120).read()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for n in z.namelist():
            if "/data/" in n and n.endswith(".csv") and n.count("/") == 3:
                rel = n.split("/data/", 1)[1]; p = os.path.join(SKAB_DIR, rel)
                os.makedirs(os.path.dirname(p), exist_ok=True); open(p, "wb").write(z.read(n))
    print("files:", len(glob.glob(os.path.join(SKAB_DIR, "**", "*.csv"), recursive=True)))


def load_skab():
    files = sorted(glob.glob(os.path.join(SKAB_DIR, "**", "*.csv"), recursive=True))
    files = [f for f in files if os.path.basename(os.path.dirname(f)) in ("anomaly-free", "other", "valve1", "valve2")]
    if len(files) != 35: sys.exit(f"expected 35 SKAB files under {SKAB_DIR}, found {len(files)}; run fetch")
    W, meta, unit = [], [], 0
    for f in files:
        grp = os.path.basename(os.path.dirname(f)); name = f"{grp}/{os.path.basename(f)}"
        d = pd.read_csv(f, sep=";", parse_dates=["datetime"]).sort_values("datetime").reset_index(drop=True)
        X = d[SKAB_SIGNALS].to_numpy(np.float64); t = d["datetime"].to_numpy()
        step = np.r_[0.0, np.diff(t).astype("timedelta64[ms]").astype(float) / 1000.0]
        an = d["anomaly"].to_numpy(float) if "anomaly" in d else np.zeros(len(d))
        blocks = [(0, len(d))] if grp != "anomaly-free" else [(a, min(a + SKAB_BLOCK, len(d))) for a in range(0, len(d), SKAB_BLOCK)]
        for a, b in blocks:
            unit += 1
            for i in range(a, b - SKAB_T + 1, SKAB_T):
                if (step[i + 1:i + SKAB_T] > SKAB_MAXSTEP).any(): continue
                W.append(X[i:i + SKAB_T])
                meta.append(dict(unit=unit, source=name, start=str(t[i])[:19], anomaly_share=float(an[i:i + SKAB_T].mean())))
    return np.stack(W), pd.DataFrame(meta)


def cache9():
    return os.path.join(OUT, "cache", f"{SKAB}.npz"), os.path.join(OUT, "cache", f"{SKAB}_windows.csv")


def prepare():
    os.makedirs(os.path.join(OUT, "cache"), exist_ok=True)
    X, meta = load_skab(); p, q = cache9()
    np.savez_compressed(p, X=X, cols=np.array(SKAB_SIGNALS)); meta.to_csv(q, index=False)
    print(f"SKAB: windows {X.shape[0]}, length {X.shape[1]}, variables {X.shape[2]}, units {meta.unit.nunique()}, "
          f"non-finite values {int((~np.isfinite(X)).sum())}, windows with anomalies {int((meta.anomaly_share > 0).sum())}")


_setting8_orig = s8.setting8
_C9 = {}


def setting9(dc, ds, seed):
    """Step 8 settings, plus SKAB with the same split and standardization rules"""
    if ds != SKAB: return _setting8_orig(dc, ds, seed)
    if (ds, seed) in _C9: return _C9[(ds, seed)]
    if ds not in _C9:
        p, q = cache9()
        if not os.path.exists(p): sys.exit("run first: python step9_extension.py prepare")
        _C9[ds] = (np.load(p)["X"], pd.read_csv(q))
    X, meta = _C9[ds]
    itr, ist, ite = s5.split_idx(ds, meta, seed); itrain = np.r_[itr, ist]
    Xa, Xte, sc = dc.standardize(X[itrain], X[ite])
    S = dict(Xa=Xa, Xte=Xte, tr=np.arange(len(itr)), st=np.arange(len(itr), len(itrain)), itrain=itrain, ite=ite,
             meta=meta, scale=sc)
    _C9[(ds, seed)] = S
    return S


s8.setting8 = setting9            # s8.parts looks the name up at call time


# ------------------------------------------------------------------------------------------- jobs
def _job(ds, mech, lv, seed, method, variant, tier, init2=False):
    j = s8._job(ds, mech, lv, seed, method, variant, tier)
    j["exp"] = "E6" if init2 else "E7"; j["init2"] = bool(init2)
    if init2: j["id"] += "|init2"
    return j


def job_list():
    jobs = []
    for seeds in (FIRST_SEEDS, [s for s in SEEDS if s not in FIRST_SEEDS]):
        for seed in seeds:                                                  # E6 first: the main open question
            for (mech, lv) in CONDS:
                for ds in ENS_DATASETS:
                    jobs += [_job(ds, mech, lv, seed, m, n, 1, init2=True) for m in ENS_MODELS for n in s8.NORMS]
        for seed in seeds:
            for (mech, lv) in CONDS:
                jobs += [_job(SKAB, mech, lv, seed, m, n, 1) for m in s8.BACKBONES for n in s8.NORMS]
                jobs += [_job(SKAB, mech, lv, seed, m, "gstat", 2) for m in s8.GSTAT]
                jobs += [_job(SKAB, mech, lv, seed, m, "own", 2) for m in s8.OWN]
    return jobs


def res_path(j, ext, out=None):
    return os.path.join(out or OUT, "results", hashlib.md5(j["id"].encode()).hexdigest() + ext)


class _Init2:
    """second model initialization: only the 'step5-init' seed key changes; splits and masks keep their keys"""
    def __enter__(self):
        self.orig = s5.sseed
        s5.sseed = lambda *k: self.orig(INIT2, *k[1:]) if k and k[0] == "step5-init" else self.orig(*k)
    def __exit__(self, *a):
        s5.sseed = self.orig


def run_job(j, dc, v2, ev, out=None):
    """the Step 8 job, with the second model initialization for E6; results go to step9_outputs"""
    with (_Init2() if j.get("init2") else _Null()):
        r = s8.run_job(dict(j), dc, v2, ev, out=out or OUT)
    r.update(exp=j["exp"], init2=j["init2"])
    return r


class _Null:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _attempt(j, dc, v2, ev, out, errdir):
    res = None
    for strict in (True, False):
        s5.strict_mode(strict)
        try:
            res = run_job(j, dc, v2, ev, out); res["deterministic"] = strict; break
        except RuntimeError as e:
            if strict and "deterministic" in str(e).lower(): continue
            open(os.path.join(errdir, hashlib.md5(j["id"].encode()).hexdigest() + ".txt"), "w").write(j["id"] + "\n" + s8.traceback.format_exc()); break
        except Exception:
            open(os.path.join(errdir, hashlib.md5(j["id"].encode()).hexdigest() + ".txt"), "w").write(j["id"] + "\n" + s8.traceback.format_exc()); break
    return res


def run(worker=0, n_workers=1):
    plan = os.path.join(OUT, "PLAN_step9.md")
    if not os.path.exists(plan): sys.exit("The plan is missing: python step9_extension.py plan")
    if not os.path.exists(cache9()[0]): sys.exit("run first: python step9_extension.py fetch, then prepare")
    if not os.path.exists(s8.cache8("CMAPSS_M")[0]): sys.exit("Step 8 cache missing: python step8_benchmark.py prepare")
    sha = script_sha()
    if sha not in open(plan, encoding="utf-8").read():
        print("WARNING: this script differs from the version recorded in PLAN_step9.md; document the change.")
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
    df = pd.DataFrame(jobs).assign(done=done)
    print(df.groupby(["exp", "method"]).done.agg(["sum", "count"]).to_string())
    for e in errs[:5]: print("  ", open(e).read().splitlines()[0])


def check():
    """(1) Step 8 reproduction with KNN; (2) the second initialization changes only the model seed"""
    os.makedirs(OUT, exist_ok=True)
    s8.check()
    a = s5.sseed("step5-init", "CMAPSS", "MCAR", 0.3, 0, "SAITS"); m1 = s5.sseed("step5-mask", "CMAPSS", "MCAR", 0.3, 0, "train")
    with _Init2():
        b = s5.sseed("step5-init", "CMAPSS", "MCAR", 0.3, 0, "SAITS"); m2 = s5.sseed("step5-mask", "CMAPSS", "MCAR", 0.3, 0, "train")
    ok = (a != b) and (m1 == m2) and s5.sseed("step5-init", "CMAPSS", "MCAR", 0.3, 0, "SAITS") == a
    print(f"model seed {a} -> {b} with the second initialization; mask seed unchanged: {m1 == m2}")
    print("seed separation check:", "passed" if ok else "FAILED")


def smoke():
    s8.FAST["on"] = True
    dc, v2, ev = s5.modules(); out = os.path.join(OUT, "smoke"); err = os.path.join(out, "errors")
    os.makedirs(err, exist_ok=True); ok = 0
    todo = [_job("CMAPSS", "MCAR", 0.3, 0, "SAITS", "global", 1, init2=True),
            _job("MetroPT", "GAP", 0.4, 0, "Transformer", "seq", 1, init2=True),
            _job("CMAPSS_M", "MNAR", 0.3, 0, "TimesNet", "seq", 1, init2=True),
            _job("CMAPSS", "MCAR", 0.7, 0, "ImputeFormer", "global", 1, init2=True)]
    todo += [_job(SKAB, "MCAR", 0.3, 0, m, n, 1) for m in s8.BACKBONES for n in s8.NORMS]
    todo += [_job(SKAB, "GAP", 0.4, 0, m, "gstat", 2) for m in s8.GSTAT] + [_job(SKAB, "MNAR", 0.3, 0, m, "own", 2) for m in s8.OWN]
    for j in todo:
        t0 = time.time(); r = _attempt(j, dc, v2, ev, out, err)
        if r is None: print(f"FAILED  {j['id']} (see {err})"); continue
        ok += 1; print(f"ok      {j['id']}: {time.time() - t0:5.1f} s, RMSE {r['rmse']:.3f}, deterministic {r['deterministic']}")
    print(f"smoke test: {ok} of {len(todo)} passed")


# ------------------------------------------------------------------------------------------- report
def _load9(j):
    p = res_path(j, ".npz")
    return (np.load(p), "step9") if os.path.exists(p) else (None, None)


def _blend(a, d, g, s_):
    return a[d] * g + (1 - a[d]) * s_


def _ens_variants(P, y_c, d_c, D, d, gn):
    """P: dict of arrays (te or xd) and cal arrays for g1, g2, s1, s2; gn: guard flag per position (None: no guard)"""
    W = lambda x, y_: s5.dual_weights(y_c, P["cal"][x], P["cal"][y_], d_c, D)
    t = P["te"]; out = {}
    a_gs = W("g1", "s1"); an = np.where(gn, 0.0, a_gs[d]) if gn is not None else a_gs[d]
    out["normblend"] = an * t["g1"] + (1 - an) * t["s1"]
    out["global"], out["seq"] = t["g1"], t["s1"]
    out["ens_global"] = _blend(W("g1", "g2"), d, t["g1"], t["g2"]); out["ens_global_eq"] = 0.5 * (t["g1"] + t["g2"])
    out["ens_seq"] = _blend(W("s1", "s2"), d, t["s1"], t["s2"]); out["ens_seq_eq"] = 0.5 * (t["s1"] + t["s2"])
    Eg, Es = ((P["cal"]["g1"] - y_c) ** 2).mean(), ((P["cal"]["s1"] - y_c) ** 2).mean()
    out["ens_valsel"] = out["ens_global"] if Eg <= Es else out["ens_seq"]
    cG, cS = 0.5 * (P["cal"]["g1"] + P["cal"]["g2"]), 0.5 * (P["cal"]["s1"] + P["cal"]["s2"])
    a2 = s5.dual_weights(y_c, cG, cS, d_c, D); a2n = np.where(gn, 0.0, a2[d]) if gn is not None else a2[d]
    out["nb_2seed"] = a2n * 0.5 * (t["g1"] + t["g2"]) + (1 - a2n) * 0.5 * (t["s1"] + t["s2"])
    return out


def evaluate(dc):
    rows, xrows, wrows, grows = [], [], [], []
    for seed in SEEDS:
        for (mech, lv) in CONDS:
            cond = f"{mech} {lv:g}"
            for ds in ENS_DATASETS + [SKAB]:
                S, Mtr, Mte, es, cal, mu, sd = s8.parts(dc, ds, mech, lv, seed)
                Za_m = dc.apply_mask((S["Xa"] - mu) / sd, Mtr); Zte_m = dc.apply_mask((S["Xte"] - mu) / sd, Mte)
                y = S["Xte"][Mte]; n, _, d = np.nonzero(Mte); D = S["Xa"].shape[2]
                y_c = S["Xa"][cal][Mtr[cal]]; _, _, d_c = np.nonzero(Mtr[cal])
                guard = s5.window_guard(Za_m[S["tr"]], Zte_m); base = dict(dataset=ds, cond=cond, seed=seed)
                if ds == SKAB:                                          # E7: the Step 8 evaluation on SKAB
                    grows.append(dict(base, guard_share=float(guard.mean())))
                    for mdl in s8.BACKBONES:
                        (zg, _), (zs, _) = _load9(_job(ds, mech, lv, seed, mdl, "global", 0)), _load9(_job(ds, mech, lv, seed, mdl, "seq", 0))
                        if zg is None or zs is None: continue
                        g, s_, cg, cs = (z[k].astype(float) for z, k in ((zg, "te"), (zs, "te"), (zg, "cal"), (zs, "cal")))
                        a = s5.dual_weights(y_c, cg, cs, d_c, D)
                        for v, p in s8._variants(g, s_, a, d, guard[n], cg, cs, y_c, d_c).items():
                            r_, a_ = s8._metrics(p, y); rows.append(dict(base, exp="E7", model=mdl, variant=v, rmse=r_, mae=a_))
                        wrows += [dict(base, model=mdl, var=k, weight=float(a[k])) for k in range(D)]
                    for mdl, var in [(m, "gstat") for m in s8.GSTAT] + [(m, "own") for m in s8.OWN]:
                        z, _ = _load9(_job(ds, mech, lv, seed, mdl, var, 0))
                        if z is None: continue
                        r_, a_ = s8._metrics(z["te"].astype(float), y)
                        rows.append(dict(base, exp="E7", model=mdl if var == "own" else f"{mdl}+stats", variant=var, rmse=r_, mae=a_))
                    continue
                xd = None                                               # E6
                if ds == "CMAPSS":
                    Xx, Mx, Sm, (mu_c, sd_c, mu_m, sd_m) = s8.cross_domain(dc, mech, lv, seed)
                    gx = s5.window_guard(Za_m[S["tr"]], dc.apply_mask((Xx - mu) / sd, Mx)); nx, _, dxv = np.nonzero(Mx)
                    to_m = lambda Q, dxv=dxv: ((Q * sd_c[dxv] + mu_c[dxv]) - mu_m[dxv]) / sd_m[dxv]
                    xd = (gx, nx, dxv, Sm["Xte"][Mx], to_m)
                for mdl in ENS_MODELS:
                    z = {"g1": s8._load_pred(s8._job(ds, mech, lv, seed, mdl, "global", 0))[0],
                         "s1": s8._load_pred(s8._job(ds, mech, lv, seed, mdl, "seq", 0))[0],
                         "g2": _load9(_job(ds, mech, lv, seed, mdl, "global", 0, init2=True))[0],
                         "s2": _load9(_job(ds, mech, lv, seed, mdl, "seq", 0, init2=True))[0]}
                    if any(v is None for v in z.values()): continue
                    P = {"te": {k: v["te"].astype(float) for k, v in z.items()}, "cal": {k: v["cal"].astype(float) for k, v in z.items()}}
                    for v, p in _ens_variants(P, y_c, d_c, D, d, guard[n]).items():
                        r_, a_ = s8._metrics(p, y); rows.append(dict(base, exp="E6", model=mdl, variant=v, rmse=r_, mae=a_))
                    if xd is not None and all("xd" in v.files for v in z.values()):
                        gx, nx, dxv, y_x, to_m = xd
                        Px = {"te": {k: to_m(v["xd"].astype(float)) for k, v in z.items()}, "cal": P["cal"]}
                        for v, p in _ens_variants(Px, y_c, d_c, D, dxv, gx[nx]).items():
                            r_, a_ = s8._metrics(p, y_x); xrows.append(dict(base, dataset="CMAPSS->CMAPSS_M", exp="E6", model=mdl, variant=v, rmse=r_, mae=a_))
    return pd.DataFrame(rows), pd.DataFrame(xrows), pd.DataFrame(wrows), pd.DataFrame(grows)


def report():
    dc, _, _ = s5.modules()
    R, X, W, G = evaluate(dc)
    if R.empty: sys.exit("no results yet")
    R.to_csv(os.path.join(OUT, "step9_runs.csv"), index=False); X.to_csv(os.path.join(OUT, "step9_cross.csv"), index=False)
    W.to_csv(os.path.join(OUT, "step9_weights.csv"), index=False); G.to_csv(os.path.join(OUT, "step9_guard.csv"), index=False)
    jr = [json.load(open(p)) for p in glob.glob(os.path.join(OUT, "results", "*.json"))]
    if jr: pd.DataFrame(jr).to_csv(os.path.join(OUT, "step9_jobs.csv"), index=False)
    L = ["# Step 9 report (descriptive)", "", f"Script versions in the results: {sorted({r.get('script_sha256') for r in jr})}", ""]
    e6 = R[R.exp == "E6"]
    if not e6.empty:
        c = e6.groupby(["dataset", "cond", "model", "variant"]).rmse.mean().unstack()
        c["oracle"] = c[["global", "seq"]].min(axis=1); c["ens_better"] = c[["ens_global", "ens_seq"]].min(axis=1)
        L += ["## E6 seed-ensemble control: geometric-mean ratio to the oracle, settings below the better ensemble", "", "```"]
        for v in ["normblend", "ens_global", "ens_seq", "ens_better", "ens_valsel", "ens_global_eq", "ens_seq_eq", "nb_2seed"]:
            if v in c: L.append(f"{v:14s} ratio to oracle {s8.gmean(c[v] / c.oracle):.3f}  NB below it in {int((c.normblend < c[v]).sum())} of {len(c)}")
        L += ["```", "", "```", c[["normblend", "ens_better", "ens_valsel", "nb_2seed"]].div(c.oracle, axis=0).groupby(level=["dataset", "model"]).agg(s8.gmean).round(3).to_string(), "```", ""]
    if not X.empty:
        L += ["## E6 cross-domain (C-MAPSS-SC -> C-MAPSS-MC), mean RMSE", "", "```",
              X.groupby(["model", "variant"]).rmse.mean().unstack().round(3).to_string(), "```", ""]
    e7 = R[R.exp == "E7"]
    if not e7.empty:
        c = e7[e7.variant.isin(["global", "seq", "normblend", "valsel"])].groupby(["cond", "model", "variant"]).rmse.mean().unstack()
        c["oracle"] = c[["global", "seq"]].min(axis=1); c["ratio"] = c.normblend / c.oracle
        L += ["## E7 SKAB", "", f"Settings: {len(c)}; NormBlend below the better normalization in {int((c.ratio < 1).sum())}; "
              f"geometric mean {s8.gmean(c.ratio):.3f}; worst {c.ratio.max():.3f}; global better in {int((c['global'] < c.seq).sum())}", "",
              "```", e7.groupby(["model", "variant"]).rmse.mean().unstack().round(4).to_string(), "```", "",
              "Guard activation on SKAB: " + G.groupby("cond").guard_share.mean().round(4).to_string().replace("\n", "; "), ""]
    if jr:
        J = pd.DataFrame(jr); L += [f"Runs without strict determinism: {int((~J.deterministic.astype(bool)).sum())} of {len(J)}", ""]
    txt = "\n".join(L); open(os.path.join(OUT, "REPORT_step9.md"), "w", encoding="utf-8").write(txt); print(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "fetch", "prepare", "check", "smoke", "run", "status", "report"])
    ap.add_argument("--worker", type=int, default=0); ap.add_argument("--n-workers", type=int, default=1)
    a = ap.parse_args()
    if a.cmd == "plan":
        os.makedirs(OUT, exist_ok=True); p = os.path.join(OUT, "PLAN_step9.md")
        if os.path.exists(p): sys.exit(f"{p} already exists; it must not be changed after results exist")
        txt = PLAN.format(date=time.strftime("%Y-%m-%d"), sha=script_sha(), sha8=sha_file(os.path.join(HERE, "step8_benchmark.py")),
                          sha5=sha_file(os.path.join(HERE, "step5_confirm.py")), sha7=sha_file(os.path.join(HERE, "step7_robustness.py")),
                          ens_models=", ".join(ENS_MODELS), ens_ds=", ".join(ENS_DATASETS), seeds=SEEDS, init2=INIT2,
                          commit=SKAB_COMMIT, signals=", ".join(SKAB_SIGNALS), block=SKAB_BLOCK, T=SKAB_T,
                          maxstep=SKAB_MAXSTEP, gap=SKAB_GAP)
        open(p, "w", encoding="utf-8").write(txt + f"\nJobs: {len(job_list())}\n")
        print(open(p, encoding="utf-8").read())
    elif a.cmd == "fetch": fetch()
    elif a.cmd == "prepare": prepare()
    elif a.cmd == "check": check()
    elif a.cmd == "smoke": smoke()
    elif a.cmd == "run": run(a.worker, a.n_workers)
    elif a.cmd == "status": status()
    else: report()
