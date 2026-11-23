"""
step5_confirm.py - Step 5: pre-registered confirmation of dual normalization on two datasets not used before.

Datasets (public; neither was opened in this project before the plan was registered)
  CMAPSS   - NASA C-MAPSS turbofan engine degradation simulation, subsets FD001 and FD003 (training files: 200 engines
             run to failure, one operating condition), 14 sensors, one value per flight cycle.
  MetroPT  - MetroPT-3 (UCI 791, CC BY 4.0): air production unit of a metro train, Feb-Aug 2020, 7 analogue signals,
             with four documented air-leak failures.

Method (frozen in Step 4, development data Siemens, ECG200, Motor, Hydraulic)
  Dual normalization: every model is trained twice, with global standardization and with per-sequence (seq)
  normalization (same model seed). On the masked early-stopping windows, a per-variable convex weight on the global
  imputation is estimated by minimizing the validation MSE (a = clip((Es - C) / (Eg + Es - 2C), 0, 1)). Window guard:
  if any variable of a test window has a visible mean or log standard deviation outside the range of the masked
  training windows by more than one range width (kappa = 1), the window uses seq only. Imputation = a*global + (1-a)*seq.

Commands (in this order):
  python step5_confirm.py plan      # writes step5_outputs/PLAN_step5.md -> register it (public GitHub commit) first
  python step5_confirm.py fetch     # downloads C-MAPSS (NASA) and MetroPT-3 (UCI) into data/
  python step5_confirm.py prepare   # builds the windows; prints structural counts only
  python step5_confirm.py run [--worker k --n-workers K]
  python step5_confirm.py status
  python step5_confirm.py report
"""
import argparse, glob, hashlib, io, json, os, platform, sys, time, traceback, zipfile
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd

OUT = "step5_outputs"
CMAPSS_DIR = os.path.join("data", "cmapss")
METRO_DIR = os.path.join("data", "metropt3")
URLS = {"cmapss": "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip",
        "metropt3": "https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip"}
EPS = 1e-5
SEEDS = list(range(10))
DATASETS = ["CMAPSS", "MetroPT"]
CONDITIONS = [("MCAR", 0.3), ("MCAR", 0.7), ("MNAR", 0.3), ("GAP", 0.4)]
T_LEN = {"CMAPSS": 30, "MetroPT": 60}
GAP_L = {"CMAPSS": 6, "MetroPT": 12}
GAP_FRAC = 0.4
MODELS = ["KNN", "SAITS", "Transformer"]
NORMS = ["global", "seq"]
TRANSFORMER = dict(revin=False, mask_input=True, mask_mode="mix")
KAPPA = 1.0
RANGE_FLOOR = 1e-6                                 # floor of the training range (constant variables)
CMAPSS_SUBSETS = ["FD001", "FD003"]
CMAPSS_SENSORS = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]
METRO_SIGNALS = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current"]
METRO_FAILURES = [("2020-04-18 00:00", "2020-04-18 23:59"), ("2020-05-29 23:30", "2020-05-30 06:00"),
                  ("2020-06-05 10:00", "2020-06-07 14:30"), ("2020-07-15 14:30", "2020-07-15 19:00")]
SPLIT_TEST, SPLIT_STOP = 0.3, 0.1
EARLY_CYCLES, LATE_CYCLES = 60, [60, 90]
WITHIN, WITHIN_SHARE, GAIN, WORST = 1.02, 0.90, 0.98, 1.10


def script_sha():
    return hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()


def sseed(*keys):
    return int(hashlib.sha256("|".join(map(str, keys)).encode()).hexdigest()[:8], 16)


PLAN = """# Step 5 - confirmation of dual normalization (pre-specified)

Written before the Step 5 datasets were downloaded or opened. Register this file (public GitHub commit) before running
`fetch`, `prepare` and `run`. Script: step5_confirm.py, SHA-256 {sha}.

## History that led to this test
Steps 1-3 showed that the advantage of the earlier DTIN model came from per-sequence normalization, that the better of
global and per-sequence normalization changes between datasets, and that a reliability-weighted shrinkage (Steps 2-2b)
did not choose between them on independent data (Step 3, pre-registered, not supported). In Step 4 (exploratory,
development data Siemens, ECG200, Motor, Hydraulic) dual normalization was developed. The variant list and choice
rule were written before Step 4, and the rule selected per-variable selection. After inspecting the Step 4 failures,
which all came from one test motor outside the training range, a window-level guard with a margin was added and the
per-variable convex combination with this guard was chosen (step4_guard_analysis.py, DEV_DECISION_step4.md). On the
development data it was within 2% of the better single normalization in 47 of 48 cells (worst ratio 1.023;
geometric-mean ratio 0.911; 47-48 of 48 cells for margins from 0.25 to 10 range widths). These figures are optimistic
because the guard was chosen on these data; this step tests the frozen method on new data.

## Method (frozen; functions dual_weights and window_guard in this script)
1. Each model is trained twice on the same masked training windows and with the same model seed: (G) global
   standardization only; (S) per-sequence normalization with the mean and standard deviation of the visible values of
   each window, inverted after imputation.
2. Both imputers impute the masked early-stopping windows. For every variable d, a_d = clip((Es - C) / (Eg + Es - 2C),
   0, 1), where Eg and Es are the validation MSEs of G and S at the removed positions of d and C is the mean product of
   their errors (a_d = 0.5 if the denominator is 0; the pooled value if d has no removed validation value).
3. Window guard: for every test window and variable with at least two visible values, the visible mean and log
   standard deviation are compared with their ranges over the masked training windows; the exceedance is the distance
   outside the range divided by the range width (floor 1e-6). If the largest exceedance of the window exceeds 1,
   a_d = 0 for all variables of that window.
4. Imputed value = a_d * G + (1 - a_d) * S at the removed positions.

## Data (fixed now, from the public documentation only)
C-MAPSS is a widely used benchmark whose general properties are known from the literature; MetroPT-3 is less used.
* CMAPSS: training files of FD001 and FD003 (200 engines, run to failure); sensors s{sens}; one value per cycle;
  per engine, non-overlapping windows of {tc} cycles from the first cycle; an incomplete last window is dropped.
  Units = engines (FD003 engine numbers offset by 1000).
* MetroPT: the seven analogue signals {msig}; 1-min means on a regular grid; non-overlapping windows of {tm} minutes
  aligned to full hours; windows with an empty minute are dropped. Units = calendar days.
* Both: global standardization with the statistics of the training and early-stopping windows of each split.

## Splits, conditions, seeds
* Seeds {seeds}; each seed gives a different split, masks and model seed (the same model seed for both normalizations).
* Units are permuted per seed: {st:.0%} test, {ss:.0%} early stopping (also used for the dual weights), the rest training.
* Missingness on training and test windows: MCAR 30%, MCAR 70%, MNAR 30% (values above the 70th percentile removed
  with probability 0.6), consecutive gaps with {gf:.0%} of each variable removed (gap length CMAPSS {gc}, MetroPT {gm}).
* Models (settings fixed in development, no tuning on these data): KNN (k = 5), SAITS (development configuration), the
  DTIN transformer (internal normalization off); linear regression on time as a reference.
* Metric: RMSE at the removed positions (standardized scale), averaged over seeds.

## Hypotheses and decision rules
Cells = dataset x condition x model = 24; oracle = min(mean RMSE global, mean RMSE seq) in the cell.
* H1 (primary): mean RMSE dual <= {w} x oracle in at least {ws:.0%} of the cells.
* H2 (secondary): the geometric mean over the cells of RMSE dual / RMSE seq <= {g}, and the same for dual / global.
* H3 (secondary): RMSE dual / oracle <= {wo} in every cell.
* Reported descriptively: per-seed and per-unit results, guard activation, the weights, linear regression, and the
  downstream tasks below (no decision rule).

## Downstream tasks (descriptive)
Features per window: mean, standard deviation, minimum, maximum, last-minus-first per variable; standardization +
logistic regression (class-balanced, max_iter 2000), trained on complete training windows, evaluated on imputed and on
complete test windows (AUROC).
* CMAPSS: early life (window ends within the first {ec} cycles) versus late life (window starts within H cycles of
  the last cycle), H in {lc}; a window meeting both is excluded.
* MetroPT: windows overlapping a documented air-leak failure versus the others (seeds whose test or training set has
  no failure window are reported as missing).
"""


# ------------------------------------------------------------------------------------------- data
def find(root, pattern):
    hits = sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True))
    if not hits: sys.exit(f"{pattern} not found under {root}; run: python step5_confirm.py fetch")
    return hits[0]


def load_cmapss():
    W, meta = [], []
    for k, sub in enumerate(CMAPSS_SUBSETS, 1):
        a = pd.read_csv(find(CMAPSS_DIR, f"train_{sub}.txt"), sep=r"\s+", header=None).to_numpy(np.float64)
        assert a.shape[1] == 26, a.shape
        for u in np.unique(a[:, 0]).astype(int):
            r = a[a[:, 0] == u]; r = r[np.argsort(r[:, 1])]
            assert (np.diff(r[:, 1]) == 1).all(), (sub, u)
            life = int(r[-1, 1]); X = r[:, [4 + s for s in CMAPSS_SENSORS]]; L = T_LEN["CMAPSS"]
            for i in range(len(r) // L):
                c0, c1 = int(r[i * L, 1]), int(r[(i + 1) * L - 1, 1])
                W.append(X[i * L:(i + 1) * L]); meta.append(dict(unit=1000 * (k - 1) + u, subset=sub, start=c0, end=c1, life=life))
    return np.stack(W), pd.DataFrame(meta), [f"s{s}" for s in CMAPSS_SENSORS]


def load_metro():
    cands = [f for f in glob.glob(os.path.join(METRO_DIR, "**", "*.csv"), recursive=True) if "metro" in os.path.basename(f).lower()]
    if len(cands) != 1: sys.exit(f"expected one MetroPT csv under {METRO_DIR}, found {cands}; run fetch")
    hdr = pd.read_csv(cands[0], nrows=0).columns
    ts = [c for c in hdr if c.strip().lower() == "timestamp"][0]
    low = {c.strip().lower(): c for c in hdr}; use = [low[s.lower()] for s in METRO_SIGNALS]
    df = pd.read_csv(cands[0], usecols=[ts] + use)
    df[ts] = pd.to_datetime(df[ts]); df = df.set_index(ts).sort_index()[use]
    g = df.resample("1min").mean(); g = g[g.index >= g.index[0].ceil("1h")]
    fails = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in METRO_FAILURES]
    W, meta, L = [], [], T_LEN["MetroPT"]
    for k in range(len(g) // L):
        w = g.iloc[k * L:(k + 1) * L]
        if w.isna().any().any(): continue
        t0, t1 = w.index[0], w.index[-1]
        W.append(w.to_numpy(np.float64))
        meta.append(dict(unit=int((t0.normalize() - pd.Timestamp("2020-01-01")).days), start=str(t0),
                         failure=int(any(t0 <= b and t1 >= a for a, b in fails))))
    return np.stack(W), pd.DataFrame(meta), METRO_SIGNALS


LOADERS = {"CMAPSS": load_cmapss, "MetroPT": load_metro}


def cache_path(ds):
    return os.path.join(OUT, "cache", f"{ds}.npz"), os.path.join(OUT, "cache", f"{ds}_windows.csv")


def prepare():
    os.makedirs(os.path.join(OUT, "cache"), exist_ok=True)
    for ds in DATASETS:
        X, meta, cols = LOADERS[ds]()
        p, q = cache_path(ds); np.savez_compressed(p, X=X, cols=np.array(cols)); meta.to_csv(q, index=False)
        print(f"{ds}: windows {X.shape[0]}, length {X.shape[1]}, variables {X.shape[2]}, units {meta.unit.nunique()}, "
              f"non-finite values {int((~np.isfinite(X)).sum())}")
        if ds == "CMAPSS":
            for H in LATE_CYCLES:
                e, l = early_late(meta, H); print(f"   early-life windows {int(e.sum())}, late-life windows (H = {H}) {int(l.sum())}")
        else:
            print("   failure windows:", int(meta.failure.sum()))


def load_cache(ds):
    p, q = cache_path(ds)
    if not os.path.exists(p): sys.exit("run first: python step5_confirm.py prepare")
    return np.load(p)["X"], pd.read_csv(q)


def early_late(meta, H):
    early = meta.end.to_numpy() <= EARLY_CYCLES
    late = (meta.life - meta.start).to_numpy() <= H
    both = early & late
    return early & ~both, late & ~both


def split_idx(ds, meta, seed):
    rng = np.random.default_rng(sseed("step5-split", ds, seed))
    units = rng.permutation(np.sort(meta.unit.unique()))
    nt, ns = round(SPLIT_TEST * len(units)), round(SPLIT_STOP * len(units))
    test, stop = set(units[:nt]), set(units[nt:nt + ns])
    role = meta.unit.map(lambda u: "test" if u in test else "stop" if u in stop else "train").to_numpy()
    return np.flatnonzero(role == "train"), np.flatnonzero(role == "stop"), np.flatnonzero(role == "test")


_CACHE = {}


def setting(dc, ds, seed):
    if (ds, seed) in _CACHE: return _CACHE[(ds, seed)]
    if ds not in _CACHE: _CACHE[ds] = load_cache(ds)
    X, meta = _CACHE[ds]
    itr, ist, ite = split_idx(ds, meta, seed); itrain = np.r_[itr, ist]
    Xa, Xte, _ = dc.standardize(X[itrain], X[ite])
    S = dict(Xa=Xa, Xte=Xte, tr=np.arange(len(itr)), st=np.arange(len(itr), len(itrain)), itrain=itrain, ite=ite, meta=meta)
    _CACHE[(ds, seed)] = S
    return S


def masks(dc, ds, S, mech, lv, seed):
    def mk(X, split):
        rng = np.random.default_rng(sseed("step5-mask", ds, mech, lv, seed, split))
        if mech == "MCAR": return dc.mask_mcar(X.shape, lv, rng)
        if mech == "MNAR": return dc.mask_mnar(X, lv, rng, dc.CFG["mnar_quantile"], dc.CFG["mnar_p_high"])
        if mech == "GAP": return dc.mask_gaps(X.shape, GAP_L[ds], int(round(GAP_FRAC * X.shape[1])), rng)
        raise ValueError(mech)
    return mk(S["Xa"], "train"), mk(S["Xte"], "test")


# ------------------------------------------------------------------------------------------- frozen method
def seq_stats(X_nan):
    v = ~np.isnan(X_nan); c = v.sum(1, keepdims=True).astype(float); x = np.nan_to_num(X_nan)
    m = (x * v).sum(1, keepdims=True) / np.maximum(c, 1)
    s = np.sqrt((((x - m) * v) ** 2).sum(1, keepdims=True) / np.maximum(c, 1) + EPS)
    return np.where(c > 0, m, 0.0), np.where(c > 1, s, 1.0)


def vis_stats(X_nan):
    v = ~np.isnan(X_nan); c = v.sum(1).astype(float); x = np.nan_to_num(X_nan)
    m = (x * v).sum(1) / np.maximum(c, 1)
    var = (((x - m[:, None, :]) * v) ** 2).sum(1) / np.maximum(c, 1)
    return m, 0.5 * np.log(var + EPS), c


def dual_weights(ys, gs, ss, d_st, D):
    """per-variable convex weight on the global imputation, minimizing the validation MSE (Step 4: comb_var)"""
    def conv(E, F):
        Eg, Es, C = (E * E).mean(), (F * F).mean(), (E * F).mean(); den = Eg + Es - 2 * C
        return 0.5 if den <= 1e-12 else float(np.clip((Es - C) / den, 0, 1))
    eg, es = gs - ys, ss - ys
    pooled = conv(eg, es); a = np.full(D, pooled)
    for d in range(D):
        k = d_st == d
        if k.sum() > 0: a[d] = conv(eg[k], es[k])
    return a


def window_guard(Xtr_m_train, Xte_m, kappa=KAPPA):
    """True for test windows with any variable outside the training range by more than kappa range widths"""
    m_tr, l_tr, c_tr = vis_stats(Xtr_m_train); ok = c_tr >= 2
    mlo, mhi = np.where(ok, m_tr, np.inf).min(0), np.where(ok, m_tr, -np.inf).max(0)
    llo, lhi = np.where(ok, l_tr, np.inf).min(0), np.where(ok, l_tr, -np.inf).max(0)
    m, l, c = vis_stats(Xte_m)
    exm = np.maximum(mlo - m, m - mhi) / np.maximum(mhi - mlo, RANGE_FLOOR)
    exl = np.maximum(llo - l, l - lhi) / np.maximum(lhi - llo, RANGE_FLOOR)
    ex = np.where(c >= 2, np.maximum(exm, exl), -np.inf).max(1)
    return ex > kappa


# ------------------------------------------------------------------------------------------- jobs
def strict_mode(strict=True):
    import torch
    torch.use_deterministic_algorithms(True, warn_only=not strict)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    for f, val in (("enable_flash_sdp", False), ("enable_mem_efficient_sdp", False), ("enable_math_sdp", True)):
        if hasattr(torch.backends.cuda, f): getattr(torch.backends.cuda, f)(val)


def modules():
    import dtin_core as dc
    dc.init(os.path.join("dtin_outputs", "config.json"))
    import dtin_v2 as v2, dtin_v2_eval as ev
    return dc, v2, ev


def job_list():
    lr, knn, heavy = [], [], []
    for ds in DATASETS:
        for mech, lv in CONDITIONS:
            for seed in SEEDS:
                base = dict(dataset=ds, mech=mech, level=lv, seed=seed)
                lr.append(dict(base, method="Linear regression", norm="global"))
                for mdl in MODELS:
                    for norm in NORMS:
                        (knn if mdl == "KNN" else heavy).append(dict(base, method=mdl, norm=norm))
    jobs = lr + knn + heavy
    for j in jobs:
        j["id"] = f'{j["dataset"]}|{j["mech"]}|{j["level"]}|{j["seed"]}|{j["method"]}|{j["norm"]}'
    return jobs


def res_path(j, ext):
    return os.path.join(OUT, "results", hashlib.md5(j["id"].encode()).hexdigest() + ext)


def run_job(j, dc, v2, ev):
    import torch
    ds, mech, lv, seed, method, norm = j["dataset"], j["mech"], j["level"], j["seed"], j["method"], j["norm"]
    S = setting(dc, ds, seed); Xa, Xte, tr, st = S["Xa"], S["Xte"], S["tr"], S["st"]
    Mtr, Mte = masks(dc, ds, S, mech, lv, seed)
    Xtr_m, Xte_m = dc.apply_mask(Xa, Mtr), dc.apply_mask(Xte, Mte)
    Xst_m = Xtr_m[st]; nst = np.setdiff1d(np.arange(len(Xa)), st)
    s = sseed("step5-init", ds, mech, lv, seed, method); info = {}
    if method == "Linear regression":
        imp_te, imp_st = dc.impute_lr_time(Xte_m), dc.impute_lr_time(Xst_m)
    else:
        if norm == "global":
            Ztr, Zte, Zst_true, Zst_m = Xtr_m, Xte_m, Xa[st], Xst_m
            back_te = back_st = (lambda Z: Z)
        else:
            (mtr, s_tr), (mte, s_te), (mst, s_st) = seq_stats(Xtr_m), seq_stats(Xte_m), seq_stats(Xst_m)
            Ztr, Zte, Zst_true, Zst_m = (Xtr_m - mtr) / s_tr, (Xte_m - mte) / s_te, (Xa[st] - mst) / s_st, (Xst_m - mst) / s_st
            back_te = lambda Z: Z * s_te + mte; back_st = lambda Z: Z * s_st + mst
        if method == "KNN":
            Zte_i = dc.impute_knn(Ztr, Zte, dc.CFG["knn_k"]); Zst_i = dc.impute_knn(Ztr[nst], Zst_m, dc.CFG["knn_k"])
        elif method == "SAITS":
            Zall, val = ev.pypots_fit_impute("SAITS", ev.select_saits()["cfg"], Ztr[tr], Zst_true, Mtr[st],
                                             np.concatenate([Zte, Zst_m]), s)
            Zte_i, Zst_i = Zall[:len(Zte)], Zall[len(Zte):]; info["val_stop_rmse_model_space"] = val
        elif method == "Transformer":
            model = v2.DTINv2Imputer(v2.imputation_config(**TRANSFORMER), seed=s).fit(Ztr[tr], Zst_true, Mtr[st])
            Zte_i, Zst_i = model.impute(Zte), model.impute(Zst_m); del model
        else:
            raise ValueError(method)
        imp_te, imp_st = back_te(Zte_i), back_st(Zst_i)
    imp_te = np.where(np.isnan(Xte_m), imp_te, Xte_m); imp_st = np.where(np.isnan(Xst_m), imp_st, Xst_m)
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    np.savez_compressed(res_path(j, ".npz"), te=imp_te[Mte].astype(np.float32), st=imp_st[Mtr[st]].astype(np.float32))
    return dict(j, rmse=dc.rmse_at(Xte, imp_te, Mte), rmse_val=dc.rmse_at(Xa[st], imp_st, Mtr[st]), **info)


def run(worker=0, n_workers=1):
    plan = os.path.join(OUT, "PLAN_step5.md")
    if not os.path.exists(plan): sys.exit("Write and register the plan first: python step5_confirm.py plan")
    sha = script_sha()
    if sha not in open(plan, encoding="utf-8").read():
        print("WARNING: this script differs from the version recorded in PLAN_step5.md; document the change as an amendment.")
    os.makedirs(os.path.join(OUT, "results"), exist_ok=True); os.makedirs(os.path.join(OUT, "errors"), exist_ok=True)
    for ds in DATASETS: load_cache(ds)
    dc, v2, ev = modules(); import torch
    jobs = [j for i, j in enumerate(job_list()) if i % n_workers == worker]
    for i, j in enumerate(jobs, 1):
        p = res_path(j, ".json")
        if os.path.exists(p): continue
        print(f"[{i}/{len(jobs)}] {j['id']}", flush=True); t0 = time.time(); res = None
        for strict in (True, False):
            strict_mode(strict)
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


def fetch():
    import urllib.request
    for key, d in (("cmapss", CMAPSS_DIR), ("metropt3", METRO_DIR)):
        os.makedirs(d, exist_ok=True); z = os.path.join(d, f"{key}.zip")
        if not os.path.exists(z):
            print("downloading", URLS[key]); urllib.request.urlretrieve(URLS[key], z + ".part"); os.replace(z + ".part", z)
        def extract(zf):
            for i in zf.infolist():
                b = os.path.basename(i.filename)
                if b.lower().endswith(".zip"): extract(zipfile.ZipFile(io.BytesIO(zf.read(i))))
                elif b.lower().endswith((".txt", ".csv")): open(os.path.join(d, b), "wb").write(zf.read(i))
        extract(zipfile.ZipFile(z))
    for s in CMAPSS_SUBSETS: find(CMAPSS_DIR, f"train_{s}.txt")
    print("data ready:", sorted(os.listdir(CMAPSS_DIR)), sorted(os.listdir(METRO_DIR)))


# ------------------------------------------------------------------------------------------- report
def feats(X):
    return np.concatenate([X.mean(1), X.std(1), X.min(1), X.max(1), X[:, -1] - X[:, 0]], axis=1)


def _clf():
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))


def downstream(ds, S, imputed):
    """AUROC per task for each imputed test array (dict name -> array) and for the complete test windows"""
    from sklearn.metrics import roc_auc_score
    meta = S["meta"]; mtr, mte = meta.iloc[S["itrain"]].reset_index(drop=True), meta.iloc[S["ite"]].reset_index(drop=True)
    tasks = {}
    if ds == "CMAPSS":
        for H in LATE_CYCLES:
            e_tr, l_tr = early_late(mtr, H); e_te, l_te = early_late(mte, H)
            tasks[f"auc_late{H}"] = (e_tr | l_tr, l_tr.astype(int), e_te | l_te, l_te.astype(int))
    else:
        y_tr, y_te = mtr.failure.to_numpy(), mte.failure.to_numpy()
        tasks["auc_failure"] = (np.ones(len(mtr), bool), y_tr, np.ones(len(mte), bool), y_te)
    out = {}
    Ftr = feats(S["Xa"])
    for t, (str_, ytr, ste, yte) in tasks.items():
        ytr, yte = ytr[str_], yte[ste]
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            for nm in list(imputed) + ["complete"]: out[(t, nm)] = np.nan
            continue
        clf = _clf().fit(Ftr[str_], ytr)
        for nm, X in list(imputed.items()) + [("complete", S["Xte"])]:
            out[(t, nm)] = float(roc_auc_score(yte, clf.predict_proba(feats(X)[ste])[:, 1]))
    return out


def gmean(x):
    x = np.asarray(x, float); return float(np.exp(np.log(x).mean()))


def report():
    dc, _, _ = modules()
    jobs = {j["id"]: j for j in job_list()}
    rows, drows, grows = [], [], []
    for ds in DATASETS:
        for mech, lv in CONDITIONS:
            cond = f"{mech} {lv:g}"
            for seed in SEEDS:
                S = setting(dc, ds, seed); Mtr, Mte = masks(dc, ds, S, mech, lv, seed)
                Xtr_m, Xte_m = dc.apply_mask(S["Xa"], Mtr), dc.apply_mask(S["Xte"], Mte)
                y = S["Xte"][Mte]; n, _, d = np.nonzero(Mte); ys = S["Xa"][S["st"]][Mtr[S["st"]]]
                _, _, d_st = np.nonzero(Mtr[S["st"]]); D = S["Xa"].shape[2]
                guard = window_guard(Xtr_m[S["tr"]], Xte_m)
                grows.append(dict(dataset=ds, cond=cond, seed=seed, guard_share=float(guard.mean())))
                unit = S["meta"].unit.to_numpy()[S["ite"]][n]
                imputed = {}
                def add(label, model, variant, pred):
                    e2 = (pred - y) ** 2
                    u = pd.DataFrame(dict(u=unit, e2=e2)).groupby("u").e2.mean() ** 0.5
                    rows.append(dict(dataset=ds, cond=cond, seed=seed, model=model, variant=variant, rmse=float(np.sqrt(e2.mean())), unit_median=float(u.median())))
                    X = Xte_m.copy(); X[Mte] = pred; imputed[label] = X
                def load(method, norm):
                    p = res_path(jobs[f"{ds}|{mech}|{lv}|{seed}|{method}|{norm}"], ".npz")
                    return np.load(p) if os.path.exists(p) else None
                z = load("Linear regression", "global")
                if z is not None: add("LR", "Linear regression", "-", z["te"].astype(float))
                for mdl in MODELS:
                    zg, zs = load(mdl, "global"), load(mdl, "seq")
                    if zg is None or zs is None: continue
                    g, s_ = zg["te"].astype(float), zs["te"].astype(float)
                    a = dual_weights(ys, zg["st"].astype(float), zs["st"].astype(float), d_st, D)
                    an = np.where(guard[n], 0.0, a[d])
                    add(f"{mdl}-global", mdl, "global", g); add(f"{mdl}-seq", mdl, "seq", s_); add(f"{mdl}-dual", mdl, "dual", an * g + (1 - an) * s_)
                    rows[-1]["weight_mean"] = float(a.mean())
                for (t, nm), v in downstream(ds, S, imputed).items():
                    drows.append(dict(dataset=ds, cond=cond, seed=seed, task=t, label=nm, auc=v))
    r, dd, gg = pd.DataFrame(rows), pd.DataFrame(drows), pd.DataFrame(grows)
    if r.empty: sys.exit("no results yet")
    nb = r[r.model != "Linear regression"]
    m = nb.groupby(["dataset", "cond", "model", "variant"]).rmse.mean().unstack()
    need = len(DATASETS) * len(CONDITIONS) * len(MODELS); complete = len(m.dropna()) == need
    m = m.dropna(); m["oracle"] = m[["global", "seq"]].min(axis=1); m["ratio"] = m.dual / m.oracle
    h1_n = int((m.ratio <= WITHIN).sum()); h1 = h1_n >= WITHIN_SHARE * need
    g_seq, g_glob = gmean(m.dual / m.seq), gmean(m.dual / m["global"]); h2 = g_seq <= GAIN and g_glob <= GAIN
    h3 = bool((m.ratio <= WORST).all())
    verdict = lambda ok: ("supported" if ok else "not supported") if complete else f"no decision yet ({len(m)} of {need} cells)"
    sha = sorted({json.load(open(p)).get("script_sha256") for p in glob.glob(os.path.join(OUT, "results", "*.json"))})
    L = ["# Step 5 report (confirmation)", "", f"Script versions in the results: {sha}", "",
         "## Mean RMSE per cell and ratio of dual to the better single normalization", "", "```",
         m.join(r[r.model == "Linear regression"].groupby(["dataset", "cond"]).rmse.mean().rename("LR"), on=["dataset", "cond"])
          .round(4).to_string(), "```", "",
         f"* H1: cells with dual <= {WITHIN} x oracle: {h1_n} of {need} (rule: at least {WITHIN_SHARE:.0%}) -> **{verdict(h1)}**",
         f"* H2: geometric mean dual / seq = {g_seq:.3f}, dual / global = {g_glob:.3f} (rule: both <= {GAIN}) -> **{verdict(h2)}**",
         f"* H3: worst ratio to the oracle = {m.ratio.max():.3f} (rule: every cell <= {WORST}) -> **{verdict(h3)}**", ""]
    ps = nb.pivot_table(index=["dataset", "cond", "model", "seed"], columns="variant", values="rmse")
    ps["ratio"] = ps.dual / ps[["global", "seq"]].min(axis=1)
    L += ["## Seed level: dual / better single normalization in the same seed", "", "```",
          ps.groupby(["dataset", "cond", "model"]).ratio.describe()[["min", "50%", "max"]].round(3).to_string(), "```", ""]
    um = nb.groupby(["dataset", "cond", "model", "variant"]).unit_median.mean().unstack()
    L += ["## Median over test units of the per-unit RMSE (mean over seeds)", "", "```", um.round(4).to_string(), "```", ""]
    L += ["## Window guard: share of test windows using seq only", "", "```",
          gg.groupby(["dataset", "cond"]).guard_share.mean().round(4).to_string(), "```", ""]
    if "weight_mean" in r:
        L += ["## Mean dual weight on global (validation estimate)", "", "```",
              r.dropna(subset=["weight_mean"]).groupby(["dataset", "cond", "model"]).weight_mean.mean().round(2).unstack().to_string(), "```", ""]
    if not dd.empty:
        L += ["## Downstream AUROC (descriptive; mean over seeds with both classes present)", "", "```",
              dd.groupby(["dataset", "task", "cond", "label"]).auc.mean().unstack("label").round(3).to_string(), "```", ""]
    jr = [json.load(open(p)) for p in glob.glob(os.path.join(OUT, "results", "*.json"))]
    if jr: L += [f"Runs without strict determinism: {int((~pd.DataFrame(jr).deterministic.astype(bool)).sum())}"]
    txt = "\n".join(L); open(os.path.join(OUT, "REPORT_step5.md"), "w", encoding="utf-8").write(txt)
    r.to_csv(os.path.join(OUT, "step5_runs.csv"), index=False); dd.to_csv(os.path.join(OUT, "step5_downstream.csv"), index=False)
    if jr: pd.DataFrame(jr).to_csv(os.path.join(OUT, "step5_jobs.csv"), index=False)
    print(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["plan", "fetch", "prepare", "run", "status", "report"])
    ap.add_argument("--worker", type=int, default=0); ap.add_argument("--n-workers", type=int, default=1)
    a = ap.parse_args()
    if a.cmd == "plan":
        os.makedirs(OUT, exist_ok=True); p = os.path.join(OUT, "PLAN_step5.md")
        if os.path.exists(p): sys.exit(f"{p} already exists; it must not be changed after results exist")
        txt = PLAN.format(sha=script_sha(), sens=", s".join(map(str, CMAPSS_SENSORS)), tc=T_LEN["CMAPSS"], msig=", ".join(METRO_SIGNALS),
                          tm=T_LEN["MetroPT"], seeds=SEEDS, st=SPLIT_TEST, ss=SPLIT_STOP, gf=GAP_FRAC, gc=GAP_L["CMAPSS"],
                          gm=GAP_L["MetroPT"], w=WITHIN, ws=WITHIN_SHARE, g=GAIN, wo=WORST, ec=EARLY_CYCLES, lc=LATE_CYCLES)
        open(p, "w", encoding="utf-8").write(txt + f"\nWritten: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        print(open(p, encoding="utf-8").read()); print("SHA-256 of the plan:", hashlib.sha256(open(p, "rb").read()).hexdigest())
    elif a.cmd == "fetch": fetch()
    elif a.cmd == "prepare": prepare()
    elif a.cmd == "run": run(a.worker, a.n_workers)
    elif a.cmd == "status": status()
    else: report()
